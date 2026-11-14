"""
backend/app/routers/prescriptions.py
Prescription image upload and OCR processing router.

Handles:
- Dynamic readiness inspection of registered OCR engines
- Chunk-streamed validation with strict 10 MB limit
- Privacy-first in-memory processing (no persistent storage by default)
- Direct single-step OCR or two-step upload-and-process workflow
"""

from __future__ import annotations

import io
import os
import uuid
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from gemma_integration import GemmaIntegrationError
from backend.app.services.image_processing import (
    ImageProcessor,
    ProcessedImageMetadata,
    read_upload_stream,
)
from backend.app.services.gemini_second_opinion import (
    GeminiSecondOpinionError,
    GeminiRequestTimeoutError,
    compare_with_gemini,
    gemini_is_configured,
)
from backend.app.services.medicine_extraction import (
    MedicineCandidateExtractor,
    MedicineExtractionResult,
    extract_gemma_medicine_candidates,
)
from backend.app.services.review_workflow import (
    DownstreamEligibleCandidate,
    InvalidReviewTransitionError,
    ReviewAction,
    ReviewSession,
    ReviewSessionStore,
    get_eligible_candidates_for_downstream,
)
from backend.app.services.medicine_normalization import (
    InvalidNormalizationTransitionError,
    NormalizationAction,
    NormalizationRecord,
    NormalizationSessionStore,
    TerminologyProvider,
    RxNavApproximateProvider,
    TerminologyProviderUnavailable,
    normalize_eligible_candidate,
)
from backend.app.services.drug_interactions import (
    InteractionProvider,
    InteractionReport,
    InteractionReportStore,
    UnavailableInteractionProvider,
    check_accepted_normalizations,
)
from backend.app.services.ocr import (
    EngineReadiness,
    OCRExtractionResult,
    get_ocr_engine,
    list_available_engines,
)

router = APIRouter(prefix="/prescriptions", tags=["prescriptions"])
DEFAULT_OCR_ENGINE = os.getenv("OCR_ENGINE", "gemma").strip().lower() or "gemma"
image_processor = ImageProcessor()
medicine_extractor = MedicineCandidateExtractor()
review_session_store = ReviewSessionStore()
normalization_session_store = NormalizationSessionStore()
terminology_provider: TerminologyProvider = RxNavApproximateProvider()
interaction_report_store = InteractionReportStore()
interaction_provider: InteractionProvider = UnavailableInteractionProvider()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EngineListResponse(BaseModel):
    """Response containing readiness information for all registered OCR engines."""

    engines: list[EngineReadiness]
    count: int
    gemini_second_opinion_configured: bool = False


class PrescriptionUploadResponse(BaseModel):
    """Response after a prescription image is validated and registered."""

    status: str = "success"
    message: str
    image: ProcessedImageMetadata


class GeminiSecondOpinionResponse(BaseModel):
    """Separate status and text for the optional Gemini comparison."""

    status: str
    model: str | None = None
    text: str | None = None
    processing_time_ms: float | None = None
    error: str | None = None


class OCRProcessingResponse(BaseModel):
    """Response returned upon OCR extraction attempt."""

    status: str = "success"
    image_id: str
    result: OCRExtractionResult
    gemini_second_opinion: GeminiSecondOpinionResponse | None = None
    medicine_extraction: MedicineExtractionResult | None = None


class MedicineExtractionRequest(BaseModel):
    """Request payload containing raw OCR text for candidate extraction."""

    ocr_text: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description="Raw OCR output text from prescription scanner",
        examples=["Rx: Amoxicillin 500mg capsules PO TID x 7 days\nTab Paracetamol 650mg PRN"],
    )


class CandidateReviewRequest(BaseModel):
    """One explicit action for one candidate in a review session."""

    model_config = ConfigDict(extra="forbid")

    action: ReviewAction
    corrected_text: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_action_fields(self) -> CandidateReviewRequest:
        if self.action == ReviewAction.CORRECT:
            if self.corrected_text is None or not self.corrected_text.strip():
                raise ValueError("A non-empty corrected_text is required for action 'correct'.")
        elif self.corrected_text is not None:
            raise ValueError("corrected_text is only valid for action 'correct'.")
        return self


class NormalizationDecisionRequest(BaseModel):
    """Explicit accept/reject/unresolved decision for a proposed concept."""

    model_config = ConfigDict(extra="forbid")

    action: NormalizationAction
    selected_concept_id: str | None = Field(default=None, min_length=1, max_length=200)
    acknowledge_attribute_mismatches: bool = False

    @model_validator(mode="after")
    def validate_decision_fields(self) -> NormalizationDecisionRequest:
        if self.action == NormalizationAction.ACCEPT:
            if self.selected_concept_id is None:
                raise ValueError("Accept requires selected_concept_id.")
        elif self.selected_concept_id is not None:
            raise ValueError("selected_concept_id is only valid for action 'accept'.")
        if self.action != NormalizationAction.ACCEPT and self.acknowledge_attribute_mismatches:
            raise ValueError("Mismatch acknowledgement is only valid for action 'accept'.")
        return self


class InteractionCheckRequest(BaseModel):
    """Request an interaction lookup using Phase 5 normalization record IDs only."""

    model_config = ConfigDict(extra="forbid")

    normalization_ids: list[UUID] = Field(min_length=1, max_length=20)

    @field_validator("normalization_ids")
    @classmethod
    def reject_duplicate_normalization_ids(cls, values: list[UUID]) -> list[UUID]:
        if len(set(values)) != len(values):
            raise ValueError("Duplicate normalization record IDs are not allowed.")
        return values


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/engines",
    response_model=EngineListResponse,
    summary="List available OCR engines and readiness",
)
async def list_engines() -> EngineListResponse:
    """Return all available OCR engines and their system readiness status."""
    engines = list_available_engines()
    return EngineListResponse(
        engines=engines,
        count=len(engines),
        gemini_second_opinion_configured=gemini_is_configured(),
    )


@router.post(
    "/upload",
    response_model=PrescriptionUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register prescription image",
)
async def upload_prescription(
    file: UploadFile = File(..., description="Prescription image file"),
) -> PrescriptionUploadResponse:
    """
    Upload and register a prescription image.
    Enforces a strict 10 MB streaming limit and performs deep image header & dimension checks.
    By default, images are retained only in memory for privacy.
    """
    filename = file.filename or "prescription.png"
    content = await read_upload_stream(file)

    try:
        metadata = image_processor.process_and_register_image(content, filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return PrescriptionUploadResponse(
        message=(
            "Prescription image validated and registered in memory (zero persistent storage)."
            if not metadata.is_persisted
            else "Prescription image validated and stored locally."
        ),
        image=metadata,
    )


@router.post(
    "/ocr",
    response_model=OCRProcessingResponse,
    summary="Run OCR text extraction on a prescription",
)
async def run_ocr(
    file: UploadFile | None = File(None, description="Direct prescription image upload (in-memory OCR)"),
    image_id: UUID | None = Form(None, description="ID of a previously uploaded prescription"),
    engine: str = Form(
        DEFAULT_OCR_ENGINE,
        description="OCR engine: local 'paddle', configured 'gemini', or Gemma API/Ollama 'gemma'",
    ),
    include_gemini_second_opinion: bool = Form(
        False,
        description="Explicitly send the image and PaddleOCR text to Gemini for comparison.",
    ),
    include_medicine_extraction: bool = Form(
        False,
        description="Extract medicine candidates from OCR text and create a review session.",
    ),
) -> OCRProcessingResponse:
    """
    Execute OCR text extraction.
    Supports either direct image upload (in-memory, single-step) or an image_id from a prior upload.
    If the selected engine dependencies are not installed, returns HTTP 503 with exact instructions.
    Gemini OCR is an independently selected engine; its use sends the image to Google.
    Gemma uses the Gemma API when configured and falls back to local Ollama.
    The separate Gemini second opinion remains off by default for PaddleOCR.
    Medicine extraction can optionally run immediately after OCR and create the
    normal mandatory-review session.
    """
    if file is None and image_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file' (direct upload) or 'image_id' must be provided.",
        )
    if file is not None and image_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either 'file' or 'image_id', not both.",
        )

    if include_gemini_second_opinion and not gemini_is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini second opinion was requested but GEMINI_API_KEY is not configured "
                "in the backend environment."
            ),
        )
    if include_gemini_second_opinion and engine.lower() != "paddle":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini second opinion is only available alongside PaddleOCR.",
        )

    try:
        selected_engine = get_ocr_engine(engine)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    target_id = str(image_id or uuid.uuid4())

    try:
        if file is not None:
            content = await read_upload_stream(file)
            validation = image_processor.validate_image_bytes(content, file.filename or "")
            if not validation.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=validation.error or "Invalid image file",
                )
            with Image.open(io.BytesIO(content)) as opened_image:
                img = opened_image.copy()
        else:
            img = image_processor.load_image(target_id)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prescription image '{target_id}' not found in cache or storage.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prescription image could not be loaded: {exc}",
        ) from exc

    try:
        readiness = selected_engine.check_readiness()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OCR engine readiness check failed: {type(exc).__name__}.",
        ) from exc

    if not readiness.is_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": f"OCR engine '{engine}' is not currently ready on this system.",
                "instructions": readiness.instructions_if_unavailable,
                "details": readiness.details,
            },
        )

    try:
        preprocessed = image_processor.preprocess_for_ocr(img)
        # CPU OCR and model initialization are blocking operations. Keep them
        # off the ASGI event loop so health/API requests remain responsive.
        extraction_result = await run_in_threadpool(
            selected_engine.extract_text,
            preprocessed,
        )
    except GeminiRequestTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "message": str(exc),
                "engine": engine,
                "error_type": type(exc).__name__,
            },
        ) from exc
    except GeminiSecondOpinionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Gemini could not complete the OCR request.",
                "engine": engine,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    except GemmaIntegrationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Gemma could not complete prescription image extraction.",
                "engine": engine,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "OCR engine inference failed; no OCR result was produced.",
                "engine": engine,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    medicine_extraction = None
    if include_medicine_extraction and extraction_result.raw_text.strip():
        try:
            if selected_engine.name == "gemma":
                structured_data = extraction_result.structured_data or {}
                medicines = structured_data.get("medicines", [])
                if not isinstance(medicines, list):
                    raise ValueError("Gemma medicine details are not a list.")
                medicine_extraction = extract_gemma_medicine_candidates(
                    extraction_result.raw_text,
                    medicines,
                )
            else:
                medicine_extraction = medicine_extractor.extract(extraction_result.raw_text)
            review_session = review_session_store.create_session(
                extraction_result.raw_text,
                medicine_extraction.candidates,
            )
            medicine_extraction.review_id = review_session.review_id
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Medicine extraction could not use the OCR output: {exc}",
            ) from exc

    gemini_result = None
    if include_gemini_second_opinion:
        try:
            model, comparison_text, processing_time_ms = await run_in_threadpool(
                compare_with_gemini,
                preprocessed,
                extraction_result.raw_text,
            )
            gemini_result = GeminiSecondOpinionResponse(
                status="success",
                model=model,
                text=comparison_text,
                processing_time_ms=processing_time_ms,
            )
        except GeminiSecondOpinionError as exc:
            gemini_result = GeminiSecondOpinionResponse(
                status="error",
                error=str(exc),
            )

    return OCRProcessingResponse(
        image_id=target_id,
        result=extraction_result,
        gemini_second_opinion=gemini_result,
        medicine_extraction=medicine_extraction,
    )


@router.post(
    "/extract-medicines",
    response_model=MedicineExtractionResult,
    summary="Extract candidate medicine mentions and attributes from OCR text",
)
async def extract_medicines(
    payload: MedicineExtractionRequest,
) -> MedicineExtractionResult:
    """
    Extract candidate medicine names, strengths, dosage forms, and administration
    frequencies from raw OCR text.

    Scope & Safety Notice:
    - Rule-based & dictionary extraction only.
    - Preserves verbatim OCR tokens without autocorrection or clinical inference.
    - Every candidate starts in the explicit 'pending_review' state.
    """
    cleaned_text = payload.ocr_text.strip()
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided OCR text is empty or contains only whitespace.",
        )

    try:
        extraction_result = medicine_extractor.extract(payload.ocr_text)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Candidate extraction failed: {exc}",
        ) from exc

    review_session = review_session_store.create_session(
        payload.ocr_text,
        extraction_result.candidates,
    )
    extraction_result.review_id = review_session.review_id
    return extraction_result


@router.get(
    "/review/{review_id}",
    response_model=ReviewSession,
    summary="Get candidate review session",
)
async def get_review_session(review_id: UUID) -> ReviewSession:
    """Return original evidence and current review states for one in-memory session."""
    session = review_session_store.get_session(str(review_id))
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review session not found or expired.",
        )
    return session


@router.post(
    "/review/{review_id}/candidates/{candidate_id}",
    response_model=ReviewSession,
    summary="Apply an explicit review action to one candidate",
)
async def review_candidate(
    review_id: UUID,
    candidate_id: UUID,
    payload: CandidateReviewRequest,
) -> ReviewSession:
    """Record a single candidate decision; reviewed candidates cannot be changed again."""
    try:
        session, _ = review_session_store.apply_action(
            review_id=str(review_id),
            candidate_id=str(candidate_id),
            action=payload.action,
            corrected_text=payload.corrected_text,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review session or candidate not found.",
        ) from exc
    except InvalidReviewTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return session


@router.post(
    "/review/{review_id}/candidates/{candidate_id}/normalization",
    response_model=NormalizationRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Request normalization for an eligible reviewed medicine candidate",
)
async def request_medicine_normalization(
    review_id: UUID,
    candidate_id: UUID,
) -> NormalizationRecord:
    """Run provider lookup only after retrieving the candidate from the Phase 4 gate."""
    review_session = review_session_store.get_session(str(review_id))
    if review_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review session not found or expired.",
        )

    review_candidate = review_session.candidates.get(str(candidate_id))
    if review_candidate is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate not found in this review session.",
        )

    eligible_candidate: DownstreamEligibleCandidate | None = next(
        (
            candidate
            for candidate in get_eligible_candidates_for_downstream(review_session)
            if candidate.candidate_id == str(candidate_id)
        ),
        None,
    )
    if eligible_candidate is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Candidate has not passed the Phase 4 review eligibility gate.",
        )
    if eligible_candidate.candidate_type != "possible_medicine_name":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only eligible medicine-name candidates can be normalized.",
        )

    try:
        record = normalize_eligible_candidate(
            review_id=str(review_id),
            candidate=eligible_candidate,
            provider=terminology_provider,
        )
        return normalization_session_store.create(record)
    except InvalidNormalizationTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except TerminologyProviderUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get(
    "/normalization/{normalization_id}",
    response_model=NormalizationRecord,
    summary="Retrieve a medicine normalization result",
)
async def get_medicine_normalization(normalization_id: UUID) -> NormalizationRecord:
    """Retrieve proposed concepts, evidence, and separate normalization review state."""
    record = normalization_session_store.get(str(normalization_id))
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Normalization record not found or expired.",
        )
    return record


@router.post(
    "/normalization/{normalization_id}/decision",
    response_model=NormalizationRecord,
    summary="Record an explicit normalization decision",
)
async def decide_medicine_normalization(
    normalization_id: UUID,
    payload: NormalizationDecisionRequest,
) -> NormalizationRecord:
    """Accept one proposed concept or explicitly reject/leave the result unresolved."""
    try:
        return normalization_session_store.decide(
            normalization_id=str(normalization_id),
            action=payload.action,
            selected_concept_id=payload.selected_concept_id,
            acknowledge_attribute_mismatches=payload.acknowledge_attribute_mismatches,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Normalization record not found or expired.",
        ) from exc
    except InvalidNormalizationTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.post(
    "/interactions/check",
    response_model=InteractionReport,
    status_code=status.HTTP_201_CREATED,
    summary="Check source-backed interaction evidence for accepted normalized concepts",
)
async def check_drug_interactions(
    payload: InteractionCheckRequest,
) -> InteractionReport:
    """Create a report using only records accepted during Phase 5 normalization review."""
    records: list[NormalizationRecord] = []
    for normalization_id in payload.normalization_ids:
        record = normalization_session_store.get(str(normalization_id))
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more normalization records were not found or have expired.",
            )
        records.append(record)

    try:
        report = check_accepted_normalizations(records, interaction_provider)
        return interaction_report_store.create(report)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get(
    "/interactions/reports/{report_id}",
    response_model=InteractionReport,
    summary="Retrieve a source-backed interaction evidence report",
)
async def get_interaction_report(report_id: UUID) -> InteractionReport:
    """Retrieve pair-level results, provenance, decisions, and coverage."""
    report = interaction_report_store.get(str(report_id))
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction report not found or expired.",
        )
    return report
