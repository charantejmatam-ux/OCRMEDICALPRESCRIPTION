"""Provider-neutral, review-gated medicine terminology normalization."""

from __future__ import annotations

import threading
import time
import uuid
import json
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.services.medicine_extraction import (
    DOSAGE_FORM_PATTERN,
    FREQUENCY_PATTERN,
    ROUTE_PATTERN,
    STRENGTH_PATTERN,
)
from backend.app.services.review_workflow import (
    DownstreamEligibleCandidate,
    ReviewStatus,
)


class NormalizationStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


class NormalizationAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MARK_AMBIGUOUS = "mark_ambiguous"
    MARK_UNMATCHED = "mark_unmatched"


class MatchMethod(str, Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    PARTIAL = "partial"


class AttributeMatch(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    NOT_COMPARABLE = "not_comparable"


class MedicineAttributes(BaseModel):
    """Verbatim prescription attributes parsed from the original evidence line."""

    strength: str | None = None
    dosage_form: str | None = None
    route: str | None = None
    frequency: str | None = None


class TerminologyConcept(BaseModel):
    """A terminology concept and source-provided provenance; never synthesized."""

    source_identifier: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=500)
    ingredient: str | None = Field(default=None, max_length=500)
    strength: str | None = Field(default=None, max_length=200)
    dosage_form: str | None = Field(default=None, max_length=200)
    route: str | None = Field(default=None, max_length=200)
    match_method: MatchMethod
    evidence: str = Field(min_length=1, max_length=2000)
    source_name: str = Field(min_length=1, max_length=200)
    source_version: str | None = Field(default=None, max_length=200)
    source_release_date: str | None = Field(default=None, max_length=100)
    attribute_match: AttributeMatch = AttributeMatch.NOT_COMPARABLE
    attribute_warnings: list[str] = Field(default_factory=list)
    match_rank: int | None = Field(default=None, ge=1)
    match_score: float | None = Field(default=None, ge=0)


class TerminologyProvider(ABC):
    """Interface for an explicitly configured terminology source."""

    provider_name: str
    source_version: str | None = None
    source_release_date: str | None = None

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether the provider has a genuine configured data source."""

    @abstractmethod
    def search(
        self,
        input_text: str,
        attributes: MedicineAttributes,
    ) -> list[TerminologyConcept]:
        """Return source-backed concepts only; never invent concepts."""


class TerminologyProviderUnavailable(RuntimeError):
    """A configured terminology service could not be reached or parsed."""


class UnavailableRxNormProvider(TerminologyProvider):
    """Offline fallback that returns no concepts rather than inventing matches."""

    provider_name = "RxNorm"
    source_version = None
    source_release_date = None

    @property
    def is_configured(self) -> bool:
        return False

    def search(
        self,
        input_text: str,
        attributes: MedicineAttributes,
    ) -> list[TerminologyConcept]:
        del input_text, attributes
        return []


class RxNavApproximateProvider(TerminologyProvider):
    """NLM RxNav approximate matching; called only after explicit review action."""

    provider_name = "RxNorm/RxNav"
    source_version = None
    source_release_date = None
    endpoint = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"

    @property
    def is_configured(self) -> bool:
        return True

    def search(
        self,
        input_text: str,
        attributes: MedicineAttributes,
    ) -> list[TerminologyConcept]:
        del attributes
        query = urlencode({"term": input_text, "maxEntries": 5})
        request = Request(
            f"{self.endpoint}?{query}",
            headers={"Accept": "application/json", "User-Agent": "PrescriptionOCRResearch/1.0"},
        )
        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise TerminologyProviderUnavailable(
                "RxNav is unavailable; no terminology result was generated."
            ) from exc

        candidates = payload.get("approximateGroup", {}).get("candidate", []) or []
        concepts: list[TerminologyConcept] = []
        for item in candidates:
            rxcui = item.get("rxcui")
            name = item.get("name")
            if not rxcui or not name:
                continue
            try:
                rank = int(item["rank"]) if item.get("rank") is not None else None
                score = float(item["score"]) if item.get("score") is not None else None
            except (TypeError, ValueError):
                rank, score = None, None
            source = item.get("source") or "RxNorm"
            concepts.append(TerminologyConcept(
                source_identifier=str(rxcui),
                name=str(name),
                ingredient=str(name),
                match_method=MatchMethod.FUZZY,
                evidence=(
                    f"RxNav approximateTerm source={source}; native ranking score="
                    f"{score}; rank={rank}. This score ranks this query's candidates; "
                    "it is not a probability of a correct medicine identification."
                ),
                source_name=self.provider_name,
                match_rank=rank,
                match_score=score,
            ))
        return concepts


class NormalizationEvidence(BaseModel):
    original_ocr_text: str
    original_candidate_text: str
    reviewer_corrected_text: str | None = None
    candidate_type: str
    start_char: int
    end_char: int
    line_number: int
    evidence_context: str
    extraction_method: str
    phase4_review_status: ReviewStatus
    phase4_review_action: str | None = None
    phase4_reviewed_at: str | None = None
    heuristic_score: float
    ocr_engine_confidence: float | None = None
    ocr_confidence_available: bool
    has_dictionary_match: bool
    has_contextual_pattern: bool
    has_spillover_or_ambiguity: bool
    normalization_input_text: str
    attributes: MedicineAttributes


class NormalizationRecord(BaseModel):
    normalization_id: str
    review_id: str
    candidate_id: str
    created_at: str
    updated_at: str
    status: NormalizationStatus
    evidence: NormalizationEvidence
    provider_name: str
    provider_configured: bool
    provider_version: str | None = None
    provider_release_date: str | None = None
    resolution_reason: str
    proposed_concepts: list[TerminologyConcept] = Field(default_factory=list)
    accepted_concept: TerminologyConcept | None = None
    decision_action: NormalizationAction | None = None
    decision_at: str | None = None
    safety_notice: str = (
        "Research prototype only. Terminology matches and review decisions do not "
        "establish medicine identity or clinical correctness."
    )


def parse_medicine_attributes(evidence_context: str) -> MedicineAttributes:
    """Extract only literal attribute spans; do not infer missing values."""
    strength = STRENGTH_PATTERN.search(evidence_context)
    dosage_form = DOSAGE_FORM_PATTERN.search(evidence_context)
    route = ROUTE_PATTERN.search(evidence_context)
    frequency = FREQUENCY_PATTERN.search(evidence_context)
    return MedicineAttributes(
        strength=strength.group() if strength else None,
        dosage_form=dosage_form.group() if dosage_form else None,
        route=route.group() if route else None,
        frequency=frequency.group() if frequency else None,
    )


def _canonical_strength(value: str) -> str:
    return "".join(value.casefold().split())


def _canonical_form(value: str) -> str:
    normalized = value.casefold().strip()
    aliases = {
        "tabs": "tablet",
        "tab": "tablet",
        "tablets": "tablet",
        "caps": "capsule",
        "cap": "capsule",
        "capsules": "capsule",
        "inj": "injection",
        "syr": "syrup",
        "susp": "suspension",
        "sol": "solution",
        "oint": "ointment",
        "crm": "cream",
        "drp": "drops",
    }
    return aliases.get(normalized, normalized)


def _compare_concept_attributes(
    concept: TerminologyConcept,
    attributes: MedicineAttributes,
) -> TerminologyConcept:
    mismatches: list[str] = []
    comparable = False

    if attributes.strength and concept.strength:
        comparable = True
        if _canonical_strength(attributes.strength) != _canonical_strength(concept.strength):
            mismatches.append(
                f"Input strength '{attributes.strength}' differs from terminology strength '{concept.strength}'."
            )

    if attributes.dosage_form and concept.dosage_form:
        comparable = True
        if _canonical_form(attributes.dosage_form) != _canonical_form(concept.dosage_form):
            mismatches.append(
                f"Input dosage form '{attributes.dosage_form}' differs from terminology form "
                f"'{concept.dosage_form}'."
            )

    return concept.model_copy(
        update={
            "attribute_match": (
                AttributeMatch.MISMATCH
                if mismatches
                else AttributeMatch.MATCH
                if comparable
                else AttributeMatch.NOT_COMPARABLE
            ),
            "attribute_warnings": mismatches,
        }
    )


def normalize_eligible_candidate(
    review_id: str,
    candidate: DownstreamEligibleCandidate,
    provider: TerminologyProvider,
) -> NormalizationRecord:
    """
    Normalize only the typed output from the Phase 4 eligibility gate.

    This function never performs string-based fallback matching. A configured
    provider supplies every concept and its source identifier and provenance.
    """
    if not isinstance(candidate, DownstreamEligibleCandidate):
        raise TypeError("Normalization requires a Phase 4 DownstreamEligibleCandidate.")
    if candidate.candidate_type != "possible_medicine_name":
        raise ValueError("Only eligible medicine-name candidates can be normalized.")
    if candidate.review_status not in {ReviewStatus.CONFIRMED, ReviewStatus.CORRECTED}:
        raise ValueError("Candidate has not passed the Phase 4 review gate.")
    if candidate.is_corrected != (candidate.review_status == ReviewStatus.CORRECTED):
        raise ValueError("Candidate review status does not match its correction state.")
    if candidate.is_corrected and not candidate.reviewer_corrected_text:
        raise ValueError("Corrected candidate has no preserved reviewer correction.")

    attributes = parse_medicine_attributes(candidate.evidence_context)
    evidence = NormalizationEvidence(
        original_ocr_text=candidate.original_ocr_text,
        original_candidate_text=candidate.original_candidate_text,
        reviewer_corrected_text=candidate.reviewer_corrected_text,
        candidate_type=candidate.candidate_type,
        start_char=candidate.start_char,
        end_char=candidate.end_char,
        line_number=candidate.line_number,
        evidence_context=candidate.evidence_context,
        extraction_method=candidate.extraction_method,
        phase4_review_status=candidate.review_status,
        phase4_review_action=(
            candidate.phase4_review_action.value
            if candidate.phase4_review_action
            else None
        ),
        phase4_reviewed_at=candidate.phase4_reviewed_at,
        heuristic_score=candidate.heuristic_score,
        ocr_engine_confidence=candidate.ocr_engine_confidence,
        ocr_confidence_available=candidate.ocr_confidence_available,
        has_dictionary_match=candidate.has_dictionary_match,
        has_contextual_pattern=candidate.has_contextual_pattern,
        has_spillover_or_ambiguity=candidate.has_spillover_or_ambiguity,
        normalization_input_text=candidate.effective_text,
        attributes=attributes,
    )
    concepts: list[TerminologyConcept] = []
    if provider.is_configured:
        concepts = [
            _compare_concept_attributes(concept, attributes)
            for concept in provider.search(candidate.effective_text, attributes)
        ]
        if any(concept.source_name != provider.provider_name for concept in concepts):
            raise ValueError("Terminology provider returned a concept with mismatched source provenance.")
        if len({concept.source_identifier for concept in concepts}) != len(concepts):
            raise ValueError("Terminology provider returned duplicate source identifiers.")

    if not provider.is_configured:
        initial_status = NormalizationStatus.UNMATCHED
        reason = "RxNorm terminology source is not configured; no concepts were generated."
    elif not concepts:
        initial_status = NormalizationStatus.UNMATCHED
        reason = "Configured terminology source returned no concepts for the reviewed input."
    elif len(concepts) > 1:
        initial_status = NormalizationStatus.AMBIGUOUS
        reason = "Multiple source-backed concepts require an explicit reviewer selection."
    else:
        initial_status = NormalizationStatus.PENDING_REVIEW
        reason = "A source-backed concept is proposed but requires explicit normalization review."

    now = datetime.now(timezone.utc).isoformat()
    return NormalizationRecord(
        normalization_id=str(uuid.uuid4()),
        review_id=review_id,
        candidate_id=candidate.candidate_id,
        created_at=now,
        updated_at=now,
        status=initial_status,
        evidence=evidence,
        provider_name=provider.provider_name,
        provider_configured=provider.is_configured,
        provider_version=provider.source_version,
        provider_release_date=provider.source_release_date,
        resolution_reason=reason,
        proposed_concepts=concepts,
    )


class InvalidNormalizationTransitionError(ValueError):
    """Raised when a normalization decision is repeated or conflicts with state."""


class NormalizationSessionStore:
    """Bounded, process-local storage for normalization results and decisions."""

    def __init__(self, max_records: int = 500, ttl_seconds: int = 3600) -> None:
        self.max_records = max_records
        self.ttl_seconds = ttl_seconds
        self._records: OrderedDict[str, tuple[float, NormalizationRecord]] = OrderedDict()
        self._by_candidate: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def create(self, record: NormalizationRecord) -> NormalizationRecord:
        with self._lock:
            self._purge_expired()
            candidate_key = (record.review_id, record.candidate_id)
            if candidate_key in self._by_candidate:
                raise InvalidNormalizationTransitionError(
                    "A normalization record already exists for this reviewed candidate."
                )
            while len(self._records) >= self.max_records:
                _, (_, evicted) = self._records.popitem(last=False)
                self._by_candidate.pop((evicted.review_id, evicted.candidate_id), None)
            self._records[record.normalization_id] = (time.time(), record)
            self._by_candidate[candidate_key] = record.normalization_id
            return record.model_copy(deep=True)

    def get(self, normalization_id: str) -> NormalizationRecord | None:
        with self._lock:
            self._purge_expired()
            stored = self._records.get(normalization_id)
            if stored is None:
                return None
            self._records[normalization_id] = (time.time(), stored[1])
            self._records.move_to_end(normalization_id)
            return stored[1].model_copy(deep=True)

    def decide(
        self,
        normalization_id: str,
        action: NormalizationAction,
        selected_concept_id: str | None = None,
        acknowledge_attribute_mismatches: bool = False,
    ) -> NormalizationRecord:
        with self._lock:
            self._purge_expired()
            stored = self._records.get(normalization_id)
            if stored is None:
                raise KeyError("Normalization record not found or expired.")

            record = stored[1]
            if record.status not in {
                NormalizationStatus.PENDING_REVIEW,
                NormalizationStatus.AMBIGUOUS,
            } or record.decision_action is not None:
                raise InvalidNormalizationTransitionError(
                    "Normalization decisions are allowed only while a concept is pending or ambiguous."
                )

            selected: TerminologyConcept | None = None
            if action == NormalizationAction.ACCEPT:
                if selected_concept_id is None:
                    raise ValueError("Accept requires a selected source concept identifier.")
                selected = next(
                    (
                        concept
                        for concept in record.proposed_concepts
                        if concept.source_identifier == selected_concept_id
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("Selected concept does not belong to this normalization result.")
                if (
                    selected.attribute_match == AttributeMatch.MISMATCH
                    and not acknowledge_attribute_mismatches
                ):
                    raise ValueError(
                        "Accepting a concept with strength or dosage-form mismatches requires explicit acknowledgement."
                    )
                record.status = NormalizationStatus.ACCEPTED
                record.accepted_concept = selected
                record.resolution_reason = (
                    "Reviewer explicitly accepted this source-backed concept; "
                    "this decision does not establish clinical correctness."
                )
            elif action == NormalizationAction.REJECT:
                if selected_concept_id is not None:
                    raise ValueError("Reject applies to the normalization result, not a selected concept.")
                record.status = NormalizationStatus.REJECTED
                record.accepted_concept = None
                record.resolution_reason = "Reviewer explicitly rejected the proposed normalization."
            elif action == NormalizationAction.MARK_AMBIGUOUS:
                if selected_concept_id is not None:
                    raise ValueError("Mark ambiguous does not accept a selected concept.")
                if len(record.proposed_concepts) < 2:
                    raise ValueError("Ambiguous state requires multiple proposed concepts.")
                record.status = NormalizationStatus.AMBIGUOUS
                record.resolution_reason = "Reviewer left multiple plausible concepts unresolved."
            elif action == NormalizationAction.MARK_UNMATCHED:
                if selected_concept_id is not None:
                    raise ValueError("Mark unmatched does not accept a selected concept.")
                record.status = NormalizationStatus.UNMATCHED
                record.accepted_concept = None
                record.resolution_reason = "Reviewer explicitly left the medicine unresolved."
            else:
                raise ValueError(f"Unsupported normalization action: {action}")

            now = datetime.now(timezone.utc).isoformat()
            record.decision_action = action
            record.decision_at = now
            record.updated_at = now
            self._records[normalization_id] = (time.time(), record)
            self._records.move_to_end(normalization_id)
            return record.model_copy(deep=True)

    def _purge_expired(self) -> None:
        now = time.time()
        expired_ids = [
            normalization_id
            for normalization_id, (created_at, _) in self._records.items()
            if now - created_at > self.ttl_seconds
        ]
        for normalization_id in expired_ids:
            _, expired = self._records.pop(normalization_id)
            self._by_candidate.pop((expired.review_id, expired.candidate_id), None)
