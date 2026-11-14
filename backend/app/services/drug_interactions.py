"""Source-backed, accepted-concept-only drug interaction evidence lookup."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field

from backend.app.services.medicine_normalization import (
    NormalizationAction,
    NormalizationRecord,
    NormalizationStatus,
)
from backend.app.services.review_workflow import ReviewAction, ReviewStatus

logger = logging.getLogger(__name__)

MAX_INTERACTION_MEDICINES = 20


class InteractionStatus(str, Enum):
    INTERACTION_FOUND = "interaction_found"
    NO_MATCHING_RECORD_FOUND = "no_matching_record_found"
    SOURCE_UNAVAILABLE = "source_unavailable"
    NOT_CONFIGURED = "not_configured"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"


class AcceptedInteractionConcept(BaseModel):
    """Auditable terminology concept proven to have passed both review phases."""

    normalization_id: str
    review_id: str
    candidate_id: str
    terminology_source: str
    terminology_version: str | None = None
    source_identifier: str
    source_name: str
    original_candidate_text: str
    normalization_input_text: str
    phase4_review_status: ReviewStatus
    phase4_review_action: ReviewAction
    phase4_reviewed_at: str
    phase5_review_action: NormalizationAction
    phase5_decision_at: str


class InteractionLookupConcept(BaseModel):
    """Minimal provider input; deliberately excludes OCR and reviewer-entered text."""

    normalization_id: str
    review_id: str
    candidate_id: str
    terminology_source: str
    terminology_version: str | None = None
    source_identifier: str
    source_name: str


class InteractionEvidenceRecord(BaseModel):
    """Evidence exactly as returned by the configured interaction source."""

    concept_identifier_a: str = Field(min_length=1, max_length=200)
    concept_identifier_b: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=10_000)
    source_severity: str | None = Field(default=None, max_length=500)
    source_name: str = Field(min_length=1, max_length=300)
    source_record_url: str | None = Field(default=None, max_length=2000)
    source_record_identifier: str | None = Field(default=None, max_length=500)
    dataset_version: str | None = Field(default=None, max_length=300)
    release_date: str | None = Field(default=None, max_length=100)
    retrieved_at: str | None = Field(default=None, max_length=100)
    limitations: list[str] = Field(default_factory=list)
    relevant_context: str | None = Field(default=None, max_length=5000)


class PairEvidenceResponse(BaseModel):
    """One provider's result for a single unordered pair."""

    status: InteractionStatus
    evidence: list[InteractionEvidenceRecord] = Field(default_factory=list)
    source_name: str
    source_version: str | None = None
    release_date: str | None = None
    retrieved_at: str | None = None
    coverage_complete: bool = True
    limitations: list[str] = Field(default_factory=list)
    message: str | None = None


class InteractionProvider(ABC):
    """Replaceable adapter contract for a legitimate interaction evidence source."""

    provider_name: str
    source_version: str | None = None
    release_date: str | None = None

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return true only when an actual evidence source is configured."""

    @abstractmethod
    def lookup_pair(
        self,
        concept_a: InteractionLookupConcept,
        concept_b: InteractionLookupConcept,
    ) -> PairEvidenceResponse:
        """Look up one pair using only the supplied standardized source identifiers."""


class UnavailableInteractionProvider(InteractionProvider):
    """Default provider; network access and external sources are disabled."""

    provider_name = "Drug interaction evidence source"
    source_version = None
    release_date = None

    @property
    def is_configured(self) -> bool:
        return False

    def lookup_pair(
        self,
        concept_a: InteractionLookupConcept,
        concept_b: InteractionLookupConcept,
    ) -> PairEvidenceResponse:
        del concept_a, concept_b
        raise RuntimeError("The drug interaction evidence provider is not configured.")


class InteractionPairResult(BaseModel):
    pair_index: int
    concept_a: AcceptedInteractionConcept
    concept_b: AcceptedInteractionConcept
    status: InteractionStatus
    evidence: list[InteractionEvidenceRecord] = Field(default_factory=list)
    source_name: str
    source_version: str | None = None
    release_date: str | None = None
    retrieved_at: str | None = None
    coverage_complete: bool
    limitations: list[str] = Field(default_factory=list)
    message: str | None = None


class InteractionReport(BaseModel):
    report_id: str
    created_at: str
    updated_at: str
    overall_status: InteractionStatus
    message: str
    provider_name: str
    provider_configured: bool
    provider_version: str | None = None
    provider_release_date: str | None = None
    submitted_normalization_ids: list[str] = Field(default_factory=list)
    checked_concepts: list[AcceptedInteractionConcept] = Field(default_factory=list)
    blocked_inputs: list[dict[str, str]] = Field(default_factory=list)
    expected_pair_count: int
    checked_pair_count: int
    pair_results: list[InteractionPairResult] = Field(default_factory=list)
    check_complete: bool
    limitations: list[str] = Field(default_factory=list)
    safety_notice: str = (
        "Research prototype only. This report is not clinical decision support. "
        "No matching record does not establish that a medicine combination is safe. "
        "Do not use this tool to make treatment decisions."
    )


def _validate_accepted_record(
    record: NormalizationRecord,
) -> tuple[AcceptedInteractionConcept | None, str | None]:
    if record.status != NormalizationStatus.ACCEPTED:
        return None, f"Normalization is not accepted (status: {record.status.value})."
    if record.decision_action != NormalizationAction.ACCEPT or not record.decision_at:
        return None, "A recorded Phase 5 accept action and timestamp are required."

    concept = record.accepted_concept
    if concept is None:
        return None, "Accepted normalization record has no accepted concept."
    proposed = next(
        (
            item
            for item in record.proposed_concepts
            if item.source_identifier == concept.source_identifier
        ),
        None,
    )
    if proposed is None or proposed != concept:
        return None, "Accepted concept does not match a proposed concept in this normalization record."
    if not record.provider_configured or concept.source_name != record.provider_name:
        return None, "Accepted concept does not have configured terminology source provenance."

    evidence = record.evidence
    expected_phase4_action = (
        ReviewAction.CORRECT
        if evidence.phase4_review_status == ReviewStatus.CORRECTED
        else ReviewAction.CONFIRM
        if evidence.phase4_review_status == ReviewStatus.CONFIRMED
        else None
    )
    if (
        expected_phase4_action is None
        or evidence.phase4_review_action != expected_phase4_action.value
        or not evidence.phase4_reviewed_at
    ):
        return None, "A valid recorded Phase 4 confirmation or correction is required."

    expected_text = (
        evidence.reviewer_corrected_text
        if evidence.phase4_review_status == ReviewStatus.CORRECTED
        else evidence.original_candidate_text
    )
    if not expected_text or evidence.normalization_input_text != expected_text:
        return None, "Normalization input does not match the Phase 4 reviewed candidate."

    return (
        AcceptedInteractionConcept(
            normalization_id=record.normalization_id,
            review_id=record.review_id,
            candidate_id=record.candidate_id,
            terminology_source=record.provider_name,
            terminology_version=record.provider_version,
            source_identifier=concept.source_identifier,
            source_name=concept.name,
            original_candidate_text=evidence.original_candidate_text,
            normalization_input_text=evidence.normalization_input_text,
            phase4_review_status=evidence.phase4_review_status,
            phase4_review_action=expected_phase4_action,
            phase4_reviewed_at=evidence.phase4_reviewed_at,
            phase5_review_action=record.decision_action,
            phase5_decision_at=record.decision_at,
        ),
        None,
    )


def _pair_key(a: AcceptedInteractionConcept, b: AcceptedInteractionConcept) -> frozenset[str]:
    return frozenset((a.source_identifier, b.source_identifier))


def _validate_provider_response(
    response: object,
    concept_a: AcceptedInteractionConcept,
    concept_b: AcceptedInteractionConcept,
) -> tuple[PairEvidenceResponse | None, str | None]:
    if not isinstance(response, PairEvidenceResponse):
        return None, "Provider returned a malformed pair response."

    if response.status in {
        InteractionStatus.BLOCKED,
        InteractionStatus.NOT_CONFIGURED,
    }:
        return None, "Provider returned a status that is not valid for a configured pair lookup."

    if response.status == InteractionStatus.INTERACTION_FOUND and not response.evidence:
        return None, "Provider reported an interaction without an evidence record."
    if response.status == InteractionStatus.NO_MATCHING_RECORD_FOUND and response.evidence:
        return None, "Provider returned evidence with a no-matching-record status."
    if response.status == InteractionStatus.NO_MATCHING_RECORD_FOUND and not response.coverage_complete:
        return None, "Provider reported no match despite incomplete pair coverage."
    if response.status == InteractionStatus.SOURCE_UNAVAILABLE and response.evidence:
        return None, "Provider returned evidence with a source-unavailable status."

    expected_pair = _pair_key(concept_a, concept_b)
    for evidence in response.evidence:
        if frozenset(
            (evidence.concept_identifier_a, evidence.concept_identifier_b)
        ) != expected_pair:
            return None, "Provider evidence identifiers do not match the queried pair."
        if evidence.source_name != response.source_name:
            return None, "Evidence record source does not match provider response provenance."

    return response, None


def _overall_status(pair_results: list[InteractionPairResult]) -> InteractionStatus:
    statuses = [result.status for result in pair_results]
    if not statuses:
        return InteractionStatus.BLOCKED
    if all(status == InteractionStatus.NOT_CONFIGURED for status in statuses):
        return InteractionStatus.NOT_CONFIGURED
    if all(status == InteractionStatus.SOURCE_UNAVAILABLE for status in statuses):
        return InteractionStatus.SOURCE_UNAVAILABLE
    if all(status == InteractionStatus.NO_MATCHING_RECORD_FOUND for status in statuses):
        return InteractionStatus.NO_MATCHING_RECORD_FOUND
    if any(status in {InteractionStatus.INCOMPLETE, InteractionStatus.SOURCE_UNAVAILABLE} for status in statuses):
        return InteractionStatus.INCOMPLETE
    if any(status == InteractionStatus.INTERACTION_FOUND for status in statuses):
        return InteractionStatus.INTERACTION_FOUND
    return InteractionStatus.INCOMPLETE


def check_accepted_normalizations(
    records: list[NormalizationRecord],
    provider: InteractionProvider,
) -> InteractionReport:
    """Create pairwise evidence results only from explicitly accepted Phase 5 concepts."""
    now = datetime.now(timezone.utc).isoformat()
    report_id = str(uuid.uuid4())
    base_limitations = [
        "A missing matching record does not establish that the combination is safe.",
        "Interaction-source coverage, identifier support, and evidence quality are source-dependent.",
        "This research prototype does not provide clinical interpretation or recommendations.",
    ]

    if not records:
        raise ValueError("At least one accepted normalization record ID is required.")
    if len(records) > MAX_INTERACTION_MEDICINES:
        raise ValueError(f"At most {MAX_INTERACTION_MEDICINES} medicines can be checked at once.")
    if len({record.normalization_id for record in records}) != len(records):
        raise ValueError("Duplicate normalization record IDs are not allowed.")

    accepted: list[AcceptedInteractionConcept] = []
    blocked: list[dict[str, str]] = []
    for record in records:
        concept, reason = _validate_accepted_record(record)
        if concept is None:
            blocked.append(
                {
                    "normalization_id": record.normalization_id,
                    "candidate_id": record.candidate_id,
                    "status": record.status.value,
                    "reason": reason or "Normalization did not pass the acceptance gate.",
                }
            )
        else:
            accepted.append(concept)

    if blocked:
        return InteractionReport(
            report_id=report_id,
            created_at=now,
            updated_at=now,
            overall_status=InteractionStatus.BLOCKED,
            message=(
                "Interaction check could not be performed because one or more requested "
                "normalizations did not pass the explicit Phase 5 acceptance gate."
            ),
            provider_name=provider.provider_name,
            provider_configured=provider.is_configured,
            provider_version=provider.source_version,
            provider_release_date=provider.release_date,
            submitted_normalization_ids=[record.normalization_id for record in records],
            blocked_inputs=blocked,
            expected_pair_count=0,
            checked_pair_count=0,
            check_complete=False,
            limitations=base_limitations,
        )

    duplicate_concept_ids = {
        concept.source_identifier
        for concept in accepted
        if sum(item.source_identifier == concept.source_identifier for item in accepted) > 1
    }
    if duplicate_concept_ids:
        raise ValueError(
            "Duplicate standardized concept identifiers are not allowed: "
            + ", ".join(sorted(duplicate_concept_ids))
        )

    pairs = list(combinations(accepted, 2))
    if not pairs:
        return InteractionReport(
            report_id=report_id,
            created_at=now,
            updated_at=now,
            overall_status=InteractionStatus.BLOCKED,
            message=(
                "Interaction check could not be performed; at least two distinct "
                "accepted standardized medicine concepts are required."
            ),
            provider_name=provider.provider_name,
            provider_configured=provider.is_configured,
            provider_version=provider.source_version,
            provider_release_date=provider.release_date,
            submitted_normalization_ids=[record.normalization_id for record in records],
            checked_concepts=accepted,
            expected_pair_count=0,
            checked_pair_count=0,
            check_complete=False,
            limitations=base_limitations,
        )

    pair_results: list[InteractionPairResult] = []
    for index, (concept_a, concept_b) in enumerate(pairs, start=1):
        if not provider.is_configured:
            pair_results.append(
                InteractionPairResult(
                    pair_index=index,
                    concept_a=concept_a,
                    concept_b=concept_b,
                    status=InteractionStatus.NOT_CONFIGURED,
                    source_name=provider.provider_name,
                    source_version=provider.source_version,
                    release_date=provider.release_date,
                    coverage_complete=False,
                    message="No drug interaction evidence source is configured.",
                )
            )
            continue

        try:
            provider_concept_a = InteractionLookupConcept(
                normalization_id=concept_a.normalization_id,
                review_id=concept_a.review_id,
                candidate_id=concept_a.candidate_id,
                terminology_source=concept_a.terminology_source,
                terminology_version=concept_a.terminology_version,
                source_identifier=concept_a.source_identifier,
                source_name=concept_a.source_name,
            )
            provider_concept_b = InteractionLookupConcept(
                normalization_id=concept_b.normalization_id,
                review_id=concept_b.review_id,
                candidate_id=concept_b.candidate_id,
                terminology_source=concept_b.terminology_source,
                terminology_version=concept_b.terminology_version,
                source_identifier=concept_b.source_identifier,
                source_name=concept_b.source_name,
            )
            raw_response: Any = provider.lookup_pair(
                provider_concept_a,
                provider_concept_b,
            )
        except Exception as exc:
            logger.exception(
                "Interaction evidence provider failed for pair %s / %s",
                concept_a.source_identifier,
                concept_b.source_identifier,
            )
            pair_results.append(
                InteractionPairResult(
                    pair_index=index,
                    concept_a=concept_a,
                    concept_b=concept_b,
                    status=InteractionStatus.SOURCE_UNAVAILABLE,
                    source_name=provider.provider_name,
                    source_version=provider.source_version,
                    release_date=provider.release_date,
                    coverage_complete=False,
                    message=f"Interaction source query failed ({type(exc).__name__}).",
                )
            )
            continue

        validated, validation_error = _validate_provider_response(
            raw_response,
            concept_a,
            concept_b,
        )
        if validated is None:
            pair_results.append(
                InteractionPairResult(
                    pair_index=index,
                    concept_a=concept_a,
                    concept_b=concept_b,
                    status=InteractionStatus.INCOMPLETE,
                    source_name=provider.provider_name,
                    source_version=provider.source_version,
                    release_date=provider.release_date,
                    coverage_complete=False,
                    message=validation_error,
                )
            )
            continue

        if validated.source_name != provider.provider_name:
            pair_results.append(
                InteractionPairResult(
                    pair_index=index,
                    concept_a=concept_a,
                    concept_b=concept_b,
                    status=InteractionStatus.INCOMPLETE,
                    evidence=validated.evidence,
                    source_name=validated.source_name,
                    source_version=validated.source_version,
                    release_date=validated.release_date,
                    retrieved_at=validated.retrieved_at,
                    coverage_complete=False,
                    limitations=validated.limitations,
                    message="Provider response source provenance did not match configured provider.",
                )
            )
            continue

        pair_status = validated.status
        pair_message = validated.message
        if not validated.coverage_complete:
            pair_status = InteractionStatus.INCOMPLETE
            pair_message = (
                "The provider marked pair coverage incomplete; any returned evidence "
                "is retained but the pair is not considered fully checked."
            )

        pair_results.append(
            InteractionPairResult(
                pair_index=index,
                concept_a=concept_a,
                concept_b=concept_b,
                status=pair_status,
                evidence=validated.evidence,
                source_name=validated.source_name,
                source_version=validated.source_version,
                release_date=validated.release_date,
                retrieved_at=validated.retrieved_at,
                coverage_complete=validated.coverage_complete,
                limitations=validated.limitations,
                message=pair_message,
            )
        )

    overall = _overall_status(pair_results)
    complete_statuses = {
        InteractionStatus.INTERACTION_FOUND,
        InteractionStatus.NO_MATCHING_RECORD_FOUND,
    }
    complete = all(
        result.status in complete_statuses and result.coverage_complete
        for result in pair_results
    )
    if overall == InteractionStatus.BLOCKED:
        message = "Interaction check could not be performed."
    elif overall == InteractionStatus.NOT_CONFIGURED:
        message = "No interaction evidence source is configured; no medicine pairs were checked."
    elif overall == InteractionStatus.SOURCE_UNAVAILABLE:
        message = "The configured interaction evidence source was unavailable for every pair."
    elif overall == InteractionStatus.NO_MATCHING_RECORD_FOUND:
        message = (
            "The configured source returned no matching records for the checked pairs. "
            "This does not establish that the combinations are safe."
        )
    elif overall == InteractionStatus.INTERACTION_FOUND:
        message = (
            "The source returned interaction evidence for at least one pair. "
            "Review the pair-level results and coverage."
        )
    else:
        message = (
            "The interaction check is incomplete; one or more required pairs "
            "could not be fully checked."
        )

    return InteractionReport(
        report_id=report_id,
        created_at=now,
        updated_at=now,
        overall_status=overall,
        message=message,
        provider_name=provider.provider_name,
        provider_configured=provider.is_configured,
        provider_version=provider.source_version,
        provider_release_date=provider.release_date,
        submitted_normalization_ids=[record.normalization_id for record in records],
        checked_concepts=accepted,
        expected_pair_count=len(pairs),
        checked_pair_count=sum(
            result.status != InteractionStatus.NOT_CONFIGURED
            for result in pair_results
        ),
        pair_results=pair_results,
        check_complete=complete,
        limitations=base_limitations,
    )


class InteractionReportStore:
    """Bounded process-local report store with inactivity expiry."""

    def __init__(self, max_reports: int = 100, ttl_seconds: int = 3600) -> None:
        self.max_reports = max_reports
        self.ttl_seconds = ttl_seconds
        self._reports: OrderedDict[str, tuple[float, InteractionReport]] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, report: InteractionReport) -> InteractionReport:
        with self._lock:
            self._purge_expired()
            while len(self._reports) >= self.max_reports:
                self._reports.popitem(last=False)
            self._reports[report.report_id] = (time.time(), report)
            return report.model_copy(deep=True)

    def get(self, report_id: str) -> InteractionReport | None:
        with self._lock:
            self._purge_expired()
            stored = self._reports.get(report_id)
            if stored is None:
                return None
            self._reports[report_id] = (time.time(), stored[1])
            self._reports.move_to_end(report_id)
            return stored[1].model_copy(deep=True)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            report_id
            for report_id, (created_at, _) in self._reports.items()
            if now - created_at > self.ttl_seconds
        ]
        for report_id in expired:
            self._reports.pop(report_id, None)
