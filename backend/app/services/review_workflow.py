"""
backend/app/services/review_workflow.py
Mandatory Human-Review Workflow, Audit Trail, and Downstream Safety Gate.

Phase 4 Scope & Safety Guarantees:
- Every candidate extracted in Phase 3 begins in ReviewStatus.PENDING_REVIEW.
- A candidate must NEVER be automatically approved or confirmed by its heuristic score.
- Reviewer actions: confirm, correct, reject, mark_uncertain.
- Any correction preserves the original OCR text and candidate text verbatim,
  storing the reviewer's correction in a separate dedicated attribute.
- Downstream Safety Gate: Only explicitly 'confirmed' (using original text) or
  'corrected' (using reviewer-supplied text) candidates can ever advance.
- Rejected, uncertain, and pending candidates are strictly BLOCKED from downstream processing.
- All sessions are kept in-memory by default (privacy-first, zero persistence of patient data).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from backend.app.services.medicine_extraction import ExtractedMention


class ReviewStatus(str, Enum):
    """Explicit review state lifecycle for a candidate mention."""

    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class ReviewAction(str, Enum):
    """Explicit action taken by a human reviewer."""

    CONFIRM = "confirm"
    CORRECT = "correct"
    REJECT = "reject"
    MARK_UNCERTAIN = "mark_uncertain"


class ReviewCandidate(BaseModel):
    """
    Reviewable candidate record preserving full original evidence,
    audit trail, and separate reviewer-entered values.
    """

    candidate_id: str
    original_ocr_text: str = Field(description="Full original OCR document or snippet context")
    original_candidate_text: str = Field(description="Original token verbatim from OCR without modification")
    candidate_type: str = Field(description="Candidate entity type (e.g. possible_medicine_name, strength)")
    start_char: int = Field(ge=0, description="Exact start character offset in original text")
    end_char: int = Field(ge=0, description="Exact end character offset in original text")
    line_number: int = Field(ge=1, description="Line number in original text")
    evidence_context: str = Field(description="Original prescription line containing the mention")
    extraction_method: str = Field(description="Explainable rule or dictionary method that flagged the candidate")

    # Heuristic evidence (explicitly uncalibrated, NOT statistical probabilities)
    heuristic_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncalibrated rule-based heuristic weight; not a verified statistical probability.",
    )
    ocr_engine_confidence: float | None = Field(
        default=None,
        description="Direct confidence value from OCR engine, or null if engine does not provide one. Never fabricated.",
    )
    ocr_confidence_available: bool = Field(
        default=False,
        description="Whether usable confidence was supplied by the OCR engine.",
    )
    has_dictionary_match: bool = False
    has_contextual_pattern: bool = False
    has_spillover_or_ambiguity: bool = False

    # Review status & audit trail
    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING_REVIEW,
        description="Mandatory state. Must start in pending_review and requires explicit human action.",
    )
    reviewer_corrected_text: str | None = Field(
        default=None,
        description="Separately entered reviewer correction. Never overwrites original_candidate_text.",
    )
    review_action: ReviewAction | None = Field(
        default=None,
        description="Latest explicit action applied by the reviewer.",
    )
    reviewed_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 timestamp when review action was recorded.",
    )
    is_downstream_eligible: bool = Field(
        default=False,
        description="Whether this candidate satisfies the safety gate to advance downstream.",
    )


class DownstreamEligibleCandidate(BaseModel):
    """
    Validated payload produced exclusively by the Downstream Safety Gate.
    Contains the eligible effective text for a future downstream workflow.
    """

    candidate_id: str
    effective_text: str = Field(
        description="The text to be used downstream (original for confirmed, or reviewer-entered for corrected)."
    )
    is_corrected: bool = Field(
        description="Whether this candidate was modified by the reviewer."
    )
    original_candidate_text: str = Field(
        description="Original uncorrected OCR text preserved for audit integrity."
    )
    candidate_type: str
    review_status: ReviewStatus
    eligibility_reason: str
    original_ocr_text: str
    start_char: int
    end_char: int
    line_number: int
    evidence_context: str
    extraction_method: str
    reviewer_corrected_text: str | None = None
    phase4_review_action: ReviewAction | None = None
    phase4_reviewed_at: str | None = None
    heuristic_score: float
    ocr_engine_confidence: float | None = None
    ocr_confidence_available: bool
    has_dictionary_match: bool
    has_contextual_pattern: bool
    has_spillover_or_ambiguity: bool


class ReviewSession(BaseModel):
    """In-memory review session holding all candidates for an OCR document."""

    review_id: str
    created_at: str
    updated_at: str
    raw_ocr_text: str
    candidates: dict[str, ReviewCandidate] = Field(default_factory=dict)
    total_candidates: int = 0
    pending_count: int = 0
    confirmed_count: int = 0
    corrected_count: int = 0
    rejected_count: int = 0
    uncertain_count: int = 0
    is_session_complete: bool = False
    safety_disclaimer: str = (
        "Research prototype only. Extracted medicine candidates may be wrong. "
        "Review status does not establish clinical correctness. "
        "Do not use this tool to make treatment decisions."
    )

    def refresh_counts(self) -> None:
        """Update summary counts and completion status."""
        self.total_candidates = len(self.candidates)
        self.pending_count = sum(1 for c in self.candidates.values() if c.review_status == ReviewStatus.PENDING_REVIEW)
        self.confirmed_count = sum(1 for c in self.candidates.values() if c.review_status == ReviewStatus.CONFIRMED)
        self.corrected_count = sum(1 for c in self.candidates.values() if c.review_status == ReviewStatus.CORRECTED)
        self.rejected_count = sum(1 for c in self.candidates.values() if c.review_status == ReviewStatus.REJECTED)
        self.uncertain_count = sum(1 for c in self.candidates.values() if c.review_status == ReviewStatus.UNCERTAIN)
        self.is_session_complete = (self.total_candidates > 0) and (self.pending_count == 0)
        self.updated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Downstream Safety Gate
# ---------------------------------------------------------------------------


def evaluate_downstream_eligibility(
    candidate: ReviewCandidate,
) -> tuple[bool, str | None, str]:
    """
    Evaluate whether a single candidate meets the strict safety criteria to proceed
    to downstream NLP phases (e.g. normalization, RxNorm matching, interaction checking).

    Rules:
    - CONFIRMED: Eligible using original_candidate_text.
    - CORRECTED: Eligible ONLY using reviewer_corrected_text (must be non-empty).
    - PENDING_REVIEW: BLOCKED. Must not advance.
    - REJECTED: BLOCKED. Must not advance.
    - UNCERTAIN: BLOCKED. Must not advance.

    Returns:
        tuple[is_eligible (bool), effective_text (str | None), reason (str)]
    """
    if candidate.review_status == ReviewStatus.CONFIRMED:
        if candidate.review_action != ReviewAction.CONFIRM or not candidate.reviewed_at:
            return (
                False,
                None,
                "BLOCKED: A confirmed state requires a recorded explicit confirm action.",
            )
        return (
            True,
            candidate.original_candidate_text,
            "Candidate explicitly confirmed as written by reviewer.",
        )

    if candidate.review_status == ReviewStatus.CORRECTED:
        corrected = candidate.reviewer_corrected_text
        if candidate.review_action != ReviewAction.CORRECT or not candidate.reviewed_at:
            return (
                False,
                None,
                "BLOCKED: A corrected state requires a recorded explicit correct action.",
            )
        if not corrected or not corrected.strip():
            return (
                False,
                None,
                "BLOCKED: Candidate marked as corrected but reviewer-entered text is empty.",
            )
        return (
            True,
            corrected,
            "Candidate is eligible for the downstream workflow using reviewer-corrected text (original preserved).",
        )

    if candidate.review_status == ReviewStatus.PENDING_REVIEW:
        return (
            False,
            None,
            "BLOCKED: Candidate is pending human review and has not been approved.",
        )

    if candidate.review_status == ReviewStatus.REJECTED:
        return (
            False,
            None,
            "BLOCKED: Candidate was explicitly rejected by the human reviewer.",
        )

    if candidate.review_status == ReviewStatus.UNCERTAIN:
        return (
            False,
            None,
            "BLOCKED: Candidate was marked uncertain and cannot proceed downstream.",
        )

    return (
        False,
        None,
        f"BLOCKED: Candidate has unrecognized review status '{candidate.review_status}'.",
    )


def get_eligible_candidates_for_downstream(
    session: ReviewSession,
) -> list[DownstreamEligibleCandidate]:
    """
    Downstream Safety Gate:
    Inspect all candidates in a review session and return only those that
    have been explicitly confirmed or corrected by a human reviewer.

    This function is the mandatory architectural gate before any Phase 5
    medicine normalization or interaction checking can consume candidates.
    """
    eligible: list[DownstreamEligibleCandidate] = []

    for candidate in session.candidates.values():
        is_ok, effective_text, reason = evaluate_downstream_eligibility(candidate)
        if is_ok and effective_text:
            eligible.append(
                DownstreamEligibleCandidate(
                    candidate_id=candidate.candidate_id,
                    effective_text=effective_text,
                    is_corrected=(candidate.review_status == ReviewStatus.CORRECTED),
                    original_candidate_text=candidate.original_candidate_text,
                    candidate_type=candidate.candidate_type,
                    review_status=candidate.review_status,
                    eligibility_reason=reason,
                    original_ocr_text=candidate.original_ocr_text,
                    start_char=candidate.start_char,
                    end_char=candidate.end_char,
                    line_number=candidate.line_number,
                    evidence_context=candidate.evidence_context,
                    extraction_method=candidate.extraction_method,
                    reviewer_corrected_text=candidate.reviewer_corrected_text,
                    phase4_review_action=candidate.review_action,
                    phase4_reviewed_at=candidate.reviewed_at,
                    heuristic_score=candidate.heuristic_score,
                    ocr_engine_confidence=candidate.ocr_engine_confidence,
                    ocr_confidence_available=candidate.ocr_confidence_available,
                    has_dictionary_match=candidate.has_dictionary_match,
                    has_contextual_pattern=candidate.has_contextual_pattern,
                    has_spillover_or_ambiguity=candidate.has_spillover_or_ambiguity,
                )
            )

    return eligible


# ---------------------------------------------------------------------------
# In-Memory Ephemeral Session Store
# ---------------------------------------------------------------------------


class ReviewSessionStore:
    """
    Thread-safe, ephemeral in-memory storage for prescription human-review sessions.
    Strictly zero persistence by default to preserve patient privacy.
    Sessions expire after ttl_seconds (default 1 hour) or when max_sessions is reached.
    """

    def __init__(self, max_sessions: int = 100, ttl_seconds: int = 3600) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, tuple[float, ReviewSession]] = OrderedDict()
        self._lock = threading.Lock()

    def create_session(
        self,
        raw_ocr_text: str,
        mentions: list[ExtractedMention],
    ) -> ReviewSession:
        """Initialize a new review session from Phase 3 candidate extraction output."""
        with self._lock:
            self._purge_expired()
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)  # LRU eviction

            now_iso = datetime.now(timezone.utc).isoformat()
            review_id = str(uuid.uuid4())

            candidates: dict[str, ReviewCandidate] = {}

            for m in mentions:
                cid = getattr(m, "candidate_id", None) or str(uuid.uuid4())
                score = m.heuristic_score

                candidate = ReviewCandidate(
                    candidate_id=cid,
                    original_ocr_text=raw_ocr_text,
                    original_candidate_text=m.text,
                    candidate_type=m.candidate_type.value if hasattr(m.candidate_type, "value") else str(m.candidate_type),
                    start_char=m.start_char,
                    end_char=m.end_char,
                    line_number=m.line_number,
                    evidence_context=m.evidence_context,
                    extraction_method=m.extraction_method.value if hasattr(m.extraction_method, "value") else str(m.extraction_method),
                    heuristic_score=round(score, 2),
                    ocr_engine_confidence=m.ocr_engine_confidence,
                    ocr_confidence_available=m.ocr_confidence_available,
                    has_dictionary_match=m.has_dictionary_match,
                    has_contextual_pattern=m.has_contextual_pattern,
                    has_spillover_or_ambiguity=m.has_spillover_or_ambiguity,
                    review_status=ReviewStatus.PENDING_REVIEW,
                    reviewer_corrected_text=None,
                    review_action=None,
                    reviewed_at=None,
                    reviewer_notes=None,
                    is_downstream_eligible=False,
                )
                candidates[cid] = candidate

            session = ReviewSession(
                review_id=review_id,
                created_at=now_iso,
                updated_at=now_iso,
                raw_ocr_text=raw_ocr_text,
                candidates=candidates,
            )
            session.refresh_counts()
            self._sessions[review_id] = (time.time(), session)
            return session

    def get_session(self, review_id: str) -> ReviewSession | None:
        """Retrieve a review session by review_id."""
        with self._lock:
            self._purge_expired()
            if review_id not in self._sessions:
                return None
            _, session = self._sessions[review_id]
            self._sessions[review_id] = (time.time(), session)
            self._sessions.move_to_end(review_id)
            return session

    def apply_action(
        self,
        review_id: str,
        candidate_id: str,
        action: ReviewAction,
        corrected_text: str | None = None,
    ) -> tuple[ReviewSession, ReviewCandidate]:
        """
        Apply an explicit reviewer action to a specific candidate in a review session.
        Enforces validation rules, state transitions, and audit records.
        """
        with self._lock:
            self._purge_expired()
            if review_id not in self._sessions:
                raise KeyError(f"Review session '{review_id}' not found.")

            _, session = self._sessions[review_id]

            if candidate_id not in session.candidates:
                raise KeyError(
                    f"Candidate '{candidate_id}' does not exist in review session '{review_id}'."
                )

            candidate = session.candidates[candidate_id]
            if candidate.review_status != ReviewStatus.PENDING_REVIEW:
                raise InvalidReviewTransitionError(
                    f"Candidate '{candidate_id}' has already been reviewed; "
                    "review actions are allowed only from pending_review."
                )
            now_iso = datetime.now(timezone.utc).isoformat()

            if action == ReviewAction.CORRECT:
                if not corrected_text or not corrected_text.strip():
                    raise ValueError(
                        "Action 'correct' requires a non-empty, non-whitespace corrected_text value."
                    )
                if len(corrected_text) > 500:
                    raise ValueError("Corrected text exceeds maximum allowed length of 500 characters.")

                candidate.review_status = ReviewStatus.CORRECTED
                candidate.reviewer_corrected_text = corrected_text
                candidate.review_action = ReviewAction.CORRECT

            elif action == ReviewAction.CONFIRM:
                candidate.review_status = ReviewStatus.CONFIRMED
                candidate.reviewer_corrected_text = None
                candidate.review_action = ReviewAction.CONFIRM

            elif action == ReviewAction.REJECT:
                candidate.review_status = ReviewStatus.REJECTED
                candidate.reviewer_corrected_text = None
                candidate.review_action = ReviewAction.REJECT

            elif action == ReviewAction.MARK_UNCERTAIN:
                candidate.review_status = ReviewStatus.UNCERTAIN
                candidate.reviewer_corrected_text = None
                candidate.review_action = ReviewAction.MARK_UNCERTAIN

            else:
                raise ValueError(f"Invalid review action '{action}'.")

            candidate.reviewed_at = now_iso
            # Re-evaluate downstream safety gate for this candidate
            is_eligible, _, _ = evaluate_downstream_eligibility(candidate)
            candidate.is_downstream_eligible = is_eligible

            # Refresh session totals
            session.refresh_counts()
            self._sessions[review_id] = (time.time(), session)
            self._sessions.move_to_end(review_id)

            return session, candidate

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            rid for rid, (created_t, _) in self._sessions.items()
            if (now - created_t) > self.ttl_seconds
        ]
        for rid in expired:
            self._sessions.pop(rid, None)


class InvalidReviewTransitionError(ValueError):
    """Raised when an action is attempted after a candidate leaves pending_review."""
