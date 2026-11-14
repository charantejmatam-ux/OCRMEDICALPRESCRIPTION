"""Tests for Phase 4 review lifecycle and downstream workflow gate."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.review_workflow import (
    ReviewAction,
    ReviewSessionStore,
    ReviewStatus,
    evaluate_downstream_eligibility,
    get_eligible_candidates_for_downstream,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def create_review(client: TestClient) -> tuple[str, dict]:
    response = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": "Amoxicillin 500mg capsule PO"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["review_id"]
    return data["review_id"], data["candidates"][0]


def review_url(review_id: str, candidate_id: str) -> str:
    return f"/prescriptions/review/{review_id}/candidates/{candidate_id}"


def test_extracted_candidates_start_pending_with_explicit_evidence(client: TestClient) -> None:
    review_id, first_candidate = create_review(client)
    extraction_response = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": "Amoxicillin 500mg capsule PO"},
    )
    assert extraction_response.status_code == 200
    extraction = extraction_response.json()
    assert all(c["review_status"] == "pending_review" for c in extraction["candidates"])

    session_response = client.get(f"/prescriptions/review/{review_id}")
    assert session_response.status_code == 200
    session = session_response.json()
    assert session["pending_count"] == session["total_candidates"]
    assert session["candidates"][first_candidate["candidate_id"]]["review_status"] == "pending_review"
    assert session["candidates"][first_candidate["candidate_id"]]["original_ocr_text"] == (
        "Amoxicillin 500mg capsule PO"
    )
    assert session["candidates"][first_candidate["candidate_id"]]["ocr_engine_confidence"] is None
    assert session["candidates"][first_candidate["candidate_id"]]["ocr_confidence_available"] is False
    assert "confidence_heuristic" not in first_candidate
    assert "probability" not in first_candidate
    assert first_candidate["heuristic_score"] is not None
    assert "not a probability" in extraction["metadata"]["heuristic_score_interpretation"]


def test_confirm_candidate_records_explicit_action(client: TestClient) -> None:
    review_id, candidate = create_review(client)
    response = client.post(
        review_url(review_id, candidate["candidate_id"]),
        json={"action": "confirm"},
    )
    assert response.status_code == 200
    reviewed = response.json()["candidates"][candidate["candidate_id"]]
    assert reviewed["review_status"] == "confirmed"
    assert reviewed["review_action"] == "confirm"
    assert reviewed["reviewed_at"]
    assert reviewed["is_downstream_eligible"] is True
    assert reviewed["original_candidate_text"] == candidate["text"]


def test_correction_preserves_original_candidate_and_ocr(client: TestClient) -> None:
    review_id, candidate = create_review(client)
    response = client.post(
        review_url(review_id, candidate["candidate_id"]),
        json={"action": "correct", "corrected_text": "Amoxicillin"},
    )
    assert response.status_code == 200
    reviewed = response.json()["candidates"][candidate["candidate_id"]]
    assert reviewed["review_status"] == "corrected"
    assert reviewed["original_candidate_text"] == candidate["text"]
    assert reviewed["original_ocr_text"] == "Amoxicillin 500mg capsule PO"
    assert reviewed["start_char"] == candidate["start_char"]
    assert reviewed["end_char"] == candidate["end_char"]
    assert reviewed["line_number"] == candidate["line_number"]
    assert reviewed["evidence_context"] == candidate["evidence_context"]
    assert reviewed["extraction_method"] == candidate["extraction_method"]
    assert reviewed["reviewer_corrected_text"] == "Amoxicillin"
    assert reviewed["reviewed_at"]
    assert reviewed["is_downstream_eligible"] is True


@pytest.mark.parametrize(
    ("action", "expected_state"),
    [
        ("reject", "rejected"),
        ("mark_uncertain", "uncertain"),
    ],
)
def test_reject_and_mark_uncertain_are_recorded(
    client: TestClient,
    action: str,
    expected_state: str,
) -> None:
    review_id, candidate = create_review(client)
    response = client.post(
        review_url(review_id, candidate["candidate_id"]),
        json={"action": action},
    )
    assert response.status_code == 200
    reviewed = response.json()["candidates"][candidate["candidate_id"]]
    assert reviewed["review_status"] == expected_state
    assert reviewed["review_action"] == action
    assert reviewed["is_downstream_eligible"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "approve"},
        {"action": "correct", "corrected_text": ""},
        {"action": "correct", "corrected_text": " \t "},
        {"action": "correct", "corrected_text": "x" * 501},
        {"action": "confirm", "corrected_text": "other value"},
    ],
)
def test_invalid_action_payloads_are_rejected(client: TestClient, payload: dict) -> None:
    review_id, candidate = create_review(client)
    response = client.post(review_url(review_id, candidate["candidate_id"]), json=payload)
    assert response.status_code == 422


def test_review_transition_cannot_be_repeated_or_changed(client: TestClient) -> None:
    review_id, candidate = create_review(client)
    url = review_url(review_id, candidate["candidate_id"])
    assert client.post(url, json={"action": "confirm"}).status_code == 200
    duplicate = client.post(url, json={"action": "confirm"})
    conflict = client.post(url, json={"action": "reject"})
    assert duplicate.status_code == 409
    assert conflict.status_code == 409


def test_unknown_review_and_candidate_ids_are_not_exposed(client: TestClient) -> None:
    unknown_review_id = str(uuid4())
    assert client.get(f"/prescriptions/review/{unknown_review_id}").status_code == 404

    review_id, candidate = create_review(client)
    response = client.post(
        review_url(review_id, str(uuid4())),
        json={"action": "confirm"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Review session or candidate not found."

    other_review_id, _ = create_review(client)
    cross_session = client.post(
        review_url(other_review_id, candidate["candidate_id"]),
        json={"action": "confirm"},
    )
    assert cross_session.status_code == 404


def test_malformed_ids_are_rejected(client: TestClient) -> None:
    assert client.get("/prescriptions/review/not-a-uuid").status_code == 422
    assert (
        client.post(
            "/prescriptions/review/not-a-uuid/candidates/not-a-uuid",
            json={"action": "confirm"},
        ).status_code
        == 422
    )


def test_workflow_eligibility_blocks_unreviewed_rejected_and_uncertain() -> None:
    store = ReviewSessionStore()
    extraction = client_extraction()
    session = store.create_session(extraction.raw_text, extraction.candidates)
    candidate_id = next(iter(session.candidates))
    candidate = session.candidates[candidate_id]

    eligible, text, _ = evaluate_downstream_eligibility(candidate)
    assert (eligible, text) == (False, None)

    candidate.review_status = ReviewStatus.CONFIRMED
    eligible, text, _ = evaluate_downstream_eligibility(candidate)
    assert (eligible, text) == (False, None)
    candidate.review_status = ReviewStatus.CORRECTED
    candidate.reviewer_corrected_text = "unverified correction"
    eligible, text, _ = evaluate_downstream_eligibility(candidate)
    assert (eligible, text) == (False, None)

    for action in (ReviewAction.REJECT, ReviewAction.MARK_UNCERTAIN):
        fresh = store.create_session(extraction.raw_text, extraction.candidates)
        fresh_id = next(iter(fresh.candidates))
        store.apply_action(fresh.review_id, fresh_id, action)
        assert get_eligible_candidates_for_downstream(fresh) == []


def test_confirmed_and_corrected_candidates_pass_gate_with_expected_text() -> None:
    store = ReviewSessionStore()
    extraction = client_extraction()

    confirmed_session = store.create_session(extraction.raw_text, extraction.candidates)
    confirmed_id = next(iter(confirmed_session.candidates))
    original_text = confirmed_session.candidates[confirmed_id].original_candidate_text
    store.apply_action(confirmed_session.review_id, confirmed_id, ReviewAction.CONFIRM)
    confirmed = get_eligible_candidates_for_downstream(confirmed_session)
    assert len(confirmed) == 1
    assert confirmed[0].effective_text == original_text
    assert confirmed[0].is_corrected is False

    corrected_session = store.create_session(extraction.raw_text, extraction.candidates)
    corrected_id = next(iter(corrected_session.candidates))
    corrected_value = "Reviewer-entered value"
    store.apply_action(
        corrected_session.review_id,
        corrected_id,
        ReviewAction.CORRECT,
        corrected_text=corrected_value,
    )
    corrected = get_eligible_candidates_for_downstream(corrected_session)
    assert len(corrected) == 1
    assert corrected[0].effective_text == corrected_value
    assert corrected[0].original_candidate_text == extraction.candidates[0].text
    assert corrected[0].is_corrected is True


def client_extraction():
    """Return deterministic synthetic Phase 3 output without invoking an OCR engine."""
    from backend.app.services.medicine_extraction import MedicineCandidateExtractor

    return MedicineCandidateExtractor().extract("Amoxicillin 500mg capsule PO")
