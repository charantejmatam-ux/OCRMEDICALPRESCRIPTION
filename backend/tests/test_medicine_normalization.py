"""Phase 5 tests use only clearly labeled synthetic terminology providers."""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import prescriptions
from backend.app.services.medicine_extraction import MedicineCandidateExtractor
from backend.app.services.medicine_normalization import (
    InvalidNormalizationTransitionError,
    MatchMethod,
    NormalizationAction,
    NormalizationSessionStore,
    TerminologyConcept,
    TerminologyProvider,
    MedicineAttributes,
    UnavailableRxNormProvider,
    RxNavApproximateProvider,
    normalize_eligible_candidate,
)
from backend.app.services import medicine_normalization
from backend.app.services.review_workflow import (
    ReviewAction,
    ReviewSessionStore,
    get_eligible_candidates_for_downstream,
)


class SyntheticTerminologyProvider(TerminologyProvider):
    """Test fixture only; these records are not real RxNorm or clinical data."""

    provider_name = "SYNTHETIC TEST DATA ONLY"
    source_version = "synthetic-fixture-v1"
    source_release_date = "not-a-real-release"

    def __init__(self, concepts: list[TerminologyConcept]) -> None:
        self.concepts = concepts
        self.search_inputs: list[tuple[str, object]] = []

    @property
    def is_configured(self) -> bool:
        return True

    def search(self, input_text: str, attributes):
        self.search_inputs.append((input_text, attributes))
        return self.concepts


def synthetic_concept(
    identifier: str,
    name: str,
    strength: str | None = "500mg",
    dosage_form: str | None = "capsule",
) -> TerminologyConcept:
    return TerminologyConcept(
        source_identifier=identifier,
        name=name,
        ingredient=name,
        strength=strength,
        dosage_form=dosage_form,
        match_method=MatchMethod.EXACT,
        evidence="Synthetic fixture record; not a real terminology entry.",
        source_name="SYNTHETIC TEST DATA ONLY",
        source_version="synthetic-fixture-v1",
        source_release_date="not-a-real-release",
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def create_review(client: TestClient, raw_text: str = "Amoxicillin 500mg capsule PO BID"):
    response = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": raw_text},
    )
    assert response.status_code == 200
    data = response.json()
    med_candidate = next(
        item for item in data["candidates"]
        if item["candidate_type"] == "possible_medicine_name"
    )
    return data["review_id"], med_candidate


def apply_phase4_action(
    client: TestClient,
    review_id: str,
    candidate: dict,
    action: str,
    corrected_text: str | None = None,
) -> None:
    payload = {"action": action}
    if corrected_text is not None:
        payload["corrected_text"] = corrected_text
    response = client.post(
        f"/prescriptions/review/{review_id}/candidates/{candidate['candidate_id']}",
        json=payload,
    )
    assert response.status_code == 200, response.text


def normalization_url(review_id: str, candidate_id: str) -> str:
    return (
        f"/prescriptions/review/{review_id}/candidates/{candidate_id}/normalization"
    )


def decision_url(normalization_id: str) -> str:
    return f"/prescriptions/normalization/{normalization_id}/decision"


@pytest.mark.parametrize(
    ("phase4_action", "correction"),
    [
        (None, None),
        ("reject", None),
        ("mark_uncertain", None),
    ],
)
def test_candidates_that_fail_phase4_gate_cannot_be_normalized(
    client: TestClient,
    phase4_action: str | None,
    correction: str | None,
) -> None:
    review_id, candidate = create_review(client)
    if phase4_action:
        apply_phase4_action(client, review_id, candidate, phase4_action, correction)
    response = client.post(normalization_url(review_id, candidate["candidate_id"]))
    assert response.status_code == 409


def test_eligible_non_medicine_attribute_is_not_normalized(client: TestClient) -> None:
    response = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": "Amoxicillin 500mg capsule PO BID"},
    )
    assert response.status_code == 200
    data = response.json()
    strength = next(
        item for item in data["candidates"] if item["candidate_type"] == "strength"
    )
    apply_phase4_action(client, data["review_id"], strength, "confirm")
    blocked = client.post(
        normalization_url(data["review_id"], strength["candidate_id"])
    )
    assert blocked.status_code == 409


@pytest.mark.parametrize(
    ("phase4_action", "correction", "expected_input"),
    [
        ("confirm", None, "Amoxicillin"),
        ("correct", "Amoxycillin reviewer value", "Amoxycillin reviewer value"),
    ],
)
def test_gate_output_uses_phase4_confirmed_or_corrected_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    phase4_action: str,
    correction: str | None,
    expected_input: str,
) -> None:
    provider = SyntheticTerminologyProvider(
        [synthetic_concept("SYNTHETIC-1", "Synthetic example concept")]
    )
    monkeypatch.setattr(prescriptions, "terminology_provider", provider)
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, phase4_action, correction)
    response = client.post(normalization_url(review_id, candidate["candidate_id"]))

    assert response.status_code == 201
    record = response.json()
    assert provider.search_inputs[0][0] == expected_input
    assert record["evidence"]["normalization_input_text"] == expected_input
    assert record["evidence"]["original_candidate_text"] == candidate["text"]
    assert record["evidence"]["reviewer_corrected_text"] == correction
    assert record["evidence"]["original_ocr_text"] == "Amoxicillin 500mg capsule PO BID"
    assert record["evidence"]["start_char"] == candidate["start_char"]
    assert record["evidence"]["end_char"] == candidate["end_char"]
    assert record["evidence"]["evidence_context"] == candidate["evidence_context"]
    assert record["evidence"]["extraction_method"] == candidate["extraction_method"]
    assert record["evidence"]["heuristic_score"] == candidate["heuristic_score"]
    assert record["evidence"]["has_dictionary_match"] == candidate["has_dictionary_match"]
    assert record["evidence"]["phase4_review_action"] == phase4_action
    assert record["evidence"]["phase4_reviewed_at"]
    if phase4_action == "correct":
        assert record["evidence"]["has_spillover_or_ambiguity"] is candidate[
            "has_spillover_or_ambiguity"
        ]
    assert record["evidence"]["attributes"] == {
        "strength": "500mg",
        "dosage_form": "capsule",
        "route": "PO",
        "frequency": "BID",
    }

    reviewed = client.get(f"/prescriptions/review/{review_id}").json()
    original_candidate = reviewed["candidates"][candidate["candidate_id"]]
    assert original_candidate["original_candidate_text"] == candidate["text"]
    assert original_candidate["reviewer_corrected_text"] == correction
    assert original_candidate["original_ocr_text"] == "Amoxicillin 500mg capsule PO BID"


@pytest.mark.parametrize("match_method", ["exact", "fuzzy", "partial"])
def test_provider_match_method_is_preserved_without_relabeling(
    match_method: str,
) -> None:
    extraction = MedicineCandidateExtractor().extract("Amoxicillin 500mg capsule")
    review_store = ReviewSessionStore()
    session = review_store.create_session(extraction.raw_text, extraction.candidates)
    med_candidate = next(
        candidate
        for candidate in session.candidates.values()
        if candidate.candidate_type == "possible_medicine_name"
    )
    review_store.apply_action(session.review_id, med_candidate.candidate_id, ReviewAction.CONFIRM)
    eligible = next(
        item
        for item in get_eligible_candidates_for_downstream(session)
        if item.candidate_id == med_candidate.candidate_id
    )
    concept = synthetic_concept("SYNTHETIC-METHOD", "Synthetic method example")
    concept.match_method = MatchMethod(match_method)
    record = normalize_eligible_candidate(
        session.review_id,
        eligible,
        SyntheticTerminologyProvider([concept]),
    )
    assert record.proposed_concepts[0].match_method.value == match_method


def test_unconfigured_rxnorm_provider_returns_unavailable_unmatched_record(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prescriptions, "terminology_provider", UnavailableRxNormProvider())
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, "confirm")
    response = client.post(normalization_url(review_id, candidate["candidate_id"]))

    assert response.status_code == 201
    record = response.json()
    assert record["provider_name"] == "RxNorm"
    assert record["provider_configured"] is False
    assert record["provider_version"] is None
    assert record["provider_release_date"] is None
    assert record["proposed_concepts"] == []
    assert record["status"] == "unmatched"
    assert "not configured" in record["resolution_reason"]
    assert UnavailableRxNormProvider().search("unknown", MedicineAttributes()) == []


def test_rxnav_provider_preserves_native_rank_and_score_without_calling_it_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return (
                b'{"approximateGroup":{"candidate":[{"rxcui":"123",'
                b'"name":"Example medicine 500 MG Oral Tablet","score":"91",'
                b'"rank":"1","source":"RxNorm"}]}}'
            )

    monkeypatch.setattr(medicine_normalization, "urlopen", lambda *_a, **_k: FakeResponse())
    concepts = RxNavApproximateProvider().search("Example medicine", MedicineAttributes())

    assert len(concepts) == 1
    assert concepts[0].source_identifier == "123"
    assert concepts[0].match_rank == 1
    assert concepts[0].match_score == 91
    assert "not a probability" in concepts[0].evidence


def test_unknown_name_with_configured_provider_is_unmatched(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SyntheticTerminologyProvider([])
    monkeypatch.setattr(prescriptions, "terminology_provider", provider)
    review_id, candidate = create_review(client, "Unknownbrand 500mg capsule")
    apply_phase4_action(client, review_id, candidate, "confirm")
    record = client.post(
        normalization_url(review_id, candidate["candidate_id"])
    ).json()
    assert record["status"] == "unmatched"
    assert record["provider_configured"] is True
    assert record["proposed_concepts"] == []


def test_multiple_concepts_stay_ambiguous_until_one_is_selected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concepts = [
        synthetic_concept("SYNTHETIC-A", "Synthetic alternative A"),
        synthetic_concept("SYNTHETIC-B", "Synthetic alternative B"),
    ]
    monkeypatch.setattr(
        prescriptions,
        "terminology_provider",
        SyntheticTerminologyProvider(concepts),
    )
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, "confirm")
    created = client.post(normalization_url(review_id, candidate["candidate_id"]))
    record = created.json()

    assert created.status_code == 201
    assert record["status"] == "ambiguous"
    assert record["accepted_concept"] is None

    accepted = client.post(
        decision_url(record["normalization_id"]),
        json={"action": "accept", "selected_concept_id": "SYNTHETIC-B"},
    )
    assert accepted.status_code == 200
    accepted_record = accepted.json()
    assert accepted_record["status"] == "accepted"
    assert accepted_record["accepted_concept"]["source_identifier"] == "SYNTHETIC-B"
    assert accepted_record["decision_action"] == "accept"
    assert accepted_record["decision_at"]
    assert accepted_record["evidence"]["phase4_review_status"] == "confirmed"

    phase4 = client.get(f"/prescriptions/review/{review_id}").json()
    assert phase4["candidates"][candidate["candidate_id"]]["review_status"] == "confirmed"


def test_strength_and_form_mismatches_are_visible_and_require_acknowledgement(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SyntheticTerminologyProvider(
        [synthetic_concept("SYNTHETIC-MISMATCH", "Synthetic mismatch", "250mg", "tablet")]
    )
    monkeypatch.setattr(prescriptions, "terminology_provider", provider)
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, "confirm")
    result = client.post(normalization_url(review_id, candidate["candidate_id"]))
    normalization_id = result.json()["normalization_id"]
    concept = result.json()["proposed_concepts"][0]
    assert concept["attribute_match"] == "mismatch"
    assert len(concept["attribute_warnings"]) == 2

    rejected_accept = client.post(
        decision_url(normalization_id),
        json={"action": "accept", "selected_concept_id": "SYNTHETIC-MISMATCH"},
    )
    assert rejected_accept.status_code == 422

    accepted = client.post(
        decision_url(normalization_id),
        json={
            "action": "accept",
            "selected_concept_id": "SYNTHETIC-MISMATCH",
            "acknowledge_attribute_mismatches": True,
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"
    assert accepted.json()["accepted_concept"]["attribute_match"] == "mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "approve"},
        {"action": "accept"},
        {"action": "accept", "selected_concept_id": "unknown"},
        {"action": "reject", "selected_concept_id": "SYNTHETIC-1"},
        {"action": "reject", "acknowledge_attribute_mismatches": True},
    ],
)
def test_invalid_normalization_decision_payloads_are_rejected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
) -> None:
    monkeypatch.setattr(
        prescriptions,
        "terminology_provider",
        SyntheticTerminologyProvider([synthetic_concept("SYNTHETIC-1", "Synthetic concept")]),
    )
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, "confirm")
    normalization_id = client.post(
        normalization_url(review_id, candidate["candidate_id"])
    ).json()["normalization_id"]
    assert client.post(decision_url(normalization_id), json=payload).status_code == 422


def test_unknown_ids_duplicate_normalization_and_repeated_actions_are_safe(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prescriptions,
        "terminology_provider",
        SyntheticTerminologyProvider([synthetic_concept("SYNTHETIC-1", "Synthetic concept")]),
    )
    assert client.get(f"/prescriptions/normalization/{uuid4()}").status_code == 404
    assert client.post(
        f"/prescriptions/normalization/{uuid4()}/decision",
        json={"action": "reject"},
    ).status_code == 404
    assert client.get("/prescriptions/normalization/not-a-uuid").status_code == 422

    review_id, candidate = create_review(client)
    assert client.post(
        normalization_url(str(uuid4()), candidate["candidate_id"])
    ).status_code == 404
    assert client.post(
        f"/prescriptions/review/{review_id}/candidates/not-a-uuid/normalization"
    ).status_code == 422
    assert client.post(
        normalization_url(review_id, str(uuid4()))
    ).status_code == 404
    apply_phase4_action(client, review_id, candidate, "confirm")
    normalization_endpoint = normalization_url(review_id, candidate["candidate_id"])
    first = client.post(normalization_endpoint)
    assert first.status_code == 201
    assert client.post(normalization_endpoint).status_code == 409

    normalization_id = first.json()["normalization_id"]
    assert client.post(
        decision_url(normalization_id),
        json={"action": "reject"},
    ).status_code == 200
    assert client.post(
        decision_url(normalization_id),
        json={"action": "reject"},
    ).status_code == 409


def test_single_candidate_has_no_default_accept_and_can_be_rejected(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prescriptions,
        "terminology_provider",
        SyntheticTerminologyProvider([synthetic_concept("SYNTHETIC-1", "Synthetic concept")]),
    )
    review_id, candidate = create_review(client)
    apply_phase4_action(client, review_id, candidate, "confirm")
    created = client.post(normalization_url(review_id, candidate["candidate_id"]))
    assert created.status_code == 201
    record = created.json()
    assert record["status"] == "pending_review"
    assert record["accepted_concept"] is None

    rejected = client.post(
        decision_url(record["normalization_id"]),
        json={"action": "reject"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"


def test_direct_service_rejects_raw_review_objects() -> None:
    extraction = MedicineCandidateExtractor().extract("Amoxicillin 500mg capsule")
    review_store = ReviewSessionStore()
    review_session = review_store.create_session(extraction.raw_text, extraction.candidates)
    medicine = next(
        candidate
        for candidate in review_session.candidates.values()
        if candidate.candidate_type == "possible_medicine_name"
    )
    review_store.apply_action(review_session.review_id, medicine.candidate_id, ReviewAction.CONFIRM)
    with pytest.raises(TypeError, match="DownstreamEligibleCandidate"):
        normalize_eligible_candidate(
            review_session.review_id,
            medicine,
            UnavailableRxNormProvider(),
        )


def test_normalization_store_marks_ambiguous_only_when_multiple_concepts_exist() -> None:
    extraction = MedicineCandidateExtractor().extract("Amoxicillin 500mg capsule")
    review_store = ReviewSessionStore()
    review_session = review_store.create_session(extraction.raw_text, extraction.candidates)
    medicine = next(
        candidate
        for candidate in review_session.candidates.values()
        if candidate.candidate_type == "possible_medicine_name"
    )
    review_store.apply_action(review_session.review_id, medicine.candidate_id, ReviewAction.CONFIRM)
    eligible = next(
        item
        for item in get_eligible_candidates_for_downstream(review_session)
        if item.candidate_id == medicine.candidate_id
    )
    record = normalize_eligible_candidate(
        review_session.review_id,
        eligible,
        SyntheticTerminologyProvider(
            [synthetic_concept("SYNTHETIC-1", "Synthetic concept")]
        ),
    )
    store = NormalizationSessionStore()
    stored = store.create(record)
    with pytest.raises(ValueError, match="multiple proposed concepts"):
        store.decide(stored.normalization_id, NormalizationAction.MARK_AMBIGUOUS)
    with pytest.raises(ValueError, match="does not belong"):
        store.decide(
            stored.normalization_id,
            NormalizationAction.ACCEPT,
            selected_concept_id="UNKNOWN",
        )

    ambiguous = normalize_eligible_candidate(
        review_session.review_id,
        eligible,
        SyntheticTerminologyProvider(
            [
                synthetic_concept("SYNTHETIC-A", "Synthetic A"),
                synthetic_concept("SYNTHETIC-B", "Synthetic B"),
            ]
        ),
    )
    ambiguous_store = NormalizationSessionStore()
    ambiguous_record = ambiguous_store.create(ambiguous)
    ambiguous_store.decide(
        ambiguous_record.normalization_id,
        NormalizationAction.MARK_AMBIGUOUS,
    )
    with pytest.raises(InvalidNormalizationTransitionError):
        ambiguous_store.decide(
            ambiguous_record.normalization_id,
            NormalizationAction.MARK_AMBIGUOUS,
        )
