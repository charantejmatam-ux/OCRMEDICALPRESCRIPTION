"""Phase 6 tests use synthetic interaction evidence only in test providers."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers import prescriptions
from backend.app.services.drug_interactions import (
    InteractionEvidenceRecord,
    InteractionLookupConcept,
    InteractionProvider,
    InteractionStatus,
    PairEvidenceResponse,
    UnavailableInteractionProvider,
    check_accepted_normalizations,
)
from backend.app.services.medicine_normalization import (
    MatchMethod,
    MedicineAttributes,
    TerminologyConcept,
    TerminologyProvider,
    UnavailableRxNormProvider,
)


class SyntheticTerminologyProvider(TerminologyProvider):
    """Test-only concepts, explicitly not real terminology records."""

    provider_name = "SYNTHETIC TEST TERMINOLOGY"
    source_version = "synthetic-only"
    source_release_date = None

    def __init__(self, *, duplicate_identifier: bool = False, multiple: bool = False) -> None:
        self.duplicate_identifier = duplicate_identifier
        self.multiple = multiple
        self.count = 0

    @property
    def is_configured(self) -> bool:
        return True

    def search(
        self,
        input_text: str,
        attributes: MedicineAttributes,
    ) -> list[TerminologyConcept]:
        del attributes
        self.count += 1
        identifier = (
            "SYNTHETIC-SAME"
            if self.duplicate_identifier
            else f"SYNTHETIC-{uuid4()}"
        )
        concepts = [
            TerminologyConcept(
                source_identifier=identifier,
                name=f"Synthetic concept {input_text}",
                ingredient=None,
                strength=None,
                dosage_form=None,
                match_method=MatchMethod.EXACT,
                evidence="Synthetic fixture; not a genuine terminology concept.",
                source_name=self.provider_name,
            )
        ]
        if self.multiple:
            concepts.append(
                TerminologyConcept(
                    source_identifier=f"{identifier}-ALT",
                    name=f"Synthetic alternative {input_text}",
                    match_method=MatchMethod.PARTIAL,
                    evidence="Synthetic fixture; not a genuine terminology concept.",
                    source_name=self.provider_name,
                )
            )
        return concepts


class SyntheticInteractionProvider(InteractionProvider):
    """Test-only interaction fixture; never a production evidence provider."""

    provider_name = "SYNTHETIC TEST INTERACTION DATA ONLY"
    source_version = "synthetic-interactions-v1"
    release_date = "not-a-real-release"

    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = outcomes or []
        self.calls: list[tuple[str, str]] = []
        self.received_concepts: list[InteractionLookupConcept] = []

    @property
    def is_configured(self) -> bool:
        return True

    def lookup_pair(
        self,
        concept_a: InteractionLookupConcept,
        concept_b: InteractionLookupConcept,
    ) -> PairEvidenceResponse:
        self.calls.append((concept_a.source_identifier, concept_b.source_identifier))
        self.received_concepts.extend((concept_a, concept_b))
        outcome = self.outcomes.pop(0) if self.outcomes else "no_match"
        if isinstance(outcome, Exception):
            raise outcome
        if outcome in {"found", "partial_found"}:
            return PairEvidenceResponse(
                status=InteractionStatus.INTERACTION_FOUND,
                evidence=[
                    InteractionEvidenceRecord(
                        concept_identifier_a=concept_a.source_identifier,
                        concept_identifier_b=concept_b.source_identifier,
                        description="Synthetic test interaction description.",
                        source_severity="Synthetic source severity label",
                        source_name=self.provider_name,
                        source_record_url=None,
                        source_record_identifier=None,
                        dataset_version=None,
                        release_date=None,
                        retrieved_at=None,
                        limitations=["Synthetic test fixture only."],
                    )
                ],
                source_name=self.provider_name,
                source_version=self.source_version,
                release_date=self.release_date,
                retrieved_at=None,
                coverage_complete=outcome == "found",
                limitations=["Synthetic test fixture only."],
            )
        if outcome == "incomplete":
            return PairEvidenceResponse(
                status=InteractionStatus.INCOMPLETE,
                source_name=self.provider_name,
                source_version=self.source_version,
                coverage_complete=False,
                message="Synthetic incomplete response.",
            )
        if outcome == "malformed_pair":
            return PairEvidenceResponse(
                status=InteractionStatus.INTERACTION_FOUND,
                evidence=[
                    InteractionEvidenceRecord(
                        concept_identifier_a="NOT-THE-QUERIED-CONCEPT",
                        concept_identifier_b=concept_b.source_identifier,
                        description="Synthetic malformed pair fixture.",
                        source_name=self.provider_name,
                    )
                ],
                source_name=self.provider_name,
            )
        if outcome == "bad_no_match":
            return PairEvidenceResponse(
                status=InteractionStatus.NO_MATCHING_RECORD_FOUND,
                source_name=self.provider_name,
                coverage_complete=False,
            )
        if outcome == "wrong_source":
            return PairEvidenceResponse(
                status=InteractionStatus.NO_MATCHING_RECORD_FOUND,
                source_name="DIFFERENT SYNTHETIC SOURCE",
            )
        if outcome == "wrong_evidence_source":
            return PairEvidenceResponse(
                status=InteractionStatus.INTERACTION_FOUND,
                evidence=[
                    InteractionEvidenceRecord(
                        concept_identifier_a=concept_a.source_identifier,
                        concept_identifier_b=concept_b.source_identifier,
                        description="Synthetic source mismatch fixture.",
                        source_name="UNRELATED SYNTHETIC EVIDENCE SOURCE",
                    )
                ],
                source_name=self.provider_name,
            )
        if outcome == "malformed":
            return object()  # type: ignore[return-value]
        return PairEvidenceResponse(
            status=InteractionStatus.NO_MATCHING_RECORD_FOUND,
            source_name=self.provider_name,
            source_version=self.source_version,
            release_date=self.release_date,
            coverage_complete=True,
            message="Synthetic fixture found no test record; not a safety conclusion.",
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


def create_accepted_normalization(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    medicine: str,
    *,
    terminology_provider: SyntheticTerminologyProvider | None = None,
    normalization_action: str = "accept",
    review_action: str = "confirm",
) -> tuple[str, str]:
    term_provider = terminology_provider or SyntheticTerminologyProvider()
    monkeypatch.setattr(prescriptions, "terminology_provider", term_provider)

    extracted = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": f"{medicine} 500mg capsule PO BID"},
    )
    assert extracted.status_code == 200
    extraction_data = extracted.json()
    medicine_candidate = next(
        candidate
        for candidate in extraction_data["candidates"]
        if candidate["candidate_type"] == "possible_medicine_name"
    )
    review_id = extraction_data["review_id"]
    review_payload = {"action": review_action}
    if review_action == "correct":
        review_payload["corrected_text"] = f"{medicine} corrected"
    reviewed = client.post(
        f"/prescriptions/review/{review_id}/candidates/{medicine_candidate['candidate_id']}",
        json=review_payload,
    )
    assert reviewed.status_code == 200

    normalization_response = client.post(
        f"/prescriptions/review/{review_id}/candidates/"
        f"{medicine_candidate['candidate_id']}/normalization"
    )
    assert normalization_response.status_code == 201
    normalization = normalization_response.json()

    if normalization_action == "pending":
        return normalization["normalization_id"], ""
    if normalization["status"] == "ambiguous":
        if normalization_action == "mark_ambiguous":
            decision = {"action": "mark_ambiguous"}
        else:
            decision = {
                "action": "accept",
                "selected_concept_id": normalization["proposed_concepts"][0]["source_identifier"],
            }
    elif normalization_action == "accept":
        decision = {
            "action": "accept",
            "selected_concept_id": normalization["proposed_concepts"][0]["source_identifier"],
        }
    else:
        decision = {"action": normalization_action}

    if normalization["status"] in {"pending_review", "ambiguous"}:
        decided = client.post(
            f"/prescriptions/normalization/{normalization['normalization_id']}/decision",
            json=decision,
        )
        expected_status = 200
        assert decided.status_code == expected_status, decided.text
        normalization = decided.json()
    return normalization["normalization_id"], normalization["accepted_concept"][
        "source_identifier"
    ] if normalization["accepted_concept"] else ""


def post_check(client: TestClient, normalization_ids: list[str]):
    return client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": normalization_ids},
    )


def test_nonaccepted_phase5_records_are_blocked(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SyntheticInteractionProvider()
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)

    pending_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
        normalization_action="pending",
    )
    pending_report_response = post_check(client, [pending_id])
    assert pending_report_response.status_code == 201
    pending_report = pending_report_response.json()
    assert pending_report["overall_status"] == "blocked"
    assert provider.calls == []

    rejected_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Metformin",
        normalization_action="reject",
    )
    rejected_report = post_check(client, [rejected_id]).json()
    assert rejected_report["overall_status"] == "blocked"
    assert rejected_report["blocked_inputs"][0]["status"] == "rejected"

    ambiguous_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Paracetamol",
        terminology_provider=SyntheticTerminologyProvider(multiple=True),
        normalization_action="mark_ambiguous",
    )
    ambiguous_report = post_check(client, [ambiguous_id]).json()
    assert ambiguous_report["overall_status"] == "blocked"
    assert ambiguous_report["blocked_inputs"][0]["status"] == "ambiguous"

    monkeypatch.setattr(
        prescriptions,
        "terminology_provider",
        UnavailableRxNormProvider(),
    )
    extracted = client.post(
        "/prescriptions/extract-medicines",
        json={"ocr_text": "Atorvastatin 500mg capsule"},
    ).json()
    candidate = next(
        item for item in extracted["candidates"]
        if item["candidate_type"] == "possible_medicine_name"
    )
    client.post(
        f"/prescriptions/review/{extracted['review_id']}/candidates/{candidate['candidate_id']}",
        json={"action": "confirm"},
    )
    unmatched = client.post(
        f"/prescriptions/review/{extracted['review_id']}/candidates/{candidate['candidate_id']}/normalization"
    ).json()
    unmatched_report = post_check(client, [unmatched["normalization_id"]]).json()
    assert unmatched_report["overall_status"] == "blocked"
    assert unmatched_report["blocked_inputs"][0]["status"] == "unmatched"
    assert provider.calls == []


def test_no_interaction_source_is_configured_and_never_queried(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, first_concept = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, second_concept = create_accepted_normalization(client, monkeypatch, "Metformin")
    assert first_concept != second_concept

    provider = UnavailableInteractionProvider()
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)
    report_response = post_check(client, [first, second])
    assert report_response.status_code == 201
    report = report_response.json()
    assert report["overall_status"] == "not_configured"
    assert report["provider_configured"] is False
    assert report["checked_pair_count"] == 0
    assert report["expected_pair_count"] == 1
    assert report["pair_results"][0]["status"] == "not_configured"
    assert report["check_complete"] is False
    assert "no medicine pairs were checked" in report["message"]
    assert report["pair_results"][0]["concept_a"]["source_identifier"] == first_concept
    assert report["pair_results"][0]["concept_b"]["source_identifier"] == second_concept
    fetched = client.get(
        f"/prescriptions/interactions/reports/{report['report_id']}"
    )
    assert fetched.status_code == 200
    assert fetched.json()["report_id"] == report["report_id"]
    assert client.get(
        f"/prescriptions/interactions/reports/{uuid4()}"
    ).status_code == 404


def test_found_pair_contains_source_provenance_and_optional_fields_stay_null(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    provider = SyntheticInteractionProvider(["found"])
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)

    report = post_check(client, [first, second]).json()
    pair = report["pair_results"][0]
    evidence = pair["evidence"][0]
    assert report["overall_status"] == "interaction_found"
    assert report["check_complete"] is True
    assert pair["status"] == "interaction_found"
    assert evidence["description"] == "Synthetic test interaction description."
    assert evidence["source_severity"] == "Synthetic source severity label"
    assert evidence["source_name"] == provider.provider_name
    assert evidence["source_record_url"] is None
    assert evidence["source_record_identifier"] is None
    assert evidence["dataset_version"] is None
    assert evidence["release_date"] is None
    assert evidence["retrieved_at"] is None
    assert pair["source_version"] == provider.source_version
    assert pair["release_date"] == provider.release_date
    concept = report["checked_concepts"][0]
    assert concept["normalization_id"] in [first, second]
    assert concept["phase4_review_action"] == "confirm"
    assert concept["phase4_reviewed_at"]
    assert concept["phase5_review_action"] == "accept"
    assert concept["phase5_decision_at"]
    assert provider.received_concepts
    assert all(
        not hasattr(item, "original_candidate_text")
        and not hasattr(item, "normalization_input_text")
        for item in provider.received_concepts
    )
    assert all(
        set(item.model_dump()) == set(InteractionLookupConcept.model_fields)
        and item.source_identifier.startswith("SYNTHETIC-")
        and item.source_name.startswith("Synthetic concept ")
        and item.terminology_source == "SYNTHETIC TEST TERMINOLOGY"
        for item in provider.received_concepts
    )


def test_no_matching_record_is_not_described_as_safe(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider(["no_match"]),
    )
    report = post_check(client, [first, second]).json()
    assert report["overall_status"] == "no_matching_record_found"
    assert report["pair_results"][0]["status"] == "no_matching_record_found"
    assert "does not establish" in report["message"]
    assert "safe" in report["safety_notice"]
    assert report["check_complete"] is True


def test_three_medicines_query_every_unique_unordered_pair_once(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [
        create_accepted_normalization(client, monkeypatch, medicine)[0]
        for medicine in ("Amoxicillin", "Metformin", "Paracetamol")
    ]
    provider = SyntheticInteractionProvider(["no_match", "no_match", "no_match"])
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)

    report = post_check(client, ids).json()
    assert report["expected_pair_count"] == 3
    assert report["checked_pair_count"] == 3
    assert len(provider.calls) == 3
    assert len({frozenset(pair) for pair in provider.calls}) == 3
    expected = {
        frozenset((a, b))
        for a, b in combinations(
            [concept["source_identifier"] for concept in report["checked_concepts"]],
            2,
        )
    }
    assert {frozenset(pair) for pair in provider.calls} == expected


def test_partial_provider_failure_is_incomplete_not_complete(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [
        create_accepted_normalization(client, monkeypatch, medicine)[0]
        for medicine in ("Amoxicillin", "Metformin", "Paracetamol")
    ]
    provider = SyntheticInteractionProvider(["found", TimeoutError(), "no_match"])
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)
    report = post_check(client, ids).json()

    assert report["overall_status"] == "incomplete"
    assert report["check_complete"] is False
    assert report["checked_pair_count"] == 3
    assert [pair["status"] for pair in report["pair_results"]] == [
        "interaction_found",
        "source_unavailable",
        "no_matching_record_found",
    ]


def test_found_evidence_with_incomplete_pair_coverage_is_not_complete(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider(["partial_found"]),
    )
    report = post_check(client, [first, second]).json()
    assert report["overall_status"] == "incomplete"
    assert report["check_complete"] is False
    pair = report["pair_results"][0]
    assert pair["status"] == "incomplete"
    assert pair["coverage_complete"] is False
    assert len(pair["evidence"]) == 1


def test_all_provider_timeouts_return_source_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider([TimeoutError()]),
    )
    report = post_check(client, [first, second]).json()
    assert report["overall_status"] == "source_unavailable"
    assert report["pair_results"][0]["status"] == "source_unavailable"
    assert report["check_complete"] is False


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        ("incomplete", "incomplete"),
        ("malformed", "incomplete"),
        ("malformed_pair", "incomplete"),
        ("bad_no_match", "incomplete"),
        ("wrong_source", "incomplete"),
        ("wrong_evidence_source", "incomplete"),
    ],
)
def test_incomplete_or_malformed_provider_data_is_never_a_no_match(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_status: str,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider([outcome]),
    )
    report = post_check(client, [first, second]).json()
    assert report["overall_status"] == expected_status
    assert report["pair_results"][0]["status"] == expected_status
    assert report["check_complete"] is False


def test_no_eligible_medicines_produces_clear_blocked_report(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider(),
    )
    pending_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
        normalization_action="pending",
    )

    report = post_check(client, [pending_id]).json()
    assert report["overall_status"] == "blocked"
    assert report["checked_concepts"] == []
    assert report["pair_results"] == []
    assert "could not be performed" in report["message"]


def test_single_accepted_medicine_is_blocked_without_pair_lookup(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
    )
    provider = SyntheticInteractionProvider()
    monkeypatch.setattr(prescriptions, "interaction_provider", provider)
    report = post_check(client, [normalization_id]).json()
    assert report["overall_status"] == "blocked"
    assert report["expected_pair_count"] == 0
    assert provider.calls == []


def test_request_validation_and_duplicate_concepts(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert client.post("/prescriptions/interactions/check", json={}).status_code == 422
    assert client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": []},
    ).status_code == 422
    assert client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": ["not-a-uuid"]},
    ).status_code == 422
    duplicate_id = str(uuid4())
    assert client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": [duplicate_id, duplicate_id]},
    ).status_code == 422
    assert client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": [duplicate_id], "medicine_names": ["arbitrary"]},
    ).status_code == 422
    assert client.post(
        "/prescriptions/interactions/check",
        json={"normalization_ids": [duplicate_id]},
    ).status_code == 404

    terminology = SyntheticTerminologyProvider(duplicate_identifier=True)
    first, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
        terminology_provider=terminology,
    )
    second, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Metformin",
        terminology_provider=terminology,
    )
    assert first != second
    response = post_check(client, [first, second])
    assert response.status_code == 422
    assert "Duplicate standardized concept" in response.json()["detail"]


def test_report_store_does_not_change_phase5_decision_state(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = create_accepted_normalization(client, monkeypatch, "Amoxicillin")
    second, _ = create_accepted_normalization(client, monkeypatch, "Metformin")
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider(["no_match"]),
    )
    report = post_check(client, [first, second]).json()
    first_normalization = client.get(
        f"/prescriptions/normalization/{first}"
    ).json()
    assert first_normalization["status"] == "accepted"
    assert first_normalization["decision_action"] == "accept"
    assert report["checked_concepts"][0]["normalization_id"] in {first, second}


def test_corrected_phase4_candidate_uses_the_correction_and_preserves_original(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corrected_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
        review_action="correct",
    )
    confirmed_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Metformin",
    )
    monkeypatch.setattr(
        prescriptions,
        "interaction_provider",
        SyntheticInteractionProvider(["no_match"]),
    )
    report = post_check(client, [corrected_id, confirmed_id]).json()
    corrected = next(
        item for item in report["checked_concepts"]
        if item["normalization_id"] == corrected_id
    )
    assert corrected["phase4_review_status"] == "corrected"
    assert corrected["phase4_review_action"] == "correct"
    assert corrected["original_candidate_text"] == "Amoxicillin"
    assert corrected["normalization_input_text"] == "Amoxicillin corrected"

    record = client.get(f"/prescriptions/normalization/{corrected_id}").json()
    assert record["evidence"]["original_candidate_text"] == "Amoxicillin"
    assert record["evidence"]["reviewer_corrected_text"] == "Amoxicillin corrected"
    assert record["evidence"]["normalization_input_text"] == "Amoxicillin corrected"


def test_service_rejects_an_accepted_concept_not_in_its_normalization_record(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization_id, _ = create_accepted_normalization(
        client,
        monkeypatch,
        "Amoxicillin",
    )
    record = prescriptions.normalization_session_store.get(normalization_id)
    assert record is not None
    forged_concept = record.proposed_concepts[0].model_copy(
        update={"source_identifier": "FORGED-NOT-PROPOSED"}
    )
    forged_record = record.model_copy(update={"accepted_concept": forged_concept})
    report = check_accepted_normalizations(
        [forged_record],
        UnavailableInteractionProvider(),
    )
    assert report.overall_status == InteractionStatus.BLOCKED
    assert "does not match a proposed concept" in report.blocked_inputs[0]["reason"]
