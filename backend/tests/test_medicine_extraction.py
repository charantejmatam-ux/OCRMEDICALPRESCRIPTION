"""
backend/tests/test_medicine_extraction.py
Unit and integration tests for Phase 3: Prescription Medicine Candidate Extraction.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.medicine_extraction import (
    CandidateType,
    ExtractionMethod,
    MedicineCandidateExtractor,
    MedicineExtractionResult,
    extract_gemma_medicine_candidates,
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Test client fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def extractor() -> MedicineCandidateExtractor:
    """Direct extractor instance."""
    return MedicineCandidateExtractor()


class TestMedicineCandidateExtractorUnit:
    """Unit tests for MedicineCandidateExtractor logic, offsets, and rules."""

    def test_extract_multiple_candidate_medicines(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        sample_text = (
            "Rx:\n"
            "1. Amoxicillin 500mg capsules PO TID x 7 days\n"
            "2. Paracetamol 650mg tab PRN pain\n"
            "3. Metformin 850 mg tablets with meals\n"
        )
        result = extractor.extract(sample_text)

        assert isinstance(result, MedicineExtractionResult)
        assert result.line_count == 4
        assert result.review_status == "pending_review"

        # Check candidate names found
        med_names = [
            c.text for c in result.candidates if c.candidate_type == CandidateType.MEDICINE_NAME
        ]
        assert "Amoxicillin" in med_names
        assert "Paracetamol" in med_names
        assert "Metformin" in med_names

        # Check strengths found
        strengths = [
            c.text for c in result.candidates if c.candidate_type == CandidateType.STRENGTH
        ]
        assert any("500mg" in s or "500" in s for s in strengths)
        assert any("650mg" in s or "650" in s for s in strengths)
        assert any("850 mg" in s or "850" in s for s in strengths)

        # Check dosage forms found
        dosage_forms = [
            c.text.lower() for c in result.candidates if c.candidate_type == CandidateType.DOSAGE_FORM
        ]
        assert "capsules" in dosage_forms
        assert "tab" in dosage_forms
        assert "tablets" in dosage_forms

    def test_gemma_candidates_use_only_structured_names_found_in_transcription(self) -> None:
        raw_text = "Rx:\nMedicine A 10 mg once daily"
        result = extract_gemma_medicine_candidates(
            raw_text,
            [
                {"medicine_name": "Medicine A", "dosage": "10 mg"},
                {"medicine_name": "Not transcribed"},
                {"medicine_name": None},
            ],
        )

        assert result.metadata["source"] == "gemma_structured_output"
        assert [candidate.text for candidate in result.candidates] == ["Medicine A"]
        candidate = result.candidates[0]
        assert candidate.extraction_method == ExtractionMethod.GEMMA_STRUCTURED
        assert raw_text[candidate.start_char : candidate.end_char] == candidate.text
        assert result.metadata["unmatched_gemma_medicine_names"] == 1

    def test_character_offsets_match_verbatim_text(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        raw = "Rx: Azithromycin 250mg tab daily for 3 days"
        result = extractor.extract(raw)

        for candidate in result.candidates:
            # Sliced substring must exactly equal candidate text
            sliced = raw[candidate.start_char : candidate.end_char]
            assert (
                sliced == candidate.text
            ), f"Offset mismatch: expected '{candidate.text}', got '{sliced}'"

    def test_strengths_and_units_variations(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        sample = (
            "Ciprofloxacin 0.5% eye drops\n"
            "Insulin 10 IU SC\n"
            "Amikacin 250 mg/ml inj\n"
            "Levothyroxine 50mcg tab\n"
            "Cough syrup 5 ml PO\n"
        )
        result = extractor.extract(sample)

        units_found = [
            c.text for c in result.candidates if c.candidate_type == CandidateType.STRENGTH
        ]
        assert any("0.5%" in u for u in units_found)
        assert any("10 IU" in u or "10" in u for u in units_found)
        assert any("250 mg/ml" in u or "mg/ml" in u for u in units_found)
        assert any("50mcg" in u for u in units_found)
        assert any("5 ml" in u for u in units_found)

    def test_unrelated_header_footer_text_not_labeled_as_medicine(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        header_text = (
            "City General Hospital, 123 Healthcare Ave\n"
            "Date: 2026-10-29\n"
            "Patient Name: John Doe\n"
            "Age: 45 Years  Sex: Male\n"
            "Dr. Sample Physician, M.D. Reg #98765\n"
            "Signature: _____________\n"
        )
        result = extractor.extract(header_text)

        med_candidates = [
            c for c in result.candidates if c.candidate_type == CandidateType.MEDICINE_NAME
        ]
        # Common clinic metadata terms must not be labeled as medicines
        assert len(med_candidates) == 0
        assert len(result.unmatched_lines) >= 5

    def test_contextual_extraction_without_dictionary(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        # "Medixolin" is not in any seed dictionary, but precedes strength
        sample = "Medixolin 100mg tab daily"
        result = extractor.extract(sample)

        med_candidate = next(
            (c for c in result.candidates if c.candidate_type == CandidateType.MEDICINE_NAME),
            None,
        )
        assert med_candidate is not None
        assert med_candidate.text == "Medixolin"
        assert med_candidate.extraction_method == ExtractionMethod.CONTEXTUAL_PATTERN
        assert med_candidate.review_status == "pending_review"

    def test_original_text_is_never_silently_corrected(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        # Even if OCR contains misspelled text or odd casing, preserve it exactly
        raw = "Amoxcilin 500mg capsul"
        result = extractor.extract(raw)

        # Candidate name should be verbatim "Amoxcilin", NOT autocorrected to "Amoxicillin"
        names = [
            c.text for c in result.candidates if c.candidate_type == CandidateType.MEDICINE_NAME
        ]
        assert "Amoxcilin" in names
        assert "Amoxicillin" not in names

    def test_repeated_mentions_get_distinct_offsets(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        raw = "Amoxicillin 500mg morning and Amoxicillin 250mg night"
        result = extractor.extract(raw)

        amox_mentions = [
            c for c in result.candidates if c.text == "Amoxicillin"
        ]
        assert len(amox_mentions) == 2
        assert amox_mentions[0].start_char != amox_mentions[1].start_char
        assert raw[amox_mentions[0].start_char : amox_mentions[0].end_char] == "Amoxicillin"
        assert raw[amox_mentions[1].start_char : amox_mentions[1].end_char] == "Amoxicillin"

    def test_contextual_offsets_preserve_leading_whitespace_and_crlf(self) -> None:
        raw = "   Medixolin 100mg tab\r\n  Othermed 50mg tablet"
        result = MedicineCandidateExtractor().extract(raw)
        medicine_candidates = [
            candidate
            for candidate in result.candidates
            if candidate.candidate_type == CandidateType.MEDICINE_NAME
        ]
        assert [candidate.text for candidate in medicine_candidates] == [
            "Medixolin",
            "Othermed",
        ]
        for candidate in medicine_candidates:
            assert raw[candidate.start_char : candidate.end_char] == candidate.text
        assert medicine_candidates[0].line_number == 1
        assert medicine_candidates[1].line_number == 2

    def test_offsets_preserve_blank_lines_and_repeated_text(self) -> None:
        raw = (
            "  Amoxicillin 500mg tab\r\n"
            "\r\n"
            " Amoxicillin 250mg tab\r\n"
            "  Amoxicillin 125mg tab"
        )
        result = MedicineCandidateExtractor().extract(raw)
        candidates = [
            candidate
            for candidate in result.candidates
            if candidate.candidate_type == CandidateType.MEDICINE_NAME
        ]

        assert [candidate.text for candidate in candidates] == ["Amoxicillin"] * 3
        assert [candidate.line_number for candidate in candidates] == [1, 3, 4]
        assert len({candidate.start_char for candidate in candidates}) == 3
        for candidate in candidates:
            assert raw[candidate.start_char : candidate.end_char] == candidate.text

    def test_unrecognized_name_is_extracted_without_reference_vocabulary(self) -> None:
        extractor = MedicineCandidateExtractor(reference_vocab=set())
        result = extractor.extract("Unlisted Brand 100mg tablet")
        candidate = next(
            candidate for candidate in result.candidates
            if candidate.candidate_type == CandidateType.MEDICINE_NAME
        )
        assert candidate.text == "Unlisted Brand"

    def test_instruction_only_text_with_frequency_returns_no_medicine_name(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        result = extractor.extract("Take once daily")

        assert result.candidates
        assert any(
            candidate.candidate_type == CandidateType.FREQUENCY_INSTRUCTION
            for candidate in result.candidates
        )
        assert not any(
            candidate.candidate_type == CandidateType.MEDICINE_NAME
            for candidate in result.candidates
        )

    def test_medicine_name_before_route_and_frequency_is_still_extracted(
        self, extractor: MedicineCandidateExtractor
    ) -> None:
        result = extractor.extract("Medixolin PO daily")
        medicine_names = [
            candidate.text
            for candidate in result.candidates
            if candidate.candidate_type == CandidateType.MEDICINE_NAME
        ]

        assert medicine_names == ["Medixolin"]

    def test_empty_string_rejected(self, extractor: MedicineCandidateExtractor) -> None:
        with pytest.raises(ValueError, match="empty"):
            extractor.extract("")

    def test_whitespace_string_rejected(self, extractor: MedicineCandidateExtractor) -> None:
        with pytest.raises(ValueError, match="empty"):
            extractor.extract("   \n\t  ")

    def test_excessive_length_rejected(self, extractor: MedicineCandidateExtractor) -> None:
        huge_text = "A" * 60_000
        with pytest.raises(ValueError, match="exceeds maximum"):
            extractor.extract(huge_text)


class TestMedicineExtractionEndpoint:
    """Integration tests for POST /prescriptions/extract-medicines."""

    def test_endpoint_success(self, client: TestClient) -> None:
        payload = {
            "ocr_text": "Rx: Ciprofloxacin 500mg tab BID x 5 days\nParacetamol 650mg tab PRN"
        }
        response = client.post("/prescriptions/extract-medicines", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["review_status"] == "pending_review"
        assert len(data["candidates"]) >= 4
        assert len(data["warnings"]) >= 1

        med_texts = [
            c["text"]
            for c in data["candidates"]
            if c["candidate_type"] == "possible_medicine_name"
        ]
        assert "Ciprofloxacin" in med_texts
        assert "Paracetamol" in med_texts

    def test_endpoint_empty_text_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/extract-medicines",
            json={"ocr_text": "   "},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_endpoint_missing_field_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/extract-medicines",
            json={},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_endpoint_excessive_length_rejected(self, client: TestClient) -> None:
        payload = {"ocr_text": "Amoxicillin 500mg\n" * 4000}  # > 50,000 chars
        response = client.post("/prescriptions/extract-medicines", json=payload)
        assert response.status_code == 422  # Exceeds max_length in Pydantic schema
