from __future__ import annotations

import io
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.services.ocr.base import EngineReadiness
from backend.app.services.ocr.gemma_engine import GemmaOCREngine
from gemma_integration import GemmaSettings


def _settings(api_key: str | None = None) -> GemmaSettings:
    return GemmaSettings(
        api_key=api_key,
        api_model="gemma-api-test",
        ollama_model="gemma-local-test:4b",
    )


def _image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (20, 12), (230, 60, 40)).save(output, format="PNG")
    return output.getvalue()


def test_readiness_uses_installed_local_gemma_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.services.ocr.gemma_engine.GemmaSettings.from_env",
        lambda: _settings(),
    )
    response = Mock()
    response.json.return_value = {"models": [{"name": "gemma-local-test:4b"}]}
    response.raise_for_status.return_value = None
    get = Mock(return_value=response)
    monkeypatch.setattr("backend.app.services.ocr.gemma_engine.requests.get", get)

    readiness = GemmaOCREngine().check_readiness()

    assert readiness.engine_name == "gemma"
    assert readiness.is_available is True
    assert readiness.details["provider"] == "Ollama"


def test_readiness_uses_cloud_gemma_when_key_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.services.ocr.gemma_engine.GemmaSettings.from_env",
        lambda: _settings(api_key="test-key"),
    )
    get = Mock()
    monkeypatch.setattr("backend.app.services.ocr.gemma_engine.requests.get", get)

    readiness = GemmaOCREngine().check_readiness()

    assert readiness.is_available is True
    assert readiness.details["api_key_configured"] is True
    get.assert_not_called()


def test_gemma_engine_maps_complete_structured_output(monkeypatch) -> None:
    source_result = {
        "extracted_text": "Medicine A 10 mg\nTake once daily",
        "medicines": [
            {
                "medicine_name": "Medicine A",
                "dosage": "10 mg",
                "frequency": "once daily",
                "duration": None,
                "instructions": "Take with food",
            }
        ],
        "medical_instructions": ["Take with food"],
        "provider": "ollama",
        "model": "gemma3:4b",
        "human_review_required": True,
    }
    process = Mock(return_value=source_result)
    monkeypatch.setattr(
        "backend.app.services.ocr.gemma_engine.process_prescription_image",
        process,
    )

    result = GemmaOCREngine().extract_text(Image.new("L", (20, 12), "white"))

    sent_image = Image.open(io.BytesIO(process.call_args.args[0]))
    assert sent_image.mode == "RGB"
    assert result.engine_name == "gemma"
    assert result.raw_text == source_result["extracted_text"]
    assert result.structured_data == {
        "extracted_text": source_result["extracted_text"],
        "medicines": source_result["medicines"],
        "medical_instructions": source_result["medical_instructions"],
        "human_verification_required": True,
    }
    assert result.metadata["provider"] == "ollama"


def test_gemma_engine_runs_through_existing_image_ocr_endpoint(monkeypatch) -> None:
    source_result = {
        "extracted_text": "Medicine A 10 mg once daily",
        "medicines": [
            {
                "medicine_name": "Medicine A",
                "dosage": "10 mg",
                "frequency": "once daily",
                "duration": None,
                "instructions": None,
            }
        ],
        "medical_instructions": [],
        "provider": "ollama",
        "model": "gemma3:4b",
        "human_review_required": True,
    }
    received_pixels = []

    def fake_process_prescription_image(image_bytes: bytes) -> dict:
        with Image.open(io.BytesIO(image_bytes)) as image:
            received_pixels.append(image.convert("RGB").getpixel((0, 0)))
        return source_result

    monkeypatch.setattr(
        "backend.app.services.ocr.gemma_engine.process_prescription_image",
        fake_process_prescription_image,
    )
    monkeypatch.setattr(
        GemmaOCREngine,
        "check_readiness",
        lambda self: EngineReadiness(
            engine_name="gemma",
            is_available=True,
            description="test",
        ),
    )
    rule_extractor = Mock(
        side_effect=AssertionError("Gemma findings must not use the rule-based extractor.")
    )
    monkeypatch.setattr(
        "backend.app.routers.prescriptions.medicine_extractor.extract",
        rule_extractor,
    )

    with TestClient(app) as client:
        response = client.post(
            "/prescriptions/ocr",
            files={"file": ("rx.png", _image_bytes(), "image/png")},
            data={"engine": "gemma", "include_medicine_extraction": "true"},
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["raw_text"] == source_result["extracted_text"]
    assert result["structured_data"]["medicines"] == source_result["medicines"]
    assert result["metadata"]["provider"] == "ollama"
    assert received_pixels == [(230, 60, 40)]
    extraction = response.json()["medicine_extraction"]
    assert extraction["metadata"]["source"] == "gemma_structured_output"
    medicine_candidates = [
        candidate
        for candidate in extraction["candidates"]
        if candidate["candidate_type"] == "possible_medicine_name"
    ]
    assert [candidate["text"] for candidate in medicine_candidates] == ["Medicine A"]
    assert medicine_candidates[0]["extraction_method"] == "gemma_structured_output"
    rule_extractor.assert_not_called()


def test_gemma_instruction_only_output_does_not_break_optional_extraction(
    monkeypatch,
) -> None:
    source_result = {
        "extracted_text": "Take once daily",
        "medicines": [],
        "medical_instructions": ["once daily"],
        "provider": "ollama",
        "model": "gemma3:4b",
        "human_review_required": True,
    }
    monkeypatch.setattr(
        "backend.app.services.ocr.gemma_engine.process_prescription_image",
        lambda _: source_result,
    )
    monkeypatch.setattr(
        GemmaOCREngine,
        "check_readiness",
        lambda self: EngineReadiness(
            engine_name="gemma",
            is_available=True,
            description="test",
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/prescriptions/ocr",
            files={"file": ("rx.png", _image_bytes(), "image/png")},
            data={"engine": "gemma", "include_medicine_extraction": "true"},
        )

    assert response.status_code == 200
    extraction = response.json()["medicine_extraction"]
    assert extraction["raw_text"] == "Take once daily"
    assert not any(
        candidate["candidate_type"] == "possible_medicine_name"
        for candidate in extraction["candidates"]
    )
