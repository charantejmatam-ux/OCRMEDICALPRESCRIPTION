from __future__ import annotations

import base64
import io
import json
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from gemma_integration.prescription import (
    GemmaIntegrationError,
    GemmaSettings,
    process_prescription_image,
)

EXPECTED = {
    "extracted_text": "Amoxicillin 500 mg\nTake one capsule three times daily for 7 days",
    "medicines": [
        {
            "medicine_name": "Amoxicillin",
            "dosage": "500 mg",
            "frequency": "three times daily",
            "duration": "7 days",
            "instructions": None,
            "confidence": None,
            "confidence_status": "Not available",
            "confidence_source": None,
        }
    ],
    "medical_instructions": ["Take with food"],
}


def _image_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color="white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _response(
    *, status_code: int = 200, text: str | None = None, body: dict[str, Any] | None = None
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.json.return_value = body or (
        {"candidates": [{"content": {"parts": [{"text": text or json.dumps(EXPECTED)}]}}]}
    )
    return response


def _settings(*, api_key: str | None = "test-key") -> GemmaSettings:
    return GemmaSettings(
        api_key=api_key,
        api_model="gemma-test",
        api_base_url="https://example.test/v1beta",
        ollama_base_url="http://localhost:11434",
        ollama_model="gemma3:test",
        timeout_seconds=1,
    )


def test_cloud_sends_image_and_returns_complete_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response()
    post = Mock(return_value=response)
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings())

    assert result == {
        **EXPECTED,
        "provider": "gemma_api",
        "model": "gemma-test",
        "human_review_required": True,
    }
    arguments = post.call_args.kwargs
    assert post.call_args.args[0] == (
        "https://example.test/v1beta/models/gemma-test:generateContent"
    )
    assert arguments["headers"] == {"x-goog-api-key": "test-key"}
    parts = arguments["json"]["contents"][0]["parts"]
    image_part = parts[1]["inlineData"]
    assert image_part["mimeType"] == "image/png"
    assert base64.b64decode(image_part["data"]).startswith(b"\x89PNG")


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_quota_or_server_error_falls_back_to_ollama_with_image(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    cloud_response = _response(status_code=status_code)
    local_response = _response(body={"response": json.dumps(EXPECTED)})
    post = Mock(side_effect=[cloud_response, local_response])
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings())

    assert result["provider"] == "ollama"
    assert result["model"] == "gemma3:test"
    local_payload = post.call_args_list[1].kwargs["json"]
    assert local_payload["format"] == "json"
    assert local_payload["options"] == {
        "temperature": 0.2,
        "repeat_penalty": 1.1,
        "num_predict": 2048,
        "num_ctx": 4096,
    }
    assert len(base64.b64decode(local_payload["images"][0])) > 0
    assert len(post.call_args_list) == 2


def test_missing_cloud_key_runs_complete_ollama_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(return_value=_response(body={"response": json.dumps(EXPECTED)}))
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings(api_key=None))

    assert result["provider"] == "ollama"
    assert {
        key: result[key]
        for key in ("extracted_text", "medicines", "medical_instructions")
    } == EXPECTED
    assert post.call_count == 1
    arguments = post.call_args.kwargs
    assert post.call_args.args[0].endswith("/api/generate")
    assert "Transcribe the image itself" in arguments["json"]["prompt"]
    assert base64.b64decode(arguments["json"]["images"][0]).startswith(b"\x89PNG")


def test_cloud_connection_failure_falls_back_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            requests.ConnectionError("offline"),
            _response(body={"response": json.dumps(EXPECTED)}),
        ]
    )
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings())

    assert result["provider"] == "ollama"
    assert post.call_count == 2


def test_local_timeout_reports_timeout_and_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(side_effect=requests.ReadTimeout("slow model"))
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    with pytest.raises(GemmaIntegrationError, match="timed out after 1 seconds"):
        process_prescription_image(_image_bytes(), _settings(api_key=None))


def test_local_model_error_is_included_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(status_code=500)
    response.text = '{"error":"prediction aborted, token repeat limit reached"}'
    monkeypatch.setattr("gemma_integration.prescription.requests.post", Mock(return_value=response))

    with pytest.raises(GemmaIntegrationError, match="token repeat limit reached"):
        process_prescription_image(_image_bytes(), _settings(api_key=None))


def test_non_quota_cloud_client_error_is_not_silently_fallbacked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(return_value=_response(status_code=401))
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    with pytest.raises(GemmaIntegrationError, match="HTTP 401"):
        process_prescription_image(_image_bytes(), _settings())
    assert post.call_count == 1


@pytest.mark.parametrize("invalid_image", [b"", b"not an image"])
def test_invalid_image_is_rejected_before_service_calls(
    monkeypatch: pytest.MonkeyPatch, invalid_image: bytes
) -> None:
    post = Mock()
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    with pytest.raises(ValueError):
        process_prescription_image(invalid_image, _settings())
    post.assert_not_called()


def test_invalid_model_json_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    post = Mock(
        return_value=_response(
            text='{"extracted_text": "words", "medicines": [], "medical_instructions":'
        )
    )
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    with pytest.raises(GemmaIntegrationError, match="invalid JSON"):
        process_prescription_image(_image_bytes(), _settings())


def test_cloud_computes_exact_token_logprobs_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_text = json.dumps(EXPECTED)
    # Reconstruct token sequence matching the JSON
    prefix = '{"extracted_text": "Amoxicillin 500 mg\\nTake one capsule three times daily for 7 days", "medicines": [{"medicine_name": "'
    suffix = '", "dosage": "500 mg", "frequency": "three times daily", "duration": "7 days", "instructions": null}], "medical_instructions": ["Take with food"]}'
    tokens = [
        {"token": prefix, "logProbability": 0.0},
        {"token": "Amox", "logProbability": -0.02},
        {"token": "icillin", "logProbability": -0.01},
        {"token": suffix, "logProbability": 0.0},
    ]
    cloud_body = {
        "candidates": [
            {
                "content": {"parts": [{"text": json_text}]},
                "logprobsResult": {"chosenCandidates": tokens},
            }
        ]
    }
    post = Mock(return_value=_response(body=cloud_body))
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings())
    assert result["provider"] == "gemma_api"
    med = result["medicines"][0]
    assert med["medicine_name"] == "Amoxicillin"
    assert med["confidence"] is not None
    # exp((-0.02 + -0.01) / 2) * 100 = exp(-0.015) * 100 = 98.5%
    assert med["confidence"] == 98.5
    assert med["confidence_status"] == "Available"
    assert med["confidence_source"] == "gemma_token_logprobs"


def test_visual_clarity_fallback_when_logprobs_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ollama_output = {
        "extracted_text": "Metformin 500mg",
        "medicines": [
            {
                "medicine_name": "Metformin",
                "dosage": "500 mg",
                "frequency": None,
                "duration": None,
                "instructions": None,
                "confidence": 92,
            }
        ],
        "medical_instructions": [],
    }
    post = Mock(return_value=_response(body={"response": json.dumps(ollama_output)}))
    monkeypatch.setattr("gemma_integration.prescription.requests.post", post)

    result = process_prescription_image(_image_bytes(), _settings(api_key=None))
    assert result["provider"] == "ollama"
    med = result["medicines"][0]
    assert med["medicine_name"] == "Metformin"
    assert med["confidence"] == 92.0
    assert med["confidence_status"] == "Available"
    assert med["confidence_source"] == "gemma_visual_clarity"


