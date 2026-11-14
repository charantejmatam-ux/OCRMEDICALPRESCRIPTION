"""Tests for the explicitly requested Gemini second-opinion service."""

from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from backend.app.services.gemini_second_opinion import (
    GeminiSecondOpinionError,
    GeminiRequestTimeoutError,
    compare_with_gemini,
    transcribe_with_gemini,
)
from backend.app.services.ocr.gemini_engine import GeminiOCREngine


def test_compare_with_gemini_sends_image_and_paddle_text() -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "Transcription and comparison"}]}}
        ]
    }

    with (
        patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"},
        ),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            return_value=response,
        ) as post,
    ):
        model, text, elapsed_ms = compare_with_gemini(
            Image.new("RGB", (4, 3), color="white"),
            "Paddle transcription",
        )

    assert model == "gemini-test"
    assert text == "Transcription and comparison"
    assert elapsed_ms >= 0
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "test-key"}
    request_parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert "Paddle transcription" in request_parts[0]["text"]
    encoded_image = request_parts[1]["inline_data"]["data"]
    assert base64.b64decode(encoded_image).startswith(b"\x89PNG")


def test_transcribe_with_gemini_requests_verbatim_transcription() -> None:
    with patch(
        "backend.app.services.ocr.gemini_engine.transcribe_with_gemini",
        return_value=(
            "gemini-test",
            "Metformin 500 mg\nTake daily",
            ["Metformin"],
            45.0,
        ),
    ):
        result = GeminiOCREngine().extract_text(Image.new("RGB", (4, 3)))

    assert result.engine_name == "gemini"
    assert result.raw_text == "Metformin 500 mg\nTake daily"
    assert [block.text for block in result.blocks] == [
        "Metformin 500 mg",
        "Take daily",
    ]
    assert [block.line_number for block in result.blocks] == [1, 2]
    assert result.metadata["human_verification_required"] is True
    assert result.structured_data == {
        "possible_medicine_names": ["Metformin"],
        "medicine_name_source": "Gemini image analysis",
        "human_verification_required": True,
    }
    assert result.confidence_avg is None


def test_transcription_prompt_asks_for_visible_text_only() -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"transcription":"Drug 5 mg",'
                                '"possible_medicine_names":["Drug"]}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            return_value=response,
        ) as post,
    ):
        model, text, medicine_names, _ = transcribe_with_gemini(
            Image.new("RGB", (2, 2))
        )

    assert model == "gemini-3.8-flash"
    assert text == "Drug 5 mg"
    assert medicine_names == ["Drug"]
    prompt = post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
    assert "verbatim" in prompt
    assert "do not infer, normalize, autocorrect" in prompt.lower()
    assert post.call_args.kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_compare_with_gemini_requires_api_key() -> None:
    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
        patch("backend.app.services.gemini_second_opinion.requests.post") as post,
    ):
        with pytest.raises(GeminiSecondOpinionError, match="GEMINI_API_KEY"):
            compare_with_gemini(Image.new("RGB", (2, 2)), "Paddle text")

    post.assert_not_called()


def test_gemini_request_uses_long_read_timeout_and_reports_timeouts_safely() -> None:
    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            side_effect=requests.ReadTimeout("provider slow"),
        ) as post,
    ):
        with pytest.raises(GeminiRequestTimeoutError, match="180 seconds"):
            compare_with_gemini(Image.new("RGB", (2, 2)), "Paddle text")

    assert post.call_args.kwargs["timeout"] == (15, 180)


def test_compare_with_gemini_reports_http_failure_without_response_body() -> None:
    response = MagicMock()
    response.ok = False
    response.status_code = 429
    response.text = "sensitive provider response"
    response.json.return_value = {
        "error": {
            "message": "Quota exceeded for test-key; try again later."
        }
    }

    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            return_value=response,
        ),
    ):
        with pytest.raises(GeminiSecondOpinionError, match="HTTP 429") as error:
            compare_with_gemini(Image.new("RGB", (2, 2)), "Paddle text")

    assert "sensitive provider response" not in str(error.value)
    assert "test-key" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_compare_with_gemini_explains_missing_model_from_provider() -> None:
    response = MagicMock()
    response.ok = False
    response.status_code = 404
    response.json.return_value = {
        "error": {
            "message": (
                "models/gemini-model-test is not found for API version v1beta."
            )
        }
    }

    with (
        patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-model-test"},
        ),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            return_value=response,
        ),
    ):
        with pytest.raises(GeminiSecondOpinionError, match="not found for API version"):
            compare_with_gemini(Image.new("RGB", (2, 2)), "Paddle text")


def test_compare_with_gemini_retries_a_retired_model_using_provider_suggestion() -> None:
    retired_response = MagicMock()
    retired_response.ok = False
    retired_response.status_code = 404
    retired_response.json.return_value = {
        "error": {
            "message": (
                "This model models/gemini-2.5-flash is no longer available to new users. "
                "Please update your code to use models/gemini-3.8-flash."
            )
        }
    }
    success_response = MagicMock()
    success_response.ok = True
    success_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                '{"transcription":"Transcription",'
                                '"possible_medicine_names":["Drug"]}'
                            )
                        }
                    ]
                }
            }
        ]
    }

    with (
        patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash"},
        ),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            side_effect=[retired_response, success_response],
        ) as post,
    ):
        model, text, medicine_names, _ = transcribe_with_gemini(
            Image.new("RGB", (2, 2))
        )

    assert model == "gemini-3.8-flash"
    assert text == "Transcription"
    assert medicine_names == ["Drug"]
    assert post.call_count == 2
    assert "gemini-2.5-flash:generateContent" in post.call_args_list[0].args[0]
    assert "gemini-3.8-flash:generateContent" in post.call_args_list[1].args[0]


def test_transcribe_with_gemini_rejects_malformed_structured_response() -> None:
    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "not json"}]}}]
    }

    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
        patch(
            "backend.app.services.gemini_second_opinion.requests.post",
            return_value=response,
        ),
    ):
        with pytest.raises(GeminiSecondOpinionError, match="invalid JSON"):
            transcribe_with_gemini(Image.new("RGB", (2, 2)))
