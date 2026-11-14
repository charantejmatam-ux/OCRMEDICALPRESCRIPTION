"""Optional Gemini comparison for PaddleOCR prescription transcription."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from urllib.parse import quote

import requests
from PIL import Image

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


class GeminiSecondOpinionError(RuntimeError):
    """A safe, user-facing error from the optional Gemini request."""


class GeminiRequestTimeoutError(GeminiSecondOpinionError):
    """Gemini did not return a response before the configured read timeout."""


def gemini_is_configured() -> bool:
    """Return whether a Gemini API key is configured in the backend environment."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def configured_gemini_model() -> str:
    """Return the effective model name without exposing any credentials."""
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _provider_error_message(response: requests.Response, api_key: str) -> str | None:
    try:
        error_data = response.json()
    except ValueError:
        return None

    provider_error = error_data.get("error", {})
    message = provider_error.get("message") if isinstance(provider_error, dict) else None
    if not isinstance(message, str):
        return None
    return message.replace(api_key, "[redacted]")[:500]


def _generate_text(
    image: Image.Image,
    prompt: str,
    *,
    expect_json: bool = False,
) -> tuple[str, str, float]:
    """Send an image and instruction to Gemini; return model, text, and elapsed time."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiSecondOpinionError(
            "Gemini is not configured. Set GEMINI_API_KEY in the backend .env file."
        )

    model = configured_gemini_model()
    allowed_model_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._"
    )
    if not model or any(
        character not in allowed_model_characters for character in model
    ):
        raise GeminiSecondOpinionError("GEMINI_MODEL contains unsupported characters.")

    image_buffer = io.BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_data,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            **({"responseMimeType": "application/json"} if expect_json else {}),
        },
    }

    started = time.perf_counter()
    response: requests.Response | None = None
    attempted_model = model
    for attempt in range(2):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(attempted_model, safe='-._')}:generateContent"
        )
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload,
                # Image transcription can take longer than a short metadata
                # request, especially for detailed handwriting images.
                timeout=(15, 180),
            )
        except requests.Timeout as exc:
            raise GeminiRequestTimeoutError(
                "Gemini did not respond within 180 seconds. Check the network and "
                "Google AI Studio service, then retry or use PaddleOCR."
            ) from exc
        except requests.RequestException as exc:
            raise GeminiSecondOpinionError(
                f"Gemini request failed ({type(exc).__name__})."
            ) from exc

        if response.ok:
            break

        provider_message = _provider_error_message(response, api_key)
        is_retired_model = (
            response.status_code == 404
            and provider_message is not None
            and "no longer available to new users" in provider_message.lower()
            and attempted_model != DEFAULT_GEMINI_MODEL
            and attempt == 0
        )
        if is_retired_model:
            suggested_model = re.search(
                r"use\s+models/([A-Za-z0-9._-]*[A-Za-z0-9_-])",
                provider_message,
                flags=re.IGNORECASE,
            )
            attempted_model = (
                suggested_model.group(1)
                if suggested_model
                else DEFAULT_GEMINI_MODEL
            )
            continue

        message = f"Gemini returned HTTP {response.status_code}"
        if provider_message:
            message += f": {provider_message}"
        raise GeminiSecondOpinionError(message + ".")

    if response is None or not response.ok:
        raise GeminiSecondOpinionError(
            "Gemini did not return a successful response after retrying a retired model."
        )

    try:
        response_data = response.json()
        text = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise GeminiSecondOpinionError(
            "Gemini returned an empty or unrecognized response."
        ) from exc

    if not text:
        raise GeminiSecondOpinionError("Gemini returned an empty response.")

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return attempted_model, text, elapsed_ms


def transcribe_with_gemini(
    image: Image.Image,
) -> tuple[str, str, list[str], float]:
    """Return a verbatim transcription and visible medicine-name candidates."""
    prompt = (
        "Analyze this prescription image and return one JSON object with exactly "
        "these fields: \"transcription\" (string) and "
        "\"possible_medicine_names\" (array of strings). In transcription, copy "
        "all visible text verbatim, preserving line breaks and spelling; mark "
        "unreadable parts as [unclear]. In possible_medicine_names, list only "
        "medicine-name candidates visibly written in the image, using the spelling "
        "you can read. Do not infer, normalize, autocorrect, or recommend a name; "
        "do not include strengths, dosage forms, or instructions in this array. "
        "If none can be identified, return an empty array. A human must verify "
        "both fields. Return valid JSON only."
    )
    model, response_text, elapsed_ms = _generate_text(
        image,
        prompt,
        expect_json=True,
    )
    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise GeminiSecondOpinionError(
            "Gemini returned invalid JSON for transcription and medicine candidates."
        ) from exc

    if not isinstance(response_data, dict):
        raise GeminiSecondOpinionError(
            "Gemini response must be a JSON object."
        )
    transcription = response_data.get("transcription")
    medicine_names = response_data.get("possible_medicine_names")
    if not isinstance(transcription, str) or not transcription.strip():
        raise GeminiSecondOpinionError(
            "Gemini response is missing a non-empty transcription."
        )
    if not isinstance(medicine_names, list) or any(
        not isinstance(name, str) or not name.strip()
        for name in medicine_names
    ):
        raise GeminiSecondOpinionError(
            "Gemini response must include possible_medicine_names as non-empty strings."
        )

    cleaned_names = list(dict.fromkeys(name.strip() for name in medicine_names))
    return model, transcription.strip(), cleaned_names, elapsed_ms


def compare_with_gemini(image: Image.Image, paddle_text: str) -> tuple[str, str, float]:
    """Ask Gemini to independently transcribe an image and compare it with PaddleOCR."""
    prompt = (
        "You are providing a second transcription opinion for a prescription image. "
        "Transcribe only text that is visibly present. Do not infer, correct, or recommend "
        "medicines or doses. Mark unreadable text as [unclear]. Compare your transcription "
        "with the PaddleOCR transcription below and list specific disagreements or uncertain "
        "medicine names, strengths, and instructions. Do not decide which transcription is "
        "correct; a human must verify both.\n\n"
        f"PaddleOCR transcription:\n{paddle_text}"
    )
    return _generate_text(image, prompt)
