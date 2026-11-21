"""Image-to-structured-data prescription extraction with Gemma."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, UnidentifiedImageError

_PROMPT = """Read this prescription image and return only a JSON object with this shape:
{
  "extracted_text": "faithful transcription of all readable text",
  "medicines": [
    {
      "medicine_name": "name as written",
      "dosage": "dose and unit, or null",
      "frequency": "frequency, or null",
      "duration": "duration, or null",
      "instructions": "medicine-specific instructions, or null",
      "confidence": "an integer from 0 to 100 representing visual certainty and legibility, or null if unclear"
    }
  ],
  "medical_instructions": ["other readable instructions"]
}

Transcribe the image itself, line by line. Preserve the written characters and
shorthand; do not infer, correct, normalize, or add medical facts. If any word
or number is uncertain, write [unclear] for that part in extracted_text. Do not
guess a medicine from a partial name or a visually similar brand. Set
medicine_name to null when it cannot be read with confidence, and set unknown
medicine fields to null. Do not classify an unclear shorthand as a dose or
frequency. If no medicine is legible, return an empty medicines array. Do not
provide medical advice."""


@dataclass(frozen=True)
class GemmaSettings:
    """Cloud and Ollama settings, normally populated from environment variables."""

    api_key: str | None = None
    api_model: str = "gemma-3-27b-it"
    api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    timeout_seconds: float = 600

    @classmethod
    def from_env(cls) -> GemmaSettings:
        try:
            from dotenv import dotenv_values
            project_env = Path(__file__).resolve().parents[1] / ".env"
            env_file = dotenv_values(project_env, interpolate=False) if project_env.is_file() else {}
        except Exception:
            env_file = {}

        raw_key = (
            os.getenv("GEMMA_API_KEY")
            or env_file.get("GEMMA_API_KEY")
            or ""
        )
        api_key = raw_key.strip() if isinstance(raw_key, str) and raw_key.strip() else None

        return cls(
            api_key=api_key,
            api_model=os.getenv("GEMMA_API_MODEL") or env_file.get("GEMMA_API_MODEL") or "gemma-3-27b-it",
            api_base_url=(
                os.getenv("GEMMA_API_BASE_URL")
                or env_file.get("GEMMA_API_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta"
            ).rstrip("/"),
            ollama_base_url=(
                os.getenv("OLLAMA_BASE_URL")
                or env_file.get("OLLAMA_BASE_URL")
                or "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_GEMMA_MODEL") or env_file.get("OLLAMA_GEMMA_MODEL") or "gemma3:4b",
            timeout_seconds=float(os.getenv("GEMMA_TIMEOUT_SECONDS") or env_file.get("GEMMA_TIMEOUT_SECONDS") or "600"),
        )


class GemmaIntegrationError(RuntimeError):
    """Raised when Gemma cannot return valid prescription data."""


class _CloudUnavailable(GemmaIntegrationError):
    """Indicates a cloud connection/quota failure that should use local Gemma."""


def _image_as_png(image: bytes | bytearray | str | Path) -> bytes:
    if isinstance(image, (str, Path)):
        try:
            image_bytes = Path(image).read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read prescription image: {exc}") from exc
    elif isinstance(image, (bytes, bytearray)):
        image_bytes = bytes(image)
    else:
        raise TypeError("Image must be bytes or a filesystem path.")

    if not image_bytes:
        raise ValueError("Prescription image is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            normalized = source.convert("RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Input must be a valid, readable image.") from exc


def _extract_token_confidences(
    response_text: str,
    logprobs_result: dict[str, Any] | None,
) -> list[float | None]:
    if not isinstance(logprobs_result, dict):
        return []

    chosen_candidates = logprobs_result.get("chosenCandidates")
    if not isinstance(chosen_candidates, list) or not chosen_candidates:
        return []

    cursor = 0
    token_spans: list[tuple[int, int, float]] = []
    for item in chosen_candidates:
        if not isinstance(item, dict):
            continue
        tok = item.get("token")
        logp = item.get("logProbability")
        if not isinstance(tok, str) or not isinstance(logp, (int, float)):
            continue
        tok_len = len(tok)
        token_spans.append((cursor, cursor + tok_len, float(logp)))
        cursor += tok_len

    if not token_spans:
        return []

    matches = list(re.finditer(r'"medicine_name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', response_text))
    confidences: list[float | None] = []

    for match in matches:
        span_start, span_end = match.start(1), match.end(1)
        if span_start == span_end:
            confidences.append(None)
            continue

        matched_logps = [
            logp for start, end, logp in token_spans
            if start < span_end and end > span_start
        ]

        if not matched_logps:
            confidences.append(None)
            continue

        mean_logp = sum(matched_logps) / len(matched_logps)
        prob = math.exp(mean_logp) * 100.0
        normalized = round(min(100.0, max(0.0, prob)), 1)
        confidences.append(normalized)

    return confidences


def _parse_result(
    response_text: str,
    provider: str,
    model: str,
    logprobs_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = response_text.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3].strip()
        if content.lower().startswith("json"):
            content = content[4:].lstrip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GemmaIntegrationError(
            f"{provider} returned invalid JSON for the prescription."
        ) from exc

    if not isinstance(result, dict):
        raise GemmaIntegrationError(f"{provider} returned a non-object JSON result.")
    extracted_text = result.get("extracted_text")
    medicines = result.get("medicines")
    instructions = result.get("medical_instructions")
    if not isinstance(extracted_text, str):
        raise GemmaIntegrationError(
            f"{provider} response is missing string field 'extracted_text'."
        )
    if not isinstance(medicines, list):
        raise GemmaIntegrationError(
            f"{provider} response is missing array field 'medicines'."
        )
    if not isinstance(instructions, list) or not all(
        isinstance(item, str) for item in instructions
    ):
        raise GemmaIntegrationError(
            f"{provider} response must include string-array 'medical_instructions'."
        )

    confidences = _extract_token_confidences(response_text, logprobs_result)

    normalized_medicines: list[dict[str, Any]] = []
    for idx, medicine in enumerate(medicines):
        if not isinstance(medicine, dict):
            raise GemmaIntegrationError(
                f"{provider} returned a medicine entry that is not an object."
            )
        name = medicine.get("medicine_name", medicine.get("name"))
        if name is not None and not isinstance(name, str):
            raise GemmaIntegrationError(
                f"{provider} returned a non-string medicine name."
            )
        conf = confidences[idx] if idx < len(confidences) else None
        conf_source = "gemma_token_logprobs" if conf is not None else None

        if conf is None:
            raw_conf = medicine.get("confidence") or medicine.get("confidence_score")
            if isinstance(raw_conf, (int, float)) and 0 <= raw_conf <= 100:
                conf = round(float(raw_conf), 1)
                conf_source = "gemma_visual_clarity"
            elif isinstance(raw_conf, str):
                try:
                    parsed = float(raw_conf.rstrip("%").strip())
                    if 0 <= parsed <= 100:
                        conf = round(parsed, 1)
                        conf_source = "gemma_visual_clarity"
                except ValueError:
                    pass

        normalized_medicines.append(
            {
                "medicine_name": name,
                "dosage": _optional_text(medicine.get("dosage")),
                "frequency": _optional_text(medicine.get("frequency")),
                "duration": _optional_text(medicine.get("duration")),
                "instructions": _optional_text(medicine.get("instructions")),
                "confidence": conf,
                "confidence_status": "Available" if conf is not None else "Not available",
                "confidence_source": conf_source,
            }
        )

    return {
        "extracted_text": extracted_text,
        "medicines": normalized_medicines,
        "medical_instructions": instructions,
        "provider": provider,
        "model": model,
        "human_review_required": True,
    }


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise GemmaIntegrationError("Gemma returned a non-string medicine detail.")
    return value


def _extract_cloud(
    image_png: bytes, settings: GemmaSettings
) -> dict[str, Any]:
    if not settings.api_key:
        raise _CloudUnavailable("Gemma API key is not configured.")

    model_path = quote(settings.api_model, safe="")
    url = f"{settings.api_base_url}/models/{model_path}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(image_png).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseLogprobs": True,
            "logprobs": 1,
        },
    }
    try:
        response = requests.post(
            url,
            headers={"x-goog-api-key": settings.api_key},
            json=payload,
            timeout=settings.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise _CloudUnavailable("Gemma API is unavailable.") from exc

    if response.status_code in {408, 429} or response.status_code >= 500:
        raise _CloudUnavailable(
            f"Gemma API is unavailable (HTTP {response.status_code})."
        )
    if not response.ok:
        raise GemmaIntegrationError(
            f"Gemma API returned HTTP {response.status_code}; check its key and model."
        )

    try:
        response_data = response.json()
        candidate = response_data["candidates"][0]
        response_text = candidate["content"]["parts"][0]["text"]
        logprobs_result = candidate.get("logprobsResult")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise GemmaIntegrationError(
            "Gemma API response did not contain generated text."
        ) from exc
    return _parse_result(
        response_text,
        "gemma_api",
        settings.api_model,
        logprobs_result=logprobs_result,
    )


def _extract_ollama(
    image_png: bytes, settings: GemmaSettings
) -> dict[str, Any]:
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": _PROMPT,
        "images": [base64.b64encode(image_png).decode("ascii")],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "num_predict": 2048,
            "num_ctx": 4096,
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=settings.timeout_seconds)
    except requests.Timeout as exc:
        raise GemmaIntegrationError(
            f"Local Ollama inference timed out after "
            f"{settings.timeout_seconds:g} seconds. Increase GEMMA_TIMEOUT_SECONDS "
            "or try a smaller image/model."
        ) from exc
    except requests.RequestException as exc:
        raise GemmaIntegrationError(
            "Could not connect to local Ollama. Confirm it is running at "
            f"{settings.ollama_base_url}."
        ) from exc
    if not response.ok:
        detail = response.text.strip()
        if detail:
            detail = f" Ollama error: {detail[:500]}"
        raise GemmaIntegrationError(
            f"Local Ollama returned HTTP {response.status_code}.{detail} "
            "check that Ollama is running and the configured Gemma model is installed."
        )
    try:
        response_text = response.json()["response"]
    except (KeyError, TypeError, ValueError) as exc:
        raise GemmaIntegrationError(
            "Local Ollama response did not contain generated text."
        ) from exc
    return _parse_result(response_text, "ollama", settings.ollama_model)


def process_prescription_image(
    image: bytes | bytearray | str | Path,
    settings: GemmaSettings | None = None,
) -> dict[str, Any]:
    """Transcribe a prescription image and extract medicine details with Gemma.

    The cloud API is preferred when configured. Connection, server, and quota
    failures fall back to local Ollama; other cloud errors remain explicit.
    """
    configuration = settings or GemmaSettings.from_env()
    image_png = _image_as_png(image)

    try:
        return _extract_cloud(image_png, configuration)
    except _CloudUnavailable as cloud_error:
        try:
            return _extract_ollama(image_png, configuration)
        except GemmaIntegrationError as local_error:
            raise GemmaIntegrationError(
                f"{cloud_error} Local fallback also failed: {local_error}"
            ) from local_error
