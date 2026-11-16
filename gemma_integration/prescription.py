"""Image-to-structured-data prescription extraction with Gemma."""

from __future__ import annotations

import base64
import io
import json
import os
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
      "instructions": "medicine-specific instructions, or null"
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
        return cls(
            api_key=os.getenv("GEMMA_API_KEY") or None,
            api_model=os.getenv("GEMMA_API_MODEL", "gemma-3-27b-it"),
            api_base_url=os.getenv(
                "GEMMA_API_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ).rstrip("/"),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_GEMMA_MODEL", "gemma3:4b"),
            timeout_seconds=float(os.getenv("GEMMA_TIMEOUT_SECONDS", "600")),
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


def _parse_result(response_text: str, provider: str, model: str) -> dict[str, Any]:
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

    normalized_medicines: list[dict[str, str | None]] = []
    for medicine in medicines:
        if not isinstance(medicine, dict):
            raise GemmaIntegrationError(
                f"{provider} returned a medicine entry that is not an object."
            )
        name = medicine.get("medicine_name", medicine.get("name"))
        if name is not None and not isinstance(name, str):
            raise GemmaIntegrationError(
                f"{provider} returned a non-string medicine name."
            )
        normalized_medicines.append(
            {
                "medicine_name": name,
                "dosage": _optional_text(medicine.get("dosage")),
                "frequency": _optional_text(medicine.get("frequency")),
                "duration": _optional_text(medicine.get("duration")),
                "instructions": _optional_text(medicine.get("instructions")),
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
        "generationConfig": {"temperature": 0.1},
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
        response_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise GemmaIntegrationError(
            "Gemma API response did not contain generated text."
        ) from exc
    return _parse_result(response_text, "gemma_api", settings.api_model)


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
            "num_predict": 1024,
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
