"""Gemma vision OCR with cloud API and local Ollama support."""

from __future__ import annotations

import io
from time import perf_counter

import requests
from PIL import Image

from gemma_integration import GemmaSettings, process_prescription_image
from backend.app.services.ocr.base import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
)


class GemmaOCREngine(OCREngine):
    """Transcribe prescription images, then identify medicines from the text."""

    @property
    def name(self) -> str:
        return "gemma"

    @property
    def description(self) -> str:
        return (
            "Gemma image transcription followed by text-only medicine extraction. "
            "Uses the Gemma API when configured and falls back to local Ollama."
        )

    def check_readiness(self) -> EngineReadiness:
        settings = GemmaSettings.from_env()
        if settings.api_key:
            return EngineReadiness(
                engine_name=self.name,
                is_available=True,
                description=self.description,
                details={
                    "provider": "Gemma API with Ollama fallback",
                    "api_model": settings.api_model,
                    "ollama_model": settings.ollama_model,
                    "api_key_configured": True,
                },
            )

        try:
            response = requests.get(
                f"{settings.ollama_base_url}/api/tags",
                timeout=min(settings.timeout_seconds, 3),
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
        except (requests.RequestException, ValueError):
            models = []

        local_model_available = any(
            isinstance(model, dict)
            and isinstance(model.get("name"), str)
            and model["name"].split(":")[0]
            == settings.ollama_model.split(":")[0]
            for model in models
        )
        return EngineReadiness(
            engine_name=self.name,
            is_available=local_model_available,
            description=self.description,
            details={
                "provider": "Ollama",
                "ollama_url": settings.ollama_base_url,
                "ollama_model": settings.ollama_model,
                "api_key_configured": False,
            },
            instructions_if_unavailable=(
                None
                if local_model_available
                else (
                    "Start Ollama and install the configured image-capable Gemma model "
                    f"with `ollama pull {settings.ollama_model}`, or configure GEMMA_API_KEY."
                )
            ),
        )

    def extract_text(self, image: Image.Image, **kwargs: Any) -> OCRExtractionResult:
        del kwargs
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Gemma OCR input must be a non-empty PIL image.")

        started = perf_counter()
        result = process_prescription_image(image_to_png(image))
        elapsed_ms = (perf_counter() - started) * 1000
        raw_text = result["extracted_text"]
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        provider = result["provider"]
        model = result["model"]

        return OCRExtractionResult(
            engine_name=self.name,
            raw_text=raw_text,
            blocks=[
                OCRTextBlock(text=line, line_number=index)
                for index, line in enumerate(lines, start=1)
            ],
            structured_data={
                "extracted_text": raw_text,
                "medicines": result["medicines"],
                "medical_instructions": result["medical_instructions"],
                "human_verification_required": result["human_review_required"],
            },
            processing_time_ms=elapsed_ms,
            metadata={
                "provider": provider,
                "model": model,
                "medicine_provider": result.get("medicine_provider"),
                "medicine_model": result.get("medicine_model"),
                "human_review_required": result["human_review_required"],
            },
        )


def image_to_png(image: Image.Image) -> bytes:
    """Serialize a validated PIL image for the standalone Gemma service."""
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


__all__ = ["GemmaOCREngine"]
