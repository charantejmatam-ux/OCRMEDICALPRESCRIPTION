"""Google Gemini vision OCR engine; requests use the configured backend API key."""

from __future__ import annotations

from typing import Any

from PIL import Image

from backend.app.services.gemini_second_opinion import (
    configured_gemini_model,
    gemini_is_configured,
    transcribe_with_gemini,
)
from backend.app.services.ocr.base import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
)


class GeminiOCREngine(OCREngine):
    """Transcribe prescription images with Gemini; no medicine is auto-corrected."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def description(self) -> str:
        return (
            "Google Gemini vision transcription; sends prescription images to Google "
            "only when selected."
        )

    def check_readiness(self) -> EngineReadiness:
        configured = gemini_is_configured()
        return EngineReadiness(
            engine_name=self.name,
            is_available=configured,
            description=self.description,
            details={
                "api_key_configured": configured,
                "model": configured_gemini_model(),
            },
            instructions_if_unavailable=(
                None
                if configured
                else "Set GEMINI_API_KEY in the backend .env file and restart the backend."
            ),
        )

    def extract_text(self, image: Image.Image, **kwargs: Any) -> OCRExtractionResult:
        del kwargs
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Gemini OCR input must be a non-empty PIL image.")

        model, raw_text, medicine_names, elapsed_ms = transcribe_with_gemini(image)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return OCRExtractionResult(
            engine_name=self.name,
            raw_text=raw_text,
            blocks=[
                OCRTextBlock(text=line, line_number=index)
                for index, line in enumerate(lines, start=1)
            ],
            structured_data={
                "possible_medicine_names": medicine_names,
                "medicine_name_source": "Gemini image analysis",
                "human_verification_required": True,
            },
            processing_time_ms=elapsed_ms,
            metadata={
                "model": model,
                "external_provider": "Google Gemini",
                "human_verification_required": True,
            },
        )


__all__ = ["GeminiOCREngine"]
