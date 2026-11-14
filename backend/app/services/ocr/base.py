"""
backend/app/services/ocr/base.py
Abstract base class and data structures for OCR engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field


class OCRTextBlock(BaseModel):
    """Represents an extracted span of text with position and confidence."""

    text: str
    confidence: float | None = None
    bounding_box: list[int] | None = Field(
        default=None,
        description="Coordinates [x_min, y_min, x_max, y_max]",
    )
    line_number: int | None = None


class OCRExtractionResult(BaseModel):
    """Normalized output from any OCR engine."""

    engine_name: str
    raw_text: str
    blocks: list[OCRTextBlock] = Field(default_factory=list)
    structured_data: dict[str, Any] | None = None
    confidence_avg: float | None = None
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineReadiness(BaseModel):
    """Health and readiness status of an OCR engine."""

    engine_name: str
    is_available: bool
    description: str
    details: dict[str, Any] = Field(default_factory=dict)
    instructions_if_unavailable: str | None = None


class OCREngine(ABC):
    """Abstract interface for prescription OCR engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the OCR engine."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the engine."""
        ...

    @abstractmethod
    def check_readiness(self) -> EngineReadiness:
        """Determine whether the engine is ready to perform inference."""
        ...

    @abstractmethod
    def extract_text(
        self,
        image: Image.Image,
        **kwargs: Any,
    ) -> OCRExtractionResult:
        """Extract text and structured elements from the provided image."""
        ...
