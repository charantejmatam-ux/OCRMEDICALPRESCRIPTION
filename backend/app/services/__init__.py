"""
backend/app/services/__init__.py
Core domain services for image processing and OCR pipelines.
"""

from __future__ import annotations

from backend.app.services.image_processing import (
    ImageProcessor,
    ImageValidationResult,
    ProcessedImageMetadata,
)
from backend.app.services.medicine_extraction import (
    CandidateType,
    ExtractedMention,
    ExtractionMethod,
    MedicineCandidateExtractor,
    MedicineExtractionResult,
    PrescriptionLineCandidate,
)

__all__ = [
    "CandidateType",
    "ExtractedMention",
    "ExtractionMethod",
    "ImageProcessor",
    "ImageValidationResult",
    "MedicineCandidateExtractor",
    "MedicineExtractionResult",
    "PrescriptionLineCandidate",
    "ProcessedImageMetadata",
]

