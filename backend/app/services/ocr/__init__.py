"""
backend/app/services/ocr/__init__.py
OCR engine factory and exports.
"""

from __future__ import annotations

import threading
from typing import Dict, Type

from backend.app.services.ocr.base import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
)
from backend.app.services.ocr.gemma_engine import GemmaOCREngine
from backend.app.services.ocr.gemini_engine import GeminiOCREngine
from backend.app.services.ocr.paddle_engine import PaddleOCREngine

_REGISTRY: Dict[str, Type[OCREngine]] = {
    "gemma": GemmaOCREngine,
    "paddle": PaddleOCREngine,
    "gemini": GeminiOCREngine,
}
_ENGINE_INSTANCES: dict[str, OCREngine] = {}
_ENGINE_LOCK = threading.Lock()


def get_ocr_engine(engine_name: str = "paddle", **kwargs: object) -> OCREngine:
    """Return a process-local engine instance; keep lazily loaded models reusable."""
    normalized_name = engine_name.lower()
    engine_cls = _REGISTRY.get(normalized_name)
    if not engine_cls:
        valid_engines = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown OCR engine '{engine_name}'. Available engines: {valid_engines}"
        )
    if kwargs:
        return engine_cls(**kwargs)

    with _ENGINE_LOCK:
        if normalized_name not in _ENGINE_INSTANCES:
            _ENGINE_INSTANCES[normalized_name] = engine_cls()
        return _ENGINE_INSTANCES[normalized_name]


def list_available_engines() -> list[EngineReadiness]:
    """Inspect and return readiness status for all registered OCR engines."""
    readiness_list: list[EngineReadiness] = []
    for name, engine_cls in _REGISTRY.items():
        try:
            engine = engine_cls()
            readiness_list.append(engine.check_readiness())
        except Exception as exc:  # noqa: BLE001
            readiness_list.append(
                EngineReadiness(
                    engine_name=name,
                    is_available=False,
                    description=f"{name} engine failed to check readiness.",
                    details={"error": str(exc)},
                    instructions_if_unavailable=f"Check logs: {exc}",
                )
            )
    return readiness_list


__all__ = [
    "EngineReadiness",
    "GemmaOCREngine",
    "GeminiOCREngine",
    "OCREngine",
    "OCRExtractionResult",
    "OCRTextBlock",
    "PaddleOCREngine",
    "get_ocr_engine",
    "list_available_engines",
]
