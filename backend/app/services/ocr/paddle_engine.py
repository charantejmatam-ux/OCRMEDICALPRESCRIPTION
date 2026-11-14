"""Local-only PaddleOCR full-page text detection and recognition."""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from backend.app.services.ocr.base import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
)

DEFAULT_DET_MODEL_PATH = (
    Path.home() / ".paddlex" / "official_models" / "PP-OCRv6_medium_det"
)
DEFAULT_REC_MODEL_PATH = (
    Path.home() / ".paddlex" / "official_models" / "PP-OCRv6_medium_rec"
)
MODEL_FILES = ("inference.yml", "inference.json", "inference.pdiparams")


def _configured_model_path(env_name: str, default_path: Path) -> Path:
    value = os.getenv(env_name, "").strip()
    return Path(value).expanduser() if value else default_path


def _is_local_model(path: Path) -> bool:
    try:
        return path.is_dir() and all(
            (path / filename).is_file() for filename in MODEL_FILES
        )
    except OSError:
        return False


class PaddleOCREngine(OCREngine):
    """Recognize handwritten and printed text across a complete page on CPU."""

    def __init__(self) -> None:
        self.det_model_path = _configured_model_path(
            "PADDLEOCR_DET_MODEL_PATH", DEFAULT_DET_MODEL_PATH
        )
        self.rec_model_path = _configured_model_path(
            "PADDLEOCR_REC_MODEL_PATH", DEFAULT_REC_MODEL_PATH
        )
        self._engine: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "paddle"

    @property
    def description(self) -> str:
        return "PaddleOCR PP-OCRv6 medium; local full-page text detection and recognition on CPU."

    def check_readiness(self) -> EngineReadiness:
        has_paddle = importlib.util.find_spec("paddle") is not None
        has_paddleocr = importlib.util.find_spec("paddleocr") is not None
        has_det_model = _is_local_model(self.det_model_path)
        has_rec_model = _is_local_model(self.rec_model_path)
        is_ready = has_paddle and has_paddleocr and has_det_model and has_rec_model

        missing: list[str] = []
        if not has_paddle or not has_paddleocr:
            missing.append("install PaddlePaddle and PaddleOCR from the project dependencies")
        if not has_det_model or not has_rec_model:
            missing.append(
                "download PP-OCRv6 medium detector and recognizer models locally, or "
                "set PADDLEOCR_DET_MODEL_PATH and PADDLEOCR_REC_MODEL_PATH"
            )

        return EngineReadiness(
            engine_name=self.name,
            is_available=is_ready,
            description=self.description,
            details={
                "paddle_installed": has_paddle,
                "paddleocr_installed": has_paddleocr,
                "device": "cpu",
                "detector_model_path": str(self.det_model_path),
                "detector_model_available": has_det_model,
                "recognizer_model_path": str(self.rec_model_path),
                "recognizer_model_available": has_rec_model,
                "model_loaded_in_memory": self._engine is not None,
                "runtime_downloads_enabled": False,
            },
            instructions_if_unavailable=("To enable PaddleOCR, " + "; and ".join(missing) + ".")
            if missing
            else None,
        )

    def _load_engine_if_needed(self) -> None:
        if self._engine is not None:
            return
        with self._load_lock:
            if self._engine is not None:
                return
            readiness = self.check_readiness()
            if not readiness.is_available:
                raise RuntimeError(
                    "PaddleOCR engine is not ready: "
                    f"{readiness.instructions_if_unavailable}"
                )

            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_dir=str(self.det_model_path),
                text_recognition_model_dir=str(self.rec_model_path),
            )

    def extract_text(self, image: Image.Image, **kwargs: Any) -> OCRExtractionResult:
        del kwargs
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("PaddleOCR input must be a non-empty PIL image.")

        import numpy as np

        started = time.perf_counter()
        cold_start = self._engine is None
        self._load_engine_if_needed()
        # PaddleOCR expects OpenCV-style BGR arrays for in-memory image input.
        image_array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        with self._inference_lock:
            results = self._engine.predict(image_array)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        result = results[0].json.get("res", {}) if results else {}
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        boxes = result.get("rec_boxes", [])
        blocks = [
            OCRTextBlock(
                text=text,
                confidence=float(scores[index]) if index < len(scores) else None,
                bounding_box=[int(value) for value in boxes[index]]
                if index < len(boxes)
                else None,
                line_number=index + 1,
            )
            for index, text in enumerate(texts)
            if text and text.strip()
        ]
        usable_scores = [block.confidence for block in blocks if block.confidence is not None]
        return OCRExtractionResult(
            engine_name=self.name,
            raw_text="\n".join(block.text for block in blocks),
            blocks=blocks,
            confidence_avg=(sum(usable_scores) / len(usable_scores))
            if usable_scores
            else None,
            processing_time_ms=elapsed_ms,
            metadata={
                "detector_model": self.det_model_path.name,
                "recognizer_model": self.rec_model_path.name,
                "device": "cpu",
                "line_count": len(blocks),
                "confidence_available": bool(usable_scores),
                "cold_start": cold_start,
            },
        )
