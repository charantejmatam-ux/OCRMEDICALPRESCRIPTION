"""
backend/app/services/ocr/tesseract_engine.py
Tesseract OCR engine integration using pytesseract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any

from PIL import Image, ImageOps

from backend.app.services.ocr.base import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
)


class TesseractEngine(OCREngine):
    """
    Standard OCR baseline engine using Tesseract.
    Provides fast, local extraction without requiring deep learning models.
    """

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def description(self) -> str:
        return (
            "Tesseract 5 LSTM OCR with contrast normalization and page-segmentation "
            "mode 6; primarily a printed-text baseline, not handwriting-specialized."
        )

    def check_readiness(self) -> EngineReadiness:
        """Check the Python wrapper, executable, version command, and English data."""
        configured_binary = os.getenv("TESSERACT_CMD", "").strip()
        tesseract_binary = shutil.which(configured_binary or "tesseract")
        tesseract_version: str | None = None
        available_languages: list[str] = []
        readiness_error: str | None = None
        try:
            import pytesseract as pytesseract_module
        except ImportError:
            pytesseract_module = None

        has_pytesseract = pytesseract_module is not None
        if pytesseract_module is not None and tesseract_binary:
            pytesseract_module.pytesseract.tesseract_cmd = tesseract_binary
            try:
                tesseract_version = str(pytesseract_module.get_tesseract_version())
                available_languages = pytesseract_module.get_languages(config="")
            except (
                OSError,
                subprocess.SubprocessError,
                pytesseract_module.TesseractError,
            ) as exc:
                readiness_error = f"{type(exc).__name__}: {exc}"

        is_ready = (
            has_pytesseract
            and tesseract_binary is not None
            and tesseract_version is not None
            and "eng" in available_languages
            and readiness_error is None
        )

        details = {
            "pytesseract_installed": has_pytesseract,
            "tesseract_executable": tesseract_binary or "Not found in PATH or TESSERACT_CMD",
            "tesseract_version": tesseract_version,
            "available_languages": available_languages,
            "english_language_available": "eng" in available_languages,
        }
        if readiness_error:
            details["readiness_error"] = readiness_error

        instructions = None
        if not is_ready:
            missing: list[str] = []
            if not has_pytesseract:
                missing.append("install the Python wrapper in the project environment: pip install pytesseract")
            if not tesseract_binary:
                missing.append(
                    "install the Tesseract system executable and add it to PATH, "
                    "or set TESSERACT_CMD to its full executable path"
                )
            elif readiness_error:
                missing.append("verify that the configured Tesseract executable can run")
            if tesseract_version is not None and "eng" not in available_languages:
                missing.append("install the Tesseract English (eng) trained data")
            instructions = "Tesseract is unavailable: " + "; ".join(missing) + "."

        return EngineReadiness(
            engine_name=self.name,
            is_available=is_ready,
            description=self.description,
            details=details,
            instructions_if_unavailable=instructions,
        )

    def extract_text(
        self,
        image: Image.Image,
        **kwargs: Any,
    ) -> OCRExtractionResult:
        """Run Tesseract extraction on the given PIL image."""
        del kwargs
        readiness = self.check_readiness()
        if not readiness.is_available:
            raise RuntimeError(
                f"Tesseract engine is not ready: {readiness.instructions_if_unavailable}"
            )

        import pytesseract

        start_time = time.perf_counter()
        with image.convert("L") as grayscale:
            normalized = ImageOps.autocontrast(grayscale, cutoff=1)
        config = "--oem 1 --psm 6"

        try:
            data = pytesseract.image_to_data(
                normalized,
                output_type=pytesseract.Output.DICT,
                lang="eng",
                config=config,
            )
        finally:
            normalized.close()

        blocks: list[OCRTextBlock] = []
        confidences: list[float] = []
        line_parts: dict[tuple[int, int, int], list[str]] = {}

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text:
                block_num = data.get("block_num", [1] * n_boxes)[i]
                par_num = data.get("par_num", [1] * n_boxes)[i]
                line_num = data["line_num"][i]
                line_parts.setdefault(
                    (int(block_num), int(par_num), int(line_num)),
                    [],
                ).append(text)
                if conf >= 0:
                    confidences.append(conf)
                blocks.append(
                    OCRTextBlock(
                        text=text,
                        confidence=conf if conf >= 0 else None,
                        bounding_box=[
                            data["left"][i],
                            data["top"][i],
                            data["left"][i] + data["width"][i],
                            data["top"][i] + data["height"][i],
                        ],
                        line_number=data["line_num"][i],
                    )
                )

        raw_text = "\n".join(
            " ".join(words)
            for words in line_parts.values()
        ).strip()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        return OCRExtractionResult(
            engine_name=self.name,
            raw_text=raw_text,
            blocks=blocks,
            confidence_avg=round(avg_conf, 2) if avg_conf is not None else None,
            processing_time_ms=round(elapsed_ms, 2),
            metadata={
                "detected_words": len(blocks),
                "has_boxes": True,
                "tesseract_version": readiness.details["tesseract_version"],
                "tesseract_executable": readiness.details["tesseract_executable"],
                "language": "eng",
                "config": config,
                "preprocessing": "grayscale with 1% autocontrast",
                "page_segmentation_mode": 6,
            },
        )
