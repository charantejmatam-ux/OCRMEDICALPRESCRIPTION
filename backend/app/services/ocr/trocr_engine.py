"""
TrOCR handwriting recognition engine with local-only, lazy-loaded CPU inference.
"""

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

DEFAULT_TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"
LOCAL_MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def segment_handwriting_lines(image: Image.Image) -> list[list[int]]:
    """Return top-to-bottom line crops as [left, top, right, bottom] boxes."""
    if not isinstance(image, Image.Image):
        raise ValueError("TrOCR input must be a PIL image.")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("TrOCR input image must have non-zero dimensions.")

    binary = image.convert("L").point(lambda value: 255 if value < 200 else 0)
    pixels = binary.tobytes()
    minimum_ink_pixels = max(2, width // 10_000)
    active_rows: list[int] = []
    for y in range(height):
        row = pixels[y * width : (y + 1) * width]
        if row.count(255) >= minimum_ink_pixels:
            active_rows.append(y)

    if not active_rows:
        return []

    gap_tolerance = max(8, min(24, height // 100))
    groups: list[tuple[int, int]] = []
    start = previous = active_rows[0]
    for y in active_rows[1:]:
        if y - previous - 1 > gap_tolerance:
            groups.append((start, previous + 1))
            start = y
        previous = y
    groups.append((start, previous + 1))

    vertical_padding = max(2, height // 200)
    horizontal_padding = max(2, width // 100)
    line_boxes: list[list[int]] = []
    for top, bottom in groups:
        line_bounds = binary.crop((0, top, width, bottom)).getbbox()
        if line_bounds is None:
            continue
        left, _, right, _ = line_bounds
        line_boxes.append(
            [
                max(0, left - horizontal_padding),
                max(0, top - vertical_padding),
                min(width, right + horizontal_padding),
                min(height, bottom + vertical_padding),
            ]
        )
    return line_boxes


class TrOCREngine(OCREngine):
    """Transcribe segmented handwriting lines using Microsoft TrOCR on the CPU."""

    def __init__(
        self,
        model_id_or_path: str | Path | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        configured_path = os.getenv("TROCR_MODEL_PATH", "").strip()
        self.model_id_or_path = str(
            model_id_or_path or configured_path or DEFAULT_TROCR_MODEL_ID
        )
        self.max_new_tokens = max_new_tokens
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "trocr"

    @property
    def description(self) -> str:
        return (
            "Microsoft TrOCR Base Handwritten; locally cached handwriting "
            "recognition with sequential CPU line inference."
        )

    @staticmethod
    def _offline_mode_enabled() -> bool:
        return any(
            os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )

    def _local_directory(self) -> Path | None:
        direct_path = Path(self.model_id_or_path)
        if direct_path.is_dir():
            return direct_path

        model_dir_candidate = LOCAL_MODEL_DIR / Path(self.model_id_or_path).name
        if model_dir_candidate.is_dir():
            return model_dir_candidate
        return None

    @staticmethod
    def _directory_has_model_files(path: Path) -> bool:
        return (path / "config.json").is_file() and any(
            (path / filename).is_file()
            for filename in WEIGHT_FILES
        )

    def _cached_model_files_available(self) -> bool:
        model_id = self.model_id_or_path
        path_like = (
            Path(model_id).is_absolute()
            or "\\" in model_id
            or model_id.startswith(".")
            or len(model_id.split("/")) != 2
        )
        if path_like:
            return False

        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False

        config_path = try_to_load_from_cache(
            model_id,
            "config.json",
        )
        weight_paths = [
            try_to_load_from_cache(
                model_id,
                filename,
            )
            for filename in WEIGHT_FILES
        ]
        return isinstance(config_path, str) and any(
            isinstance(path, str) for path in weight_paths
        )

    def _has_local_model_files(self) -> bool:
        local_directory = self._local_directory()
        if local_directory is not None:
            return self._directory_has_model_files(local_directory)
        return self._cached_model_files_available()

    def _runtime_dependencies(self) -> tuple[bool, bool]:
        return (
            importlib.util.find_spec("torch") is not None,
            importlib.util.find_spec("transformers") is not None,
        )

    def check_readiness(self) -> EngineReadiness:
        """Check package availability and local model files without loading weights."""
        has_torch, has_transformers = self._runtime_dependencies()
        has_local_weights = self._has_local_model_files()
        offline_mode = self._offline_mode_enabled()

        if not has_local_weights:
            readiness_status = "not_configured"
        elif not has_torch or not has_transformers:
            readiness_status = "dependency_unavailable"
        else:
            readiness_status = "ready"

        is_ready = readiness_status == "ready"
        local_directory = self._local_directory()
        details: dict[str, Any] = {
            "status": readiness_status,
            "model_id": self.model_id_or_path,
            "resolved_model_path": str(local_directory) if local_directory else None,
            "has_local_weights": has_local_weights,
            "torch_installed": has_torch,
            "transformers_installed": has_transformers,
            "device": "cpu",
            "offline_mode": offline_mode,
            "model_loaded_in_memory": self._model is not None,
        }

        instructions = None
        if readiness_status == "not_configured":
            instructions = (
                "TrOCR is not configured: place microsoft/trocr-base-handwritten "
                "in the Hugging Face cache or provide its local directory with "
                "TROCR_MODEL_PATH. Model downloads are never started by the application."
            )
            if offline_mode:
                instructions += " Offline mode is enabled; only local weights can be used."
        elif readiness_status == "dependency_unavailable":
            instructions = (
                "TrOCR dependencies are unavailable: install the optional trocr "
                "dependencies with a PyTorch CPU build."
            )

        return EngineReadiness(
            engine_name=self.name,
            is_available=is_ready,
            description=self.description,
            details=details,
            instructions_if_unavailable=instructions,
        )

    def _load_model_if_needed(self) -> None:
        """Load processor and weights on first inference, strictly from local files."""
        if self._model is not None and self._processor is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return

            readiness = self.check_readiness()
            if not readiness.is_available:
                raise RuntimeError(
                    f"TrOCR engine is not ready: {readiness.instructions_if_unavailable}"
                )

            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            target = str(self._local_directory() or self.model_id_or_path)
            processor = TrOCRProcessor.from_pretrained(
                target,
                local_files_only=True,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                target,
                local_files_only=True,
            )
            model.to("cpu")
            model.eval()
            self._processor = processor
            self._model = model

    def extract_text(
        self,
        image: Image.Image,
        **kwargs: Any,
    ) -> OCRExtractionResult:
        """Recognize each detected handwriting line independently on the CPU."""
        del kwargs
        line_boxes = segment_handwriting_lines(image)
        if not line_boxes:
            return OCRExtractionResult(
                engine_name=self.name,
                raw_text="",
                metadata={
                    "model_id": self.model_id_or_path,
                    "device": "cpu",
                    "line_count": 0,
                    "line_boxes": [],
                    "confidence_available": False,
                },
            )

        with self._inference_lock:
            self._load_model_if_needed()
            import torch

            start_time = time.perf_counter()
            source = image.convert("RGB")
            blocks: list[OCRTextBlock] = []
            recognized_lines: list[str] = []
            for line_number, box in enumerate(line_boxes, start=1):
                line_image = source.crop(tuple(box))
                inputs = self._processor(line_image, return_tensors="pt")
                pixel_values = inputs.pixel_values
                if pixel_values.shape[0] != 1:
                    raise RuntimeError("TrOCR line inference requires batch size 1.")
                pixel_values = pixel_values.to("cpu")

                with torch.inference_mode():
                    generated = self._model.generate(
                        pixel_values,
                        max_new_tokens=self.max_new_tokens,
                        num_beams=1,
                    )

                decoded_lines = self._processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
                if len(decoded_lines) != 1:
                    raise RuntimeError("TrOCR must return exactly one result per line.")
                text = decoded_lines[0].strip()
                recognized_lines.append(text)
                blocks.append(
                    OCRTextBlock(
                        text=text,
                        confidence=None,
                        bounding_box=box,
                        line_number=line_number,
                    )
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return OCRExtractionResult(
                engine_name=self.name,
                raw_text="\n".join(recognized_lines),
                blocks=blocks,
                confidence_avg=None,
                processing_time_ms=round(elapsed_ms, 2),
                metadata={
                    "model_id": self.model_id_or_path,
                    "device": "cpu",
                    "line_count": len(blocks),
                    "line_boxes": line_boxes,
                    "batch_size": 1,
                    "confidence_available": False,
                },
            )
