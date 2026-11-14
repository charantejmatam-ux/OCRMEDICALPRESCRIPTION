"""
backend/app/services/ocr/donut_engine.py
Donut (Document Understanding Transformer) OCR engine.

Matches the reference repository architecture (chinmays18/medical-prescription-ocr / naver-clova-ix/donut-base).
Employs lazy-loading so heavy deep-learning weights are never instantiated on module import.
"""

from __future__ import annotations

import os
import re
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

DEFAULT_DONUT_MODEL_ID = "chinmays18/medical-prescription-ocr"
LOCAL_MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))


class DonutEngine(OCREngine):
    """
    End-to-End Vision Encoder-Decoder transformer engine based on Donut.
    Transcribes handwritten prescriptions directly into structured representations.
    """

    def __init__(
        self,
        model_id_or_path: str | Path | None = None,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_id_or_path = str(model_id_or_path or DEFAULT_DONUT_MODEL_ID)
        self.max_length = max_length
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "donut"

    @property
    def description(self) -> str:
        return (
            "Donut Vision Encoder-Decoder transformer "
            "(fine-tuned for handwritten medical prescriptions)."
        )

    def _resolve_model_path(self) -> str:
        """Find a locally installed model directory without fetching remote weights."""
        # 1. Check local models directory for a local copy
        candidate = LOCAL_MODEL_DIR / Path(self.model_id_or_path).name
        if candidate.is_dir():
            return str(candidate)

        # 2. Check direct path
        direct_candidate = Path(self.model_id_or_path)
        if direct_candidate.is_dir():
            return str(direct_candidate)

        return self.model_id_or_path

    @staticmethod
    def _has_local_model_files(model_path: str) -> bool:
        path = Path(model_path)
        has_config = (path / "config.json").is_file()
        has_weights = any(
            candidate.is_file()
            for pattern in ("pytorch_model*.bin", "*.safetensors")
            for candidate in path.glob(pattern)
        )
        return has_config and has_weights

    def check_readiness(self) -> EngineReadiness:
        """Check whether PyTorch, Transformers, and Donut dependencies are installed."""
        has_torch = False
        has_transformers = False
        torch_version = None
        cuda_available = False

        try:
            import torch

            has_torch = True
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        try:
            import transformers  # noqa: F401

            has_transformers = True
        except ImportError:
            pass

        resolved_path = self._resolve_model_path()
        has_local_weights = self._has_local_model_files(resolved_path)

        is_ready = has_torch and has_transformers and has_local_weights

        details = {
            "torch_installed": has_torch,
            "torch_version": torch_version,
            "cuda_available": cuda_available,
            "transformers_installed": has_transformers,
            "resolved_model_target": resolved_path,
            "has_local_weights": has_local_weights,
            "model_loaded_in_memory": self._model is not None,
        }

        instructions = None
        if not is_ready:
            missing = []
            if not has_torch or not has_transformers:
                missing.append("install PyTorch and Hugging Face Transformers (pip install torch transformers)")
            if not has_local_weights:
                missing.append(
                    "place the Donut model config.json and local model weights in MODEL_DIR "
                    "(or provide a local model directory); remote downloads are disabled"
                )
            instructions = "To enable Donut, " + "; and ".join(missing) + "."

        return EngineReadiness(
            engine_name=self.name,
            is_available=is_ready,
            description=self.description,
            details=details,
            instructions_if_unavailable=instructions,
        )

    def _load_model_if_needed(self) -> None:
        """Lazy loader for model and processor."""
        if self._model is not None and self._processor is not None:
            return

        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return

            readiness = self.check_readiness()
            if not readiness.is_available:
                raise RuntimeError(
                    f"Donut engine is not ready: {readiness.instructions_if_unavailable}"
                )

            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            target = self._resolve_model_path()

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            processor = DonutProcessor.from_pretrained(
                target,
                local_files_only=True,
            )
            model = VisionEncoderDecoderModel.from_pretrained(
                target,
                local_files_only=True,
            )
            model.to(self._device)
            model.eval()
            self._processor = processor
            self._model = model

    def extract_text(
        self,
        image: Image.Image,
        **kwargs: Any,
    ) -> OCRExtractionResult:
        """Run vision-encoder-decoder inference on prescription image."""
        self._load_model_if_needed()

        import torch

        start_time = time.perf_counter()

        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self._processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        task_prompt = kwargs.get("task_prompt", "<s_prescription>")
        decoder_input_ids = self._processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        seq = self._processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(self._processor.tokenizer.eos_token, "").replace(
            self._processor.tokenizer.pad_token, ""
        )
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove start task prompt

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Attempt to parse output format if processor provides token2json
        structured: dict[str, Any] | None = None
        structured_parse_error: str | None = None
        if hasattr(self._processor, "token2json"):
            try:
                structured = self._processor.token2json(seq)
            except Exception as exc:
                structured = None
                structured_parse_error = type(exc).__name__

        metadata = {
            "model_id": self.model_id_or_path,
            "device": self._device,
        }
        if structured_parse_error is not None:
            metadata["structured_data_parse_error"] = structured_parse_error

        return OCRExtractionResult(
            engine_name=self.name,
            raw_text=seq,
            blocks=[OCRTextBlock(text=seq, confidence=None)],
            structured_data=structured,
            processing_time_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )
