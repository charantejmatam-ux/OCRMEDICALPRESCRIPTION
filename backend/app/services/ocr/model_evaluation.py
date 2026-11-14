"""Offline, sequential comparison of locally available handwriting OCR models."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from backend.app.services.image_processing import ImageProcessor
from backend.app.services.ocr.trocr_engine import segment_handwriting_lines

MODEL_IDS = {
    "trocr-small": "microsoft/trocr-small-handwritten",
    "trocr-base": "microsoft/trocr-base-handwritten",
    "tcocr-doctor": "galihkjaya/tcOCR-doctor-handwritten-recognition",
    "doctor-handwriting": "codinggeek101/doctor-handwriting-ocr",
}
SUPPORTED_ENCODERS = {"deit", "vit"}


class ModelCompatibilityError(ValueError):
    """Raised when a local model config cannot use the shared TrOCR pipeline."""


@dataclass(frozen=True)
class ImageCase:
    path: Path
    line_boxes: list[list[int]]
    diagnostics: dict[str, Any]
    reference: dict[str, Any] | None


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate case-insensitive CER after collapsing whitespace."""
    expected = _normalized_text(reference)
    actual = _normalized_text(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate WER on whitespace-separated words, case-insensitively."""
    expected = _normalized_text(reference).split()
    actual = _normalized_text(hypothesis).split()
    if not expected:
        return 0.0 if not actual else 1.0
    return _word_edit_distance(expected, actual) / len(expected)


def _word_edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_word in enumerate(left, start=1):
        current = [row]
        for column, right_word in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def check_model_compatibility(model_dir: Path) -> dict[str, Any]:
    """Validate the declared architecture before asking Transformers to load it."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ModelCompatibilityError("Missing config.json.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"Cannot read model config: {exc}") from exc

    architectures = config.get("architectures", [])
    encoder_type = config.get("encoder", {}).get("model_type")
    decoder_type = config.get("decoder", {}).get("model_type")
    if (
        config.get("model_type") != "vision-encoder-decoder"
        or "VisionEncoderDecoderModel" not in architectures
        or encoder_type not in SUPPORTED_ENCODERS
        or decoder_type != "trocr"
    ):
        raise ModelCompatibilityError(
            "Expected VisionEncoderDecoderModel with a ViT/DeiT encoder and "
            "TrOCR decoder; found "
            f"model_type={config.get('model_type')!r}, architectures={architectures!r}, "
            f"encoder={encoder_type!r}, decoder={decoder_type!r}."
        )

    has_weights = any(
        (model_dir / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        raise ModelCompatibilityError("No supported model weights found locally.")

    processor_config = config.get("processor_class")
    for filename in ("processor_config.json", "preprocessor_config.json"):
        candidate = model_dir / filename
        if candidate.is_file():
            try:
                details = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelCompatibilityError(
                    f"Cannot read {filename}: {exc}"
                ) from exc
            processor_config = processor_config or details.get("processor_class")
            break
    if processor_config != "TrOCRProcessor":
        raise ModelCompatibilityError(
            f"Expected TrOCRProcessor metadata; found {processor_config!r}."
        )

    return {
        "model_type": config["model_type"],
        "architecture": architectures[0],
        "encoder": encoder_type,
        "decoder": decoder_type,
        "processor": processor_config,
    }


def _resolve_local_model(model_id: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {explicit_path}")
        return explicit_path

    model_dir = Path(os.getenv("MODEL_DIR", "./models")) / model_id.rsplit("/", 1)[-1]
    if model_dir.is_dir():
        return model_dir

    from huggingface_hub import snapshot_download

    # local_files_only prevents all network access, including implicit downloads.
    return Path(snapshot_download(model_id, local_files_only=True))


def _load_model(model_dir: Path) -> tuple[Any, Any]:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model.to("cpu")
    model.eval()
    return processor, model


class _WindowsMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _private_bytes() -> int | None:
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    counters = _WindowsMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowsMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    process = get_current_process()
    succeeded = get_process_memory_info(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        raise OSError("GetProcessMemoryInfo failed.")
    return counters.PrivateUsage


class _MemorySampler:
    def __init__(self) -> None:
        self.baseline = _private_bytes()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.baseline is None:
            return

        def sample() -> None:
            while not self._stop.wait(0.1):
                current = _private_bytes()
                if current is not None:
                    self.peak = max(self.peak or current, current)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        current = _private_bytes()
        if current is not None:
            self.peak = max(self.peak or current, current)
        return current


def _mb(value: int | None) -> float | None:
    return round(value / (1024 * 1024), 2) if value is not None else None


def _read_references(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reference JSON must be an object keyed by image filename.")
    references: dict[str, dict[str, Any]] = {}
    for filename, value in payload.items():
        if isinstance(value, str):
            references[filename] = {"text": value}
        elif isinstance(value, dict) and isinstance(value.get("text"), str):
            for field in ("medicine_names", "dosage_frequency"):
                annotation = value.get(field)
                if annotation is not None and (
                    not isinstance(annotation, list)
                    or not all(isinstance(item, str) for item in annotation)
                ):
                    raise ValueError(
                        f"Reference field {field!r} for {filename!r} must be "
                        "a list of strings."
                    )
            references[filename] = value
        else:
            raise ValueError(
                f"Reference for {filename!r} must be a transcription string or "
                "an object containing a text string."
            )
    return references


def _image_case(
    path: Path,
    reference: dict[str, Any] | None,
    crops_dir: Path | None,
    image_number: int,
) -> ImageCase:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    processor = ImageProcessor(persist_images=False)
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).copy()
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc

    original_image = image
    image = processor.preprocess_for_ocr(original_image)
    original_image.close()
    gray = image.convert("L")
    histogram = gray.histogram()
    total = sum(histogram)
    cumulative = 0
    percentile_values: dict[int, int] = {}
    for value, count in enumerate(histogram):
        cumulative += count
        for percentile in (5, 95):
            if percentile not in percentile_values and cumulative >= total * percentile / 100:
                percentile_values[percentile] = value
    line_boxes = segment_handwriting_lines(image)
    crop_files: list[str] = []
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
        for line_number, box in enumerate(line_boxes, start=1):
            crop_path = crops_dir / (
                f"{image_number:03d}_{path.stem}_line_{line_number:03d}.png"
            )
            image.crop(tuple(box)).save(crop_path, format="PNG")
            crop_files.append(str(crop_path.resolve()))
    diagnostics = {
        "image_size": list(image.size),
        "grayscale_stddev": round(ImageStat.Stat(gray).stddev[0], 2),
        "grayscale_p05": percentile_values.get(5),
        "grayscale_p95": percentile_values.get(95),
        "ink_fraction_below_200": round(
            sum(histogram[:200]) / total if total else 0.0,
            4,
        ),
        "segmentation_line_count": len(line_boxes),
        "line_boxes": line_boxes,
        "crop_preview_files": crop_files,
        "skew_assessment": "not corrected; visually inspect source and line crops",
        "segmentation_assessment": (
            "visual review required; boxes are heuristic row-projection segments"
        ),
    }
    image.close()
    gray.close()
    return ImageCase(path, line_boxes, diagnostics, reference)


def _recognized_lines(
    processor: Any,
    model: Any,
    image: Image.Image,
    boxes: list[list[int]],
) -> list[str]:
    import torch

    lines: list[str] = []
    rgb_image = image.convert("RGB")
    try:
        for box in boxes:
            with rgb_image.crop(tuple(box)) as crop:
                inputs = processor(crop, return_tensors="pt")
                pixel_values = inputs.pixel_values
                if pixel_values.shape[0] != 1:
                    raise RuntimeError("Evaluation inference requires batch size 1.")
                with torch.inference_mode():
                    generated = model.generate(
                        pixel_values.to("cpu"),
                        max_new_tokens=128,
                        num_beams=1,
                    )
                decoded = processor.batch_decode(generated, skip_special_tokens=True)
                if len(decoded) != 1:
                    raise RuntimeError("Model must return exactly one result per line.")
                lines.append(decoded[0].strip())
    finally:
        rgb_image.close()
    return lines


def _annotation_errors(
    text: str,
    reference: dict[str, Any] | None,
    field: str,
) -> dict[str, Any] | None:
    if reference is None or field not in reference:
        return None
    expected = reference[field]
    if not isinstance(expected, list) or not all(
        isinstance(item, str) for item in expected
    ):
        raise ValueError(f"Reference field {field!r} must be a list of strings.")
    normalized_text = _normalized_text(text)
    missing = [
        item
        for item in expected
        if _normalized_text(item) not in normalized_text
    ]
    return {"expected": expected, "missing": missing, "error_count": len(missing)}


def evaluate_models(
    image_paths: list[Path],
    model_names: list[str],
    references_path: Path | None = None,
    model_paths: dict[str, Path] | None = None,
    crops_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate local models on shared, deterministically segmented line crops."""
    references = _read_references(references_path)
    cases = [
        _image_case(path, references.get(path.name), crops_dir, index)
        for index, path in enumerate(image_paths, start=1)
    ]
    model_paths = model_paths or {}
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "device": "cpu",
            "batch_size": 1,
            "network_access": "disabled; local model files only",
        },
        "preprocessing": {
            "implementation": "ImageProcessor.preprocess_for_ocr",
            "line_segmenter": "segment_handwriting_lines",
            "shared_crop_boxes_across_models": True,
            "automatic_deskew_or_contrast_enhancement": False,
            "crop_previews_saved_to": str(crops_dir.resolve()) if crops_dir else None,
            "image_diagnostics": [
                {"image": str(case.path), **case.diagnostics}
                for case in cases
            ],
        },
        "models": [],
    }

    for model_name in model_names:
        if model_name not in MODEL_IDS:
            raise ValueError(
                f"Unknown model {model_name!r}. Choose from {', '.join(MODEL_IDS)}."
            )
        model_id = MODEL_IDS[model_name]
        result: dict[str, Any] = {
            "name": model_name,
            "model_id": model_id,
            "status": "not_evaluated",
        }
        try:
            model_dir = _resolve_local_model(model_id, model_paths.get(model_name))
        except (FileNotFoundError, OSError) as exc:
            result.update(
                status="not_locally_available",
                reason=str(exc),
            )
            report["models"].append(result)
            continue

        try:
            architecture = check_model_compatibility(model_dir)
        except ModelCompatibilityError as exc:
            result.update(status="incompatible", reason=str(exc))
            report["models"].append(result)
            continue

        result["architecture"] = architecture
        sampler = _MemorySampler()
        sampler.start()
        load_start = time.perf_counter()
        processor = None
        model = None
        try:
            processor, model = _load_model(model_dir)
            load_seconds = time.perf_counter() - load_start

            image_results: list[dict[str, Any]] = []
            inference_seconds = 0.0
            for case in cases:
                with Image.open(case.path) as opened:
                    image = ImageOps.exif_transpose(opened).copy()
                    image.load()
                original_image = image
                image = ImageProcessor(persist_images=False).preprocess_for_ocr(
                    original_image
                )
                original_image.close()
                inference_start = time.perf_counter()
                try:
                    recognized_lines = _recognized_lines(
                        processor,
                        model,
                        image,
                        case.line_boxes,
                    )
                finally:
                    image.close()
                elapsed = time.perf_counter() - inference_start
                inference_seconds += elapsed
                recognized_text = "\n".join(recognized_lines)
                image_result: dict[str, Any] = {
                    "image": str(case.path),
                    "recognized_text": recognized_text,
                    "recognized_lines": recognized_lines,
                    "inference_time_seconds": round(elapsed, 4),
                    "segmentation_line_count": len(case.line_boxes),
                }
                if case.reference is not None:
                    reference_text = case.reference["text"]
                    image_result.update(
                        cer=round(calculate_cer(reference_text, recognized_text), 6),
                        wer=round(calculate_wer(reference_text, recognized_text), 6),
                        medicine_name_errors=_annotation_errors(
                            recognized_text,
                            case.reference,
                            "medicine_names",
                        ),
                        dosage_frequency_errors=_annotation_errors(
                            recognized_text,
                            case.reference,
                            "dosage_frequency",
                        ),
                    )
                image_results.append(image_result)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            current_memory = sampler.stop()
            result.update(
                status="evaluation_failed",
                reason=f"{type(exc).__name__}: {exc}",
                memory_mb={
                    "private_bytes_before_model_load": _mb(sampler.baseline),
                    "private_bytes_after_failure": _mb(current_memory),
                    "peak_private_bytes_sampled": _mb(sampler.peak),
                    "peak_sample_interval_seconds": 0.1,
                },
            )
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            report["models"].append(result)
            continue
        current_memory = sampler.stop()
        result.update(
            status="evaluated",
            model_loading_time_seconds=round(load_seconds, 4),
            total_inference_time_seconds=round(inference_seconds, 4),
            memory_mb={
                "private_bytes_before_model_load": _mb(sampler.baseline),
                "private_bytes_after_evaluation": _mb(current_memory),
                "peak_private_bytes_sampled": _mb(sampler.peak),
                "peak_sample_interval_seconds": 0.1,
            },
            images=image_results,
        )

        del model
        del processor
        gc.collect()
        report["models"].append(result)

    evaluated = [
        model
        for model in report["models"]
        if model["status"] == "evaluated"
    ]
    scored = [
        model
        for model in evaluated
        if any("cer" in image for image in model["images"])
    ]
    if scored:
        for model in scored:
            scored_images = [image for image in model["images"] if "cer" in image]
            model["mean_cer"] = round(
                sum(image["cer"] for image in scored_images) / len(scored_images),
                6,
            )
            model["mean_wer"] = round(
                sum(image["wer"] for image in scored_images) / len(scored_images),
                6,
            )
        winner = min(
            scored,
            key=lambda model: (
                model["mean_cer"],
                model["mean_wer"],
                model["total_inference_time_seconds"],
            ),
        )
        report["best_performing_model"] = {
            "name": winner["name"],
            "selection_basis": "lowest mean CER, then mean WER, then inference time",
        }
    else:
        report["best_performing_model"] = None
        report["best_model_limitation"] = (
            "No evaluated model has manual reference transcriptions; accuracy "
            "cannot be ranked from recognition output alone."
        )
    report["known_limitations"] = [
        "The model comparison does not establish clinical accuracy or suitability.",
        "A model name or repository description is not evidence of medical training.",
        "Skew and segmentation must be visually checked using the original images.",
        "OCR output remains subject to mandatory human review before "
        "normalization or interaction checking.",
    ]
    return report


def _parse_model_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in MODEL_IDS or not path:
        raise argparse.ArgumentTypeError(
            "Use MODEL_NAME=LOCAL_DIRECTORY, where MODEL_NAME is one of: "
            + ", ".join(MODEL_IDS)
        )
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare locally available handwriting OCR models. This tool never "
            "downloads model weights or sends images to external services."
        )
    )
    parser.add_argument("--image", action="append", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_IDS,
        help="Select model(s); defaults to all candidates.",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="JSON file with manually transcribed references, keyed by image filename.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        type=_parse_model_path,
        default=[],
        metavar="MODEL=PATH",
        help="Use a local model directory instead of the Hugging Face cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr-model-evaluation.json"),
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        help=(
            "Optional directory for local line-crop previews. These may contain "
            "sensitive prescription data."
        ),
    )
    args = parser.parse_args(argv)
    report = evaluate_models(
        image_paths=args.image,
        model_names=args.model or list(MODEL_IDS),
        references_path=args.references,
        model_paths=dict(args.model_path),
        crops_dir=args.crops_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
