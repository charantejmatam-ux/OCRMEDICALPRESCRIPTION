"""Unit tests for local-only, sequential CPU TrOCR inference."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from PIL import Image, ImageDraw

from backend.app.services.ocr import get_ocr_engine
from backend.app.services.ocr.tesseract_engine import TesseractEngine
from backend.app.services.ocr.trocr_engine import TrOCREngine


def create_local_model_directory(tmp_path: Path) -> Path:
    model_dir = tmp_path / "trocr"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"mock weights")
    return model_dir


def create_handwriting_lines() -> Image.Image:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line((15, 25, 180, 25), fill="black", width=5)
    draw.line((25, 82, 205, 82), fill="black", width=5)
    return image


class FakeTensor:
    shape = (1, 3, 32, 128)

    def __init__(self) -> None:
        self.to = MagicMock(return_value=self)


@pytest.fixture
def mock_trocr_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    pixel_values = FakeTensor()
    processor = MagicMock()
    processor.side_effect = lambda image, **kwargs: SimpleNamespace(
        pixel_values=pixel_values
    )
    processor.batch_decode.side_effect = [
        ["recognized line one"],
        ["recognized line two"],
    ]

    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = object()

    processor_loader = MagicMock(return_value=processor)
    model_loader = MagicMock(return_value=model)
    transformers = ModuleType("transformers")
    transformers.TrOCRProcessor = SimpleNamespace(from_pretrained=processor_loader)
    transformers.VisionEncoderDecoderModel = SimpleNamespace(
        from_pretrained=model_loader
    )
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    return {
        "pixel_values": pixel_values,
        "processor": processor,
        "processor_loader": processor_loader,
        "model": model,
        "model_loader": model_loader,
    }


def test_paddle_is_registered_and_trocr_is_not() -> None:
    from backend.app.services.ocr import PaddleOCREngine

    assert isinstance(get_ocr_engine("paddle"), PaddleOCREngine)
    with pytest.raises(
        ValueError,
        match="Available engines: gemma, paddle, gemini",
    ):
        get_ocr_engine("trocr")


def test_unavailable_runtime_dependencies_report_engine_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = create_local_model_directory(tmp_path)
    engine = TrOCREngine(model_dir)
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (False, False))

    readiness = engine.check_readiness()

    assert readiness.is_available is False
    assert readiness.details["status"] == "dependency_unavailable"
    assert readiness.details["device"] == "cpu"


def test_missing_local_weights_report_not_configured_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = TrOCREngine(tmp_path / "missing-model")
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))
    cache_lookup = MagicMock(return_value=False)
    monkeypatch.setattr(engine, "_cached_model_files_available", cache_lookup)

    readiness = engine.check_readiness()

    assert readiness.is_available is False
    assert readiness.details["status"] == "not_configured"
    assert readiness.details["has_local_weights"] is False
    assert "never started" in readiness.instructions_if_unavailable
    cache_lookup.assert_called_once()


def test_offline_mode_reports_missing_cached_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = TrOCREngine(tmp_path / "missing-model")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))
    monkeypatch.setattr(engine, "_cached_model_files_available", lambda: False)

    readiness = engine.check_readiness()

    assert readiness.details["offline_mode"] is True
    assert readiness.details["status"] == "not_configured"
    assert "Offline mode is enabled" in readiness.instructions_if_unavailable


def test_cached_model_is_detected_without_network_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached_files = {
        "config.json": str(tmp_path / "cached-config.json"),
        "model.safetensors": str(tmp_path / "cached-weights.safetensors"),
    }
    cache_lookup = MagicMock(side_effect=lambda model_id, filename: cached_files.get(filename))
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.try_to_load_from_cache = cache_lookup
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setattr(
        "backend.app.services.ocr.trocr_engine.LOCAL_MODEL_DIR",
        tmp_path / "models",
    )
    engine = TrOCREngine()
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))

    readiness = engine.check_readiness()

    assert readiness.is_available is True
    assert readiness.details["status"] == "ready"
    assert cache_lookup.call_count == 5


def test_line_segmentation_keeps_small_vertical_glyph_gaps_together() -> None:
    image = Image.new("RGB", (100, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.point((20, 10), fill="black")
    draw.line((20, 19, 20, 35), fill="black", width=2)

    from backend.app.services.ocr.trocr_engine import segment_handwriting_lines

    assert len(segment_handwriting_lines(image)) == 1


def test_model_loading_is_lazy_and_inference_is_cpu_batch_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_trocr_dependencies: dict[str, MagicMock],
) -> None:
    model_dir = create_local_model_directory(tmp_path)
    engine = TrOCREngine(model_dir)
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))

    assert engine._model is None
    readiness = engine.check_readiness()
    assert readiness.is_available is True
    assert readiness.details["model_loaded_in_memory"] is False
    mock_trocr_dependencies["processor_loader"].assert_not_called()
    mock_trocr_dependencies["model_loader"].assert_not_called()

    result = engine.extract_text(create_handwriting_lines())

    assert result.raw_text == "recognized line one\nrecognized line two"
    assert [block.line_number for block in result.blocks] == [1, 2]
    assert all(block.confidence is None for block in result.blocks)
    assert all(block.bounding_box for block in result.blocks)
    assert result.confidence_avg is None
    assert result.metadata["line_count"] == 2
    assert result.metadata["batch_size"] == 1
    assert result.metadata["device"] == "cpu"
    assert mock_trocr_dependencies["model"].to.call_args_list[0].args == ("cpu",)
    assert mock_trocr_dependencies["model"].generate.call_count == 2
    assert mock_trocr_dependencies["processor"].call_count == 2
    assert mock_trocr_dependencies["pixel_values"].to.call_args_list == [
        call("cpu"),
        call("cpu"),
    ]
    assert mock_trocr_dependencies["processor_loader"].call_args.kwargs[
        "local_files_only"
    ] is True
    assert mock_trocr_dependencies["model_loader"].call_args.kwargs[
        "local_files_only"
    ] is True


def test_blank_image_returns_no_lines_without_loading_model(
    mock_trocr_dependencies: dict[str, MagicMock],
) -> None:
    engine = TrOCREngine()

    result = engine.extract_text(Image.new("RGB", (100, 80), "white"))

    assert result.raw_text == ""
    assert result.blocks == []
    assert result.metadata["line_count"] == 0
    mock_trocr_dependencies["model_loader"].assert_not_called()


@pytest.mark.parametrize("image", ["not an image", None])
def test_invalid_image_is_rejected_before_model_loading(
    image: object,
    mock_trocr_dependencies: dict[str, MagicMock],
) -> None:
    engine = TrOCREngine()

    with pytest.raises(ValueError, match="PIL image"):
        engine.extract_text(image)  # type: ignore[arg-type]

    mock_trocr_dependencies["model_loader"].assert_not_called()


def test_model_loading_failure_is_propagated_without_partial_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_trocr_dependencies: dict[str, MagicMock],
) -> None:
    model_dir = create_local_model_directory(tmp_path)
    engine = TrOCREngine(model_dir)
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))
    mock_trocr_dependencies["model_loader"].side_effect = OSError("bad local weights")

    with pytest.raises(OSError, match="bad local weights"):
        engine.extract_text(create_handwriting_lines())

    assert engine._model is None
    assert engine._processor is None


def test_inference_failure_is_propagated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_trocr_dependencies: dict[str, MagicMock],
) -> None:
    model_dir = create_local_model_directory(tmp_path)
    engine = TrOCREngine(model_dir)
    monkeypatch.setattr(engine, "_runtime_dependencies", lambda: (True, True))
    mock_trocr_dependencies["model"].generate.side_effect = RuntimeError(
        "inference failed"
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        engine.extract_text(create_handwriting_lines())
