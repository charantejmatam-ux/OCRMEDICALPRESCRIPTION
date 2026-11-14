from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw

from backend.app.services.ocr.model_evaluation import (
    ModelCompatibilityError,
    _image_case,
    calculate_cer,
    calculate_wer,
    check_model_compatibility,
    evaluate_models,
)
import backend.app.services.ocr.model_evaluation as evaluator


def _write_compatible_model(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "vision-encoder-decoder",
                "architectures": ["VisionEncoderDecoderModel"],
                "encoder": {"model_type": "vit"},
                "decoder": {"model_type": "trocr"},
                "processor_class": "TrOCRProcessor",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"test weights")


def test_cer_and_wer_normalize_case_and_whitespace() -> None:
    assert calculate_cer("Amoxicillin 500 mg", "amoxicillin 500 mg") == 0
    assert calculate_wer("Amoxicillin 500 mg", "amoxicillin 500 mg") == 0
    assert calculate_wer("longword", "longerword") == 1


def test_model_architecture_is_checked_before_loading(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_compatible_model(model_dir)

    details = check_model_compatibility(model_dir)

    assert details == {
        "model_type": "vision-encoder-decoder",
        "architecture": "VisionEncoderDecoderModel",
        "encoder": "vit",
        "decoder": "trocr",
        "processor": "TrOCRProcessor",
    }


def test_incompatible_architecture_is_rejected(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_compatible_model(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["encoder"]["model_type"] = "donut"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ModelCompatibilityError, match="Expected VisionEncoderDecoderModel"):
        check_model_compatibility(model_dir)


def test_segmentation_diagnostics_and_crop_previews_use_shared_boxes(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "handwritten.png"
    image = Image.new("RGB", (240, 100), "white")
    ImageDraw.Draw(image).line((20, 35, 200, 35), fill="black", width=4)
    image.save(image_path)
    crops_dir = tmp_path / "crops"

    case = _image_case(image_path, None, crops_dir, 1)

    assert len(case.line_boxes) == 1
    assert case.diagnostics["segmentation_line_count"] == 1
    assert case.diagnostics["grayscale_stddev"] > 0
    assert case.diagnostics["grayscale_p05"] <= case.diagnostics["grayscale_p95"]
    crop_files = case.diagnostics["crop_preview_files"]
    assert len(crop_files) == 1
    assert Path(crop_files[0]).is_file()


def test_report_without_manual_references_does_not_claim_a_winner(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blank.png"
    Image.new("RGB", (40, 40), "white").save(image_path)

    report = evaluate_models([image_path], [], model_paths={})

    assert report["models"] == []
    assert report["best_performing_model"] is None
    assert "manual reference transcriptions" in report["best_model_limitation"]


def test_evaluation_records_reference_metrics_and_annotation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "trocr"
    _write_compatible_model(model_dir)
    image_path = tmp_path / "rx.png"
    image = Image.new("RGB", (240, 80), "white")
    ImageDraw.Draw(image).line((20, 35, 200, 35), fill="black", width=4)
    image.save(image_path)
    references_path = tmp_path / "references.json"
    references_path.write_text(
        json.dumps(
            {
                image_path.name: {
                    "text": "amoxicillin 500 mg",
                    "medicine_names": ["amoxicillin"],
                    "dosage_frequency": ["500 mg twice daily"],
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeTensor:
        shape = (1, 3, 384, 384)

        def to(self, device: str) -> FakeTensor:
            assert device == "cpu"
            return self

    pixel_values = FakeTensor()

    class FakeProcessor:
        def __call__(self, image: Image.Image, **kwargs: object) -> SimpleNamespace:
            del image, kwargs
            return SimpleNamespace(pixel_values=pixel_values)

        def batch_decode(
            self,
            generated: object,
            skip_special_tokens: bool,
        ) -> list[str]:
            del generated
            assert skip_special_tokens is True
            return ["amoxicillin 500 mg"]

    fake_processor = FakeProcessor()
    fake_model = MagicMock()

    def resolve_local_model(model_id: str, explicit_path: Path | None) -> Path:
        del model_id, explicit_path
        return model_dir

    def load_model(path: Path) -> tuple[FakeProcessor, MagicMock]:
        del path
        return fake_processor, fake_model

    monkeypatch.setattr(
        evaluator,
        "_resolve_local_model",
        resolve_local_model,
    )
    monkeypatch.setattr(
        evaluator,
        "_load_model",
        load_model,
    )
    monkeypatch.setattr(evaluator, "_private_bytes", lambda: None)

    report = evaluate_models(
        [image_path],
        ["trocr-base"],
        references_path=references_path,
    )

    model = report["models"][0]
    image_result = model["images"][0]
    assert model["status"] == "evaluated"
    assert image_result["recognized_text"] == "amoxicillin 500 mg"
    assert image_result["cer"] == 0
    assert image_result["wer"] == 0
    assert image_result["medicine_name_errors"]["error_count"] == 0
    assert image_result["dosage_frequency_errors"]["missing"] == [
        "500 mg twice daily"
    ]
    assert report["best_performing_model"]["name"] == "trocr-base"
    fake_model.generate.assert_called_once()
    assert fake_model.generate.call_args.kwargs == {
        "max_new_tokens": 128,
        "num_beams": 1,
    }
