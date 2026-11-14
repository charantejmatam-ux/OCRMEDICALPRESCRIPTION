"""Run shubham879/trocr-prescription locally on the CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

MODEL_ID = "shubham879/trocr-prescription"
MODEL_DIR = Path(__file__).resolve().parent / "model"


def load_model(*, allow_download: bool) -> tuple[AutoProcessor, VisionEncoderDecoderModel]:
    """Load only cached model files unless downloading was explicitly requested."""
    try:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_files_only=not allow_download,
        )
    except LocalEntryNotFoundError as exc:
        raise FileNotFoundError(
            f"{MODEL_ID} is not fully cached locally. No model files were downloaded. "
            "To explicitly download the model, rerun with --download."
        ) from exc

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model.to(torch.device("cpu"))
    model.eval()
    return processor, model


def transcribe(
    image_path: Path,
    processor: AutoProcessor,
    model: VisionEncoderDecoderModel,
) -> str:
    """Return the model's raw transcription for one image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.inference_mode():
        generated_ids = model.generate(pixel_values.to(torch.device("cpu")))
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one prescription image using shubham879/trocr-prescription "
            "on CPU. Model downloads are disabled unless --download is supplied."
        )
    )
    parser.add_argument("image", type=Path, help="Path to a prescription image")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly allow downloading model files from Hugging Face if needed",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")

    try:
        processor, model = load_model(allow_download=args.download)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    text = transcribe(args.image, processor, model)
    if not text:
        print("The model returned no text for this image.", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
