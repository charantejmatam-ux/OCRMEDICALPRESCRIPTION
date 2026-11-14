"""Segment a prescription image into line regions and transcribe them with TrOCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "images"
    / "1_3xUyINxRtDf2qowd-kkGQA.jpg"
)


class PrescriptionOCR:
    def __init__(self, model_path: Path = MODEL_DIR) -> None:
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Model files were not found in {model_path}. "
                "Download them first using run_ocr.py with --download."
            )

        print(f"Loading TrOCR model from {model_path}...")
        self.processor = TrOCRProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

    @staticmethod
    def _read_image(image_path: Path) -> np.ndarray:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            with Image.open(image_path) as source:
                return np.asarray(source.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image: {image_path}") from exc

    @staticmethod
    def _line_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)
        _, threshold = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel_width = max(15, min(60, width // 30))
        kernel_height = max(3, height // 400)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, kernel_height)
        )
        dilated = cv2.dilate(threshold, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = sorted(
            (cv2.boundingRect(contour) for contour in contours),
            key=lambda box: box[1],
        )

        min_height = max(14, height // 100)
        max_height = max(160, height // 5)
        padding = max(2, width // 500)
        line_boxes = []
        for x, y, box_width, box_height in boxes:
            if box_width < 50 or box_height < min_height or box_height > max_height:
                continue
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + box_width + padding)
            y2 = min(height, y + box_height + padding)
            line_boxes.append((x1, y1, x2, y2))
        return line_boxes

    @staticmethod
    def _save_debug_boxes(
        image: np.ndarray,
        line_boxes: list[tuple[int, int, int, int]],
        output_path: Path,
    ) -> None:
        annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        for line_number, (x1, y1, x2, y2) in enumerate(line_boxes, start=1):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(
                annotated,
                str(line_number),
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save debug image: {output_path}")

    def segment_and_ocr(self, image_path: Path) -> list[tuple[int, str]]:
        image = self._read_image(image_path)
        line_boxes = self._line_boxes(image)
        print(f"Found {len(line_boxes)} candidate line regions.")

        results = []
        for line_number, (x1, y1, x2, y2) in enumerate(line_boxes, start=1):
            cropped_image = Image.fromarray(image[y1:y2, x1:x2])
            pixel_values = self.processor(
                images=cropped_image,
                return_tensors="pt",
            ).pixel_values.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    pixel_values,
                    num_beams=5,
                    early_stopping=True,
                    max_new_tokens=40,
                )
            text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()
            if text:
                results.append((line_number, text))
        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe detected handwritten line regions with local TrOCR."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Prescription image path (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--debug-boxes",
        action="store_true",
        help="Save detected boxes beside the image and exit without loading the model",
    )
    args = parser.parse_args()

    try:
        if args.debug_boxes:
            image = PrescriptionOCR._read_image(args.image)
            line_boxes = PrescriptionOCR._line_boxes(image)
            output_path = args.image.with_name(f"{args.image.stem}_debug_boxes.jpg")
            PrescriptionOCR._save_debug_boxes(image, line_boxes, output_path)
            print(f"Found {len(line_boxes)} candidate regions.")
            print(f"Saved box visualization to {output_path}")
            return 0

        results = PrescriptionOCR().segment_and_ocr(args.image)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    if not results:
        print("No text was transcribed. Try a clearer image or adjust the crop.")
        return 1

    print("\nDetected handwriting (review against the image):")
    for line_number, text in results:
        print(f"[Line {line_number}] {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
