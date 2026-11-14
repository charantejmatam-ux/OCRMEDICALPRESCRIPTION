"""
backend/tests/test_prescriptions.py
Unit and integration tests for prescription upload, privacy guarantees, and OCR engine router.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from dotenv import dotenv_values

from backend.app.main import app
from backend.app.services.image_processing import (
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    ImageProcessor,
    InMemoryImageCache,
)
from backend.app.services.gemini_second_opinion import (
    GeminiRequestTimeoutError,
    GeminiSecondOpinionError,
)
from backend.app.services.ocr import (
    EngineReadiness,
    OCREngine,
    OCRExtractionResult,
    OCRTextBlock,
    get_ocr_engine,
    list_available_engines,
)
from backend.app.services.ocr.donut_engine import DonutEngine
from backend.app.services.ocr.tesseract_engine import TesseractEngine


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Synchronous HTTP client fixture."""
    with TestClient(app) as c:
        yield c


def create_test_image_bytes(
    format: str = "PNG", size: tuple[int, int] = (120, 80), color: str = "white"
) -> bytes:
    """Helper to generate a real, valid image in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestPrescriptionsRouter:
    """Integration tests for /prescriptions router endpoints."""

    def test_list_engines(self, client: TestClient) -> None:
        with patch(
            "backend.app.routers.prescriptions.gemini_is_configured",
            return_value=False,
        ), patch(
            "backend.app.services.ocr.gemini_engine.gemini_is_configured",
            return_value=False,
        ):
            response = client.get("/prescriptions/engines")
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert data["count"] == 3
        assert data["gemini_second_opinion_configured"] is False
        engine_names = [e["engine_name"] for e in data["engines"]]
        assert engine_names == ["gemma", "paddle", "gemini"]
        gemini = next(engine for engine in data["engines"] if engine["engine_name"] == "gemini")
        assert gemini["is_available"] is False

    def test_upload_valid_image_in_memory_default(self, client: TestClient) -> None:
        """By default, images are registered in-memory and not written to disk."""
        image_bytes = create_test_image_bytes("PNG")
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("rx_sample.png", image_bytes, "image/png")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        img_info = data["image"]
        assert img_info["format"].upper() == "PNG"
        assert img_info["width"] == 120
        assert img_info["height"] == 80
        assert img_info["is_persisted"] is False
        assert img_info["file_path"] is None
        assert "image_id" in img_info

    def test_upload_oversized_file_rejected_413(self, client: TestClient) -> None:
        """Uploads exceeding 10 MB must be rejected with HTTP 413."""
        oversized_data = b"X" * (MAX_FILE_SIZE_BYTES + 1024)  # 10 MB + 1 KB
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("large_scan.png", oversized_data, "image/png")},
        )
        assert response.status_code == 413
        assert "10 mb" in response.json()["detail"].lower()

    def test_upload_jpeg_extension_accepts_webp_content(self, client: TestClient) -> None:
        """Trust decoded image content rather than a misleading .jpeg suffix."""
        webp_bytes = create_test_image_bytes("WEBP")
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("prescription.jpeg", webp_bytes, "image/jpeg")},
        )
        assert response.status_code == 201
        assert response.json()["image"]["format"] == "WEBP"

    def test_upload_corrupted_image_rejected(self, client: TestClient) -> None:
        """Corrupt or random binary data must be rejected with HTTP 400."""
        corrupt_data = b"NOT_A_VALID_IMAGE_HEADER_12345"
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("broken.png", corrupt_data, "image/png")},
        )
        assert response.status_code == 400
        assert "corrupted" in response.json()["detail"].lower()

    def test_upload_empty_file_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_invalid_extension_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/upload",
            files={"file": ("malicious.exe", b"fake-executable", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_ocr_unknown_engine_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/ocr",
            data={
                "image_id": "00000000-0000-0000-0000-000000000001",
                "engine": "unsupported_engine",
            },
        )
        assert response.status_code == 400
        assert "unknown" in response.json()["detail"].lower()

    def test_ocr_rejects_non_uuid_image_ids(self, client: TestClient, tmp_path: Path) -> None:
        arbitrary_path = tmp_path / "local.png"
        arbitrary_path.write_bytes(create_test_image_bytes("PNG"))
        response = client.post(
            "/prescriptions/ocr",
            data={"image_id": str(arbitrary_path), "engine": "paddle"},
        )
        assert response.status_code == 422

    def test_ocr_rejects_malformed_and_unknown_image_ids(self, client: TestClient) -> None:
        malformed = client.post(
            "/prescriptions/ocr",
            data={"image_id": "not-a-uuid", "engine": "paddle"},
        )
        unknown = client.post(
            "/prescriptions/ocr",
            data={
                "image_id": "00000000-0000-0000-0000-000000000001",
                "engine": "paddle",
            },
        )
        assert malformed.status_code == 422
        assert unknown.status_code == 404

    def test_ocr_rejects_both_file_and_image_id(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/ocr",
            files={"file": ("rx.png", create_test_image_bytes("PNG"), "image/png")},
            data={
                "image_id": "00000000-0000-0000-0000-000000000001",
                "engine": "paddle",
            },
        )
        assert response.status_code == 400
        assert "not both" in response.json()["detail"]

    def test_ocr_validates_image_before_reporting_engine_unavailable(
        self,
        client: TestClient,
    ) -> None:
        response = client.post(
            "/prescriptions/ocr",
            files={"file": ("broken.png", b"not an image", "image/png")},
            data={"engine": "paddle"},
        )
        assert response.status_code == 400
        assert "image data" in response.json()["detail"].lower()

    def test_ocr_truncated_image_is_rejected_before_engine_readiness(
        self,
        client: TestClient,
    ) -> None:
        truncated = create_test_image_bytes("PNG")[:-12]
        response = client.post(
            "/prescriptions/ocr",
            files={"file": ("truncated.png", truncated, "image/png")},
            data={"engine": "paddle"},
        )
        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()

    def test_ocr_neither_file_nor_id_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/prescriptions/ocr",
            data={"engine": "donut"},
        )
        assert response.status_code == 400
        assert "either" in response.json()["detail"].lower()

    def test_ocr_missing_engine_returns_503(self, client: TestClient) -> None:
        """Unavailable engines return HTTP 503 with setup details, independent of host tools."""
        image_bytes = create_test_image_bytes("PNG")
        unavailable_engine = MagicMock(spec=OCREngine)
        unavailable_engine.check_readiness.return_value = EngineReadiness(
            engine_name="tesseract",
            is_available=False,
            description="Test engine unavailable",
            instructions_if_unavailable="Install the test engine dependencies.",
        )
        with patch(
            "backend.app.routers.prescriptions.get_ocr_engine",
            return_value=unavailable_engine,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={"engine": "tesseract"},
            )
        assert response.status_code == 503
        data = response.json()
        assert "not currently ready" in data["detail"]["message"].lower()
        assert "instructions" in data["detail"]

    def test_ocr_readiness_check_failure_returns_503(self, client: TestClient) -> None:
        image_bytes = create_test_image_bytes("PNG")
        unavailable_engine = MagicMock(spec=OCREngine)
        unavailable_engine.check_readiness.side_effect = RuntimeError("runtime detail")

        with patch(
            "backend.app.routers.prescriptions.get_ocr_engine",
            return_value=unavailable_engine,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={"engine": "tesseract"},
            )

        assert response.status_code == 503
        assert response.json()["detail"] == "OCR engine readiness check failed: RuntimeError."

    def test_ocr_mocked_success(self, client: TestClient) -> None:
        """Verify the full success pipeline using a mocked engine returning real-shaped OCR text."""
        image_bytes = create_test_image_bytes("PNG")

        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.name = "mock_donut"
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="mock_donut",
            is_available=True,
            description="Mock engine",
        )
        mock_engine.extract_text.return_value = OCRExtractionResult(
            engine_name="mock_donut",
            raw_text="Amoxicillin 500mg PO TID x 7 days",
            blocks=[
                OCRTextBlock(text="Amoxicillin 500mg PO TID x 7 days", confidence=0.95)
            ],
            structured_data={"medication": "Amoxicillin", "dosage": "500mg"},
            processing_time_ms=120.5,
        )

        with (
            patch("backend.app.routers.prescriptions.get_ocr_engine", return_value=mock_engine),
            patch(
                "backend.app.routers.prescriptions.compare_with_gemini"
            ) as compare,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={"engine": "mock_donut"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            res = data["result"]
            assert res["raw_text"] == "Amoxicillin 500mg PO TID x 7 days"
            assert res["processing_time_ms"] == 120.5
            assert len(res["blocks"]) == 1
            assert data["gemini_second_opinion"] is None
            assert data["medicine_extraction"] is None
            compare.assert_not_called()

    def test_ocr_can_run_medicine_extraction_in_the_same_request(
        self,
        client: TestClient,
    ) -> None:
        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.name = "mock_paddle"
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="mock_paddle",
            is_available=True,
            description="Mock OCR engine",
        )
        mock_engine.extract_text.return_value = OCRExtractionResult(
            engine_name="mock_paddle",
            raw_text="Newbrand medicine 500mg capsule",
        )

        with patch(
            "backend.app.routers.prescriptions.get_ocr_engine",
            return_value=mock_engine,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", create_test_image_bytes("PNG"), "image/png")},
                data={
                    "engine": "paddle",
                    "include_medicine_extraction": "true",
                },
            )

        assert response.status_code == 200
        extraction = response.json()["medicine_extraction"]
        assert extraction["review_id"]
        medicine_names = [
            candidate["text"] for candidate in extraction["candidates"]
            if candidate["candidate_type"] == "possible_medicine_name"
        ]
        assert medicine_names == ["Newbrand medicine"]
        review_response = client.get(
            f"/prescriptions/review/{extraction['review_id']}"
        )
        assert review_response.status_code == 200

    def test_gemini_second_opinion_requires_backend_key(self, client: TestClient) -> None:
        with patch(
            "backend.app.routers.prescriptions.gemini_is_configured",
            return_value=False,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", create_test_image_bytes("PNG"), "image/png")},
                data={
                    "engine": "paddle",
                    "include_gemini_second_opinion": "true",
                },
            )

        assert response.status_code == 503
        assert "GEMINI_API_KEY" in response.json()["detail"]

    def test_gemini_selected_as_primary_ocr_engine(
        self,
        client: TestClient,
    ) -> None:
        image_bytes = create_test_image_bytes("PNG")
        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.name = "gemini"
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="gemini",
            is_available=True,
            description="Mock Gemini engine",
        )
        mock_engine.extract_text.return_value = OCRExtractionResult(
            engine_name="gemini",
            raw_text="Amoxicillin 500 mg capsules",
            blocks=[OCRTextBlock(text="Amoxicillin 500 mg capsules")],
            structured_data={
                "possible_medicine_names": ["Amoxicillin"],
                "human_verification_required": True,
            },
            metadata={"model": "gemini-test"},
        )

        with patch(
            "backend.app.routers.prescriptions.get_ocr_engine",
            return_value=mock_engine,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={"engine": "gemini"},
            )

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["engine_name"] == "gemini"
        assert result["raw_text"] == "Amoxicillin 500 mg capsules"
        assert result["structured_data"]["possible_medicine_names"] == ["Amoxicillin"]
        assert response.json()["gemini_second_opinion"] is None

    def test_gemini_timeout_returns_gateway_timeout_instead_of_server_error(
        self,
        client: TestClient,
    ) -> None:
        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="gemini",
            is_available=True,
            description="Mock Gemini engine",
        )
        mock_engine.extract_text.side_effect = GeminiRequestTimeoutError(
            "Gemini did not respond within 180 seconds."
        )

        with patch(
            "backend.app.routers.prescriptions.get_ocr_engine",
            return_value=mock_engine,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", create_test_image_bytes("PNG"), "image/png")},
                data={"engine": "gemini"},
            )

        assert response.status_code == 504
        detail = response.json()["detail"]
        assert detail["error_type"] == "GeminiRequestTimeoutError"
        assert "180 seconds" in detail["message"]

    def test_gemini_second_opinion_is_opt_in_and_separate(
        self,
        client: TestClient,
    ) -> None:
        image_bytes = create_test_image_bytes("PNG")
        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.name = "paddle"
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="paddle",
            is_available=True,
            description="Mock Paddle engine",
        )
        mock_engine.extract_text.return_value = OCRExtractionResult(
            engine_name="paddle",
            raw_text="Amoxcillin 500mg",
            blocks=[OCRTextBlock(text="Amoxcillin 500mg")],
        )

        with (
            patch("backend.app.routers.prescriptions.get_ocr_engine", return_value=mock_engine),
            patch("backend.app.routers.prescriptions.gemini_is_configured", return_value=True),
            patch(
                "backend.app.routers.prescriptions.compare_with_gemini",
                return_value=("gemini-3.8-flash", "Possible reading: Amoxicillin 500 mg.", 234.5),
            ) as compare,
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={
                    "engine": "paddle",
                    "include_gemini_second_opinion": "true",
                },
            )

        assert response.status_code == 200
        result = response.json()
        assert result["result"]["raw_text"] == "Amoxcillin 500mg"
        assert result["gemini_second_opinion"] == {
            "status": "success",
            "model": "gemini-3.8-flash",
            "text": "Possible reading: Amoxicillin 500 mg.",
            "processing_time_ms": 234.5,
            "error": None,
        }
        compare.assert_called_once()

    def test_gemini_failure_is_reported_without_hiding_paddle_result(
        self,
        client: TestClient,
    ) -> None:
        image_bytes = create_test_image_bytes("PNG")
        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.name = "paddle"
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="paddle",
            is_available=True,
            description="Mock Paddle engine",
        )
        mock_engine.extract_text.return_value = OCRExtractionResult(
            engine_name="paddle",
            raw_text="Paddle text",
        )

        with (
            patch("backend.app.routers.prescriptions.get_ocr_engine", return_value=mock_engine),
            patch("backend.app.routers.prescriptions.gemini_is_configured", return_value=True),
            patch(
                "backend.app.routers.prescriptions.compare_with_gemini",
                side_effect=GeminiSecondOpinionError("Gemini returned HTTP 429."),
            ),
        ):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={
                    "engine": "paddle",
                    "include_gemini_second_opinion": "true",
                },
            )

        assert response.status_code == 200
        result = response.json()
        assert result["result"]["raw_text"] == "Paddle text"
        assert result["gemini_second_opinion"]["status"] == "error"
        assert result["gemini_second_opinion"]["error"] == "Gemini returned HTTP 429."

    def test_ocr_inference_failure_handled_as_500(self, client: TestClient) -> None:
        """Engine inference runtime errors must return 500 without crashing the server."""
        image_bytes = create_test_image_bytes("PNG")

        mock_engine = MagicMock(spec=OCREngine)
        mock_engine.check_readiness.return_value = EngineReadiness(
            engine_name="broken_engine",
            is_available=True,
            description="Mock engine",
        )
        mock_engine.extract_text.side_effect = RuntimeError("CUDA Out of Memory during inference")

        with patch("backend.app.routers.prescriptions.get_ocr_engine", return_value=mock_engine):
            response = client.post(
                "/prescriptions/ocr",
                files={"file": ("rx.png", image_bytes, "image/png")},
                data={"engine": "broken_engine"},
            )
            assert response.status_code == 500
            detail = response.json()["detail"]
            assert detail["message"] == "OCR engine inference failed; no OCR result was produced."
            assert detail["engine"] == "broken_engine"
            assert detail["error_type"] == "RuntimeError"
            assert "cuda out of memory" in detail["error"].lower()


class TestImageProcessingService:
    """Unit tests for the ImageProcessor service and privacy guarantees."""

    def test_validate_valid_image(self) -> None:
        processor = ImageProcessor(persist_images=False)
        img_bytes = create_test_image_bytes("JPEG", (200, 150))
        result = processor.validate_image_bytes(img_bytes, "test.jpg")
        assert result.is_valid is True
        assert result.width == 200
        assert result.height == 150
        assert result.format == "JPEG"

    def test_excessive_dimensions_rejected(self) -> None:
        processor = ImageProcessor(persist_images=False)
        # Create an image exceeding MAX_IMAGE_DIMENSION (8000)
        # We can construct a minimal valid header or mock size check
        img_bytes = create_test_image_bytes("PNG", (MAX_IMAGE_DIMENSION + 50, 50))
        result = processor.validate_image_bytes(img_bytes, "huge.png")
        assert result.is_valid is False
        assert "exceed maximum allowed limit" in result.error

    def test_pixel_limit_is_enforced_even_when_each_dimension_is_allowed(self) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        first_open.__exit__.return_value = None
        second_open = MagicMock()
        second_open.__enter__.return_value.format = "PNG"
        second_open.__enter__.return_value.size = (6000, 5000)
        second_open.__enter__.return_value.getbands.return_value = ("R", "G", "B")
        second_open.__exit__.return_value = None

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", "large.png")

        assert result.is_valid is False
        assert "pixel count" in result.error.lower()

    @pytest.mark.parametrize(
        ("size", "valid"),
        [
            ((5000, 4999), True),
            ((5000, 5000), True),
            ((5000, 5001), False),
            ((8000, 3125), True),
            ((8000, 3126), False),
        ],
    )
    def test_pixel_limit_boundaries_and_extreme_aspect_ratio(
        self,
        size: tuple[int, int],
        valid: bool,
    ) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        second_open = MagicMock()
        image = second_open.__enter__.return_value
        image.format = "PNG"
        image.size = size
        image.getbands.return_value = ("R", "G", "B")

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", "boundary.png")

        assert result.is_valid is valid
        assert size[0] * size[1] <= MAX_IMAGE_PIXELS if valid else size[0] * size[1] > MAX_IMAGE_PIXELS

    def test_unsupported_format_is_rejected_without_an_extension(self) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        first_open.__exit__.return_value = None
        second_open = MagicMock()
        second_open.__enter__.return_value.format = "GIF"
        second_open.__enter__.return_value.size = (10, 10)
        second_open.__exit__.return_value = None

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image")

        assert result.is_valid is False
        assert "unsupported image format" in result.error.lower()

    def test_unsupported_format_with_misleading_extension_is_rejected(self) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        second_open = MagicMock()
        second_open.__enter__.return_value.format = "GIF"
        second_open.__enter__.return_value.size = (10, 10)

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", "misleading.png")

        assert result.is_valid is False
        assert "unsupported image format" in result.error.lower()

    @pytest.mark.parametrize(
        ("filename", "format_name"),
        [("wrong.png", "JPEG"), ("wrong.jpeg", "PNG")],
    )
    def test_supported_content_is_accepted_despite_filename_extension(
        self,
        filename: str,
        format_name: str,
    ) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        second_open = MagicMock()
        image = second_open.__enter__.return_value
        image.format = format_name
        image.size = (10, 10)
        image.getbands.return_value = ("R", "G", "B")

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", filename)

        assert result.is_valid is True
        assert result.format == format_name

    @pytest.mark.parametrize("failure_stage", ["verify", "inspect"])
    def test_image_contexts_close_on_success_and_failure(self, failure_stage: str) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        second_open = MagicMock()
        first_image = first_open.__enter__.return_value
        second_image = second_open.__enter__.return_value
        first_image.verify.return_value = None
        second_image.format = "PNG"
        second_image.size = (10, 10)
        second_image.getbands.return_value = ("R", "G", "B")

        if failure_stage == "verify":
            first_image.verify.side_effect = OSError("synthetic corrupt data")
        else:
            second_image.getbands.side_effect = OSError("synthetic inspection failure")

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", "test.png")

        assert result.is_valid is False
        if failure_stage == "verify":
            first_open.__exit__.assert_called_once()
            second_open.__enter__.assert_not_called()
        else:
            first_open.__exit__.assert_called_once()
            second_open.__exit__.assert_called_once()

    def test_image_contexts_close_after_success(self) -> None:
        processor = ImageProcessor(persist_images=False)
        first_open = MagicMock()
        second_open = MagicMock()
        first_open.__enter__.return_value.verify.return_value = None
        second_open.__enter__.return_value.format = "PNG"
        second_open.__enter__.return_value.size = (10, 10)
        second_open.__enter__.return_value.getbands.return_value = ("R", "G", "B")

        with patch(
            "backend.app.services.image_processing.Image.open",
            side_effect=[first_open, second_open],
        ):
            result = processor.validate_image_bytes(b"synthetic-image", "test.png")

        assert result.is_valid is True
        first_open.__exit__.assert_called_once()
        second_open.__exit__.assert_called_once()

    def test_in_memory_processing_creates_no_files(self, tmp_path: Path) -> None:
        """Verify that default processing creates zero files in storage_dir."""
        processor = ImageProcessor(storage_dir=tmp_path, persist_images=False)
        img_bytes = create_test_image_bytes("PNG", (100, 100))

        metadata = processor.process_and_register_image(img_bytes, "sample.png")
        assert metadata.is_persisted is False
        assert metadata.file_path is None
        assert metadata.stored_filename is None

        # Verify no files were created in tmp_path
        files_created = list(tmp_path.glob("*"))
        assert len(files_created) == 0

        # Load back from memory cache
        loaded = processor.load_image(metadata.image_id)
        assert loaded.size == (100, 100)

    def test_persistent_storage_when_explicitly_opted_in(self, tmp_path: Path) -> None:
        """When persist_images=True, saves file to storage_dir."""
        processor = ImageProcessor(storage_dir=tmp_path, persist_images=True)
        img_bytes = create_test_image_bytes("PNG", (80, 80))

        metadata = processor.process_and_register_image(img_bytes, "saved.png")
        assert metadata.is_persisted is True
        assert metadata.file_path is not None
        assert Path(metadata.file_path).is_file()

        # Load back
        loaded = processor.load_image(metadata.image_id)
        assert loaded.size == (80, 80)

    def test_persistent_storage_extension_uses_detected_format(
        self,
        tmp_path: Path,
    ) -> None:
        processor = ImageProcessor(storage_dir=tmp_path, persist_images=True)
        image_bytes = create_test_image_bytes("WEBP")

        metadata = processor.process_and_register_image(
            image_bytes,
            "prescription.jpeg",
        )

        assert metadata.format == "WEBP"
        assert metadata.stored_filename is not None
        assert metadata.stored_filename.endswith(".webp")

    def test_in_memory_cache_eviction_lru(self) -> None:
        cache = InMemoryImageCache(max_items=2, ttl_seconds=60)
        validation = MagicMock()

        cache.put("id_1", b"data1", validation)
        cache.put("id_2", b"data2", validation)
        cache.put("id_3", b"data3", validation)  # Should evict id_1

        assert cache.get("id_1") is None
        assert cache.get("id_2") == b"data2"
        assert cache.get("id_3") == b"data3"

    def test_preprocess_scaling(self) -> None:
        processor = ImageProcessor(persist_images=False)
        img = Image.new("RGB", (3000, 1500), color="blue")
        processed = processor.preprocess_for_ocr(img, max_dimension=1000)
        assert max(processed.size) == 1000
        assert processed.mode == "L"  # Converted to grayscale


class TestOCREnginesReadiness:
    """Unit tests for OCR engine readiness reporting."""

    def test_donut_engine_readiness_structure(self) -> None:
        engine = DonutEngine()
        readiness = engine.check_readiness()
        assert readiness.engine_name == "donut"
        assert "torch_installed" in readiness.details
        assert "transformers_installed" in readiness.details
        assert "resolved_model_target" in readiness.details
        assert "has_local_weights" in readiness.details

    def test_donut_readiness_requires_local_config_and_weights(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys
        from types import ModuleType, SimpleNamespace

        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "test"
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        model_dir = tmp_path / "local-model"
        model_dir.mkdir()
        engine = DonutEngine(model_id_or_path=model_dir)
        assert engine.check_readiness().is_available is False

        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"synthetic test weights")
        readiness = engine.check_readiness()
        assert readiness.is_available is True
        assert readiness.details["has_local_weights"] is True

    def test_donut_model_loader_disables_remote_downloads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys
        from types import ModuleType, SimpleNamespace

        model_dir = tmp_path / "donut"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"synthetic test weights")

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_transformers = ModuleType("transformers")
        processor_loader = MagicMock(return_value=MagicMock())
        model_loader = MagicMock(return_value=MagicMock())
        fake_transformers.DonutProcessor = SimpleNamespace(from_pretrained=processor_loader)
        fake_transformers.VisionEncoderDecoderModel = SimpleNamespace(
            from_pretrained=model_loader
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = DonutEngine(model_id_or_path=model_dir)
        monkeypatch.setattr(
            engine,
            "check_readiness",
            lambda: EngineReadiness(
                engine_name="donut",
                is_available=True,
                description="Synthetic local model for test",
            ),
        )
        engine._load_model_if_needed()

        processor_loader.assert_called_once_with(str(model_dir), local_files_only=True)
        model_loader.assert_called_once_with(str(model_dir), local_files_only=True)

    def test_unavailable_local_donut_does_not_attempt_model_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys
        from types import ModuleType, SimpleNamespace

        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "test"
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_transformers = ModuleType("transformers")
        processor_loader = MagicMock()
        model_loader = MagicMock()
        fake_transformers.DonutProcessor = SimpleNamespace(from_pretrained=processor_loader)
        fake_transformers.VisionEncoderDecoderModel = SimpleNamespace(
            from_pretrained=model_loader
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = DonutEngine(model_id_or_path=tmp_path / "missing-model")
        with pytest.raises(RuntimeError, match="not ready"):
            engine._load_model_if_needed()

        processor_loader.assert_not_called()
        model_loader.assert_not_called()

    def test_donut_lazy_model_initialization_is_thread_safe(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys
        from types import ModuleType, SimpleNamespace

        model_dir = tmp_path / "concurrent-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"synthetic test weights")

        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_transformers = ModuleType("transformers")
        processor = MagicMock()
        model = MagicMock()
        processor_loader = MagicMock(side_effect=lambda *args, **kwargs: (time.sleep(0.02), processor)[1])
        model_loader = MagicMock(return_value=model)
        fake_transformers.DonutProcessor = SimpleNamespace(from_pretrained=processor_loader)
        fake_transformers.VisionEncoderDecoderModel = SimpleNamespace(
            from_pretrained=model_loader
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        engine = DonutEngine(model_id_or_path=model_dir)
        monkeypatch.setattr(
            engine,
            "check_readiness",
            lambda: EngineReadiness(engine_name="donut", is_available=True, description="test"),
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: engine._load_model_if_needed(), range(16)))

        processor_loader.assert_called_once_with(str(model_dir), local_files_only=True)
        model_loader.assert_called_once_with(str(model_dir), local_files_only=True)
        assert engine._model is model
        assert engine._processor is processor

    def test_tesseract_engine_readiness_structure(self) -> None:
        engine = TesseractEngine()
        readiness = engine.check_readiness()
        assert readiness.engine_name == "tesseract"
        assert "tesseract_executable" in readiness.details
        assert "tesseract_version" in readiness.details
        assert "english_language_available" in readiness.details

    def test_tesseract_readiness_verifies_configured_executable_and_language(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_integration = SimpleNamespace(tesseract_cmd=None)
        fake_module.pytesseract = fake_integration
        fake_module.TesseractError = RuntimeError
        fake_module.get_tesseract_version = MagicMock(return_value="5.4.1")
        fake_module.get_languages = MagicMock(return_value=["eng", "osd"])
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.setenv("TESSERACT_CMD", r"C:\OCR Runtime\tesseract.exe")
        which = MagicMock(return_value=r"C:\OCR Runtime\tesseract.exe")
        monkeypatch.setattr("backend.app.services.ocr.tesseract_engine.shutil.which", which)

        readiness = TesseractEngine().check_readiness()

        assert readiness.is_available is True
        assert fake_integration.tesseract_cmd == r"C:\OCR Runtime\tesseract.exe"
        assert readiness.details["tesseract_version"] == "5.4.1"
        assert readiness.details["english_language_available"] is True
        which.assert_called_once_with(r"C:\OCR Runtime\tesseract.exe")

    def test_dotenv_preserves_unquoted_windows_tesseract_path(self) -> None:
        values = dotenv_values(
            stream=StringIO(r"TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe")
        )
        assert values["TESSERACT_CMD"] == r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def test_tesseract_readiness_is_unavailable_without_executable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_module.pytesseract = SimpleNamespace(tesseract_cmd=None)
        fake_module.TesseractError = RuntimeError
        fake_module.get_tesseract_version = MagicMock()
        fake_module.get_languages = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        monkeypatch.setattr("backend.app.services.ocr.tesseract_engine.shutil.which", lambda _: None)

        readiness = TesseractEngine().check_readiness()

        assert readiness.is_available is False
        assert readiness.details["tesseract_executable"] == "Not found in PATH or TESSERACT_CMD"
        fake_module.get_tesseract_version.assert_not_called()
        fake_module.get_languages.assert_not_called()

    def test_tesseract_readiness_is_unavailable_without_english_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_module.pytesseract = SimpleNamespace(tesseract_cmd=None)
        fake_module.TesseractError = RuntimeError
        fake_module.get_tesseract_version = MagicMock(return_value="5.4.1")
        fake_module.get_languages = MagicMock(return_value=["osd"])
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        monkeypatch.setattr(
            "backend.app.services.ocr.tesseract_engine.shutil.which",
            lambda _: r"C:\Tesseract-OCR\tesseract.exe",
        )

        readiness = TesseractEngine().check_readiness()

        assert readiness.is_available is False
        assert readiness.details["english_language_available"] is False
        assert "English" in readiness.instructions_if_unavailable

    def test_tesseract_readiness_reports_nonworking_binary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_module.pytesseract = SimpleNamespace(tesseract_cmd=None)
        fake_module.TesseractError = RuntimeError
        fake_module.get_tesseract_version = MagicMock(side_effect=RuntimeError("cannot execute"))
        fake_module.get_languages = MagicMock()
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        monkeypatch.setattr(
            "backend.app.services.ocr.tesseract_engine.shutil.which",
            lambda _: r"C:\Tesseract-OCR\tesseract.exe",
        )

        readiness = TesseractEngine().check_readiness()

        assert readiness.is_available is False
        assert "cannot execute" in readiness.details["readiness_error"]
        fake_module.get_languages.assert_not_called()

    def test_tesseract_adapter_returns_only_text_from_python_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_module.pytesseract = SimpleNamespace(tesseract_cmd=None)
        fake_module.TesseractError = RuntimeError
        fake_module.Output = SimpleNamespace(DICT="dict")
        fake_module.get_tesseract_version = MagicMock(return_value="5.4.1")
        fake_module.get_languages = MagicMock(return_value=["eng"])
        fake_module.image_to_data = MagicMock(
            return_value={
                "text": ["Amoxicillin", "500", "mg", "capsule"],
                "conf": ["92.5", "90", "88", "85"],
                "left": [10, 200, 260, 320],
                "top": [20, 20, 20, 20],
                "width": [180, 50, 40, 100],
                "height": [30, 30, 30, 30],
                "block_num": [1, 1, 1, 1],
                "par_num": [1, 1, 1, 1],
                "line_num": [1, 1, 1, 1],
            }
        )
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        monkeypatch.setattr(
            "backend.app.services.ocr.tesseract_engine.shutil.which",
            lambda _: r"C:\Tesseract-OCR\tesseract.exe",
        )

        # This is an adapter unit test with an explicitly mocked wrapper, not OCR evidence.
        result = TesseractEngine().extract_text(Image.new("RGB", (640, 240), "white"))

        assert result.raw_text == "Amoxicillin 500 mg capsule"
        assert len(result.blocks) == 4
        assert result.confidence_avg == 88.88
        assert result.metadata["tesseract_version"] == "5.4.1"
        assert result.metadata["tesseract_executable"] == r"C:\Tesseract-OCR\tesseract.exe"
        assert result.metadata["page_segmentation_mode"] == 6
        assert result.metadata["language"] == "eng"
        assert result.raw_text == "Amoxicillin 500 mg capsule"
        fake_module.image_to_data.assert_called_once()
        assert fake_module.image_to_data.call_args.kwargs["config"] == "--oem 1 --psm 6"

    def test_tesseract_reconstructs_multiple_lines_from_word_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_module = ModuleType("pytesseract")
        fake_module.pytesseract = SimpleNamespace(tesseract_cmd=None)
        fake_module.TesseractError = RuntimeError
        fake_module.Output = SimpleNamespace(DICT="dict")
        fake_module.get_tesseract_version = MagicMock(return_value="5.4.1")
        fake_module.get_languages = MagicMock(return_value=["eng"])
        fake_module.image_to_data = MagicMock(
            return_value={
                "text": ["Take", "one", "daily"],
                "conf": ["90", "88", "85"],
                "left": [10, 70, 120],
                "top": [10, 10, 40],
                "width": [50, 40, 80],
                "height": [20, 20, 20],
                "block_num": [1, 1, 1],
                "par_num": [1, 1, 1],
                "line_num": [1, 1, 2],
            }
        )
        monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        monkeypatch.setattr(
            "backend.app.services.ocr.tesseract_engine.shutil.which",
            lambda _: r"C:\Tesseract-OCR\tesseract.exe",
        )

        result = TesseractEngine().extract_text(Image.new("RGB", (240, 100), "white"))

        assert result.raw_text == "Take one\ndaily"
        assert [block.line_number for block in result.blocks] == [1, 1, 2]

    def test_list_available_engines_returns_all(self) -> None:
        engines = list_available_engines()
        names = {e.engine_name for e in engines}
        assert names == {"paddle", "gemma", "gemini"}

    def test_default_engine_instances_are_reused(self) -> None:
        assert get_ocr_engine("paddle") is get_ocr_engine("PADDLE")

    def test_ocr_engine_factory_is_thread_safe(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            engines = list(executor.map(lambda _: get_ocr_engine("paddle"), range(24)))
        assert len({id(engine) for engine in engines}) == 1
        with ThreadPoolExecutor(max_workers=8) as executor:
            readiness = list(executor.map(lambda engine: engine.check_readiness(), engines))
        assert all(item.engine_name == "paddle" for item in readiness)


def test_cors_allows_frontend_methods_and_rejects_unneeded_methods(
    client: TestClient,
) -> None:
    allowed = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    rejected = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "PUT",
        },
    )
    untrusted_origin = client.options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert "access-control-allow-credentials" not in allowed.headers
    assert "POST" in allowed.headers["access-control-allow-methods"]
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in untrusted_origin.headers
