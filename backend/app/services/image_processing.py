"""
backend/app/services/image_processing.py
Prescription image validation, sanitization, in-memory buffering, and opt-in local storage.

Privacy & Security Guarantees:
- In-memory processing by default (PERSIST_IMAGES=False). No patient images are written to disk.
- Bounded in-memory ephemeral cache with TTL to support two-step upload-and-process workflows.
- Max file size strictly enforced at 10 MB.
- Decompression-bomb and excessive-dimension protection (Image.MAX_IMAGE_PIXELS).
- Validation of actual decoded image format; filenames are not trusted as format indicators.
- Internal filesystem paths are never exposed in API responses.
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

# Security and sizing boundaries
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB strict limit
MAX_IMAGE_DIMENSION = 8000  # Max 8000 x 8000 pixels
MAX_IMAGE_PIXELS = 25_000_000  # 25 megapixels (decompression bomb protection)

# Configure Pillow's decompression bomb limit
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
EXTENSION_FORMAT_MAP = {
    ".png": {"PNG"},
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".webp": {"WEBP"},
    ".tiff": {"TIFF"},
    ".tif": {"TIFF"},
    ".bmp": {"BMP"},
}
SUPPORTED_FORMATS = set().union(*EXTENSION_FORMAT_MAP.values())
FORMAT_EXTENSION_MAP = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "TIFF": ".tiff",
    "BMP": ".bmp",
}

DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PERSIST_IMAGES_DEFAULT = os.getenv("PERSIST_IMAGES", "false").lower() in ("true", "1", "yes")


class ImageValidationResult(BaseModel):
    """Result of image format, extension, and safety validation."""

    is_valid: bool
    format: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    file_size_bytes: int = 0
    error: str | None = None


class ProcessedImageMetadata(BaseModel):
    """Metadata describing a validated prescription image."""

    image_id: str
    original_filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int
    is_persisted: bool = False
    stored_filename: str | None = None
    file_path: str | None = None


class InMemoryImageCache:
    """Thread-safe, bounded, TTL-based in-memory cache for ephemeral prescription images."""

    def __init__(self, max_items: int = 50, ttl_seconds: int = 600) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, bytes, ImageValidationResult]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, image_id: str, content: bytes, validation: ImageValidationResult) -> None:
        with self._lock:
            self._purge_expired()
            if len(self._cache) >= self.max_items:
                self._cache.popitem(last=False)  # Evict oldest (LRU)
            self._cache[image_id] = (time.time(), content, validation)

    def get(self, image_id: str) -> bytes | None:
        with self._lock:
            self._purge_expired()
            if image_id not in self._cache:
                return None
            _, content, _ = self._cache[image_id]
            self._cache.move_to_end(image_id)
            return content

    def remove(self, image_id: str) -> None:
        with self._lock:
            self._cache.pop(image_id, None)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [k for k, (t, _, _) in self._cache.items() if (now - t) > self.ttl_seconds]
        for k in expired:
            self._cache.pop(k, None)


async def read_upload_stream(
    file: UploadFile,
    max_bytes: int = MAX_FILE_SIZE_BYTES,
) -> bytes:
    """
    Read an uploaded file in chunks, enforcing the max size limit dynamically.
    Avoids loading excessively large payloads into memory before rejecting.
    """
    chunks: list[bytes] = []
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    return b"".join(chunks)


class ImageProcessor:
    """Service for validating, preprocessing, and managing prescription images."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        persist_images: bool | None = None,
    ) -> None:
        self.persist_images = (
            persist_images if persist_images is not None else PERSIST_IMAGES_DEFAULT
        )
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_DATA_DIR
        self._mem_cache = InMemoryImageCache()

        if self.persist_images:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def validate_image_bytes(
        self, content: bytes, original_filename: str = ""
    ) -> ImageValidationResult:
        """
        Thoroughly validate raw image bytes.
        Checks:
        - Non-zero length and max file size (10 MB)
        - Allowed file extension
        - Uncorrupted image stream (Pillow verify)
        - Decompression bomb safety (max pixels)
        - Max dimensions (width/height <= 8000)
        - Supported format identified from decoded image content
        """
        file_size = len(content)
        if file_size == 0:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=0,
                error="Uploaded file is empty (0 bytes).",
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
            )

        ext = Path(original_filename).suffix.lower() if original_filename else ""
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        # Step 1: verify integrity
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Image.DecompressionBombError:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error="Image exceeds safe pixel density limits (decompression bomb protection).",
            )
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error=f"Corrupted or invalid image data: {exc}",
            )

        # Step 2: decode and inspect attributes
        try:
            with Image.open(io.BytesIO(content)) as img:
                fmt = (img.format or "UNKNOWN").upper()
                width, height = img.size

                if fmt not in SUPPORTED_FORMATS:
                    return ImageValidationResult(
                        is_valid=False,
                        file_size_bytes=file_size,
                        error=f"Unsupported image format '{fmt}'.",
                    )

                if width * height > MAX_IMAGE_PIXELS:
                    return ImageValidationResult(
                        is_valid=False,
                        file_size_bytes=file_size,
                        error=(
                            f"Image pixel count ({width * height}) exceeds maximum allowed "
                            f"limit of {MAX_IMAGE_PIXELS}."
                        ),
                    )

                # Check excessive dimensions
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return ImageValidationResult(
                        is_valid=False,
                        file_size_bytes=file_size,
                        error=(
                            f"Image dimensions ({width}x{height}) exceed maximum allowed limit "
                            f"of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}."
                        ),
                    )

                channels = len(img.getbands()) if hasattr(img, "getbands") else 1
                return ImageValidationResult(
                    is_valid=True,
                    format=fmt,
                    width=width,
                    height=height,
                    channels=channels,
                    file_size_bytes=file_size,
                )
        except Image.DecompressionBombError:
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error="Image exceeds safe pixel density limits (decompression bomb protection).",
            )
        except Exception as exc:  # noqa: BLE001
            return ImageValidationResult(
                is_valid=False,
                file_size_bytes=file_size,
                error=f"Failed to inspect image attributes: {exc}",
            )

    def process_and_register_image(
        self, content: bytes, original_filename: str
    ) -> ProcessedImageMetadata:
        """
        Validate and register an image.
        Default: kept in memory only, not persisted to disk.
        If persistent storage is explicitly opted-in (PERSIST_IMAGES=True), saves to storage_dir.
        """
        validation = self.validate_image_bytes(content, original_filename)
        if not validation.is_valid:
            raise ValueError(validation.error or "Invalid image file")

        image_id = str(uuid.uuid4())
        image_format = validation.format or "PNG"
        ext = FORMAT_EXTENSION_MAP[image_format]

        stored_filename = None
        file_path = None

        if self.persist_images:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            stored_filename = f"rx_{image_id}{ext}"
            dest = self.storage_dir / stored_filename
            with open(dest, "wb") as f:
                f.write(content)
            file_path = str(dest.resolve())
        else:
            # Ephemeral in-memory caching
            self._mem_cache.put(image_id, content, validation)

        return ProcessedImageMetadata(
            image_id=image_id,
            original_filename=original_filename,
            format=validation.format or "UNKNOWN",
            width=validation.width or 0,
            height=validation.height or 0,
            file_size_bytes=validation.file_size_bytes,
            is_persisted=self.persist_images,
            stored_filename=stored_filename,
            file_path=file_path if self.persist_images else None,
        )

    def load_image_bytes(self, image_id_or_path: str | Path) -> bytes:
        """Retrieve raw image bytes from in-memory cache or disk (if persisted)."""
        str_id = str(image_id_or_path)

        # 1. Check in-memory cache
        mem_data = self._mem_cache.get(str_id)
        if mem_data is not None:
            return mem_data

        # 2. Check disk if path or file exists
        path = Path(str_id)
        if path.is_file():
            return path.read_bytes()

        if self.storage_dir.is_dir():
            matches = list(self.storage_dir.glob(f"rx_{str_id}.*"))
            if matches:
                return matches[0].read_bytes()

        raise FileNotFoundError(f"Prescription image '{str_id}' not found in memory or storage.")

    def load_image(self, image_id_or_path: str | Path) -> Image.Image:
        """Load a PIL Image given an image ID or file path."""
        data = self.load_image_bytes(image_id_or_path)
        with Image.open(io.BytesIO(data)) as img:
            return ImageOps.exif_transpose(img).copy()

    def preprocess_for_ocr(
        self,
        image: Image.Image,
        to_grayscale: bool = True,
        max_dimension: int = 2048,
    ) -> Image.Image:
        """
        Normalize and preprocess prescription image in memory for OCR.
        - Keeps aspect ratio while constraining maximum dimension
        - Optional grayscale conversion
        """
        processed = image.copy()
        processed = ImageOps.exif_transpose(processed)

        w, h = processed.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            processed = processed.resize(new_size, Image.Resampling.LANCZOS)

        if to_grayscale and processed.mode not in ("L", "1"):
            processed = processed.convert("L")

        return processed
