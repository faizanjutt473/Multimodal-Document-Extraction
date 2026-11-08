"""Loading documents (images / PDFs) and preparing images for the vision model."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}


def list_documents(directory: Path) -> list[Path]:
    """Return all supported documents in a directory, sorted by name."""
    if not directory.exists():
        raise FileNotFoundError(f"Folder not found: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_pages(path: Path, max_pages: int = 3, pdf_dpi: int = 200) -> list[Image.Image]:
    """Load an image or PDF as a list of RGB PIL images (one per page)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, max_pages, pdf_dpi)
    if suffix in IMAGE_EXTENSIONS:
        with Image.open(path) as img:
            return [ImageOps.exif_transpose(img).convert("RGB")]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_pdf(path: Path, max_pages: int, dpi: int) -> list[Image.Image]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover - older versions
        import fitz

    pages: list[Image.Image] = []
    scale = dpi / 72
    with fitz.open(path) as pdf:
        for index in range(min(len(pdf), max_pages)):
            pixmap = pdf.load_page(index).get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
    if not pages:
        raise ValueError(f"PDF has no pages: {path}")
    return pages


def encode_for_llm(image: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    """Resize (keeping aspect ratio) and encode as base64 JPEG. Returns (data, media_type)."""
    img = image.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii"), "image/jpeg"
