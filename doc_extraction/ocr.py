"""OCR with Tesseract: preprocessing, text extraction and word-level confidence."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    mean_confidence: float | None  # 0-100, None if no words were found
    word_count: int


def configure_tesseract(cmd: str | None) -> None:
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def _estimate_skew(binary_inv: np.ndarray) -> float:
    """Find the small rotation angle (degrees) that makes text rows most 'peaky'."""
    scale = 600 / max(binary_inv.shape[1], 1)
    small = cv2.resize(binary_inv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    centre = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-5.0, 5.01, 0.5):
        matrix = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotated = cv2.warpAffine(small, matrix, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def preprocess(image: Image.Image) -> np.ndarray:
    """Grayscale -> upscale small images -> denoise -> deskew -> binarise (Otsu)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    if max(height, width) < 1600:
        factor = min(2.0, 1600 / max(height, width))
        gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    angle = _estimate_skew(255 - binary)
    if abs(angle) >= 0.5:
        h, w = binary.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        binary = cv2.warpAffine(
            binary, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return binary


def run_ocr(image: Image.Image, config: str = "--oem 3 --psm 6") -> OCRResult:
    """Run Tesseract on one page and return text + mean word confidence."""
    processed = preprocess(image)
    data = pytesseract.image_to_data(processed, config=config, output_type=Output.DICT)

    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf >= 0:
            confidences.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    return OCRResult(
        text=text,
        mean_confidence=round(mean(confidences), 1) if confidences else None,
        word_count=sum(len(w) for w in lines.values()),
    )


def run_ocr_pages(images: list[Image.Image], config: str = "--oem 3 --psm 6") -> OCRResult:
    """OCR several pages and merge the results."""
    results = [run_ocr(img, config) for img in images]
    confs = [r.mean_confidence for r in results if r.mean_confidence is not None]
    return OCRResult(
        text="\n\n".join(r.text for r in results),
        mean_confidence=round(mean(confs), 1) if confs else None,
        word_count=sum(r.word_count for r in results),
    )
