"""End-to-end pipeline: load -> OCR -> vision LLM -> validate -> score confidence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from PIL import Image

from . import ocr
from .config import Settings, get_settings
from .confidence import compute_grounding, score_fields
from .documents import load_pages
from .llm import LLMOutput, create_extractor
from .parsing import parse_date
from .schema import FIELD_NAMES, DocumentResult, ExtractedFields
from .validation import validate_fields

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    def extract(self, images: list[Image.Image], ocr_text: str | None) -> LLMOutput: ...


class DocumentPipeline:
    def __init__(self, settings: Settings | None = None, extractor: Extractor | None = None):
        self.settings = settings or get_settings()
        ocr.configure_tesseract(self.settings.tesseract_cmd)
        self.extractor = extractor or create_extractor(self.settings)

    # ------------------------------------------------------------------ OCR
    def _run_ocr(self, pages: list[Image.Image]) -> ocr.OCRResult | None:
        try:
            return ocr.run_ocr_pages(pages, self.settings.ocr_config)
        except Exception as exc:  # Tesseract missing / crashed: continue with vision only
            logger.warning("OCR unavailable (%s). Continuing with the vision model only.", exc)
            return None

    # -------------------------------------------------------------- process
    def process(self, path: Path) -> DocumentResult:
        path = Path(path)
        try:
            pages = load_pages(path, max_pages=self.settings.max_pdf_pages)
            ocr_result = self._run_ocr(pages)
            ocr_text = ocr_result.text if ocr_result else None

            output = self.extractor.extract(pages, ocr_text)
            raw_values = dict(output.values)

            # Resolve DD/MM vs MM/DD with the configured preference, then validate the schema.
            parsed_date = parse_date(raw_values.get("invoice_date"), self.settings.date_dayfirst)
            payload = {name: raw_values.get(name) for name in FIELD_NAMES}
            payload["invoice_date"] = parsed_date.isoformat() if parsed_date else None
            fields = ExtractedFields.model_validate(payload)

            issues = validate_fields(fields, raw_values, self.settings)
            grounding = compute_grounding(fields, ocr_text, self.settings)
            confidence, reasons, flagged = score_fields(
                fields, output.confidences, grounding, issues, self.settings
            )
            return DocumentResult(
                source_file=path.name,
                data=fields,
                field_confidence=confidence,
                review_reasons={k: v for k, v in reasons.items() if v},
                fields_to_review=flagged,
                needs_review=bool(flagged),
                ocr_mean_confidence=ocr_result.mean_confidence if ocr_result else None,
                model=self.settings.model,
            )
        except Exception as exc:
            logger.error("Failed to process %s: %s", path.name, exc)
            return DocumentResult(
                source_file=path.name,
                needs_review=True,
                fields_to_review=list(FIELD_NAMES),
                review_reasons={"document": [f"processing failed: {exc}"]},
                model=self.settings.model,
                error=str(exc),
            )
