"""Confidence scoring.

Final confidence of a field combines three independent signals:
  1. the confidence the vision model reports for that field,
  2. "grounding": does the extracted value actually appear in the independent OCR text?
  3. rule validation: any failed business rule caps the confidence.

    confidence = 0.6 * llm_confidence + 0.4 * grounding      (grounding skipped if OCR is unavailable)
    if validation issues:   confidence = min(confidence, 0.5)
    if value is missing:    confidence = 0.0

Fields below `settings.review_threshold` are flagged for human review.
"""
from __future__ import annotations

import re
from datetime import date

from rapidfuzz import fuzz

from .config import Settings
from .parsing import normalise_alnum, normalise_words, parse_amount, parse_date
from .schema import FIELD_NAMES, ExtractedFields

LLM_WEIGHT = 0.6
GROUNDING_WEIGHT = 0.4
VALIDATION_CAP = 0.5

_NUMBER_RE = re.compile(r"\d[\d.,]*")
_DATE_RE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def ocr_numbers(ocr_text: str) -> set[float]:
    values = set()
    for token in _NUMBER_RE.findall(ocr_text):
        number = parse_amount(token.rstrip(".,"))
        if number is not None:
            values.add(round(number, 2))
    return values


def grounding_id(value: str, ocr_text: str) -> float:
    """1.0 if the ID appears in the OCR text, otherwise a low score (a near miss is suspicious)."""
    wanted = normalise_alnum(value)
    if not wanted:
        return 0.0
    if wanted in normalise_alnum(ocr_text):
        return 1.0
    return 0.4 * fuzz.partial_ratio(wanted, normalise_alnum(ocr_text)) / 100


def grounding_text(value: str, ocr_text: str) -> float:
    wanted, haystack = normalise_words(value), normalise_words(ocr_text)
    if not wanted:
        return 0.0
    if wanted in haystack:
        return 1.0
    return fuzz.partial_ratio(wanted, haystack) / 100


def grounding_number(value: float, numbers: set[float]) -> float:
    return 1.0 if round(value, 2) in numbers else 0.0


def grounding_date(iso_value: str, ocr_text: str, dayfirst: bool) -> float:
    target = date.fromisoformat(iso_value)
    for token in _DATE_RE.findall(ocr_text):
        for order in (dayfirst, not dayfirst):
            if parse_date(token, dayfirst=order) == target:
                return 1.0
    lowered = ocr_text.lower()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        if target.strftime(fmt).lower().replace(" 0", " ") in lowered:
            return 1.0
    return 0.0


def compute_grounding(
    fields: ExtractedFields, ocr_text: str | None, settings: Settings
) -> dict[str, float | None]:
    """Grounding score per field, or None when OCR text is unavailable / the value is missing."""
    result: dict[str, float | None] = {name: None for name in FIELD_NAMES}
    if not ocr_text or not ocr_text.strip():
        return result

    numbers = ocr_numbers(ocr_text)
    if fields.invoice_number:
        result["invoice_number"] = grounding_id(fields.invoice_number, ocr_text)
    if fields.invoice_date:
        result["invoice_date"] = grounding_date(fields.invoice_date, ocr_text, settings.date_dayfirst)
    if fields.vendor:
        result["vendor"] = grounding_text(fields.vendor, ocr_text)
    if fields.total is not None:
        result["total"] = grounding_number(fields.total, numbers)
    amounts = [i.amount for i in fields.line_items if i.amount is not None]
    if amounts:
        result["line_items"] = sum(grounding_number(a, numbers) for a in amounts) / len(amounts)
    return result


def _is_missing(fields: ExtractedFields, name: str) -> bool:
    value = getattr(fields, name)
    return value is None or value == [] or value == ""


def score_fields(
    fields: ExtractedFields,
    llm_confidence: dict[str, float],
    grounding: dict[str, float | None],
    issues: dict[str, list[str]],
    settings: Settings,
) -> tuple[dict[str, float], dict[str, list[str]], list[str]]:
    """Return (confidence per field, review reasons per field, list of fields to review)."""
    confidence: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    flagged: list[str] = []

    for name in FIELD_NAMES:
        field_reasons: list[str] = []
        if _is_missing(fields, name):
            score = 0.0
            field_reasons.append("value not found in the document")
        else:
            llm_score = llm_confidence.get(name, 0.5)
            ground = grounding.get(name)
            if ground is None:
                score = llm_score
            else:
                score = LLM_WEIGHT * llm_score + GROUNDING_WEIGHT * ground
                if ground < 0.5:
                    field_reasons.append("value not confirmed by independent OCR text")
            if llm_score < settings.review_threshold:
                field_reasons.append(f"model reported low confidence ({llm_score:.2f})")
            if issues.get(name):
                score = min(score, VALIDATION_CAP)
        field_reasons.extend(i for i in issues.get(name, []) if i not in field_reasons)

        confidence[name] = round(score, 3)
        reasons[name] = field_reasons
        if score < settings.review_threshold:
            flagged.append(name)
            if not field_reasons:
                reasons[name] = [f"confidence {score:.2f} below threshold"]
    return confidence, reasons, flagged
