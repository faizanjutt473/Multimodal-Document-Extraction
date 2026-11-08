"""Pydantic models that define (and enforce) the output JSON schema."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .parsing import parse_amount, parse_date

FIELD_NAMES: tuple[str, ...] = ("invoice_number", "invoice_date", "vendor", "line_items", "total")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


class LineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None  # per unit, before tax
    amount: float | None = None      # line total as printed (gross if tax is shown)

    @field_validator("description", mode="before")
    @classmethod
    def _description(cls, value: Any) -> str | None:
        return _clean_text(value)

    @field_validator("quantity", "unit_price", "amount", mode="before")
    @classmethod
    def _numbers(cls, value: Any) -> float | None:
        return parse_amount(value)


class ExtractedFields(BaseModel):
    """The structured data we extract from one invoice / receipt."""

    model_config = ConfigDict(extra="ignore")

    invoice_number: str | None = None
    invoice_date: str | None = Field(default=None, description="ISO format YYYY-MM-DD")
    vendor: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    total: float | None = None

    @field_validator("invoice_number", "vendor", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str | None:
        return _clean_text(value)

    @field_validator("invoice_date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> str | None:
        # Day/month order is resolved by the pipeline before validation (see pipeline.py);
        # here we only accept ISO strings or anything parse_date can read with the US default.
        parsed = parse_date(value)
        return parsed.isoformat() if parsed else None

    @field_validator("total", mode="before")
    @classmethod
    def _total(cls, value: Any) -> float | None:
        return parse_amount(value)

    @field_validator("line_items", mode="before")
    @classmethod
    def _items(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]


class DocumentResult(BaseModel):
    """Final output for one document: data + confidence + review flags."""

    source_file: str
    data: ExtractedFields = Field(default_factory=ExtractedFields)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    review_reasons: dict[str, list[str]] = Field(default_factory=dict)
    fields_to_review: list[str] = Field(default_factory=list)
    needs_review: bool = False
    ocr_mean_confidence: float | None = None
    model: str | None = None
    error: str | None = None
