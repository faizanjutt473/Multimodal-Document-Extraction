"""Business-rule validation. Returns a list of issues per field (empty list = OK)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .config import Settings
from .schema import FIELD_NAMES, ExtractedFields


def validate_fields(
    fields: ExtractedFields, raw_values: dict[str, Any], settings: Settings
) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {name: [] for name in FIELD_NAMES}

    # invoice_number
    if not fields.invoice_number:
        issues["invoice_number"].append("missing")
    elif len(fields.invoice_number) > 40:
        issues["invoice_number"].append("unusually long for an invoice number")

    # invoice_date
    raw_date = raw_values.get("invoice_date")
    if not fields.invoice_date:
        issues["invoice_date"].append(
            "date could not be parsed" if raw_date not in (None, "") else "missing"
        )
    else:
        parsed = date.fromisoformat(fields.invoice_date)
        if parsed.year < 1990 or parsed > date.today() + timedelta(days=366):
            issues["invoice_date"].append(f"date {fields.invoice_date} is outside a plausible range")

    # vendor
    if not fields.vendor:
        issues["vendor"].append("missing")
    elif len(fields.vendor) < 2:
        issues["vendor"].append("vendor name too short")

    # total
    if fields.total is None:
        issues["total"].append("missing")
    elif fields.total <= 0:
        issues["total"].append("total is not positive")

    # line items
    if not fields.line_items:
        issues["line_items"].append("no line items found")
    else:
        for index, item in enumerate(fields.line_items, start=1):
            if item.amount is None:
                issues["line_items"].append(f"item {index}: amount missing")
            elif item.amount < 0:
                issues["line_items"].append(f"item {index}: negative amount")

    # cross-check: sum of line items vs. total
    amounts = [item.amount for item in fields.line_items if item.amount is not None]
    if fields.total is not None and amounts and len(amounts) == len(fields.line_items):
        line_sum = round(sum(amounts), 2)
        tolerance = max(settings.sum_tolerance_abs, settings.sum_tolerance_rel * abs(fields.total))
        if abs(line_sum - fields.total) > tolerance:
            message = f"sum of line items ({line_sum}) does not match total ({fields.total})"
            issues["total"].append(message)
            issues["line_items"].append(message)

    return issues
