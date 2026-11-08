"""Convert the raw dataset annotations (katanaml invoices-donut-data-v1) to our own schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ExtractedFields


def _load_entry(entry: Any) -> dict[str, Any]:
    """Each raw entry is a JSON string like {"gt_parse": {...}} (or already a dict)."""
    if isinstance(entry, str):
        entry = json.loads(entry)
    return entry.get("gt_parse", entry)


def to_schema(entry: Any) -> dict[str, Any]:
    gt = _load_entry(entry)
    header = gt.get("header", {}) or {}
    summary = gt.get("summary", {}) or {}
    items = gt.get("items", []) or []

    fields = ExtractedFields.model_validate(
        {
            "invoice_number": header.get("invoice_no") or header.get("invoice_number"),
            "invoice_date": header.get("invoice_date"),
            "vendor": header.get("seller") or header.get("vendor"),
            "line_items": [
                {
                    "description": item.get("item_desc"),
                    "quantity": item.get("item_qty"),
                    "unit_price": item.get("item_net_price"),
                    "amount": item.get("item_gross_worth"),
                }
                for item in items
            ],
            "total": summary.get("total_gross_worth"),
        }
    )
    return fields.model_dump()


def build_ground_truth(raw_path: Path, out_path: Path) -> dict[str, dict[str, Any]]:
    """Image names follow the download script: inv_000.png, inv_001.png, ... (same order)."""
    raw_entries = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    truth = {f"inv_{index:03d}.png": to_schema(entry) for index, entry in enumerate(raw_entries)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return truth
