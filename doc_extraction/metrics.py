"""Field-level comparison between predictions and ground truth."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from .parsing import normalise_alnum, normalise_words

COMPARED_FIELDS = ("invoice_number", "invoice_date", "vendor", "line_items", "total")

AMOUNT_TOLERANCE = 0.01
VENDOR_THRESHOLD = 90      # partial fuzzy match: the dataset's seller text also contains the address
ITEM_DESC_THRESHOLD = 80


def amounts_match(a: float | None, b: float | None, tolerance: float = AMOUNT_TOLERANCE) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def ids_match(pred: str | None, truth: str | None) -> bool:
    if not pred and not truth:
        return True
    if not pred or not truth:
        return False
    return normalise_alnum(pred) == normalise_alnum(truth)


def dates_match(pred: str | None, truth: str | None) -> bool:
    return (pred or None) == (truth or None)


def vendors_match(pred: str | None, truth: str | None) -> bool:
    if not pred and not truth:
        return True
    if not pred or not truth:
        return False
    p, t = normalise_words(pred), normalise_words(truth)
    if len(p) < 3:
        return False
    return fuzz.partial_ratio(p, t) >= VENDOR_THRESHOLD


def _items_equal(pred: dict[str, Any], truth: dict[str, Any]) -> bool:
    if not amounts_match(pred.get("amount"), truth.get("amount")):
        return False
    p_desc, t_desc = pred.get("description"), truth.get("description")
    if p_desc and t_desc:
        return fuzz.token_set_ratio(normalise_words(p_desc), normalise_words(t_desc)) >= ITEM_DESC_THRESHOLD
    return True


@dataclass
class ItemMatch:
    matched: int
    n_pred: int
    n_truth: int

    @property
    def exact(self) -> bool:
        return self.matched == self.n_pred == self.n_truth


def match_line_items(pred: list[dict[str, Any]], truth: list[dict[str, Any]]) -> ItemMatch:
    """Greedy one-to-one matching of predicted rows to ground-truth rows."""
    used: set[int] = set()
    matched = 0
    for truth_item in truth:
        for index, pred_item in enumerate(pred):
            if index not in used and _items_equal(pred_item, truth_item):
                used.add(index)
                matched += 1
                break
    return ItemMatch(matched=matched, n_pred=len(pred), n_truth=len(truth))


def compare_field(field: str, pred: dict[str, Any], truth: dict[str, Any]) -> bool:
    if field == "invoice_number":
        return ids_match(pred.get("invoice_number"), truth.get("invoice_number"))
    if field == "invoice_date":
        return dates_match(pred.get("invoice_date"), truth.get("invoice_date"))
    if field == "vendor":
        return vendors_match(pred.get("vendor"), truth.get("vendor"))
    if field == "total":
        return amounts_match(pred.get("total"), truth.get("total"))
    if field == "line_items":
        return match_line_items(pred.get("line_items", []), truth.get("line_items", [])).exact
    raise ValueError(f"Unknown field: {field}")
