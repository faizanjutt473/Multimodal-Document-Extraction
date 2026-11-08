"""Small, dependency-free helpers to normalise numbers, dates and text."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

_ORDINAL_RE = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)

_ISO_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d")
_TEXT_FORMATS = (
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
)
_MDY_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y")
_DMY_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y")


def parse_amount(value: Any) -> float | None:
    """Convert strings such as '$ 1,209.15', '209,15' or '(12.00)' to a float.

    The decimal separator is detected automatically:
      * if both ',' and '.' appear, the one that appears last is the decimal mark;
      * a single ',' followed by 1-2 digits is a decimal mark ('209,15');
      * otherwise separators are treated as thousands separators.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    cleaned = re.sub(r"[^\d,.]", "", text)
    if not re.search(r"\d", cleaned):
        return None

    last_comma, last_dot = cleaned.rfind(","), cleaned.rfind(".")
    decimal_index = -1
    if last_comma != -1 and last_dot != -1:
        decimal_index = max(last_comma, last_dot)
    elif last_comma != -1:
        decimals = len(cleaned) - last_comma - 1
        if cleaned.count(",") == 1 and decimals in (1, 2):
            decimal_index = last_comma
    elif last_dot != -1 and cleaned.count(".") == 1:
        decimal_index = last_dot

    if decimal_index == -1:
        number = re.sub(r"[,.]", "", cleaned)
    else:
        integer = re.sub(r"[,.]", "", cleaned[:decimal_index]) or "0"
        fraction = re.sub(r"[,.]", "", cleaned[decimal_index + 1:])
        number = f"{integer}.{fraction or '0'}"
    try:
        result = float(number)
    except ValueError:
        return None
    return -result if negative else result


def parse_date(value: Any, dayfirst: bool = False) -> date | None:
    """Parse many common date notations. Returns None if nothing matches."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = _ORDINAL_RE.sub("", str(value).strip())
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    numeric = _DMY_FORMATS + _MDY_FORMATS if dayfirst else _MDY_FORMATS + _DMY_FORMATS
    for fmt in _ISO_FORMATS + _TEXT_FORMATS + numeric:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalise_words(text: Any) -> str:
    """Lower-case, strip punctuation and collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def normalise_alnum(text: Any) -> str:
    """Lower-case and keep only letters and digits (used for IDs)."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())
