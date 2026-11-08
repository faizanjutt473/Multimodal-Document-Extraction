from datetime import date

import pytest

from doc_extraction.parsing import parse_amount, parse_date


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$ 209,15", 209.15),
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("1,234", 1234.0),
        ("12,5", 12.5),
        ("USD 99", 99.0),
        ("(12.00)", -12.0),
        (7, 7.0),
        ("abc", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize(
    "raw, dayfirst, expected",
    [
        ("2024-03-05", False, date(2024, 3, 5)),
        ("03/05/2024", False, date(2024, 3, 5)),
        ("03/05/2024", True, date(2024, 5, 3)),
        ("25/12/2023", False, date(2023, 12, 25)),  # impossible as MM/DD -> falls back to DD/MM
        ("March 5, 2024", False, date(2024, 3, 5)),
        ("5th Mar 2024", False, date(2024, 3, 5)),
        ("not a date", False, None),
        (None, False, None),
    ],
)
def test_parse_date(raw, dayfirst, expected):
    assert parse_date(raw, dayfirst) == expected
