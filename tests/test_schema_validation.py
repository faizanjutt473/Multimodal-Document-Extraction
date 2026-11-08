from doc_extraction.config import Settings
from doc_extraction.schema import ExtractedFields
from doc_extraction.validation import validate_fields

SETTINGS = Settings()


def make(**overrides):
    base = {
        "invoice_number": "INV-001",
        "invoice_date": "2024-03-05",
        "vendor": "Acme Ltd",
        "line_items": [
            {"description": "Widget", "quantity": "2,00", "unit_price": "$ 10,00", "amount": "$ 22,00"},
            {"description": "Gadget", "quantity": 1, "unit_price": 8, "amount": 8},
        ],
        "total": "$ 30,00",
    }
    base.update(overrides)
    return ExtractedFields.model_validate(base)


def test_schema_normalises_numbers_and_dates():
    fields = make(invoice_date="03/05/2024")
    assert fields.total == 30.0
    assert fields.line_items[0].quantity == 2.0
    assert fields.line_items[0].amount == 22.0
    assert fields.invoice_date == "2024-03-05"


def test_valid_document_has_no_issues():
    fields = make()
    issues = validate_fields(fields, {"invoice_date": "2024-03-05"}, SETTINGS)
    assert all(not v for v in issues.values())


def test_sum_mismatch_is_reported():
    fields = make(total=99)
    issues = validate_fields(fields, {}, SETTINGS)
    assert any("does not match" in m for m in issues["total"])
    assert any("does not match" in m for m in issues["line_items"])


def test_unparseable_date_and_missing_fields():
    fields = make(invoice_date="someday", invoice_number=None, line_items=[])
    issues = validate_fields(fields, {"invoice_date": "someday"}, SETTINGS)
    assert issues["invoice_date"] == ["date could not be parsed"]
    assert issues["invoice_number"] == ["missing"]
    assert issues["line_items"] == ["no line items found"]
