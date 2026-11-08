from doc_extraction.config import Settings
from doc_extraction.confidence import compute_grounding, score_fields
from doc_extraction.llm import ExtractionError, parse_llm_json
from doc_extraction.metrics import compare_field, match_line_items
from doc_extraction.schema import ExtractedFields

SETTINGS = Settings()
OCR_TEXT = "Acme Ltd\nInvoice no: 51109338\nDate of issue: 04/13/2013\nWidget 2,00 10,00 22,00\nTotal $ 22,00"


def fields(**kw):
    base = {
        "invoice_number": "51109338",
        "invoice_date": "2013-04-13",
        "vendor": "Acme Ltd",
        "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 10, "amount": 22}],
        "total": 22,
    }
    base.update(kw)
    return ExtractedFields.model_validate(base)


def test_grounded_values_get_high_confidence():
    f = fields()
    grounding = compute_grounding(f, OCR_TEXT, SETTINGS)
    conf, reasons, flagged = score_fields(f, {k: 0.95 for k in grounding}, grounding, {k: [] for k in grounding}, SETTINGS)
    assert flagged == []
    assert min(conf.values()) > 0.9


def test_value_missing_from_ocr_is_flagged():
    f = fields(invoice_number="99999999")
    grounding = compute_grounding(f, OCR_TEXT, SETTINGS)
    conf, reasons, flagged = score_fields(f, {k: 0.95 for k in grounding}, grounding, {k: [] for k in grounding}, SETTINGS)
    assert "invoice_number" in flagged
    assert "not confirmed" in reasons["invoice_number"][0]


def test_validation_issue_caps_confidence():
    f = fields()
    issues = {k: [] for k in ("invoice_number", "invoice_date", "vendor", "line_items", "total")}
    issues["total"] = ["sum mismatch"]
    conf, _, flagged = score_fields(f, {k: 1.0 for k in issues}, {k: 1.0 for k in issues}, issues, SETTINGS)
    assert conf["total"] <= 0.5 and "total" in flagged


def test_no_ocr_falls_back_to_llm_confidence():
    f = fields()
    grounding = compute_grounding(f, None, SETTINGS)
    assert all(v is None for v in grounding.values())


def test_metrics():
    truth = fields().model_dump()
    assert compare_field("invoice_number", {"invoice_number": "5110-9338"}, truth)
    assert compare_field("vendor", {"vendor": "Acme"}, {"vendor": "Acme Ltd 12 Main St, Springfield"})
    assert not compare_field("total", {"total": 23}, truth)
    result = match_line_items(truth["line_items"], truth["line_items"])
    assert result.exact


def test_llm_json_parsing_handles_code_fences():
    reply = '```json\n{"invoice_number": {"value": "A1", "confidence": 0.9}, "total": 5}\n```'
    output = parse_llm_json(reply)
    assert output.values["invoice_number"] == "A1"
    assert output.confidences["invoice_number"] == 0.9
    assert output.confidences["total"] == 0.5  # bare value -> neutral confidence


def test_llm_json_parsing_rejects_garbage():
    import pytest

    with pytest.raises(ExtractionError):
        parse_llm_json("sorry, I cannot do that")
