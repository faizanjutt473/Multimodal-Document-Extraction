"""Pipeline test with a fake extractor, so no API key or network is needed."""
from PIL import Image, ImageDraw

from doc_extraction.config import Settings
from doc_extraction.llm import LLMOutput
from doc_extraction.pipeline import DocumentPipeline


class FakeExtractor:
    def extract(self, images, ocr_text):
        return LLMOutput(
            values={
                "invoice_number": "12345",
                "invoice_date": "03/05/2024",
                "vendor": "Acme Ltd",
                "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 10, "amount": 10}],
                "total": 99,  # deliberately wrong: does not match the line items
            },
            confidences={k: 0.95 for k in ("invoice_number", "invoice_date", "vendor", "line_items", "total")},
        )


def test_pipeline_flags_inconsistent_total(tmp_path):
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 50), "Acme Ltd  Invoice 12345  03/05/2024  Total 10.00", fill="black")
    path = tmp_path / "doc.png"
    image.save(path)

    result = DocumentPipeline(Settings(), extractor=FakeExtractor()).process(path)

    assert result.error is None
    assert result.data.invoice_date == "2024-03-05"
    assert "total" in result.fields_to_review
    assert result.needs_review
