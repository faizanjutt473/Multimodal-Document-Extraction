# Multimodal Document Extraction System

Extracts structured data from scanned invoices / receipts (images or PDFs) using **OCR + a vision LLM**.
The output is **schema-validated JSON**, low-confidence fields are **flagged for human review**, and
**field-level accuracy** is measured against ground truth on 30+ documents.

## 1. How it works

```
 invoice (png/jpg/pdf)
        |
        +--> OpenCV preprocessing --> Tesseract OCR --> text + word confidence ---+
        |                                                                         |
        +--> vision LLM (image + OCR text as a hint) --> JSON + per-field confidence
                                                                                  |
                              Pydantic schema + business rules (validation.py) <--+
                                                                                  |
                    confidence scoring (LLM confidence + OCR agreement + rules)   |
                                                                                  v
                      outputs/predictions/<doc>.json   (data, confidence, review flags)
                                                                                  |
                       evaluate.py --> reports/accuracy_report.md  (field-level accuracy)
                       review_app.py --> human review of flagged fields (Streamlit)
```

**LLM provider:** Google Gemini (free tier, used for this project). Anthropic Claude is supported as an optional alternative (`LLM_PROVIDER`).

**Extracted fields:** `invoice_number`, `invoice_date` (ISO), `vendor`, `line_items`
(description, quantity, unit_price, amount) and `total`.

### Confidence and review flags
For every field: `confidence = 0.6 * model_confidence + 0.4 * grounding`, where *grounding* checks whether the
extracted value is also found in the independent OCR text. Any failed business rule (missing value, unparseable
date, sum of line items != total, ...) caps the confidence at 0.5. Fields below `REVIEW_THRESHOLD`
(default 0.75) get `needs_review = true` and a human-readable reason.

## 2. Setup

1. **Python 3.10+** and a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Install the Tesseract OCR program** (separate from pip). Windows installer: UB Mannheim build of Tesseract.
   Default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically;
   otherwise set `TESSERACT_CMD` in `.env`. If Tesseract is missing the pipeline still runs
   (vision model only) but loses the OCR cross-check.
3. **API key (Google Gemini, free tier):** get a key from Google AI Studio, then copy `.env.example` to `.env` and set
   `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` and `GEMINI_MODEL=...` (model names change; use one available to your key,
   this project was run with `gemini-3.8-flash`). The Gemini package is `google-genai` (already in `requirements.txt`).
   *Optional:* an Anthropic Claude key can be used instead (`LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...`, `pip install anthropic`).

## 3. Run

```powershell
python download_data.py          # (once) 40 sample invoices -> data/images + data/ground_truth_raw.json
python prepare_ground_truth.py   # -> data/ground_truth.json (our schema)
python run_extraction.py --limit 3   # quick test on 3 documents
python run_extraction.py             # all documents (resumable; use --overwrite to redo)
python evaluate.py                   # -> reports/accuracy_report.md
streamlit run review_app.py          # optional: review flagged documents
pytest                               # unit tests (no API key needed)
```

## 4. Output example

```json
{
  "source_file": "inv_000.png",
  "data": {
    "invoice_number": "51109338",
    "invoice_date": "2013-04-13",
    "vendor": "Andrews, Kirby and Valdez",
    "line_items": [{"description": "Wine Glasses", "quantity": 5.0, "unit_price": 12.99, "amount": 71.44}],
    "total": 71.44
  },
  "field_confidence": {"invoice_number": 0.98, "invoice_date": 0.97, "vendor": 0.95, "line_items": 0.9, "total": 0.99},
  "review_reasons": {},
  "fields_to_review": [],
  "needs_review": false
}
```

## 5. Evaluation method
See section 4 of `reports/accuracy_report.md` for the exact matching rules per field. Besides accuracy the report
shows how useful the flagging is: accuracy of values that were *not* flagged and the share of errors that *were* flagged.

## 6. Results status and known issue (free API tier)

This project was run with the **free tier of the Google Gemini API**. During the first full run the API returned
`429 RESOURCE_EXHAUSTED` (free-tier quota used up) and some documents failed. Because `run_extraction.py` is resumable,
the run was continued afterwards: finished documents were skipped and the failed ones were retried.

- Final evaluation: **31 documents** (the task requires at least 30). The exact number is printed at the top of
  `reports/accuracy_report.md`, and section 6 of that report lists anything that was excluded.
- 3 of the 40 dataset records have incomplete ground truth (missing invoice number or total) and are excluded from evaluation.
- Documents that fail because of API errors are never counted as model mistakes; they are excluded and listed with the error message.
- If you hit the free-tier limit again, wait for the quota reset (or use a paid plan / another key) and run
  `python run_extraction.py` followed by `python evaluate.py`.

## 7. Limitations
- The sample dataset is synthetic and clean, and the measured accuracy on it is very high. This does **not** show that the system
  is perfect on real-world scans (blur, stamps, handwriting, odd layouts), which will score lower.
- The dataset is public, so the vision model may have seen similar invoices. Treat the numbers as a baseline, not a guarantee.
- Confidence is a heuristic, not a calibrated probability.
- Only one invoice per file; multi-page PDFs are limited to the first 3 pages.
- Requires network access and an API key for the vision model (Gemini by default; see `doc_extraction/llm.py` to add another provider).

## 8. Project structure
```
doc_extraction/   config, parsing, schema, documents, ocr, llm, validation, confidence, pipeline, ground_truth, metrics
download_data.py  prepare_ground_truth.py  run_extraction.py  evaluate.py  review_app.py
tests/            parsing, schema/validation, confidence/metrics, pipeline (fake extractor)
data/             images + ground truth      outputs/  predictions      reports/  accuracy report
```
