# Accuracy Report

Generated: 2026-11-08  |  Documents evaluated: **31**  |  Review threshold: **0.75**

## 1. Field-level accuracy

| Field | Correct | Accuracy |
|---|---|---|
| invoice_number | 31/31 | 100.0% |
| invoice_date | 31/31 | 100.0% |
| vendor | 31/31 | 100.0% |
| line_items | 31/31 | 100.0% |
| total | 31/31 | 100.0% |

- **Mean field accuracy:** 100.0%
- **Documents with every field correct:** 31/31 (100.0%)
- **Line items, row level:** precision 100.0%, recall 100.0%, F1 100.0%

## 2. Human-review flagging

| Field | Flagged for review | Accuracy of NOT-flagged values | Share of errors that were flagged |
|---|---|---|---|
| invoice_number | 0.0% | 100.0% | n/a |
| invoice_date | 0.0% | 100.0% | n/a |
| vendor | 0.0% | 100.0% | n/a |
| line_items | 9.7% | 100.0% | n/a |
| total | 38.7% | 100.0% | n/a |

- Documents with at least one flagged field: 12/31 (38.7%)
- A useful flagging system has **high accuracy on not-flagged values** and a **high share of errors caught**,
  while keeping the flagged rate low enough that review stays cheap.

## 3. Where the system fails

No errors found.

## 4. How accuracy was measured

- Ground truth: the dataset's own annotations, converted to the project schema (`prepare_ground_truth.py`).
- `invoice_number`: equal after lower-casing and removing punctuation/spaces.
- `invoice_date`: equal ISO date (YYYY-MM-DD).
- `vendor`: fuzzy partial match >= 90 (the dataset's seller text also contains an address, the model returns the name only).
- `total`: equal within 0.01.
- `line_items`: a document counts as correct only if **every** row matches (amount within 0.01 and similar description) and no extra rows exist.
- Documents that failed because of API errors are excluded and listed in section 6.

## 5. Limitations

- The dataset consists of synthetic, clean invoices; real-world scans (blur, stamps, handwriting) will score lower.
- Fuzzy vendor matching is lenient by design; a stricter metric would lower the vendor score.
- Confidence is a heuristic (model self-report + OCR agreement + rule checks), not a calibrated probability.

## 6. Run notes and known issues

- Documents evaluated: **31** (the task asks for at least 30).
- 1 documents failed during extraction (inv_033.png). They are **excluded** from the accuracy numbers, because the failure came from the API and not from the extraction quality. First error message: `503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please `
- 5 documents have no prediction because the run was stopped (first ones: inv_035.png, inv_036.png, inv_037.png, inv_038.png, inv_039.png).
- 3 documents (inv_008.png, inv_009.png, inv_011.png) were excluded because the dataset's own annotation is incomplete (missing invoice number or total).
