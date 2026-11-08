"""Compute field-level accuracy against the ground truth and write the accuracy report."""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from doc_extraction.config import get_settings
from doc_extraction.metrics import COMPARED_FIELDS, compare_field, match_line_items


def load_json_dir(directory: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        results[record["source_file"]] = record
    return results


def build_table(predictions: dict, truth: dict) -> pd.DataFrame:
    rows = []
    for name, prediction in predictions.items():
        if name not in truth:
            continue
        pred_data, true_data = prediction["data"], truth[name]
        flagged_fields = set(prediction.get("fields_to_review", []))
        for field in COMPARED_FIELDS:
            rows.append(
                {
                    "document": name,
                    "field": field,
                    "correct": compare_field(field, pred_data, true_data),
                    "confidence": prediction.get("field_confidence", {}).get(field, 0.0),
                    "flagged": field in flagged_fields,
                    "predicted": json.dumps(pred_data.get(field), ensure_ascii=False),
                    "truth": json.dumps(true_data.get(field), ensure_ascii=False),
                    "reasons": "; ".join(prediction.get("review_reasons", {}).get(field, [])),
                }
            )
    return pd.DataFrame(rows)


def item_level_scores(predictions: dict, truth: dict) -> tuple[float, float, float]:
    matched = n_pred = n_truth = 0
    for name, prediction in predictions.items():
        if name in truth:
            result = match_line_items(prediction["data"].get("line_items", []), truth[name]["line_items"])
            matched, n_pred, n_truth = matched + result.matched, n_pred + result.n_pred, n_truth + result.n_truth
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def pct(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(table: pd.DataFrame, predictions: dict, truth: dict, threshold: float) -> tuple[str, pd.DataFrame]:
    docs = table["document"].nunique()
    summary_rows, field_rows = [], []
    for field in COMPARED_FIELDS:
        sub = table[table.field == field]
        wrong = sub[~sub.correct]
        kept, flagged = sub[~sub.flagged], sub[sub.flagged]
        field_rows.append(
            {
                "field": field,
                "correct": int(sub.correct.sum()),
                "total": len(sub),
                "accuracy": sub.correct.mean(),
                "flagged_rate": sub.flagged.mean(),
                "accuracy_not_flagged": kept.correct.mean() if len(kept) else None,
                "errors_caught_by_flagging": wrong.flagged.mean() if len(wrong) else None,
            }
        )
    fields_df = pd.DataFrame(field_rows)
    for row in field_rows:
        summary_rows.append(
            [row["field"], f"{row['correct']}/{row['total']}", pct(row["accuracy"])]
        )

    per_doc = table.groupby("document").agg(all_correct=("correct", "all"), any_flag=("flagged", "any"))
    precision, recall, f1 = item_level_scores(predictions, truth)
    field_avg = fields_df.accuracy.mean()

    review_rows = [
        [r["field"], pct(r["flagged_rate"]), pct(r["accuracy_not_flagged"]), pct(r["errors_caught_by_flagging"])]
        for r in field_rows
    ]

    errors = table[~table.correct].sort_values("confidence", ascending=False)
    error_lines = []
    for field in COMPARED_FIELDS:
        sub = errors[errors.field == field].head(5)
        if sub.empty:
            continue
        error_lines.append(f"\n**{field}** ({int((~table[table.field == field].correct).sum())} errors, first {len(sub)} shown)\n")
        error_lines.append(
            markdown_table(
                ["document", "predicted", "truth", "confidence", "flagged", "reasons"],
                [
                    [r.document, r.predicted[:60], r.truth[:60], f"{r.confidence:.2f}", "yes" if r.flagged else "no", r.reasons[:80]]
                    for r in sub.itertuples()
                ],
            )
        )

    report = f"""# Accuracy Report

Generated: {date.today().isoformat()}  |  Documents evaluated: **{docs}**  |  Review threshold: **{threshold}**

## 1. Field-level accuracy

{markdown_table(["Field", "Correct", "Accuracy"], summary_rows)}

- **Mean field accuracy:** {pct(field_avg)}
- **Documents with every field correct:** {int(per_doc.all_correct.sum())}/{docs} ({pct(per_doc.all_correct.mean())})
- **Line items, row level:** precision {pct(precision)}, recall {pct(recall)}, F1 {pct(f1)}

## 2. Human-review flagging

{markdown_table(["Field", "Flagged for review", "Accuracy of NOT-flagged values", "Share of errors that were flagged"], review_rows)}

- Documents with at least one flagged field: {int(per_doc.any_flag.sum())}/{docs} ({pct(per_doc.any_flag.mean())})
- A useful flagging system has **high accuracy on not-flagged values** and a **high share of errors caught**,
  while keeping the flagged rate low enough that review stays cheap.

## 3. Where the system fails
{"".join(chr(10) + line for line in error_lines) if error_lines else chr(10) + "No errors found."}

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
"""
    return report, fields_df


def run_notes(evaluated: int, failed: dict, not_processed: list, excluded: list, minimum: int = 30) -> str:
    """Honest, auto-generated notes about what the numbers in this report are based on."""
    lines = ["", "## 6. Run notes and known issues", ""]
    lines.append(f"- Documents evaluated: **{evaluated}** (the task asks for at least {minimum}).")
    if evaluated < minimum:
        lines.append(
            f"- **The {minimum}-document requirement was NOT met.** Accuracy numbers are based on only "
            f"{evaluated} documents and should be read as indicative, not statistically solid."
        )
    if failed:
        names = ", ".join(sorted(failed))
        sample = str(next(iter(failed.values())).get("error", ""))[:150].replace("\n", " ")
        lines.append(
            f"- {len(failed)} documents failed during extraction ({names}). They are **excluded** from the "
            "accuracy numbers, because the failure came from the API and not from the extraction quality. "
            f"First error message: `{sample}`"
        )
        quota = [n for n, p in failed.items() if any(k in str(p.get("error", "")) for k in ("429", "RESOURCE_EXHAUSTED"))]
        if quota:
            lines.append(
                f"- {len(quota)} of these failures are HTTP 429 `RESOURCE_EXHAUSTED`: the **free-tier quota of the "
                "Gemini API was used up**. A paid plan or a quota reset would allow the remaining documents to "
                "be processed (`run_extraction.py` is resumable)."
            )
    if not_processed:
        lines.append(
            f"- {len(not_processed)} documents have no prediction because the run was stopped "
            f"(first ones: {', '.join(sorted(not_processed)[:5])})."
        )
    if excluded:
        lines.append(
            f"- {len(excluded)} documents ({', '.join(excluded)}) were excluded because the dataset's own "
            "annotation is incomplete (missing invoice number or total)."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()

    settings = get_settings()
    predictions = load_json_dir(args.predictions or settings.predictions_dir)
    truth = json.loads((args.truth or settings.ground_truth).read_text(encoding="utf-8"))

    # Skip documents whose own ground-truth annotation is incomplete (missing invoice number or total).
    excluded = [n for n, r in truth.items() if not r["invoice_number"] or r["total"] is None]
    truth = {n: r for n, r in truth.items() if n not in excluded}
    if excluded:
        print(f"Excluded {len(excluded)} documents with incomplete ground truth: {', '.join(excluded)}")

    # Separate documents that failed during extraction (API errors) from real predictions.
    failed = {n: p for n, p in predictions.items() if p.get("error") and n in truth}
    predictions = {n: p for n, p in predictions.items() if not p.get("error")}
    not_processed = [n for n in truth if n not in predictions and n not in failed]
    if not predictions:
        raise SystemExit("No successful predictions to evaluate.")
    if not predictions:
        raise SystemExit("No predictions found. Run run_extraction.py first.")

    common = [name for name in predictions if name in truth]
    if len(common) < 30:
        print(f"WARNING: only {len(common)} documents have both prediction and ground truth (30 required).")

    table = build_table(predictions, truth)
    report, fields_df = build_report(table, predictions, truth, settings.review_threshold)
    report += run_notes(table["document"].nunique(), failed, not_processed, excluded)

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.report_dir / "accuracy_report.md").write_text(report, encoding="utf-8")
    fields_df.to_csv(settings.report_dir / "field_accuracy.csv", index=False)
    table.to_csv(settings.output_dir / "evaluation_details.csv", index=False)

    print(report.split("## 2.")[0])
    print(f"Full report: {settings.report_dir / 'accuracy_report.md'}")


if __name__ == "__main__":
    main()
