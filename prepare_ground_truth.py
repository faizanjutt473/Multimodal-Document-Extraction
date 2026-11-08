"""Convert data/ground_truth_raw.json into data/ground_truth.json (our schema)."""
import json

from doc_extraction.config import get_settings
from doc_extraction.ground_truth import build_ground_truth


def main() -> None:
    settings = get_settings()
    truth = build_ground_truth(settings.raw_ground_truth, settings.ground_truth)
    print(f"Wrote {len(truth)} records to {settings.ground_truth}")

    name, record = next(iter(truth.items()))
    print(f"\nSanity check - first record ({name}):")
    print(json.dumps(record, indent=2)[:900])

    empty = [n for n, r in truth.items() if not r["invoice_number"] or r["total"] is None]
    if empty:
        print(f"\nWARNING: {len(empty)} records have a missing invoice number or total: {empty[:5]}")


if __name__ == "__main__":
    main()
