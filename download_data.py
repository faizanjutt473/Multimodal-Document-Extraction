"""Download 40 sample invoices (+ ground truth) from Hugging Face into ./data."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

from doc_extraction.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    settings = get_settings()
    dataset = load_dataset("katanaml-org/invoices-donut-data-v1", split=f"train[:{args.count}]")
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(dataset):
        row["image"].convert("RGB").save(settings.images_dir / f"inv_{index:03d}.png")
    settings.raw_ground_truth.write_text(
        json.dumps([row["ground_truth"] for row in dataset]), encoding="utf-8"
    )
    print(f"Saved {len(dataset)} images to {settings.images_dir}")


if __name__ == "__main__":
    main()
