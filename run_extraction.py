"""Run the extraction pipeline over a folder of invoices and save one JSON per document."""
import argparse
import json
import logging
import time
from pathlib import Path

from doc_extraction.config import get_settings
from doc_extraction.documents import list_documents
from doc_extraction.pipeline import DocumentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Folder with invoices (default: data/images)")
    parser.add_argument("--output", type=Path, help="Where to save JSON (default: outputs/predictions)")
    parser.add_argument("--limit", type=int, help="Only process the first N documents")
    parser.add_argument("--overwrite", action="store_true", help="Re-process files that already have output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    input_dir = args.input or settings.images_dir
    output_dir = args.output or settings.predictions_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = list_documents(input_dir)
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        raise SystemExit(f"No documents found in {input_dir}")

    pipeline = DocumentPipeline(settings)
    processed = flagged = failed = 0
    started = time.time()

    for number, path in enumerate(documents, start=1):
        target = output_dir / f"{path.stem}.json"
        if target.exists() and not args.overwrite:
            previous = json.loads(target.read_text(encoding="utf-8"))
            if not previous.get("error"):
                print(f"[{number}/{len(documents)}] {path.name}: skipped (already done)")
                continue

        result = pipeline.process(path)
        target.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        processed += 1
        failed += bool(result.error)
        flagged += result.needs_review
        status = "FAILED" if result.error else ("REVIEW " + ",".join(result.fields_to_review) if result.needs_review else "ok")
        print(f"[{number}/{len(documents)}] {path.name}: {status}")

    print(
        f"\nDone in {time.time() - started:.0f}s. Processed {processed}, "
        f"flagged for review {flagged}, failed {failed}. Output: {output_dir}"
    )


if __name__ == "__main__":
    main()