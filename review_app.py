"""Human-review UI for flagged documents.  Run with:  streamlit run review_app.py"""
import json

import streamlit as st

from doc_extraction.config import get_settings
from doc_extraction.documents import load_pages
from doc_extraction.schema import FIELD_NAMES

settings = get_settings()
settings.reviewed_dir.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Invoice review", layout="wide")
st.title("Human review of low-confidence fields")

records = {}
for path in sorted(settings.predictions_dir.glob("*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("needs_review"):
        records[record["source_file"]] = record

if not records:
    st.success("Nothing to review. Run run_extraction.py first, or all documents passed.")
    st.stop()

done = {p.stem for p in settings.reviewed_dir.glob("*.json")}
name = st.sidebar.selectbox(
    f"Documents to review ({len(records)})",
    list(records),
    format_func=lambda n: f"{'[done] ' if n.rsplit('.', 1)[0] in done else ''}{n}",
)
record = records[name]
left, right = st.columns(2)

with left:
    image_path = settings.images_dir / name
    if image_path.exists():
        st.image(load_pages(image_path)[0], use_container_width=True)

with right:
    st.subheader(name)
    data = record["data"]
    edited = {}
    for field in FIELD_NAMES:
        confidence = record["field_confidence"].get(field, 0.0)
        flagged = field in record["fields_to_review"]
        label = f"{'⚠️ ' if flagged else ''}{field}  (confidence {confidence:.2f})"
        value = json.dumps(data[field], indent=2, ensure_ascii=False) if field == "line_items" else str(data[field] or "")
        edited[field] = st.text_area(label, value, height=200 if field == "line_items" else 70, key=f"{name}-{field}")
        for reason in record["review_reasons"].get(field, []):
            st.caption(f"↳ {reason}")

    if st.button("Save corrected values", type="primary"):
        try:
            corrected = {
                "source_file": name,
                "data": {
                    **{k: edited[k] or None for k in ("invoice_number", "invoice_date", "vendor")},
                    "total": float(edited["total"]) if edited["total"] else None,
                    "line_items": json.loads(edited["line_items"] or "[]"),
                },
                "reviewed": True,
            }
            (settings.reviewed_dir / f"{name.rsplit('.', 1)[0]}.json").write_text(
                json.dumps(corrected, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            st.success("Saved to outputs/reviewed/")
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Invalid input: {exc}")
