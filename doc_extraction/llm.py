"""Vision-LLM extractor (Anthropic). Sends the page image(s) plus the OCR text as a hint."""
from __future__ import annotations

import json,base64,time
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .config import Settings
from .documents import encode_for_llm
from .schema import FIELD_NAMES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise data-extraction engine for scanned invoices and receipts.
You read the document image and return ONLY a JSON object. Never add explanations or markdown.

Extraction rules:
- Copy values exactly as printed. Never invent or guess a value; use null if it is not visible.
- invoice_number: the invoice / receipt / bill number (as a string).
- invoice_date: the date the invoice was issued, exactly as printed.
- vendor: the NAME of the seller / company that issued the invoice (name only, no address,
  no tax id). The seller is NOT the client / bill-to / customer.
- line_items: every purchased row. For each row:
    description  - item description
    quantity     - number of units
    unit_price   - price per single unit, before tax
    amount       - the line total as printed in the last amount column (gross, including tax, if tax is shown)
- total: the grand total payable (gross, including tax).
- Write all numbers as plain numbers with '.' as the decimal separator and no currency symbols
  or thousands separators (example: "$ 1 209,15" becomes 1209.15).

Confidence rules: give every field a confidence between 0 and 1.
  0.95+ = clearly printed and unambiguous; 0.7-0.9 = readable but slightly unclear;
  below 0.6 = blurry, partly hidden, or you are unsure. Be honest: low confidence is useful.
  For line_items give ONE confidence for the whole list.

Return exactly this JSON shape:
{
  "invoice_number": {"value": "...", "confidence": 0.0},
  "invoice_date":   {"value": "...", "confidence": 0.0},
  "vendor":         {"value": "...", "confidence": 0.0},
  "line_items":     {"value": [{"description": "...", "quantity": 0, "unit_price": 0, "amount": 0}], "confidence": 0.0},
  "total":          {"value": 0, "confidence": 0.0}
}"""


@dataclass
class LLMOutput:
    values: dict[str, Any] = field(default_factory=dict)
    confidences: dict[str, float] = field(default_factory=dict)
    raw_text: str = ""


class ExtractionError(RuntimeError):
    """Raised when the model response cannot be turned into the expected JSON."""


def parse_llm_json(text: str) -> LLMOutput:
    """Turn the model's text reply into values + confidences. Tolerates code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ExtractionError("No JSON object found in model reply")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractionError("Model JSON is not an object")

    output = LLMOutput(raw_text=text)
    for name in FIELD_NAMES:
        entry = payload.get(name)
        if isinstance(entry, dict) and "value" in entry:
            value, confidence = entry.get("value"), entry.get("confidence")
        else:  # model returned a bare value
            value, confidence = entry, None
        output.values[name] = value
        try:
            output.confidences[name] = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            output.confidences[name] = 0.5
    return output


class LLMExtractor:
    """Calls the vision model. Replace this class to use another provider."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Create a .env file (see .env.example)."
            )
        import anthropic

        self._settings = settings
        self._client = anthropic.Anthropic(api_key=settings.api_key, max_retries=3)

    def extract(self, images: list[Image.Image], ocr_text: str | None) -> LLMOutput:
        content: list[dict[str, Any]] = []
        for image in images:
            data, media_type = encode_for_llm(image, self._settings.max_image_side)
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
            )

        instruction = "Extract the invoice data from the document image(s) as specified."
        if ocr_text and ocr_text.strip():
            instruction += (
                "\n\nOCR text of the same document (may contain recognition errors; "
                "the image is the source of truth):\n<ocr>\n" + ocr_text.strip()[:6000] + "\n</ocr>"
            )
        content.append({"type": "text", "text": instruction})

        last_error: Exception | None = None
        for attempt in range(1, self._settings.llm_attempts + 1):
            response = self._client.messages.create(
                model=self._settings.model,
                max_tokens=self._settings.max_output_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            reply = "".join(block.text for block in response.content if block.type == "text")
            try:
                return parse_llm_json(reply)
            except ExtractionError as exc:
                last_error = exc
                logger.warning("Attempt %d: %s", attempt, exc)
        raise ExtractionError(str(last_error))
    
class GeminiExtractor:
    """Google Gemini vision extractor."""

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai

        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def extract(self, images: list[Image.Image], ocr_text: str | None) -> LLMOutput:
        from google.genai import types

        parts: list[Any] = []
        for image in images:
            data, media_type = encode_for_llm(image, self._settings.max_image_side)
            parts.append(types.Part.from_bytes(data=base64.b64decode(data), mime_type=media_type))

        instruction = "Extract the invoice data from the document image(s) as specified."
        if ocr_text and ocr_text.strip():
            instruction += (
                "\n\nOCR text of the same document (may contain recognition errors; "
                "the image is the source of truth):\n<ocr>\n" + ocr_text.strip()[:6000] + "\n</ocr>"
            )
        parts.append(instruction)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=4096,
        )

        last_error: Exception | None = None
        for attempt in range(1, 5):
            try:
                response = self._client.models.generate_content(
                    model=self._settings.gemini_model, contents=parts, config=config
                )
                return parse_llm_json(response.text or "")
            except ExtractionError as exc:
                last_error = exc
            except Exception as exc:  # rate limit (429) on the free tier, temporary errors
                last_error = exc
                logger.warning("Attempt %d failed: %s. Waiting 20s...", attempt, exc)
                time.sleep(20)
        raise ExtractionError(str(last_error))


def create_extractor(settings: Settings):
    if settings.provider == "gemini":
        return GeminiExtractor(settings)
    return LLMExtractor(settings)