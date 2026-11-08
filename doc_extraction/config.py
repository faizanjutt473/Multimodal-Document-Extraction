"""Central configuration. Values come from environment variables / the .env file."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def _find_tesseract() -> str | None:
    explicit = os.getenv("TESSERACT_CMD")
    if explicit:
        return explicit
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    if _WINDOWS_TESSERACT.exists():
        return str(_WINDOWS_TESSERACT)
    return None


@dataclass(frozen=True)
class Settings:
    # Paths
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    raw_ground_truth: Path = PROJECT_ROOT / "data" / "ground_truth_raw.json"
    ground_truth: Path = PROJECT_ROOT / "data" / "ground_truth.json"
    output_dir: Path = PROJECT_ROOT / "outputs"
    report_dir: Path = PROJECT_ROOT / "reports"

    # LLM
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    max_output_tokens: int = 2000
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic").lower())
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    max_image_side: int = 1568
    llm_attempts: int = 2

    # OCR
    tesseract_cmd: str | None = field(default_factory=_find_tesseract)
    ocr_config: str = "--oem 3 --psm 6"
    max_pdf_pages: int = 3

    # Review / validation
    review_threshold: float = field(default_factory=lambda: float(os.getenv("REVIEW_THRESHOLD", "0.75")))
    date_dayfirst: bool = field(default_factory=lambda: _env_bool("DATE_DAYFIRST", False))
    sum_tolerance_abs: float = 0.05
    sum_tolerance_rel: float = 0.02

    @property
    def active_model(self) -> str:
        return self.gemini_model if self.provider == "gemini" else self.model
    
     
    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def reviewed_dir(self) -> Path:
        return self.output_dir / "reviewed"


def get_settings() -> Settings:
    return Settings()
