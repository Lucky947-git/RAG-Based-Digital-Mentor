"""Environment and path configuration. Load dotenv before other app modules."""
from __future__ import annotations
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
def _path(value: str | Path) -> Path:
    result = Path(value).expanduser()
    return result if result.is_absolute() else ROOT / result

DATA_PATH = _path(os.getenv("DATA_PATH", "data"))
CHROMA_PATH = _path(os.getenv("CHROMA_PATH", os.getenv("VIVEKA_CHROMA_DIR", str(DATA_PATH / "chroma"))))
PDF_PATH = _path(os.getenv("PDF_PATH", os.getenv("VIVEKA_PDF_DIR", str(DATA_PATH / "pdfs"))))
SESSION_PATH = _path(os.getenv("SESSION_PATH", os.getenv("VIVEKA_SESSIONS_DIR", str(DATA_PATH / "sessions"))))
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if x.strip()]
