"""Configuration locale du projet Gemini."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_DELAYS = tuple(
    float(value.strip())
    for value in os.getenv("GEMINI_RETRY_DELAYS", "2,5,10").split(",")
    if value.strip()
)
if GEMINI_MAX_RETRIES < 0:
    raise ValueError("GEMINI_MAX_RETRIES doit etre positif ou nul.")
if len(GEMINI_RETRY_DELAYS) < GEMINI_MAX_RETRIES or any(
    delay < 0 for delay in GEMINI_RETRY_DELAYS
):
    raise ValueError(
        "GEMINI_RETRY_DELAYS doit fournir un delai positif ou nul par retry."
    )
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "5"))
if PDF_CHUNK_SIZE <= 0:
    raise ValueError("PDF_CHUNK_SIZE doit etre strictement positif.")

MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "20"))
if MIN_CHUNK_WORDS <= 0:
    raise ValueError("MIN_CHUNK_WORDS doit etre strictement positif.")

# Modele multilingue compact (384 dimensions), adapte a une execution sur CPU.
EMBEDDING_MODEL = (
    os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RAW_DIR = PROJECT_ROOT / "output" / "raw"
STRUCTURED_DIR = PROJECT_ROOT / "output" / "structured"
REPORT_DIR = PROJECT_ROOT / "output" / "reports"
