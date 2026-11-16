"""Extraction locale du texte d'un PDF, page par page, sans OCR."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import pymupdf


MIN_TEXT_CHARACTERS = 20
MIN_VISUAL_INK_RATIO = 0.0025
MAX_INK_GRAYSCALE = 210
PageStatus = Literal["READABLE", "EMPTY_OR_BLANK", "FAILED"]


class ExtractedPage(TypedDict):
    page_number: int
    text: str
    needs_ocr_or_vision: bool
    extraction_method: Literal[
        "pymupdf", "gemini_vision", "gemini_vision_failed"
    ]
    page_status: PageStatus
    status: NotRequired[PageStatus]
    error_type: NotRequired[str]
    error_message: NotRequired[str]


def has_meaningful_visual_content(page: pymupdf.Page) -> bool:
    """Detecte une densite visuelle minimale sur un rendu basse resolution."""
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(0.5, 0.5),
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    samples = memoryview(pixmap.samples)
    if not samples:
        return False
    ink_pixels = sum(value < MAX_INK_GRAYSCALE for value in samples)
    return ink_pixels / len(samples) >= MIN_VISUAL_INK_RATIO


def page_has_meaningful_visual_content(
    pdf_value: str | Path, page_number: int
) -> bool:
    pdf_path = Path(pdf_value).expanduser().resolve()
    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError("Numero de page hors limites.")
        return has_meaningful_visual_content(document[page_number - 1])


def extract_pdf_pages(
    pdf_value: str | Path,
    *,
    min_text_characters: int = MIN_TEXT_CHARACTERS,
) -> list[ExtractedPage]:
    """Retourne le texte natif de chaque page et signale les pages sans texte."""
    pdf_path = Path(pdf_value).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Le fichier fourni doit avoir l'extension .pdf.")

    if min_text_characters < 1:
        raise ValueError("min_text_characters doit etre strictement positif.")

    pages: list[ExtractedPage] = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            useful_characters = len(re.sub(r"\s+", "", text))
            readable = useful_characters >= min_text_characters
            visually_nonempty = readable or has_meaningful_visual_content(page)
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "needs_ocr_or_vision": not readable and visually_nonempty,
                    "extraction_method": "pymupdf",
                    "page_status": (
                        "READABLE"
                        if readable
                        else "FAILED"
                        if visually_nonempty
                        else "EMPTY_OR_BLANK"
                    ),
                }
            )
    return pages
