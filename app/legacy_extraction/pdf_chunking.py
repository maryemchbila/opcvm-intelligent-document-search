"""Decoupage temporaire et fusion deterministe d'extractions PDF."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Hashable

from pypdf import PdfReader, PdfWriter

from .schemas import DocumentExtraction, Provenance


@dataclass(frozen=True, slots=True)
class PDFChunk:
    path: Path
    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


def split_pdf(pdf_path: Path, output_dir: Path, chunk_size: int) -> list[PDFChunk]:
    """Cree des PDF contigus dans ``output_dir`` avec des pages basees sur 1."""
    if chunk_size <= 0:
        raise ValueError("chunk_size doit etre strictement positif.")
    reader = PdfReader(str(pdf_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[PDFChunk] = []
    for zero_start in range(0, len(reader.pages), chunk_size):
        zero_end = min(zero_start + chunk_size, len(reader.pages))
        writer = PdfWriter()
        for page_index in range(zero_start, zero_end):
            writer.add_page(reader.pages[page_index])
        chunk_path = output_dir / f"chunk_{zero_start + 1:05d}_{zero_end:05d}.pdf"
        with chunk_path.open("wb") as chunk_file:
            writer.write(chunk_file)
        chunks.append(PDFChunk(chunk_path, zero_start + 1, zero_end))
    return chunks


def correct_chunk_page_numbers(payload: object, chunk: PDFChunk) -> object:
    """Convertit recursivement les numeros relatifs d'un bloc en pages originales."""
    if isinstance(payload, list):
        return [correct_chunk_page_numbers(item, chunk) for item in payload]
    if not isinstance(payload, dict):
        return payload
    corrected: dict[object, object] = {}
    for key, value in payload.items():
        if (
            key == "page_number"
            and isinstance(value, int)
            and not isinstance(value, bool)
            and 1 <= value <= chunk.page_count
        ):
            corrected[key] = value + chunk.start_page - 1
        else:
            corrected[key] = correct_chunk_page_numbers(value, chunk)
    return corrected


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _best(values: list[Provenance]) -> Provenance | None:
    if not values:
        return None
    return max(
        values,
        key=lambda value: (
            getattr(value.source_method, "value", value.source_method) == "explicit",
            value.confidence,
            -value.page_number,
        ),
    )


def _deduplicate(items: list[object], key: Callable[[object], Hashable]) -> list[object]:
    output: list[object] = []
    seen: set[Hashable] = set()
    for item in items:
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        output.append(item)
    return output


def _semantic_key(item: object, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalized(getattr(item, field, None)) for field in fields)


def merge_extractions(extractions: list[DocumentExtraction]) -> DocumentExtraction:
    """Fusionne localement des blocs valides, sans completer ni reconstruire de donnees."""
    if not extractions:
        return DocumentExtraction()

    merged: dict[str, object] = {}
    for field in ("document_type", "document_title", "document_date"):
        candidates = [value for item in extractions if (value := getattr(item, field)) is not None]
        selected = _best(candidates)
        merged[field] = selected.model_dump(mode="json") if selected is not None else None

    fund_fields: dict[str, object] = {}
    for field in (
        "manager", "depositary", "distributor", "constitution_date",
        "valuation_frequency", "fund_category", "capitalization_type", "valuation_date",
    ):
        candidates = [
            value
            for item in extractions
            if item.fund_information is not None
            and (value := getattr(item.fund_information, field)) is not None
        ]
        selected = _best(candidates)
        fund_fields[field] = selected.model_dump(mode="json") if selected is not None else None
    merged["fund_information"] = fund_fields if any(fund_fields.values()) else None

    meetings = [item.meeting for item in extractions if item.meeting is not None]
    meeting = _best(meetings)
    merged["meeting"] = meeting.model_dump(mode="json") if meeting is not None else None

    all_items = lambda field: [value for item in extractions for value in getattr(item, field)]
    rules: dict[str, Callable[[object], Hashable]] = {
        "opcvm": lambda item: _normalized(item.name),
        "participants": lambda item: _normalized(item.name),
        "decisions": lambda item: (
            _normalized(item.text), item.page_number, _normalized(item.source_text),
            item.confidence, _normalized(item.source_method.value),
        ),
        "resolutions": lambda item: _semantic_key(item, ("resolution_number", "text", "status")),
        "tasks": lambda item: _semantic_key(
            item, ("title", "description", "assigned_to", "due_date", "status", "priority")
        ),
        "mandates": lambda item: _semantic_key(
            item,
            ("person", "role", "mandate_type", "start_date", "end_date", "appointment_date", "renewal_date"),
        ),
        "financial_information": lambda item: (
            _normalized(item.metric_type.value), _normalized(item.label),
            _normalized(item.raw_value), item.page_number,
        ),
        "investment_policy": lambda item: _semantic_key(
            item, ("category", "raw_text", "percentage", "minimum", "maximum")
        ),
        "historical_performance": lambda item: _semantic_key(
            item, ("date", "raw_value", "numeric_value", "unit")
        ),
    }
    for field, key in rules.items():
        merged[field] = [
            item.model_dump(mode="json") for item in _deduplicate(all_items(field), key)
        ]
    return DocumentExtraction.model_validate(merged)
