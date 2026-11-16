"""Decoupage deterministe du texte PDF sans melanger les pages."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypedDict

from app.config import MIN_CHUNK_WORDS


DEFAULT_CHUNK_SIZE = 1200
MAX_MERGED_CHUNK_FACTOR = 1.2


@dataclass
class ChunkingStats:
    micro_chunks_merged: int = 0
    micro_chunks_ignored: int = 0


class TextChunk(TypedDict):
    chunk_id: str
    text: str
    source: str
    source_path: str
    page_number: int
    category: str
    opcvm: str
    extraction_method: str


def _split_page_text(text: str, max_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts: list[str] = []
    remaining = normalized
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _word_count(text: str) -> int:
    """Compte les mots Unicode en gardant les apostrophes internes."""
    return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text, flags=re.UNICODE))


def _filter_micro_chunks(
    parts: list[str],
    *,
    min_chunk_words: int,
    max_merged_chars: int,
    stats: ChunkingStats,
) -> list[str]:
    filtered: list[str] = []
    for part in parts:
        if _word_count(part) >= min_chunk_words:
            filtered.append(part)
            continue

        if filtered and len(filtered[-1]) + 1 + len(part) <= max_merged_chars:
            filtered[-1] = f"{filtered[-1]} {part}"
            stats.micro_chunks_merged += 1
        else:
            stats.micro_chunks_ignored += 1
    return filtered


def _document_identity(chunk: Mapping[str, object]) -> tuple[str, str] | None:
    source_path = chunk.get("source_path")
    if isinstance(source_path, str) and source_path.strip():
        return ("source_path", source_path)
    source = chunk.get("source")
    if isinstance(source, str) and source.strip():
        return ("source", source)
    return None


def normalize_existing_chunks(
    chunks: Iterable[Mapping[str, object]],
    *,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    stats: ChunkingStats | None = None,
) -> list[dict[str, object]]:
    """Fusionne ou retire les micro-chunks adjacents, sans changer de page."""
    if min_chunk_words <= 0:
        raise ValueError("min_chunk_words doit etre strictement positif.")
    if max_chars <= 0:
        raise ValueError("max_chars doit etre strictement positif.")

    chunking_stats = stats if stats is not None else ChunkingStats()
    max_merged_chars = max(max_chars, int(max_chars * MAX_MERGED_CHUNK_FACTOR))
    normalized_chunks: list[dict[str, object]] = []

    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise ValueError(f"Le chunk {position} est invalide.")
        normalized_chunk = dict(chunk)
        text = normalized_chunk.get("text")
        page_number = normalized_chunk.get("page_number")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Le chunk {position} ne contient pas de texte valide.")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(f"Le chunk {position} a une page invalide.")

        normalized_chunk["text"] = text.strip()
        if _word_count(text) >= min_chunk_words:
            normalized_chunks.append(normalized_chunk)
            continue

        current_document = _document_identity(normalized_chunk)
        if normalized_chunks and current_document is not None:
            previous = normalized_chunks[-1]
            previous_text = previous.get("text")
            same_page = previous.get("page_number") == page_number
            same_document = _document_identity(previous) == current_document
            if isinstance(previous_text, str) and same_page and same_document:
                merged_text = f"{previous_text.rstrip()} {text.strip()}"
                if len(merged_text) <= max_merged_chars:
                    previous["text"] = merged_text
                    chunking_stats.micro_chunks_merged += 1
                    continue

        chunking_stats.micro_chunks_ignored += 1

    return normalized_chunks


def chunk_pages(
    pages: Iterable[Mapping[str, object]],
    *,
    source: str,
    source_path: str,
    category: str,
    opcvm: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    min_chunk_words: int = MIN_CHUNK_WORDS,
    stats: ChunkingStats | None = None,
) -> list[TextChunk]:
    """Decoupe chaque page independamment et enrichit les chunks de metadonnees."""
    if max_chars <= 0:
        raise ValueError("max_chars doit etre strictement positif.")
    if min_chunk_words <= 0:
        raise ValueError("min_chunk_words doit etre strictement positif.")

    document_key = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    chunking_stats = stats if stats is not None else ChunkingStats()
    max_merged_chars = max(max_chars, int(max_chars * MAX_MERGED_CHUNK_FACTOR))
    chunks: list[TextChunk] = []
    for page in pages:
        page_number = page.get("page_number")
        text = page.get("text")
        extraction_method = page.get("extraction_method")
        page_status = page.get("page_status")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise ValueError("Chaque page doit avoir un page_number entier positif.")
        if not isinstance(text, str):
            raise ValueError("Chaque page doit avoir un champ text.")
        if extraction_method not in {
            "pymupdf",
            "gemini_vision",
            "gemini_vision_failed",
        }:
            raise ValueError("Chaque page doit avoir une extraction_method valide.")
        if page_status not in {"READABLE", "EMPTY_OR_BLANK", "FAILED"}:
            raise ValueError("Chaque page doit avoir un page_status valide.")
        if page_status != "READABLE":
            continue

        page_chunks = _filter_micro_chunks(
            _split_page_text(text, max_chars),
            min_chunk_words=min_chunk_words,
            max_merged_chars=max_merged_chars,
            stats=chunking_stats,
        )
        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": (
                        f"{document_key}-p{page_number:05d}-c{chunk_number:04d}"
                    ),
                    "text": chunk_text,
                    "source": source,
                    "source_path": source_path,
                    "page_number": page_number,
                    "category": category,
                    "opcvm": opcvm,
                    "extraction_method": extraction_method,
                }
            )
    return chunks
