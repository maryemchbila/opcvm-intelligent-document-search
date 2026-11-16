"""Index lexical BM25 reconstruit en memoire depuis les chunks techniques."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import TypedDict

from rank_bm25 import BM25Okapi


class BM25SearchResult(TypedDict):
    rank: int
    score: float
    chunk_id: object
    text: object
    source: object
    page_number: object
    category: object
    opcvm: object
    extraction_method: object


def tokenize_french(text: str) -> list[str]:
    """Tokenise sans stemming ni suppression agressive de mots courants."""
    if not isinstance(text, str):
        raise TypeError("Le texte a tokeniser doit etre une chaine.")
    normalized = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


class BM25SearchIndex:
    def __init__(self, chunks: Iterable[Mapping[str, object]]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Le corpus BM25 est vide.")

        tokenized_corpus: list[list[str]] = []
        for index, chunk in enumerate(self.chunks):
            text = chunk.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Le chunk {index} ne contient pas de texte valide.")
            tokenized_corpus.append(tokenize_french(text))
        if not any(tokenized_corpus):
            raise ValueError("Le corpus BM25 ne contient aucun texte exploitable.")

        self._index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[BM25SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k doit etre strictement positif.")
        query_tokens = tokenize_french(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked = sorted(
            (
                (float(score), index)
                for index, score in enumerate(scores)
                if float(score) > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]

        results: list[BM25SearchResult] = []
        for rank, (score, index) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text"),
                    "source": chunk.get("source"),
                    "page_number": chunk.get("page_number"),
                    "category": chunk.get("category"),
                    "opcvm": chunk.get("opcvm"),
                    "extraction_method": chunk.get("extraction_method"),
                }
            )
        return results
