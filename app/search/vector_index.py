"""Index semantique FAISS reconstruit en memoire depuis les chunks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, TypedDict

import numpy as np

from app.config import EMBEDDING_MODEL


class VectorSearchResult(TypedDict):
    rank: int
    score: float
    chunk_id: object
    text: object
    source: object
    page_number: object
    category: object
    opcvm: object
    extraction_method: object


class _EmbeddingModel(Protocol):
    def encode(self, sentences: object, **kwargs: object) -> object: ...


class _VectorIndex(Protocol):
    def add(self, vectors: np.ndarray) -> None: ...

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[object, object]: ...


def _load_model(model_name: str) -> _EmbeddingModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers est requis pour la recherche semantique."
        ) from exc
    return SentenceTransformer(model_name)


def _create_faiss_index(dimension: int) -> _VectorIndex:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu est requis pour la recherche semantique.") from exc
    return faiss.IndexFlatIP(dimension)


def _encode_for_retrieval(
    model: _EmbeddingModel,
    values: list[str],
    *,
    query: bool,
) -> np.ndarray:
    """Encode avec l'API retrieval, avec repli pour les anciens modeles."""
    method_name = "encode_query" if query else "encode_document"
    encoder: Callable[..., Any] = getattr(model, method_name, model.encode)
    encoded = encoder(
        values,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = np.asarray(encoded, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] != len(values) or vectors.shape[1] == 0:
        raise ValueError("Le modele a retourne des embeddings de forme invalide.")
    if not np.isfinite(vectors).all():
        raise ValueError("Le modele a retourne des embeddings non finis.")
    return np.ascontiguousarray(vectors, dtype=np.float32)


class VectorSearchIndex:
    """Recherche cosine via produit scalaire sur des embeddings normalises."""

    def __init__(
        self,
        chunks: Iterable[Mapping[str, object]],
        model_name: str = EMBEDDING_MODEL,
        *,
        model: _EmbeddingModel | None = None,
        index_factory: Callable[[int], _VectorIndex] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Le corpus vectoriel est vide.")

        texts: list[str] = []
        for position, chunk in enumerate(self.chunks):
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Le chunk {position} ne contient pas de texte exploitable."
                )
            texts.append(text)

        self.model_name = model_name
        self._model = model if model is not None else _load_model(model_name)
        embeddings = _encode_for_retrieval(self._model, texts, query=False)
        factory = index_factory if index_factory is not None else _create_faiss_index
        self._index = factory(int(embeddings.shape[1]))
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k doit etre strictement positif.")
        if not isinstance(query, str):
            raise TypeError("La requete doit etre une chaine.")
        if not query.strip():
            return []

        query_embedding = _encode_for_retrieval(self._model, [query], query=True)
        limit = min(top_k, len(self.chunks))
        raw_scores, raw_indexes = self._index.search(query_embedding, limit)
        scores = np.asarray(raw_scores)
        indexes = np.asarray(raw_indexes)

        results: list[VectorSearchResult] = []
        for chunk_index, score in zip(indexes[0], scores[0]):
            position = int(chunk_index)
            if position < 0 or position >= len(self.chunks):
                continue
            chunk = self.chunks[position]
            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
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
