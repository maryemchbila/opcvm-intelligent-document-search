from __future__ import annotations

import copy
import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from app.search.vector_index import VectorSearchIndex
from scripts.semantic_search import main


def _chunk(identifier: str, text: str, *, page: int = 1) -> dict[str, object]:
    return {
        "chunk_id": identifier,
        "text": text,
        "source": "document.pdf",
        "source_path": "Fichiers/Categorie/Fonds/document.pdf",
        "page_number": page,
        "category": "Categorie",
        "opcvm": "AMEN PREMIERE",
        "extraction_method": "gemini_vision",
    }


class FakeRetrievalModel:
    def __init__(self) -> None:
        self.document_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []

    def _vectors(self, values: list[str]) -> np.ndarray:
        vectors = []
        for value in values:
            lowered = value.casefold()
            if "reconduit" in lowered or "mandat" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "rendement" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float64)

    def encode_document(self, values: list[str], **kwargs: object) -> np.ndarray:
        self.document_calls.append(kwargs)
        return self._vectors(values)

    def encode_query(self, values: list[str], **kwargs: object) -> np.ndarray:
        self.query_calls.append(kwargs)
        return self._vectors(values)

    def encode(self, values: list[str], **kwargs: object) -> np.ndarray:
        raise AssertionError("L'API retrieval doit etre utilisee lorsqu'elle existe.")


class FakeFlatIPIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.copy()

    def search(self, vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        similarities = vectors @ self.vectors.T
        indexes = np.argsort(-similarities, axis=1)[:, :top_k]
        scores = np.take_along_axis(similarities, indexes, axis=1)
        return scores.astype(np.float32), indexes.astype(np.int64)


def _build_index(chunks: list[dict[str, object]]) -> VectorSearchIndex:
    return VectorSearchIndex(
        chunks,
        model=FakeRetrievalModel(),
        index_factory=FakeFlatIPIndex,
    )


class VectorSearchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _chunk(
                "A",
                "M. X a ete reconduit dans ses fonctions pour un nouveau mandat.",
                page=3,
            ),
            _chunk("B", "Le fonds affiche un rendement annuel de 6,2 %."),
            _chunk("C", "Les liquidites disponibles doivent etre optimisees."),
        ]

    def test_semantic_query_ranks_renewed_mandate_first(self) -> None:
        results = _build_index(self.chunks).search(
            "Quels mandats ont ete renouveles ?"
        )
        self.assertEqual(results[0]["chunk_id"], "A")
        self.assertEqual(results[0]["rank"], 1)
        self.assertAlmostEqual(results[0]["score"], 1.0)

    def test_retrieval_api_normalizes_documents_and_query(self) -> None:
        model = FakeRetrievalModel()
        index = VectorSearchIndex(
            self.chunks, model=model, index_factory=FakeFlatIPIndex
        )
        index.search("Quels mandats ont ete renouveles ?")
        for call in (model.document_calls[0], model.query_calls[0]):
            self.assertIs(call["normalize_embeddings"], True)
            self.assertIs(call["show_progress_bar"], False)
            self.assertIs(call["convert_to_numpy"], True)

    def test_top_k_is_limited_to_corpus_size(self) -> None:
        index = _build_index(self.chunks)
        self.assertEqual(len(index.search("rendement", top_k=2)), 2)
        self.assertEqual(len(index.search("rendement", top_k=20)), 3)

    def test_result_preserves_metadata_and_chunk_content(self) -> None:
        original = copy.deepcopy(self.chunks)
        result = _build_index(self.chunks).search("mandat", top_k=1)[0]
        self.assertEqual(self.chunks, original)
        self.assertEqual(result["text"], self.chunks[0]["text"])
        for field in (
            "chunk_id",
            "source",
            "page_number",
            "category",
            "opcvm",
            "extraction_method",
        ):
            self.assertEqual(result[field], self.chunks[0][field])

    def test_empty_corpus_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "corpus vectoriel est vide"):
            VectorSearchIndex(
                [], model=FakeRetrievalModel(), index_factory=FakeFlatIPIndex
            )

    def test_empty_query_is_controlled(self) -> None:
        self.assertEqual(_build_index(self.chunks).search("   "), [])


class SemanticSearchCLITests(unittest.TestCase):
    def test_cli_loads_pickle_and_prints_results(self) -> None:
        chunks = [
            _chunk("A", "M. X a ete reconduit pour un nouveau mandat.", page=3),
            _chunk("B", "Le fonds affiche un rendement annuel."),
            _chunk("C", "Les liquidites doivent etre optimisees."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_path = root / "indexes" / "dev_chunks.pkl"
            chunks_path.parent.mkdir(parents=True)
            with chunks_path.open("wb") as chunks_file:
                pickle.dump(chunks, chunks_file)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    ["Quels mandats ont ete renouveles ?", "--top-k", "1"],
                    project_root=root,
                    index_factory=_build_index,
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Query : Quels mandats ont ete renouveles ?", output)
        self.assertIn("Resultats : 1", output)
        self.assertIn("Document : document.pdf", output)
        self.assertIn("Page : 3", output)


if __name__ == "__main__":
    unittest.main()
