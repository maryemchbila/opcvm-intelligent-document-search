from __future__ import annotations

import copy
import io
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.search.bm25_index import BM25SearchIndex, tokenize_french
from scripts.search import main


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


class BM25SearchIndexTests(unittest.TestCase):
    def test_french_tokenizer_is_deterministic_and_keeps_accents_and_numbers(self) -> None:
        self.assertEqual(
            tokenize_french("Gestion active des liquidités disponibles, 2025."),
            ["gestion", "active", "des", "liquidités", "disponibles", "2025"],
        )

    def test_exact_lexical_search_ranks_expected_chunk_first(self) -> None:
        chunks = [
            _chunk("A", "gestion active des liquidités disponibles"),
            _chunk("B", "participants du comité d'investissement"),
            _chunk("C", "rendement annuel du fonds"),
        ]
        results = BM25SearchIndex(chunks).search("gestion des liquidités")
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "A")
        self.assertGreater(results[0]["score"], 0)

    def test_proper_names_find_their_chunks(self) -> None:
        chunks = [
            _chunk("person", "Le participant Sami Hariga présente le dossier"),
            _chunk("fund", "Le fonds AMEN PREMIERE respecte ses limites"),
            _chunk("other", "Rendement annuel et valeur liquidative"),
        ]
        index = BM25SearchIndex(chunks)
        self.assertEqual(index.search("Sami Hariga")[0]["chunk_id"], "person")
        self.assertEqual(index.search("AMEN PREMIERE")[0]["chunk_id"], "fund")

    def test_absent_query_returns_no_zero_score_results(self) -> None:
        index = BM25SearchIndex(
            [_chunk("A", "alpha"), _chunk("B", "beta"), _chunk("C", "gamma")]
        )
        self.assertEqual(index.search("mottotalementabsent"), [])

    def test_top_k_limits_results(self) -> None:
        chunks = [
            _chunk("A", "alpha information"),
            _chunk("B", "beta information"),
            _chunk("C", "gamma information"),
            _chunk("D", "delta sujet"),
        ]
        results = BM25SearchIndex(chunks).search("alpha beta gamma", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([result["rank"] for result in results], [1, 2])

    def test_result_preserves_metadata_without_modifying_chunks(self) -> None:
        chunks = [
            _chunk("target", "allocation obligataire tunisienne", page=3),
            _chunk("other-1", "participant réunion"),
            _chunk("other-2", "performance fonds"),
        ]
        original = copy.deepcopy(chunks)
        result = BM25SearchIndex(chunks).search("allocation obligataire")[0]
        self.assertEqual(chunks, original)
        for field in (
            "chunk_id",
            "source",
            "page_number",
            "category",
            "opcvm",
            "extraction_method",
        ):
            self.assertEqual(result[field], chunks[0][field])

    def test_empty_corpus_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "corpus BM25 est vide"):
            BM25SearchIndex([])


class SearchCLITests(unittest.TestCase):
    def test_cli_loads_pickle_and_prints_results(self) -> None:
        chunks = [
            _chunk("A", "gestion active des liquidités disponibles", page=3),
            _chunk("B", "participants du comité d'investissement"),
            _chunk("C", "rendement annuel du fonds"),
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
                    ["gestion des liquidités", "--top-k", "3"], project_root=root
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Query : gestion des liquidités", stdout.getvalue())
        self.assertIn("Document : document.pdf", stdout.getvalue())
        self.assertIn("Page : 3", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
