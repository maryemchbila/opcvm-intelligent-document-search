from __future__ import annotations

import io
import os
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.extraction.chunker import ChunkingStats, normalize_existing_chunks
from scripts.normalize_chunks import main


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


def _normal_text(prefix: str = "mot") -> str:
    return " ".join(f"{prefix}{index}" for index in range(20))


def _write_index(root: Path, chunks: list[dict[str, object]]) -> Path:
    path = root / "indexes" / "dev_chunks.pkl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as chunks_file:
        pickle.dump(chunks, chunks_file)
    return path


class NormalizeExistingChunksTests(unittest.TestCase):
    def test_micro_chunk_is_merged_with_previous_chunk(self) -> None:
        normal = _normal_text()
        stats = ChunkingStats()
        normalized = normalize_existing_chunks(
            [_chunk("A", normal), _chunk("B", "d'investissement")],
            stats=stats,
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["text"], f"{normal} d'investissement")
        self.assertEqual(stats.micro_chunks_merged, 1)
        self.assertEqual(stats.micro_chunks_ignored, 0)

    def test_micro_chunk_is_not_merged_across_pages(self) -> None:
        normalized = normalize_existing_chunks(
            [_chunk("A", _normal_text(), page=1), _chunk("B", "fin courte", page=2)]
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["page_number"], 1)
        self.assertNotIn("fin courte", normalized[0]["text"])

    def test_short_chunk_without_neighbor_is_ignored(self) -> None:
        stats = ChunkingStats()
        normalized = normalize_existing_chunks(
            [_chunk("A", "d'investissement")], stats=stats
        )

        self.assertEqual(normalized, [])
        self.assertEqual(stats.micro_chunks_ignored, 1)

    def test_metadata_from_previous_chunk_is_preserved(self) -> None:
        previous = _chunk("A", _normal_text(), page=3)
        micro = _chunk("B", "phrase finale", page=3)
        normalized = normalize_existing_chunks([previous, micro])

        for field, value in previous.items():
            if field != "text":
                self.assertEqual(normalized[0][field], value)
        self.assertEqual(previous["text"], _normal_text())


class NormalizeChunksCLITests(unittest.TestCase):
    def test_save_is_atomic_and_backup_is_created_once(self) -> None:
        chunks = [_chunk("A", _normal_text()), _chunk("B", "fin courte")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_path = _write_index(root, chunks)
            original_bytes = chunks_path.read_bytes()
            real_replace = os.replace

            def replace(source: str | Path, destination: str | Path) -> None:
                source_path = Path(source)
                self.assertEqual(source_path.name, "dev_chunks.pkl.tmp")
                self.assertTrue(source_path.is_file())
                real_replace(source, destination)

            with patch("scripts.normalize_chunks.os.replace", side_effect=replace) as mocked:
                with redirect_stdout(io.StringIO()):
                    exit_code = main(project_root=root)

            backup = root / "indexes" / "dev_chunks.before_normalization.pkl"
            self.assertEqual(backup.read_bytes(), original_bytes)
            backup.write_bytes(b"existing-backup")
            with redirect_stdout(io.StringIO()):
                second_exit_code = main(project_root=root)

            self.assertEqual(second_exit_code, 0)
            self.assertEqual(backup.read_bytes(), b"existing-backup")
            self.assertFalse((root / "indexes" / "dev_chunks.pkl.tmp").exists())

        self.assertEqual(exit_code, 0)
        self.assertEqual(second_exit_code, 0)
        mocked.assert_called_once()

    def test_cli_makes_no_gemini_call_and_reports_counts(self) -> None:
        chunks = [
            _chunk("A", _normal_text("alpha")),
            _chunk("B", "d'investissement"),
            _chunk("C", _normal_text("beta"), page=2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_index(root, chunks)
            stdout = io.StringIO()
            with patch(
                "google.genai.Client",
                side_effect=AssertionError("Gemini ne doit pas etre appele"),
            ) as gemini_client:
                with redirect_stdout(stdout):
                    exit_code = main(project_root=root)

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        gemini_client.assert_not_called()
        self.assertIn("Chunks avant : 3", output)
        self.assertIn("Micro-chunks fusionnés : 1", output)
        self.assertIn("Micro-chunks ignorés : 0", output)
        self.assertIn("Chunks après : 2", output)


if __name__ == "__main__":
    unittest.main()
