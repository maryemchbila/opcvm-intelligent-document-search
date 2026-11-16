from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pypdf import PdfReader, PdfWriter

from app.legacy_extraction.extractor import ExtractionError, PDFExtractor
from app.legacy_extraction.gemini_client import GeminiClientError, GeminiGeneration
from app.legacy_extraction.pdf_chunking import (
    PDFChunk,
    correct_chunk_page_numbers,
    merge_extractions,
    split_pdf,
)
from app.legacy_extraction.schemas import DocumentExtraction


def _pdf(path: Path, page_count: int) -> Path:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    with path.open("wb") as output:
        writer.write(output)
    return path


def _generation(payload: dict, tokens: int | None = None) -> GeminiGeneration:
    return GeminiGeneration(
        text=json.dumps(payload),
        input_tokens=tokens,
        output_tokens=tokens,
        total_tokens=tokens,
    )


class PDFChunkingTests(unittest.TestCase):
    def test_split_pdf_creates_expected_page_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = split_pdf(_pdf(root / "source.pdf", 17), root / "chunks", 5)
            self.assertEqual(
                [(chunk.start_page, chunk.end_page) for chunk in chunks],
                [(1, 5), (6, 10), (11, 15), (16, 17)],
            )
            self.assertEqual(
                [len(PdfReader(str(chunk.path)).pages) for chunk in chunks],
                [5, 5, 5, 2],
            )

    def test_relative_page_numbers_are_corrected_recursively(self) -> None:
        chunk = PDFChunk(Path("chunk.pdf"), 6, 10)
        payload = {
            "participants": [{"page_number": 2}],
            "fund_information": {"manager": {"page_number": 5}},
            "already_absolute": {"page_number": 8},
        }
        corrected = correct_chunk_page_numbers(payload, chunk)
        self.assertEqual(corrected["participants"][0]["page_number"], 7)
        self.assertEqual(corrected["fund_information"]["manager"]["page_number"], 10)
        self.assertEqual(corrected["already_absolute"]["page_number"], 8)

    def test_merge_deduplicates_with_configured_business_keys(self) -> None:
        participant = {
            "name": "Élodie Martin", "page_number": 2,
            "source_text": "Élodie Martin", "confidence": 0.9,
        }
        metric = {
            "metric_type": "NAV", "label": "VL", "raw_value": "10,5",
            "page_number": 3, "source_text": "VL 10,5", "confidence": 0.9,
        }
        first = DocumentExtraction.model_validate(
            {
                "document_title": {
                    "value": "Titre", "page_number": 4,
                    "source_text": "Titre", "confidence": 0.7,
                },
                "participants": [participant],
                "financial_information": [metric],
            }
        )
        second = DocumentExtraction.model_validate(
            {
                "document_title": {
                    "value": "Titre fiable", "page_number": 1,
                    "source_text": "Titre fiable", "confidence": 0.95,
                },
                "participants": [{**participant, "name": " elodie   martin "}],
                "financial_information": [metric],
            }
        )
        merged = merge_extractions([first, second])
        self.assertEqual(merged.document_title.value, "Titre fiable")
        self.assertEqual(len(merged.participants), 1)
        self.assertEqual(len(merged.financial_information), 1)


class PDFExtractorFallbackTests(unittest.TestCase):
    def _extractor(self, root: Path, client: Mock, chunk_size: int = 5) -> PDFExtractor:
        return PDFExtractor(
            client=client,
            raw_dir=root / "raw",
            structured_dir=root / "structured",
            report_dir=root / "reports",
            pdf_chunk_size=chunk_size,
        )

    def test_504_triggers_fallback_and_chunked_report(self) -> None:
        client = Mock(model="gemini-test")
        client.generate_pdf.side_effect = [
            GeminiClientError("HTTP 504 DEADLINE_EXCEEDED\nmessage: deadline"),
            _generation(
                {"opcvm": [{"name": "Fonds A", "page_number": 1,
                              "source_text": "Fonds A", "confidence": 0.9}]},
                10,
            ),
            _generation(
                {"financial_information": [{"metric_type": "NAV", "raw_value": "42",
                                              "page_number": 2, "source_text": "VL 42",
                                              "confidence": 0.9}]},
                20,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._extractor(root, client).extract(_pdf(root / "source.pdf", 7))
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(client.generate_pdf.call_count, 3)
        self.assertEqual(result.extraction.financial_information[0].page_number, 7)
        self.assertEqual(report["processing_mode"], "CHUNKED_FALLBACK")
        self.assertEqual(report["chunk_size"], 5)
        self.assertEqual(report["chunk_count"], 2)
        self.assertIn("HTTP 504 DEADLINE_EXCEEDED", report["initial_error"])
        self.assertEqual(report["total_tokens"], 30)

    def test_successful_single_call_does_not_split_pdf(self) -> None:
        client = Mock(model="gemini-test")
        client.generate_pdf.return_value = _generation({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._extractor(root, client).extract(_pdf(root / "source.pdf", 8))
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(client.generate_pdf.call_count, 1)
        self.assertEqual(report["processing_mode"], "SINGLE_CALL")
        self.assertNotIn("chunk_count", report)

    def test_temporary_chunk_files_are_removed_after_success(self) -> None:
        client = Mock(model="gemini-test")
        chunk_paths: list[Path] = []

        def generate(path: Path, prompt: str, schema: object) -> GeminiGeneration:
            if not chunk_paths:
                chunk_paths.append(Path("initial-call-marker"))
                raise GeminiClientError("HTTP 504 DEADLINE_EXCEEDED")
            chunk_paths.append(Path(path))
            return _generation({})

        client.generate_pdf.side_effect = generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._extractor(root, client).extract(_pdf(root / "source.pdf", 6))
        self.assertTrue(chunk_paths[1:])
        self.assertTrue(all(not path.exists() for path in chunk_paths[1:]))

    def test_one_failed_chunk_fails_whole_extraction_and_cleans_temp_files(self) -> None:
        client = Mock(model="gemini-test")
        chunk_paths: list[Path] = []
        calls = 0

        def generate(path: Path, prompt: str, schema: object) -> GeminiGeneration:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise GeminiClientError("HTTP 504 DEADLINE_EXCEEDED")
            chunk_paths.append(Path(path))
            if calls == 3:
                raise GeminiClientError("HTTP 500 INTERNAL")
            return _generation({})

        client.generate_pdf.side_effect = generate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = self._extractor(root, client)
            with self.assertRaisesRegex(ExtractionError, "HTTP 500 INTERNAL"):
                extractor.extract(_pdf(root / "source.pdf", 6))
            report = json.loads(
                next((root / "reports").glob("*.report.json")).read_text(encoding="utf-8")
            )
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["processing_mode"], "CHUNKED_FALLBACK")
        self.assertEqual(report["chunk_count"], 2)
        self.assertTrue(all(not path.exists() for path in chunk_paths))


if __name__ == "__main__":
    unittest.main()
