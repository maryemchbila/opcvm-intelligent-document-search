from __future__ import annotations

import io
import json
import os
import pickle
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types
from google.genai.errors import APIError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.extraction.chunker import ChunkingStats, chunk_pages
from app.extraction.pdf_text_extractor import extract_pdf_pages
from app.extraction.vision_text_extractor import (
    VisionExtractionError,
    VisionTextExtractor,
)
from scripts.build_chunks import main


def _synthetic_pdf(
    path: Path,
    texts: list[str | None],
    *,
    visual_pages: set[int] | None = None,
) -> Path:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    visual_pages = visual_pages or set()
    for page_number, text in enumerate(texts, start=1):
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            if page_number in visual_pages:
                stream = DecodedStreamObject()
                stream.set_data(b"0 0 0 rg 72 620 300 100 re f")
                page[NameObject("/Contents")] = writer._add_object(stream)
            continue
        resources = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)
    return path


def _vision_response(*texts: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=json.dumps(
            {
                "pages": [
                    {"page_number": index, "text": text}
                    for index, text in enumerate(texts, start=1)
                ]
            }
        )
    )


class PDFTextExtractorTests(unittest.TestCase):
    def test_extracts_text_page_by_page_and_flags_empty_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(
                Path(directory) / "sample.pdf",
                ["PAGE ONE WITH A COMPLETE LOCAL TEXT LAYER", None],
            )
            pages = extract_pdf_pages(pdf)

        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertIn("PAGE ONE", pages[0]["text"])
        self.assertFalse(pages[0]["needs_ocr_or_vision"])
        self.assertEqual(pages[0]["extraction_method"], "pymupdf")
        self.assertEqual(pages[0]["page_status"], "READABLE")
        self.assertFalse(pages[1]["needs_ocr_or_vision"])
        self.assertEqual(pages[1]["page_status"], "EMPTY_OR_BLANK")


class ChunkerTests(unittest.TestCase):
    def test_chunks_never_mix_pages(self) -> None:
        pages = [
            {
                "page_number": 1,
                "text": "PAGE_ONE " * 20,
                "extraction_method": "pymupdf",
                "page_status": "READABLE",
            },
            {
                "page_number": 2,
                "text": "PAGE_TWO " * 20,
                "extraction_method": "gemini_vision",
                "page_status": "READABLE",
            },
            {
                "page_number": 3,
                "text": "",
                "extraction_method": "pymupdf",
                "page_status": "EMPTY_OR_BLANK",
            },
            {
                "page_number": 4,
                "text": "SHOULD NEVER BE CHUNKED",
                "extraction_method": "gemini_vision_failed",
                "page_status": "FAILED",
            },
        ]
        chunks = chunk_pages(
            pages,
            source="source.pdf",
            source_path="Fichiers/category/opcvm/source.pdf",
            category="category",
            opcvm="opcvm",
            max_chars=45,
            min_chunk_words=1,
        )

        self.assertGreater(len(chunks), 2)
        self.assertEqual({chunk["page_number"] for chunk in chunks}, {1, 2})
        for chunk in chunks:
            if chunk["page_number"] == 1:
                self.assertIn("PAGE_ONE", chunk["text"])
                self.assertNotIn("PAGE_TWO", chunk["text"])
                self.assertEqual(chunk["extraction_method"], "pymupdf")
            else:
                self.assertIn("PAGE_TWO", chunk["text"])
                self.assertNotIn("PAGE_ONE", chunk["text"])
                self.assertEqual(chunk["extraction_method"], "gemini_vision")

    @staticmethod
    def _page(page_number: int, text: str) -> dict[str, object]:
        return {
            "page_number": page_number,
            "text": text,
            "extraction_method": "pymupdf",
            "page_status": "READABLE",
        }

    def _chunk(self, pages: list[dict[str, object]], **kwargs: object) -> list[dict]:
        return chunk_pages(
            pages,
            source="source.pdf",
            source_path="Fichiers/category/opcvm/source.pdf",
            category="category",
            opcvm="opcvm",
            **kwargs,
        )

    def test_micro_chunk_is_merged_with_previous_chunk(self) -> None:
        normal = " ".join(f"mot{i}" for i in range(20))
        stats = ChunkingStats()
        chunks = self._chunk(
            [self._page(1, f"{normal} d'investissement")],
            max_chars=len(normal),
            stats=stats,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], f"{normal} d'investissement")
        self.assertEqual(stats.micro_chunks_merged, 1)
        self.assertEqual(stats.micro_chunks_ignored, 0)

    def test_isolated_micro_chunk_is_ignored(self) -> None:
        stats = ChunkingStats()
        chunks = self._chunk(
            [self._page(1, "d'investissement")], stats=stats
        )

        self.assertEqual(chunks, [])
        self.assertEqual(stats.micro_chunks_merged, 0)
        self.assertEqual(stats.micro_chunks_ignored, 1)

    def test_micro_chunk_is_never_merged_across_pages(self) -> None:
        normal = " ".join(f"contenu{i}" for i in range(20))
        stats = ChunkingStats()
        chunks = self._chunk(
            [self._page(1, normal), self._page(2, "d'investissement")],
            stats=stats,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertNotIn("investissement", chunks[0]["text"])
        self.assertEqual(stats.micro_chunks_ignored, 1)

    def test_normal_chunk_is_preserved(self) -> None:
        normal = " ".join(f"information{i}" for i in range(20))
        chunks = self._chunk([self._page(1, normal)])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], normal)

    def test_short_final_sentence_is_attached_to_previous_chunk(self) -> None:
        normal = " ".join(f"analyse{i}" for i in range(22))
        ending = "Cette decision clot la seance."
        chunks = self._chunk(
            [self._page(1, f"{normal} {ending}")], max_chars=len(normal)
        )

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["text"].endswith(ending))

    def test_no_final_chunk_is_below_minimum_word_count(self) -> None:
        first = " ".join(f"alpha{i}" for i in range(20))
        second = " ".join(f"beta{i}" for i in range(20))
        chunks = self._chunk(
            [self._page(1, f"{first} {second} fin courte")],
            max_chars=len(first),
        )

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk["text"].split()) >= 20 for chunk in chunks))


class BuildChunksCLITests(unittest.TestCase):
    def test_text_pdf_never_calls_vision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root
                / "Fichiers"
                / "PV comité et Présentations"
                / "AMEN PREMIERE"
                / "PV comité SICAV 01-09-2025.pdf",
                [
                    "PREMIERE PAGE " + "aa " * 25,
                    "DEUXIEME PAGE " + "bb " * 25,
                    None,
                ],
            )
            vision = Mock()
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [str(pdf)], project_root=root, vision_extractor=vision
                )

            output_path = root / "indexes" / "dev_chunks.pkl"
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            with output_path.open("rb") as chunk_file:
                chunks = pickle.load(chunk_file)

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["category"], "PV comité et Présentations")
        self.assertEqual(chunks[0]["opcvm"], "AMEN PREMIERE")
        self.assertEqual(chunks[0]["source"], "PV comité SICAV 01-09-2025.pdf")
        self.assertEqual(chunks[0]["source_path"], str(pdf.resolve()))
        self.assertEqual(chunks[0]["extraction_method"], "pymupdf")
        self.assertIn("Chunks créés :", stdout.getvalue())
        self.assertIn("Pages texte PyMuPDF : 2", stdout.getvalue())
        self.assertIn("Pages vides détectées localement : 1", stdout.getvalue())
        self.assertIn("Pages envoyées à Gemini Vision : 0", stdout.getvalue())
        self.assertIn("Pages Gemini réussies : 0", stdout.getvalue())
        self.assertIn("Pages exploitables : 2", stdout.getvalue())
        self.assertIn("Pages vides : 1", stdout.getvalue())
        self.assertIn("Pages échouées : 0", stdout.getvalue())
        self.assertIn("Pages vides : [3]", stdout.getvalue())
        self.assertIn("Micro-chunks fusionnés : 0", stdout.getvalue())
        self.assertIn("Micro-chunks ignorés : 0", stdout.getvalue())
        vision.extract_pages.assert_not_called()

    def test_image_only_pdf_uses_one_mocked_vision_call_for_all_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "scan.pdf",
                [None, None],
                visual_pages={1, 2},
            )
            vision = Mock()
            vision.extract_pages.return_value = [
                {
                    "page_number": 1,
                    "text": "VISION PAGE ONE WITH USEFUL COMPLETE CONTENT " * 4,
                    "needs_ocr_or_vision": False,
                    "extraction_method": "gemini_vision",
                    "page_status": "READABLE",
                },
                {
                    "page_number": 2,
                    "text": "VISION PAGE TWO WITH USEFUL COMPLETE CONTENT " * 4,
                    "needs_ocr_or_vision": False,
                    "extraction_method": "gemini_vision",
                    "page_status": "READABLE",
                },
            ]
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [str(pdf)], project_root=root, vision_extractor=vision
                )
            with (root / "indexes" / "dev_chunks.pkl").open("rb") as chunk_file:
                chunks = pickle.load(chunk_file)

        self.assertEqual(exit_code, 0)
        vision.extract_pages.assert_called_once_with(pdf.resolve(), [1, 2], total_pages=2)
        self.assertEqual({chunk["page_number"] for chunk in chunks}, {1, 2})
        self.assertTrue(
            all(chunk["extraction_method"] == "gemini_vision" for chunk in chunks)
        )

    def test_mixed_pdf_sends_only_unreadable_page_to_vision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "mixed.pdf",
                ["LOCAL PAGE ONE " + "aa " * 25, None],
                visual_pages={2},
            )
            vision = Mock()
            vision.extract_pages.return_value = [
                {
                    "page_number": 2,
                    "text": "VISION PAGE TWO WITH USEFUL COMPLETE CONTENT " * 4,
                    "needs_ocr_or_vision": False,
                    "extraction_method": "gemini_vision",
                    "page_status": "READABLE",
                }
            ]
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [str(pdf)], project_root=root, vision_extractor=vision
                )
            with (root / "indexes" / "dev_chunks.pkl").open("rb") as chunk_file:
                chunks = pickle.load(chunk_file)

        self.assertEqual(exit_code, 0)
        vision.extract_pages.assert_called_once_with(pdf.resolve(), [2], total_pages=2)
        by_page = {chunk["page_number"]: chunk for chunk in chunks}
        self.assertEqual(by_page[1]["extraction_method"], "pymupdf")
        self.assertEqual(by_page[2]["extraction_method"], "gemini_vision")
        self.assertIn("LOCAL PAGE ONE", by_page[1]["text"])
        self.assertNotIn("VISION PAGE TWO", by_page[1]["text"])

    def test_missing_path_returns_nonzero_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "Fichiers" / "missing.pdf"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main([str(missing)], project_root=root)

        self.assertNotEqual(exit_code, 0)
        self.assertIn("fichier introuvable", stderr.getvalue().lower())

    def test_failed_vision_page_does_not_create_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "partial.pdf",
                [None, None],
                visual_pages={1, 2},
            )
            vision = Mock(last_mode="FULL_DOCUMENT", last_fallback="PAGE_BY_PAGE")
            vision.extract_pages.return_value = [
                {
                    "page_number": 1,
                    "text": "RECOVERED PAGE ONE WITH USEFUL COMPLETE CONTENT " * 4,
                    "needs_ocr_or_vision": False,
                    "extraction_method": "gemini_vision",
                    "page_status": "READABLE",
                },
                {
                    "page_number": 2,
                    "text": "",
                    "needs_ocr_or_vision": True,
                    "extraction_method": "gemini_vision_failed",
                    "page_status": "FAILED",
                    "status": "FAILED",
                    "error_type": "HTTP_504",
                    "error_message": "Deadline expired before operation could complete.",
                },
            ]
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [str(pdf)], project_root=root, vision_extractor=vision
                )
            with (root / "indexes" / "dev_chunks.pkl").open("rb") as chunk_file:
                chunks = pickle.load(chunk_file)

        self.assertEqual(exit_code, 0)
        self.assertEqual({chunk["page_number"] for chunk in chunks}, {1})
        self.assertIn("Vision fallback : PAGE_BY_PAGE", stdout.getvalue())
        self.assertIn("Pages Gemini échouées : 1", stdout.getvalue())
        self.assertIn("Pages exploitables : 1", stdout.getvalue())
        self.assertIn("Pages vides : 0", stdout.getvalue())
        self.assertIn("Pages échouées : 1", stdout.getvalue())
        self.assertIn(
            "- Page 2 : HTTP_504 - Deadline expired before operation could complete.",
            stdout.getvalue(),
        )

    def test_zero_chunks_preserves_existing_index_and_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "blank.pdf", [None]
            )
            output_path = root / "indexes" / "dev_chunks.pkl"
            output_path.parent.mkdir(parents=True)
            previous_content = pickle.dumps([{"chunk_id": "previous-valid"}])
            output_path.write_bytes(previous_content)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([str(pdf)], project_root=root)

            self.assertNotEqual(exit_code, 0)
            self.assertEqual(output_path.read_bytes(), previous_content)
            self.assertFalse((root / "indexes" / "dev_chunks.pkl.tmp").exists())

        self.assertIn("Aucun chunk exploitable généré.", stdout.getvalue())
        self.assertIn(
            "Index existant conservé : indexes/dev_chunks.pkl", stdout.getvalue()
        )

    def test_zero_chunks_without_previous_index_creates_no_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "blank.pdf", [None]
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main([str(pdf)], project_root=root)

            self.assertNotEqual(exit_code, 0)
            self.assertFalse((root / "indexes" / "dev_chunks.pkl").exists())
            self.assertFalse((root / "indexes" / "dev_chunks.pkl.tmp").exists())

        self.assertIn("Aucun index précédent disponible.", stdout.getvalue())

    def test_valid_chunks_are_saved_with_atomic_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = _synthetic_pdf(
                root / "Fichiers" / "Category" / "Fund" / "text.pdf",
                ["VALID PAGE " + "aa " * 25],
            )
            real_replace = os.replace
            temporary_seen: list[Path] = []

            def replace(source: str | Path, destination: str | Path) -> None:
                source_path = Path(source)
                self.assertTrue(source_path.is_file())
                self.assertEqual(source_path.name, "dev_chunks.pkl.tmp")
                temporary_seen.append(source_path)
                real_replace(source, destination)

            with patch("scripts.build_chunks.os.replace", side_effect=replace) as mocked:
                with redirect_stdout(io.StringIO()):
                    exit_code = main([str(pdf)], project_root=root)

            output_path = root / "indexes" / "dev_chunks.pkl"
            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()
            self.assertTrue(temporary_seen)
            self.assertTrue(output_path.is_file())
            self.assertFalse((root / "indexes" / "dev_chunks.pkl.tmp").exists())
            with output_path.open("rb") as chunk_file:
                self.assertGreater(len(pickle.load(chunk_file)), 0)


class VisionTextExtractorTests(unittest.TestCase):
    def test_full_scanned_pdf_uses_one_mocked_call_and_restores_page_numbers(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/vision",
            uri="https://files.invalid/vision",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "pages": [
                        {"page_number": 1, "text": "FIRST PAGE WITH USEFUL EXTRACTED TEXT"},
                        {"page_number": 2, "text": "SECOND PAGE WITH USEFUL EXTRACTED TEXT"},
                    ]
                }
            )
        )
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None, None])
            pages = extractor.extract_pages(pdf, [1, 2], total_pages=2)

        self.assertEqual(sdk.models.generate_content.call_count, 1)
        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertEqual(
            [page["page_status"] for page in pages], ["READABLE", "READABLE"]
        )
        self.assertTrue(
            all(page["extraction_method"] == "gemini_vision" for page in pages)
        )
        config = sdk.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNone(config.response_schema)
        self.assertIsInstance(config.response_json_schema, dict)
        self.assertIn("pages", config.response_json_schema["properties"])
        sdk.files.delete.assert_called_once_with(name="files/vision")

    def test_global_504_falls_back_page_by_page_and_cleans_temp_files(self) -> None:
        sdk = Mock()
        uploaded_paths: list[Path] = []

        def upload(*, file: object, config: object) -> SimpleNamespace:
            uploaded_paths.append(Path(file.name))
            index = len(uploaded_paths)
            return SimpleNamespace(
                name=f"files/vision-{index}",
                uri=f"https://files.invalid/vision-{index}",
                state=types.FileState.ACTIVE,
            )

        sdk.files.upload.side_effect = upload
        sdk.models.generate_content.side_effect = [
            APIError(
                code=504,
                response_json={"error": {"status": "DEADLINE_EXCEEDED"}},
            ),
            _vision_response("FIRST PAGE WITH USEFUL EXTRACTED TEXT"),
            _vision_response("SECOND PAGE WITH USEFUL EXTRACTED TEXT"),
        ]
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None, None])
            pages = extractor.extract_pages(pdf, [1, 2], total_pages=2)
            self.assertTrue(pdf.exists())
            self.assertTrue(all(not path.exists() for path in uploaded_paths[1:]))

        self.assertEqual(sdk.models.generate_content.call_count, 3)
        self.assertEqual(extractor.last_mode, "FULL_DOCUMENT")
        self.assertEqual(extractor.last_fallback, "PAGE_BY_PAGE")
        self.assertEqual([page["page_number"] for page in pages], [1, 2])
        self.assertEqual(
            [page["page_status"] for page in pages], ["READABLE", "READABLE"]
        )

    def test_page_503_retries_then_succeeds(self) -> None:
        sdk = Mock()
        sdk.files.upload.side_effect = [
            SimpleNamespace(
                name=f"files/vision-{index}",
                uri=f"https://files.invalid/vision-{index}",
                state=types.FileState.ACTIVE,
            )
            for index in range(1, 4)
        ]
        sdk.models.generate_content.side_effect = [
            APIError(
                code=504,
                response_json={"error": {"status": "DEADLINE_EXCEEDED"}},
            ),
            APIError(
                code=503,
                response_json={"error": {"status": "UNAVAILABLE"}},
            ),
            _vision_response("FIRST PAGE AFTER RETRY WITH USEFUL TEXT"),
            _vision_response("SECOND PAGE WITH USEFUL EXTRACTED TEXT"),
        ]
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=1,
            retry_delays=(0,),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None, None])
            with patch("app.extraction.vision_text_extractor.time.sleep") as sleep:
                pages = extractor.extract_pages(pdf, [1, 2], total_pages=2)

        self.assertEqual(sdk.models.generate_content.call_count, 4)
        self.assertEqual(pages[0]["page_status"], "READABLE")
        sleep.assert_called_once_with(0)

    def test_individual_page_failure_is_marked_and_other_pages_continue(self) -> None:
        sdk = Mock()
        sdk.files.upload.side_effect = [
            SimpleNamespace(
                name=f"files/vision-{index}",
                uri=f"https://files.invalid/vision-{index}",
                state=types.FileState.ACTIVE,
            )
            for index in range(1, 4)
        ]
        sdk.models.generate_content.side_effect = [
            APIError(
                code=504,
                response_json={"error": {"status": "DEADLINE_EXCEEDED"}},
            ),
            APIError(
                code=400,
                response_json={"error": {"status": "INVALID_ARGUMENT"}},
            ),
            _vision_response("SECOND PAGE WITH USEFUL EXTRACTED TEXT"),
        ]
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None, None])
            pages = extractor.extract_pages(pdf, [1, 2], total_pages=2)

        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["text"], "")
        self.assertTrue(pages[0]["needs_ocr_or_vision"])
        self.assertEqual(pages[0]["extraction_method"], "gemini_vision_failed")
        self.assertEqual(pages[0]["page_status"], "FAILED")
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertEqual(pages[1]["page_status"], "READABLE")

    def test_failed_pages_keep_sanitized_503_and_504_details(self) -> None:
        sdk = Mock()
        sdk.files.upload.side_effect = [
            SimpleNamespace(
                name=f"files/vision-{index}",
                uri=f"https://files.invalid/vision-{index}",
                state=types.FileState.ACTIVE,
            )
            for index in range(1, 4)
        ]
        sdk.models.generate_content.side_effect = [
            APIError(
                code=504,
                response_json={
                    "error": {
                        "status": "DEADLINE_EXCEEDED",
                        "message": "Global deadline exceeded.",
                    }
                },
            ),
            APIError(
                code=504,
                response_json={
                    "error": {
                        "status": "DEADLINE_EXCEEDED",
                        "message": "Deadline expired before operation could complete.",
                    }
                },
            ),
            APIError(
                code=503,
                response_json={
                    "error": {
                        "status": "UNAVAILABLE",
                        "message": "This model is currently experiencing high demand.",
                    }
                },
            ),
        ]
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None, None])
            pages = extractor.extract_pages(pdf, [1, 2], total_pages=2)

        self.assertEqual(pages[0]["status"], "FAILED")
        self.assertEqual(pages[0]["error_type"], "HTTP_504")
        self.assertEqual(
            pages[0]["error_message"],
            "Deadline expired before operation could complete.",
        )
        self.assertEqual(pages[1]["status"], "FAILED")
        self.assertEqual(pages[1]["error_type"], "HTTP_503")
        self.assertEqual(
            pages[1]["error_message"],
            "This model is currently experiencing high demand.",
        )
        self.assertNotIn("unit-test-key", str(pages))

    def test_empty_vision_result_on_blank_page_is_not_failed(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/vision",
            uri="https://files.invalid/vision",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = _vision_response("")
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "blank.pdf", [None])
            pages = extractor.extract_pages(pdf, [1], total_pages=1)

        self.assertEqual(pages[0]["page_status"], "EMPTY_OR_BLANK")
        self.assertFalse(pages[0]["needs_ocr_or_vision"])

    def test_invalid_json_response_raises_explicit_vision_error(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/vision",
            uri="https://files.invalid/vision",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = SimpleNamespace(text="not json")
        extractor = VisionTextExtractor(
            api_key="unit-test-key",
            client=sdk,
            max_retries=0,
            retry_delays=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = _synthetic_pdf(Path(directory) / "scan.pdf", [None])
            with self.assertRaisesRegex(
                VisionExtractionError, "JSON Gemini vision invalide"
            ):
                extractor.extract_pages(pdf, [1], total_pages=1)

        sdk.files.delete.assert_called_once_with(name="files/vision")


if __name__ == "__main__":
    unittest.main()
