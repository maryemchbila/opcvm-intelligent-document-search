from __future__ import annotations

import tempfile
import unittest
from contextlib import chdir
from pathlib import Path

from app.legacy_extraction.validators import validate_extraction, validate_pdf_path
from app.legacy_extraction.schemas import DocumentExtraction


class ValidatorTests(unittest.TestCase):
    def test_pdf_path_requires_existing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            self.assertEqual(validate_pdf_path(pdf), pdf.resolve())
            text_file = Path(directory) / "sample.txt"
            text_file.write_text("not a pdf", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_pdf_path(text_file)
            with self.assertRaises(FileNotFoundError):
                validate_pdf_path(Path(directory) / "missing.pdf")

    def test_pdf_header_is_checked_without_reading_document_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "invalid.pdf"
            pdf.write_bytes(b"not a pdf")
            with self.assertRaisesRegex(ValueError, "PDF valide"):
                validate_pdf_path(pdf)

    def test_unicode_pdf_paths_are_accepted(self) -> None:
        names = (
            "PV comité SICAV 01-09-2025.pdf",
            "PV AGO AP (signé).pdf.pdf",
            "Fiche été 2026.pdf",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                with self.subTest(name=name):
                    pdf = root / name
                    pdf.write_bytes(b"%PDF-1.7\n")
                    self.assertEqual(validate_pdf_path(pdf), pdf.resolve())

    def test_relative_pdf_path_can_point_to_any_explicit_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files_dir = root / "Fichiers" / "Sous dossier"
            files_dir.mkdir(parents=True)
            pdf = files_dir / "document.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            with chdir(root):
                resolved = validate_pdf_path(
                    Path("Fichiers") / "Sous dossier" / "document.pdf"
                )
            self.assertEqual(resolved, pdf.resolve())

    def test_extraction_validator_keeps_pydantic_result(self) -> None:
        extraction = DocumentExtraction.model_validate({})
        self.assertIs(validate_extraction(extraction), extraction)

    def test_extraction_validation_does_not_reclassify_gemini_values(self) -> None:
        extraction = DocumentExtraction.model_validate(
            {
                "financial_information": [
                    {
                        "metric_type": "ALLOCATION",
                        "label": "Liquidités",
                        "raw_value": "12.18%",
                        "percentage": 12.18,
                        "page_number": 1,
                        "source_text": "Liquidités 12.18%",
                        "confidence": 0.95,
                    }
                ]
            }
        )
        before = extraction.model_dump()
        validate_extraction(extraction)
        self.assertEqual(extraction.model_dump(), before)


if __name__ == "__main__":
    unittest.main()
