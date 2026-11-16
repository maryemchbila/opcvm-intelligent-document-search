from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.legacy_extraction.extractor import ExtractionError, PDFExtractor
from app.legacy_extraction.gemini_client import GeminiUnavailableError


class ExtractorTests(unittest.TestCase):
    def _write_pdf(self, root: Path) -> Path:
        pdf = root / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        return pdf

    def test_raw_structured_and_report_outputs_are_saved(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/temp", uri="https://files.invalid/temp", mime_type="application/pdf"
        )
        sdk.models.generate_content.return_value = SimpleNamespace(
            text="{}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=23, candidates_token_count=10, total_token_count=33
            ),
        )
        client = Mock()
        client.model = "gemini-2.5-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text="{}", input_tokens=23, output_tokens=10, total_tokens=33,
            retry_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            ).extract(self._write_pdf(root))

            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(result.raw_path.read_text(encoding="utf-8"), "{}")
            self.assertEqual(json.loads(result.structured_path.read_text(encoding="utf-8"))["opcvm"], [])
            self.assertEqual(report["status"], "SUCCESS")
            self.assertEqual(report["validation_status"], "SUCCESS")
            self.assertEqual(report["validation_warnings"], [])
            self.assertEqual(report["input_tokens"], 23)
            self.assertEqual(report["total_tokens"], 33)
            self.assertEqual(report["retry_count"], 2)

    def test_invalid_structured_json_is_reported_and_raw_is_preserved(self) -> None:
        client = Mock()
        client.model = "gemini-2.5-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text="not json", input_tokens=None, output_tokens=None, total_tokens=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            )
            with self.assertRaises(ExtractionError):
                extractor.extract(self._write_pdf(root))
            raw_path = next((root / "raw").glob("*.json"))
            report = json.loads(next((root / "reports").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(raw_path.read_text(encoding="utf-8"), "not json")
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["error_type"], "PYDANTIC_VALIDATION_ERROR")
            self.assertTrue(report["validation_errors"])
            self.assertNotIn("input_tokens", report)

    def test_pydantic_error_location_type_message_and_input_are_reported(self) -> None:
        payload = {
            "financial_information": [
                {
                    "metric_type": "INVALID_METRIC",
                    "raw_value": "12.18%",
                    "page_number": 1,
                    "source_text": "Liquidités 12.18%",
                    "confidence": 0.95,
                }
            ]
        }
        raw_text = json.dumps(payload, ensure_ascii=False)
        client = Mock()
        client.model = "gemini-3.8-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text=raw_text, input_tokens=None, output_tokens=None, total_tokens=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            )
            with self.assertRaises(ExtractionError) as caught:
                extractor.extract(self._write_pdf(root))
            raw_path = next((root / "raw").glob("*.json"))
            saved_raw_text = raw_path.read_text(encoding="utf-8")
            report = json.loads(
                next((root / "reports").glob("*.report.json")).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(saved_raw_text, raw_text)
        self.assertIn("financial_information.0.metric_type", str(caught.exception))
        self.assertIn("INVALID_METRIC", str(caught.exception))
        self.assertEqual(report["error_type"], "PYDANTIC_VALIDATION_ERROR")
        detail = report["validation_errors"][0]
        self.assertEqual(detail["location"], ["financial_information", 0, "metric_type"])
        self.assertEqual(detail["type"], "enum")
        self.assertIn("Input should be", detail["message"])
        self.assertEqual(detail["input"], "INVALID_METRIC")

    def test_repeated_run_does_not_overwrite_prior_outputs(self) -> None:
        client = Mock()
        client.model = "gemini-2.5-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text="{}", input_tokens=None, output_tokens=None, total_tokens=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            )
            first = extractor.extract(self._write_pdf(root))
            first_content = first.structured_path.read_text(encoding="utf-8")
            second = extractor.extract(root / "sample.pdf")
            self.assertIn("run2", second.structured_path.name)
            self.assertEqual(first.structured_path.read_text(encoding="utf-8"), first_content)

    def test_missing_api_key_is_reported_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = PDFExtractor(
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ExtractionError, "GEMINI_API_KEY est absente"):
                    extractor.extract(self._write_pdf(root))
            report = json.loads(next((root / "reports").glob("*.report.json")).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["error"], "GEMINI_API_KEY est absente.")

    def test_exhausted_503_is_reported_as_gemini_unavailable(self) -> None:
        client = Mock()
        client.model = "gemini-test"
        client.generate_pdf.side_effect = GeminiUnavailableError(
            "HTTP 503 UNAVAILABLE\nmessage: High demand", retry_count=3
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            )
            with self.assertRaisesRegex(ExtractionError, "HTTP 503 UNAVAILABLE"):
                extractor.extract(self._write_pdf(root))
            report = json.loads(
                next((root / "reports").glob("*.report.json")).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error_type"], "GEMINI_UNAVAILABLE")
        self.assertEqual(report["retry_count"], 3)
        self.assertEqual(report["processing_mode"], "SINGLE_CALL")

    def test_business_warning_sets_review_required_without_failing(self) -> None:
        payload = {
            "financial_information": [
                {
                    "metric_type": "BOND_ALLOCATION",
                    "label": "Obligation",
                    "raw_value": "58.30%",
                    "percentage": 58.30,
                    "page_number": 1,
                    "source_text": "Obligation 58.30%",
                    "confidence": 0.95,
                }
            ]
        }
        client = Mock()
        client.model = "gemini-3.8-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text=json.dumps(payload),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            ).extract(self._write_pdf(root))
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["validation_status"], "REVIEW_REQUIRED")
        self.assertTrue(report["validation_warnings"])
        self.assertIsNone(report["error"])

    def test_unicode_document_name_is_preserved_in_report(self) -> None:
        client = Mock()
        client.model = "gemini-3.8-flash"
        client.generate_pdf.return_value = SimpleNamespace(
            text="{}", input_tokens=None, output_tokens=None, total_tokens=None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "Fiche été 2026.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            result = PDFExtractor(
                client=client,
                raw_dir=root / "raw",
                structured_dir=root / "structured",
                report_dir=root / "reports",
            ).extract(pdf)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["document"], "Fiche été 2026.pdf")
        self.assertTrue(result.report_path.name.startswith("Fiche_ete_2026."))


if __name__ == "__main__":
    unittest.main()
