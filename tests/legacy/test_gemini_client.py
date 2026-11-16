from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from app.legacy_extraction.gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiConfigurationError,
    GeminiUnavailableError,
    safe_remote_filename,
    sanitize_schema_for_gemini,
    validate_required_properties,
)
from app.legacy_extraction.schemas import DocumentExtraction


class SmokeSchema(BaseModel):
    value: str


class GeminiClientTests(unittest.TestCase):
    def test_unicode_names_have_safe_remote_filenames(self) -> None:
        cases = {
            "PV comité SICAV 01-09-2025.pdf": "PV_comite_SICAV_01-09-2025.pdf",
            "PV AGO AP (signé).pdf.pdf": "PV_AGO_AP_signe.pdf.pdf",
            "Fiche été 2026.pdf": "Fiche_ete_2026.pdf",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                remote_name = safe_remote_filename(Path(original))
                self.assertEqual(remote_name, expected)
                remote_name.encode("ascii")

    def test_full_document_schema_has_only_defined_required_properties(self) -> None:
        schema = sanitize_schema_for_gemini(DocumentExtraction.model_json_schema())
        validate_required_properties(schema)

        task_reference = schema["properties"]["tasks"]["items"]["$ref"]
        task_schema = schema
        for segment in task_reference.removeprefix("#/").split("/"):
            task_schema = task_schema[segment]

        self.assertIn("title", task_schema["properties"])
        self.assertLessEqual(
            set(task_schema.get("required", [])), set(task_schema["properties"])
        )

    def test_incoherent_required_property_is_rejected_locally(self) -> None:
        schema = {
            "type": "object",
            "properties": {"defined": {"type": "string"}},
            "required": ["missing"],
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            sanitize_schema_for_gemini(schema)

    def test_missing_api_key_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(GeminiConfigurationError):
                GeminiClient()

    def test_pdf_is_uploaded_as_pdf_and_json_schema_is_used(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/temp",
            uri="https://files.invalid/temp",
            mime_type="application/pdf",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = SimpleNamespace(
            text='{"value":"ok"}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=12, candidates_token_count=3, total_token_count=15
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            client = GeminiClient(api_key="unit-test-key", client=sdk)
            response = client.generate_pdf(pdf, 'Retourne {"value":"ok"}.', SmokeSchema)

        self.assertEqual(response.text, '{"value":"ok"}')
        self.assertEqual((response.input_tokens, response.output_tokens, response.total_tokens), (12, 3, 15))
        sdk.files.upload.assert_called_once()
        upload_config = sdk.files.upload.call_args.kwargs["config"]
        self.assertEqual(upload_config.mime_type, "application/pdf")
        self.assertEqual(upload_config.display_name, "sample.pdf")
        uploaded_stream = sdk.files.upload.call_args.kwargs["file"]
        self.assertTrue(uploaded_stream.closed)
        self.assertEqual(Path(uploaded_stream.name), pdf)
        sdk.files.delete.assert_called_once_with(name="files/temp")
        config = sdk.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNone(config.response_schema)
        self.assertEqual(config.response_json_schema["type"], "object")
        self.assertEqual(
            config.thinking_config.thinking_level,
            types.ThinkingLevel.LOW,
        )

    def test_processing_file_is_polled_until_active(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/temp",
            uri="https://files.invalid/temp",
            state=types.FileState.PROCESSING,
        )
        sdk.files.get.return_value = SimpleNamespace(
            name="files/temp",
            uri="https://files.invalid/temp",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = SimpleNamespace(text='{"value":"ok"}')
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            client = GeminiClient(
                api_key="unit-test-key", client=sdk, file_poll_seconds=0
            )
            client.generate_pdf(pdf, "Extrais le PDF.", SmokeSchema)

        sdk.files.get.assert_called_once_with(name="files/temp")

    def test_generate_pdf_validates_extension_before_upload(self) -> None:
        sdk = Mock()
        with tempfile.TemporaryDirectory() as directory:
            text_file = Path(directory) / "sample.txt"
            text_file.write_text("not a pdf", encoding="utf-8")
            client = GeminiClient(api_key="unit-test-key", client=sdk)
            with self.assertRaisesRegex(ValueError, "extension .pdf"):
                client.generate_pdf(text_file, "Extrais le PDF.", SmokeSchema)
        sdk.files.upload.assert_not_called()

    def test_unicode_local_path_is_uploaded_with_ascii_display_name(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/temp",
            uri="https://files.invalid/temp",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.return_value = SimpleNamespace(text='{"value":"ok"}')
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "PV comité (signé).pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            client = GeminiClient(api_key="unit-test-key", client=sdk)
            client.generate_pdf(pdf, "Extrais le PDF.", SmokeSchema)

        uploaded_stream = sdk.files.upload.call_args.kwargs["file"]
        upload_config = sdk.files.upload.call_args.kwargs["config"]
        self.assertEqual(Path(uploaded_stream.name), pdf)
        self.assertEqual(upload_config.display_name, "PV_comite_signe.pdf")
        upload_config.display_name.encode("ascii")

    def test_429_is_not_retried(self) -> None:
        sdk = Mock()
        sdk.models.get.side_effect = APIError(
            code=429, response_json={"error": {"status": "RESOURCE_EXHAUSTED"}}
        )
        client = GeminiClient(
            api_key="unit-test-key", client=sdk, max_retries=3
        )
        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            with self.assertRaises(GeminiClientError):
                client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 1)
        sleep.assert_not_called()

    def test_503_then_success_retries_after_first_configured_delay(self) -> None:
        sdk = Mock()
        sdk.models.get.side_effect = [
            APIError(code=503, response_json={"error": {"status": "UNAVAILABLE"}}),
            object(),
        ]
        client = GeminiClient(api_key="unit-test-key", client=sdk)
        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_pdf_generation_exposes_retry_count_after_503(self) -> None:
        sdk = Mock()
        sdk.files.upload.return_value = SimpleNamespace(
            name="files/temp",
            uri="https://files.invalid/temp",
            state=types.FileState.ACTIVE,
        )
        sdk.models.generate_content.side_effect = [
            APIError(code=503, response_json={"error": {"status": "UNAVAILABLE"}}),
            SimpleNamespace(text='{"value":"ok"}'),
        ]
        client = GeminiClient(api_key="unit-test-key", client=sdk)
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7\n")
            with patch("app.legacy_extraction.gemini_client.time.sleep"):
                generation = client.generate_pdf(pdf, "Extrais.", SmokeSchema)

        self.assertEqual(sdk.models.generate_content.call_count, 2)
        self.assertEqual(generation.retry_count, 1)

    def test_two_503_then_success_use_delays_in_order(self) -> None:
        sdk = Mock()
        unavailable = lambda: APIError(
            code=503, response_json={"error": {"status": "UNAVAILABLE"}}
        )
        sdk.models.get.side_effect = [unavailable(), unavailable(), object()]
        client = GeminiClient(api_key="unit-test-key", client=sdk)
        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.0, 5.0])

    def test_503_exhaustion_reports_three_retries(self) -> None:
        sdk = Mock()
        sdk.models.get.side_effect = APIError(
            code=503,
            response_json={
                "error": {"status": "UNAVAILABLE", "message": "High demand"}
            },
        )
        client = GeminiClient(api_key="unit-test-key", client=sdk)
        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            with self.assertRaises(GeminiUnavailableError) as caught:
                client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 4)
        self.assertEqual(caught.exception.retry_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list], [2.0, 5.0, 10.0]
        )

    def test_504_is_not_retried(self) -> None:
        sdk = Mock()
        sdk.models.get.side_effect = APIError(
            code=504,
            response_json={"error": {"status": "DEADLINE_EXCEEDED"}},
        )
        client = GeminiClient(api_key="unit-test-key", client=sdk)
        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            with self.assertRaises(GeminiClientError):
                client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 1)
        sleep.assert_not_called()

    def test_api_error_preserves_provider_message_without_key(self) -> None:
        key = "test-secret-key"
        sdk = Mock()
        sdk.models.get.side_effect = APIError(
            code=400,
            response_json={
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": f"Invalid request; key={key}",
                    "details": [
                        {
                            "fieldViolations": [
                                {"field": "responseJsonSchema", "description": "Unsupported const"}
                            ]
                        }
                    ],
                }
            },
        )
        client = GeminiClient(api_key=key, client=sdk, max_retries=3)

        with patch("app.legacy_extraction.gemini_client.time.sleep") as sleep:
            with self.assertRaises(GeminiClientError) as caught:
                client.check_connection()

        message = str(caught.exception)
        self.assertIn("HTTP 400 INVALID_ARGUMENT", message)
        self.assertIn("Invalid request", message)
        self.assertIn("responseJsonSchema: Unsupported const", message)
        self.assertNotIn(key, message)
        self.assertEqual(sdk.models.get.call_count, 1)
        sleep.assert_not_called()

    def test_timeout_is_not_retried(self) -> None:
        sdk = Mock()
        sdk.models.get.side_effect = TimeoutError("private document")
        client = GeminiClient(
            api_key="unit-test-key", client=sdk, max_retries=3
        )
        with patch("app.legacy_extraction.gemini_client.time.sleep"):
            with self.assertRaisesRegex(GeminiClientError, "timeout"):
                client.check_connection()
        self.assertEqual(sdk.models.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
