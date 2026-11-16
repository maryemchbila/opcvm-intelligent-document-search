"""Orchestration PDF -> Gemini -> validation Pydantic -> sorties locales."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from ..config import GEMINI_MODEL, PDF_CHUNK_SIZE, RAW_DIR, REPORT_DIR, STRUCTURED_DIR
from .gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiGeneration,
    GeminiUnavailableError,
    safe_remote_filename,
)
from .pdf_chunking import correct_chunk_page_numbers, merge_extractions, split_pdf
from .prompts import PDF_EXTRACTION_PROMPT, pdf_chunk_prompt
from .schemas import DocumentExtraction
from .validators import (
    BusinessValidationResult,
    validate_allocation_business_rules,
    validate_extraction,
    validate_pdf_path,
)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    extraction: DocumentExtraction
    raw_path: Path
    structured_path: Path
    report_path: Path
    generation: GeminiGeneration
    duration_seconds: float
    business_validation: BusinessValidationResult


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ChunkFailure(Exception):
    error: Exception
    chunk_count: int
    completed_chunk_retry_count: int


def _safe_stem(path: Path) -> str:
    remote_name = safe_remote_filename(path)
    return Path(remote_name).stem


class PDFExtractor:
    def __init__(
        self,
        client: GeminiClient | None = None,
        raw_dir: Path = RAW_DIR,
        structured_dir: Path = STRUCTURED_DIR,
        report_dir: Path = REPORT_DIR,
        pdf_chunk_size: int = PDF_CHUNK_SIZE,
    ) -> None:
        if pdf_chunk_size <= 0:
            raise ValueError("pdf_chunk_size doit etre strictement positif.")
        self.client = client
        self.raw_dir = Path(raw_dir)
        self.structured_dir = Path(structured_dir)
        self.report_dir = Path(report_dir)
        self.pdf_chunk_size = pdf_chunk_size

    def _output_paths(self, pdf_path: Path) -> tuple[Path, Path, Path]:
        stem = _safe_stem(pdf_path)
        run = 1
        while any(
            path.exists()
            for path in (
                self.raw_dir / f"{stem}.run{run}.json",
                self.structured_dir / f"{stem}.run{run}.json",
                self.report_dir / f"{stem}.run{run}.report.json",
            )
        ):
            run += 1
        return (
            self.raw_dir / f"{stem}.run{run}.json",
            self.structured_dir / f"{stem}.run{run}.json",
            self.report_dir / f"{stem}.run{run}.report.json",
        )

    def extract(self, pdf_value: str | Path) -> ExtractionResult:
        pdf_path = validate_pdf_path(pdf_value)
        client = self.client
        raw_path, structured_path, report_path = self._output_paths(pdf_path)
        for directory in (self.raw_dir, self.structured_dir, self.report_dir):
            directory.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        generation: GeminiGeneration | None = None
        extraction: DocumentExtraction | None = None
        business_validation: BusinessValidationResult | None = None
        failure: Exception | None = None
        processing_mode = "SINGLE_CALL"
        initial_error: str | None = None
        chunk_count = 0
        retry_count = 0
        try:
            if client is None:
                client = GeminiClient()
            try:
                generation = client.generate_pdf(pdf_path, PDF_EXTRACTION_PROMPT, DocumentExtraction)
                retry_count += getattr(generation, "retry_count", 0)
                # Conserve la reponse monobloc meme si sa validation echoue ensuite.
                raw_path.write_text(generation.text, encoding="utf-8")
                extraction = DocumentExtraction.model_validate_json(generation.text)
            except GeminiClientError as exc:
                if not _is_deadline_exceeded(exc):
                    raise
                processing_mode = "CHUNKED_FALLBACK"
                initial_error = str(exc)
                retry_count += exc.retry_count
                try:
                    generation, extraction, chunk_count = self._extract_in_chunks(client, pdf_path)
                    retry_count += generation.retry_count
                except _ChunkFailure as chunk_failure:
                    chunk_count = chunk_failure.chunk_count
                    retry_count += chunk_failure.completed_chunk_retry_count
                    raise chunk_failure.error
            if processing_mode == "CHUNKED_FALLBACK":
                raw_path.write_text(generation.text, encoding="utf-8")
            extraction = validate_extraction(DocumentExtraction.model_validate(extraction))
            business_validation = validate_allocation_business_rules(extraction)
            structured_path.write_text(
                extraction.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            failure = exc
            if isinstance(exc, GeminiClientError):
                retry_count += exc.retry_count
        duration = round(time.perf_counter() - started, 3)
        report: dict[str, object] = {
            "document": pdf_path.name,
            "model": client.model if client is not None else GEMINI_MODEL,
            "started_at": started_at,
            "duration_seconds": duration,
            "processing_mode": processing_mode,
            "retry_count": retry_count,
            "status": (
                "FAILED"
                if failure
                else business_validation.validation_status
                if business_validation is not None
                else "SUCCESS"
            ),
            "error": _safe_failure(failure) if failure else None,
            "validation_status": (
                business_validation.validation_status
                if business_validation is not None
                else None
            ),
            "validation_warnings": (
                [warning.as_dict() for warning in business_validation.validation_warnings]
                if business_validation is not None
                else []
            ),
        }
        if processing_mode == "CHUNKED_FALLBACK":
            report.update(
                {
                    "chunk_size": self.pdf_chunk_size,
                    "chunk_count": chunk_count,
                    "initial_error": initial_error,
                }
            )
        if isinstance(failure, ValidationError):
            report["error_type"] = "PYDANTIC_VALIDATION_ERROR"
            report["validation_errors"] = _pydantic_validation_errors(failure)
        elif isinstance(failure, GeminiUnavailableError):
            report["error_type"] = "GEMINI_UNAVAILABLE"
        if generation is not None:
            if generation.input_tokens is not None:
                report["input_tokens"] = generation.input_tokens
            if generation.output_tokens is not None:
                report["output_tokens"] = generation.output_tokens
            if generation.total_tokens is not None:
                report["total_tokens"] = generation.total_tokens
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if failure is not None:
            if isinstance(failure, GeminiClientError):
                raise ExtractionError(str(failure)) from None
            if isinstance(failure, ValidationError):
                raise ExtractionError(_format_pydantic_validation_error(failure)) from None
            if isinstance(failure, (OSError, ValueError)):
                raise ExtractionError(str(failure)) from None
            raise ExtractionError(f"Echec inattendu ({type(failure).__name__}).") from None
        if generation is None or extraction is None or business_validation is None:
            raise ExtractionError("Gemini n'a pas produit de resultat exploitable.")
        return ExtractionResult(
            extraction=extraction,
            raw_path=raw_path,
            structured_path=structured_path,
            report_path=report_path,
            generation=generation,
            duration_seconds=duration,
            business_validation=business_validation,
        )

    def _extract_in_chunks(
        self, client: GeminiClient, pdf_path: Path
    ) -> tuple[GeminiGeneration, DocumentExtraction, int]:
        generations: list[GeminiGeneration] = []
        extractions: list[DocumentExtraction] = []
        with tempfile.TemporaryDirectory(prefix="pdf_chunks_") as directory:
            chunks = split_pdf(pdf_path, Path(directory), self.pdf_chunk_size)
            try:
                for chunk in chunks:
                    chunk_generation = client.generate_pdf(
                        chunk.path,
                        pdf_chunk_prompt(chunk.start_page, chunk.end_page),
                        DocumentExtraction,
                    )
                    relative = DocumentExtraction.model_validate_json(chunk_generation.text)
                    corrected_payload = correct_chunk_page_numbers(
                        relative.model_dump(mode="json"), chunk
                    )
                    extractions.append(DocumentExtraction.model_validate(corrected_payload))
                    generations.append(chunk_generation)
            except Exception as exc:
                raise _ChunkFailure(
                    exc,
                    len(chunks),
                    sum(
                        getattr(generation, "retry_count", 0)
                        for generation in generations
                    ),
                ) from exc

        merged = merge_extractions(extractions)
        text = merged.model_dump_json(indent=2)
        return (
            GeminiGeneration(
                text=text,
                input_tokens=_sum_token_counts(generations, "input_tokens"),
                output_tokens=_sum_token_counts(generations, "output_tokens"),
                total_tokens=_sum_token_counts(generations, "total_tokens"),
                retry_count=sum(
                    getattr(generation, "retry_count", 0)
                    for generation in generations
                ),
            ),
            merged,
            len(chunks),
        )


def _safe_failure(error: Exception) -> str:
    if isinstance(error, GeminiClientError):
        return str(error)
    if isinstance(error, ValidationError):
        return f"Validation Pydantic invalide ({error.error_count()} erreur(s))."
    return f"{type(error).__name__}: extraction interrompue."


def _is_deadline_exceeded(error: GeminiClientError) -> bool:
    message = str(error).upper()
    return "HTTP 504" in message or "DEADLINE_EXCEEDED" in message


def _sum_token_counts(
    generations: list[GeminiGeneration], attribute: str
) -> int | None:
    values = [getattr(generation, attribute) for generation in generations]
    return sum(value for value in values if value is not None) if any(
        value is not None for value in values
    ) else None


def _json_safe_input(value: object) -> object:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _pydantic_validation_errors(error: ValidationError) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for item in error.errors():
        details.append(
            {
                "location": list(item.get("loc", ())),
                "type": str(item.get("type", "unknown")),
                "message": str(item.get("msg", "Validation invalide.")),
                "input": _json_safe_input(item.get("input")),
            }
        )
    return details


def _format_pydantic_validation_error(error: ValidationError) -> str:
    lines = ["Validation Pydantic invalide :"]
    for detail in _pydantic_validation_errors(error):
        location = ".".join(str(part) for part in detail["location"]) or "$"
        lines.extend(
            (
                location,
                str(detail["message"]),
                f"type: {detail['type']}",
                f"input: {detail['input']!r}",
            )
        )
    return "\n".join(lines)
