"""Extraction de texte Gemini pour les seules pages illisibles localement."""

from __future__ import annotations

import os
import re
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pymupdf
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import (
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RETRY_DELAYS,
    GEMINI_TIMEOUT_SECONDS,
    MAX_OUTPUT_TOKENS,
)
from .pdf_text_extractor import (
    MIN_TEXT_CHARACTERS,
    ExtractedPage,
    page_has_meaningful_visual_content,
)


class VisionExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        status: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.provider_message = provider_message


class VisionPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class VisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: list[VisionPage]


def _safe_filename(path: Path) -> str:
    stem = unicodedata.normalize("NFKD", path.stem)
    stem = stem.encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"{stem}.pdf"


def _safe_provider_message(message: object, api_key: str) -> str:
    text = message if isinstance(message, str) else "Message provider absent."
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    text = re.sub(
        r"(?i)(?:authorization|x-goog-api-key|gemini_api_key)\s*[:=]\s*"
        r"(?:['\"]?)(?:bearer\s+)?[^\s,'\";}]+",
        "[AUTH REDACTED]",
        text,
    )
    return re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text).strip()


class VisionTextExtractor:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        timeout: float = GEMINI_TIMEOUT_SECONDS,
        max_retries: int = GEMINI_MAX_RETRIES,
        retry_delays: tuple[float, ...] = GEMINI_RETRY_DELAYS,
        client: object | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        if not key or not key.strip():
            raise VisionExtractionError("GEMINI_API_KEY est absente pour le fallback vision.")
        if timeout <= 0 or max_retries < 0 or len(retry_delays) < max_retries:
            raise ValueError("Configuration Gemini vision invalide.")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = tuple(retry_delays)
        self._api_key = key.strip()
        self.last_mode = "NOT_USED"
        self.last_fallback = "NONE"
        self._client = client or genai.Client(
            api_key=key.strip(),
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    def _execute(self, operation: Callable[[], object]) -> object:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except errors.APIError as exc:
                if exc.code == 503 and attempt < self.max_retries:
                    time.sleep(self.retry_delays[attempt])
                    continue
                status = str(exc.status or "UNKNOWN")
                message = _safe_provider_message(exc.message, self._api_key)
                raise VisionExtractionError(
                    f"HTTP {exc.code} {status}\nmessage: {message}",
                    code=exc.code,
                    status=status,
                    provider_message=message,
                ) from None
        raise VisionExtractionError("Echec de l'appel Gemini vision.")

    def _wait_until_ready(self, uploaded: object) -> object:
        current = uploaded
        deadline = time.monotonic() + self.timeout
        while True:
            state = getattr(current, "state", None)
            state_value = getattr(state, "value", state)
            if state_value in (None, "ACTIVE"):
                return current
            if state_value != "PROCESSING":
                raise VisionExtractionError(
                    f"Etat de fichier Gemini vision inattendu : {state_value}."
                )
            if time.monotonic() >= deadline:
                raise VisionExtractionError("Preparation Gemini vision expiree.")
            name = getattr(current, "name", None)
            if not name:
                raise VisionExtractionError("Nom de fichier Gemini vision absent.")
            time.sleep(1)
            current = self._execute(lambda: self._client.files.get(name=name))

    def extract_pages(
        self,
        pdf_value: str | Path,
        page_numbers: list[int],
        *,
        total_pages: int,
    ) -> list[ExtractedPage]:
        pdf_path = Path(pdf_value).expanduser().resolve()
        requested_pages = sorted(set(page_numbers))
        if not requested_pages:
            return []
        if requested_pages[0] < 1 or requested_pages[-1] > total_pages:
            raise ValueError("Numero de page hors limites pour le fallback vision.")

        self.last_mode = "FULL_DOCUMENT"
        self.last_fallback = "NONE"
        try:
            return self._extract_batch(
                pdf_path,
                requested_pages,
                total_pages=total_pages,
                force_temporary=False,
            )
        except VisionExtractionError as exc:
            if not _is_deadline_exceeded(exc):
                raise

        self.last_fallback = "PAGE_BY_PAGE"
        pages: list[ExtractedPage] = []
        for page_number in requested_pages:
            try:
                pages.extend(
                    self._extract_batch(
                        pdf_path,
                        [page_number],
                        total_pages=total_pages,
                        force_temporary=True,
                    )
                )
            except VisionExtractionError as exc:
                pages.append(_failed_page(page_number, exc))
        return pages

    def _extract_batch(
        self,
        pdf_path: Path,
        requested_pages: list[int],
        *,
        total_pages: int,
        force_temporary: bool,
    ) -> list[ExtractedPage]:

        with _vision_pdf(
            pdf_path,
            requested_pages,
            total_pages,
            force_temporary=force_temporary,
        ) as vision_pdf:
            mapping = ", ".join(
                f"page PDF {index} = page originale {original}"
                for index, original in enumerate(requested_pages, start=1)
            )
            prompt = f"""Retourne le texte integral page par page.
Ne resume pas.
Ne reformule pas.
Conserve les tableaux autant que possible.
Indique clairement les numeros de pages originales.
Le mapping des pages est : {mapping}.
Retourne exactement une entree par page PDF, dans le meme ordre."""

            def upload() -> object:
                with vision_pdf.open("rb") as pdf_file:
                    return self._client.files.upload(
                        file=pdf_file,
                        config=types.UploadFileConfig(
                            mime_type="application/pdf",
                            display_name=_safe_filename(vision_pdf),
                        ),
                    )

            uploaded = self._execute(upload)
            file_name = getattr(uploaded, "name", None)
            if not file_name:
                raise VisionExtractionError("Upload Gemini vision incomplet.")
            try:
                uploaded = self._wait_until_ready(uploaded)
                uri = getattr(uploaded, "uri", None)
                if not uri:
                    raise VisionExtractionError("URI Gemini vision absente.")
                response = self._execute(
                    lambda: self._client.models.generate_content(
                        model=self.model,
                        contents=[
                            prompt,
                            types.Part.from_uri(
                                file_uri=uri, mime_type="application/pdf"
                            ),
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_json_schema=VisionExtraction.model_json_schema(),
                            temperature=0.0,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        ),
                    )
                )
                text = getattr(response, "text", None)
                if not isinstance(text, str) or not text.strip():
                    raise VisionExtractionError("Reponse Gemini vision vide.")
                try:
                    result = VisionExtraction.model_validate_json(text)
                except ValidationError as exc:
                    raise VisionExtractionError(
                        "JSON Gemini vision invalide : "
                        f"{exc.error_count()} erreur(s) de validation."
                    ) from None
            finally:
                try:
                    self._execute(lambda: self._client.files.delete(name=file_name))
                except VisionExtractionError:
                    pass

        if len(result.pages) != len(requested_pages):
            raise VisionExtractionError(
                "Gemini vision n'a pas retourne exactement une entree par page."
            )
        pages: list[ExtractedPage] = []
        for original_page, returned_page in zip(requested_pages, result.pages):
            text = returned_page.text.strip()
            useful_characters = len(re.sub(r"\s+", "", text))
            if useful_characters >= MIN_TEXT_CHARACTERS:
                status = "READABLE"
                needs_vision = False
            elif page_has_meaningful_visual_content(pdf_path, original_page):
                status = "FAILED"
                needs_vision = True
            else:
                status = "EMPTY_OR_BLANK"
                needs_vision = False
            page: ExtractedPage = {
                "page_number": original_page,
                "text": text,
                "needs_ocr_or_vision": needs_vision,
                "extraction_method": "gemini_vision",
                "page_status": status,
            }
            if status == "FAILED":
                page.update(
                    {
                        "status": "FAILED",
                        "error_type": "NO_TEXT_EXTRACTED",
                        "error_message": (
                            "Contenu visible mais aucun texte exploitable recupere."
                        ),
                    }
                )
            pages.append(page)
        return pages


@contextmanager
def _vision_pdf(
    pdf_path: Path,
    requested_pages: list[int],
    total_pages: int,
    *,
    force_temporary: bool = False,
) -> Iterator[Path]:
    if not force_temporary and requested_pages == list(range(1, total_pages + 1)):
        yield pdf_path
        return

    with tempfile.TemporaryDirectory(prefix="vision_pages_") as directory:
        subset_path = Path(directory) / "vision_pages.pdf"
        with pymupdf.open(pdf_path) as source, pymupdf.open() as subset:
            for page_number in requested_pages:
                subset.insert_pdf(
                    source, from_page=page_number - 1, to_page=page_number - 1
                )
            subset.save(subset_path)
        yield subset_path


def _is_deadline_exceeded(error: VisionExtractionError) -> bool:
    return (
        error.code == 504
        or (error.status or "").upper() == "DEADLINE_EXCEEDED"
        or "DEADLINE_EXCEEDED" in str(error).upper()
    )


def _failed_page(
    page_number: int, error: VisionExtractionError
) -> ExtractedPage:
    error_type = f"HTTP_{error.code}" if error.code is not None else "VISION_ERROR"
    error_message = error.provider_message or str(error).splitlines()[0]
    return {
        "page_number": page_number,
        "text": "",
        "needs_ocr_or_vision": True,
        "extraction_method": "gemini_vision_failed",
        "page_status": "FAILED",
        "status": "FAILED",
        "error_type": error_type,
        "error_message": error_message,
    }
