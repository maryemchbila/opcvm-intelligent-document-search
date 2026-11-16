"""Client Gemini pour un PDF et une reponse JSON structuree."""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type

import httpx2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ..config import (
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RETRY_DELAYS,
    GEMINI_TIMEOUT_SECONDS,
    MAX_OUTPUT_TOKENS,
)
from .validators import validate_pdf_path

LOGGER = logging.getLogger(__name__)
_DROP_SCHEMA_KEYS = {
    "$comment",
    "$id",
    "$schema",
    "additionalProperties",
    "default",
    "deprecated",
    "example",
    "examples",
    "title",
}
_COMPLEX_PATTERN = re.compile(r"\(\?(?:[=!<]|P=)|\\[1-9]|\\k[<{]|\\g[<{]|\(\?>")


@dataclass(frozen=True, slots=True)
class GeminiGeneration:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    retry_count: int = 0


class GeminiClientError(RuntimeError):
    """Erreur Gemini sans cle ni contenu complet du PDF."""

    def __init__(self, message: str, *, retry_count: int = 0) -> None:
        super().__init__(message)
        self.retry_count = retry_count


class GeminiUnavailableError(GeminiClientError):
    """Erreur 503 epuisee apres les retries configures."""


class GeminiConfigurationError(GeminiClientError):
    """Configuration Gemini invalide."""


def safe_remote_filename(path: Path) -> str:
    """Construit un nom ASCII distant sans modifier le chemin local Unicode."""
    path = Path(path)
    stem = unicodedata.normalize("NFKD", path.stem)
    stem = stem.encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+\.", ".", stem).strip("._") or "document"
    return f"{stem}{path.suffix.lower()}"


def sanitize_schema_for_gemini(schema: dict) -> dict:
    """Adapte les schemas Pydantic au sous-ensemble JSON Schema de Gemini."""
    validate_required_properties(schema)

    def sanitize(value: object, *, named_schema_map: bool = False) -> object:
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if not isinstance(value, dict):
            return value
        output: dict = {}
        for key, item in value.items():
            # Dans `properties` et `$defs`, les cles sont des noms metier/de
            # modeles, pas des mots-cles JSON Schema. Un champ metier nomme
            # `title` doit donc etre conserve.
            if not named_schema_map and key in _DROP_SCHEMA_KEYS:
                continue
            if not named_schema_map and key == "const":
                allowed = value.get("enum")
                if allowed is not None and item not in allowed:
                    raise ValueError("Schema invalide: const absent de enum.")
                output["enum"] = [item]
                continue
            if not named_schema_map and key == "enum" and "const" in value:
                continue
            if not named_schema_map and key == "pattern" and (
                not isinstance(item, str)
                or len(item) > 128
                or _COMPLEX_PATTERN.search(item) is not None
            ):
                continue
            output[key] = sanitize(
                item,
                named_schema_map=(not named_schema_map and key in {"properties", "$defs"}),
            )
        return output

    result = sanitize(schema)
    if not isinstance(result, dict):
        raise TypeError("Le schema Gemini doit etre un objet.")
    validate_required_properties(result)
    return result


def validate_required_properties(schema: dict) -> None:
    """Verifie recursivement que chaque champ requis existe dans properties."""
    if not isinstance(schema, dict):
        raise TypeError("Le schema JSON doit etre un objet.")

    def validate(value: object, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                validate(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return

        required = value.get("required")
        if required is not None:
            if not isinstance(required, list) or not all(
                isinstance(name, str) for name in required
            ):
                raise ValueError(f"Schema JSON invalide a {path}: required doit etre une liste.")
            properties = value.get("properties")
            property_names = set(properties) if isinstance(properties, dict) else set()
            missing = [name for name in required if name not in property_names]
            if missing:
                names = ", ".join(missing)
                raise ValueError(
                    f"Schema JSON incoherent a {path}: champs requis absents de properties: {names}."
                )

        for key, item in value.items():
            validate(item, f"{path}.{key}")

    validate(schema, "$")


def _safe_error_text(value: object, key: str) -> str:
    text = value if isinstance(value, str) else ""
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(
        r"(?i)(?:authorization|x-goog-api-key|gemini_api_key)\s*[:=]\s*"
        r"(?:['\"]?)(?:bearer\s+)?[^\s,'\";}]+",
        "[AUTH REDACTED]",
        text,
    )
    return re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)


def _format_api_error(error: errors.APIError, key: str) -> str:
    status = _safe_error_text(error.status, key) or "UNKNOWN"
    message = _safe_error_text(error.message, key) or "Message provider absent."
    lines = [f"HTTP {error.code} {status}", f"message: {message}"]
    details = error.details
    error_body = details.get("error") if isinstance(details, dict) else None
    violations = error_body.get("details") if isinstance(error_body, dict) else None
    messages: list[str] = []
    if isinstance(violations, list):
        for detail in violations:
            if not isinstance(detail, dict):
                continue
            for violation in detail.get("fieldViolations", []):
                if isinstance(violation, dict):
                    field = _safe_error_text(violation.get("field"), key)
                    description = _safe_error_text(violation.get("description"), key)
                    if field or description:
                        messages.append(f"{field or 'unknown field'}: {description}")
    if messages:
        lines.append("fieldViolations: " + "; ".join(messages))
    return "\n".join(lines)


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        timeout: float = GEMINI_TIMEOUT_SECONDS,
        max_retries: int = GEMINI_MAX_RETRIES,
        retry_delays: tuple[float, ...] = GEMINI_RETRY_DELAYS,
        file_poll_seconds: float = 1.0,
        client: object | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        if not key or not key.strip():
            raise GeminiConfigurationError("GEMINI_API_KEY est absente.")
        if not model or not model.strip():
            raise GeminiConfigurationError("GEMINI_MODEL est absente.")
        if (
            timeout <= 0
            or max_retries < 0
            or file_poll_seconds < 0
            or len(retry_delays) < max_retries
            or any(delay < 0 for delay in retry_delays)
        ):
            raise ValueError(
                "Timeout positif, retries non negatif et un delai positif par retry requis."
            )
        self.model = model.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = tuple(retry_delays)
        self.file_poll_seconds = file_poll_seconds
        self._api_key = key.strip()
        self._client = client or genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    def _execute(
        self,
        operation: Callable[[], object],
        retry_counter: list[int] | None = None,
    ) -> object:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except errors.APIError as exc:
                if exc.code == 503 and attempt < self.max_retries:
                    if retry_counter is not None:
                        retry_counter[0] += 1
                    time.sleep(self.retry_delays[attempt])
                    continue
                error_type = GeminiUnavailableError if exc.code == 503 else GeminiClientError
                retry_count = retry_counter[0] if retry_counter is not None else attempt
                raise error_type(
                    _format_api_error(exc, self._api_key), retry_count=retry_count
                ) from None
            except (TimeoutError, httpx2.TimeoutException):
                raise GeminiClientError(
                    f"Gemini timeout apres {self.timeout:g} secondes."
                ) from None
            except httpx2.TransportError:
                raise GeminiClientError("Erreur reseau lors de l'appel Gemini.") from None
        raise GeminiClientError("Echec de l'appel Gemini.")

    def check_connection(self) -> None:
        self._execute(lambda: self._client.models.get(model=self.model))

    def _wait_for_file_ready(
        self, uploaded: object, retry_counter: list[int] | None = None
    ) -> object:
        """Attend la fin du traitement Files API, dans la limite du timeout client."""
        current = uploaded
        deadline = time.monotonic() + self.timeout
        while True:
            state = getattr(current, "state", None)
            state_value = getattr(state, "value", state)
            if state_value in (None, "ACTIVE"):
                return current
            if state_value == "FAILED":
                raise GeminiClientError("Gemini n'a pas pu preparer le fichier PDF.")
            if state_value != "PROCESSING":
                raise GeminiClientError(f"Etat de fichier Gemini inattendu: {state_value}.")
            if time.monotonic() >= deadline:
                raise GeminiClientError(
                    f"Preparation du PDF Gemini expiree apres {self.timeout:g} secondes."
                )
            time.sleep(self.file_poll_seconds)
            file_name = getattr(current, "name", None)
            if not file_name:
                raise GeminiClientError("Le fichier Gemini ne possede pas de nom distant.")
            current = self._execute(
                lambda: self._client.files.get(name=file_name), retry_counter
            )

    def generate_pdf(
        self,
        pdf_path: Path,
        prompt: str,
        schema: Type[BaseModel],
    ) -> GeminiGeneration:
        """Upload un unique PDF, genere le JSON et supprime le fichier distant."""
        pdf_path = validate_pdf_path(pdf_path)
        remote_filename = safe_remote_filename(pdf_path)
        retry_counter = [0]

        def upload_pdf() -> object:
            # Un flux empeche le SDK de copier le basename Unicode dans le
            # header ASCII X-Goog-Upload-File-Name. Chaque retry rouvre le PDF.
            with pdf_path.open("rb") as pdf_file:
                return self._client.files.upload(
                    file=pdf_file,
                    config=types.UploadFileConfig(
                        mime_type="application/pdf",
                        display_name=remote_filename,
                    ),
                )

        uploaded = self._execute(upload_pdf, retry_counter)
        file_name = getattr(uploaded, "name", None)
        if not file_name:
            raise GeminiClientError("L'upload Gemini n'a pas retourne les informations du fichier.")

        generation_succeeded = False
        try:
            uploaded = self._wait_for_file_ready(uploaded, retry_counter)
            file_uri = getattr(uploaded, "uri", None)
            if not file_uri:
                raise GeminiClientError("Gemini n'a pas retourne l'URI du fichier prepare.")
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=sanitize_schema_for_gemini(schema.model_json_schema()),
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            response = self._execute(
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=[
                        prompt,
                        types.Part.from_uri(file_uri=file_uri, mime_type="application/pdf"),
                    ],
                    config=config,
                ),
                retry_counter,
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise GeminiClientError("Gemini a retourne une reponse vide.")
            generation_succeeded = True
            usage = getattr(response, "usage_metadata", None)
            return GeminiGeneration(
                text=text,
                input_tokens=_token_count(getattr(usage, "prompt_token_count", None)),
                output_tokens=_token_count(getattr(usage, "candidates_token_count", None)),
                total_tokens=_token_count(getattr(usage, "total_token_count", None)),
                retry_count=retry_counter[0],
            )
        finally:
            active_exception = sys.exc_info()[0] is not None
            try:
                self._execute(lambda: self._client.files.delete(name=file_name))
            except GeminiClientError as exc:
                if generation_succeeded or not active_exception:
                    raise
                LOGGER.error("Suppression du fichier temporaire Gemini impossible (%s).", type(exc).__name__)

    def close(self) -> None:
        self._client.close()


def _token_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
