"""Analyse un seul PDF avec Gemini, sans OCR local ni traitement de dossier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.legacy_extraction.extractor import ExtractionError, PDFExtractor  # noqa: E402
from app.legacy_extraction.gemini_client import (  # noqa: E402
    GeminiClient,
    GeminiClientError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Chemin relatif ou absolu d'un unique fichier PDF",
    )
    parser.add_argument(
        "--check-gemini",
        action="store_true",
        help="Verifie la cle API, la connexion et l'acces au modele",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_gemini:
        client: GeminiClient | None = None
        try:
            client = GeminiClient()
            client.check_connection()
        except GeminiClientError as exc:
            print(f"Gemini: FAILED - {exc}", file=sys.stderr)
            return 1
        finally:
            if client is not None:
                client.close()
        print("Connexion Gemini : OK")
        print("Modele disponible : OK")
        return 0

    if not args.pdf:
        build_parser().error("indiquez le chemin d'un seul PDF ou utilisez --check-gemini")

    try:
        result = PDFExtractor().extract(args.pdf)
    except ExtractionError as exc:
        print(f"Extraction: FAILED - {exc}", file=sys.stderr)
        return 1

    print(f"Extraction: {result.report_path}")
    print(f"JSON structure: {result.structured_path}")
    print(f"Reponse brute: {result.raw_path}")
    print(f"Duree: {result.duration_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
