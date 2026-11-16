"""Extrait et decoupe un PDF, puis sauvegarde les chunks techniques en pickle."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.extraction.chunker import ChunkingStats, chunk_pages  # noqa: E402
from app.extraction.pdf_text_extractor import extract_pdf_pages  # noqa: E402
from app.extraction.vision_text_extractor import (  # noqa: E402
    VisionExtractionError,
    VisionTextExtractor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Chemin d'un unique fichier PDF")
    return parser


def derive_path_metadata(pdf_path: Path) -> tuple[str, str, str]:
    """Deduit categorie, OPCVM et source depuis l'arborescence du document."""
    parts = pdf_path.parts
    fichiers_indexes = [
        index for index, part in enumerate(parts) if part.casefold() == "fichiers"
    ]
    if fichiers_indexes:
        index = fichiers_indexes[-1]
        if len(parts) >= index + 4:
            return parts[index + 1], parts[index + 2], pdf_path.name
    if len(pdf_path.parents) >= 2:
        return pdf_path.parent.parent.name, pdf_path.parent.name, pdf_path.name
    raise ValueError(
        "Le chemin doit contenir une categorie et un OPCVM avant le nom du PDF."
    )


def save_chunks_atomically(chunks: list[dict], output_path: Path) -> None:
    """Ecrit le pickle complet avant de remplacer atomiquement l'index courant."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as output_file:
            pickle.dump(chunks, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(
    argv: list[str] | None = None,
    *,
    project_root: Path | None = None,
    vision_extractor: VisionTextExtractor | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"Erreur : fichier introuvable : {pdf_path}", file=sys.stderr)
        return 1

    try:
        category, opcvm, source = derive_path_metadata(pdf_path)
        pages = extract_pdf_pages(pdf_path)
        local_pages = pages
        vision_mode = "NOT_USED"
        vision_fallback = "NONE"
        vision_page_numbers = [
            page["page_number"] for page in pages if page["needs_ocr_or_vision"]
        ]
        if vision_page_numbers:
            extractor = vision_extractor or VisionTextExtractor()
            vision_pages = extractor.extract_pages(
                pdf_path,
                vision_page_numbers,
                total_pages=len(pages),
            )
            vision_by_number = {
                page["page_number"]: page for page in vision_pages
            }
            if set(vision_by_number) != set(vision_page_numbers):
                raise VisionExtractionError(
                    "Pagination Gemini vision incomplete ou incoherente."
                )
            pages = [
                vision_by_number.get(page["page_number"], page) for page in pages
            ]
            reported_mode = getattr(extractor, "last_mode", None)
            reported_fallback = getattr(extractor, "last_fallback", None)
            vision_mode = (
                reported_mode if isinstance(reported_mode, str) else "FULL_DOCUMENT"
            )
            vision_fallback = (
                reported_fallback if isinstance(reported_fallback, str) else "NONE"
            )
        chunking_stats = ChunkingStats()
        chunks = chunk_pages(
            pages,
            source=source,
            source_path=str(pdf_path),
            category=category,
            opcvm=opcvm,
            stats=chunking_stats,
        )
        output_path = root / "indexes" / "dev_chunks.pkl"
        if chunks:
            save_chunks_atomically(chunks, output_path)
    except (OSError, ValueError, VisionExtractionError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    pymupdf_text_pages = sum(
        page["page_status"] == "READABLE" for page in local_pages
    )
    locally_empty_pages = sum(
        page["page_status"] == "EMPTY_OR_BLANK" for page in local_pages
    )
    gemini_pages = sum(
        page["extraction_method"] == "gemini_vision"
        and page["page_status"] != "FAILED"
        for page in pages
    )
    failed_gemini_pages = sum(
        page["extraction_method"] == "gemini_vision_failed" for page in pages
    )
    readable_page_numbers = [
        page["page_number"] for page in pages if page["page_status"] == "READABLE"
    ]
    empty_page_numbers = [
        page["page_number"]
        for page in pages
        if page["page_status"] == "EMPTY_OR_BLANK"
    ]
    failed_page_numbers = [
        page["page_number"] for page in pages if page["page_status"] == "FAILED"
    ]
    print(f"Document : {source}")
    print(f"Pages totales : {len(pages)}")
    print(f"Vision mode : {vision_mode}")
    print(f"Vision fallback : {vision_fallback}")
    print(f"Pages texte PyMuPDF : {pymupdf_text_pages}")
    print(f"Pages vides détectées localement : {locally_empty_pages}")
    print(f"Pages envoyées à Gemini Vision : {len(vision_page_numbers)}")
    print(f"Pages Gemini réussies : {gemini_pages}")
    print(f"Pages Gemini échouées : {failed_gemini_pages}")
    print(f"Pages exploitables : {len(readable_page_numbers)}")
    print(f"Pages vides : {len(empty_page_numbers)}")
    print(f"Pages échouées : {len(failed_page_numbers)}")
    print(f"Pages vides : {empty_page_numbers}")
    for page in pages:
        if page["page_status"] == "FAILED":
            error_type = page.get("error_type", "VISION_ERROR")
            error_message = page.get("error_message", "Erreur vision non detaillee.")
            print(f"- Page {page['page_number']} : {error_type} - {error_message}")
    print(f"Chunks créés : {len(chunks)}")
    print(f"Micro-chunks fusionnés : {chunking_stats.micro_chunks_merged}")
    print(f"Micro-chunks ignorés : {chunking_stats.micro_chunks_ignored}")
    if chunks:
        print(f"Output : {output_path.resolve()}")
        return 0

    print("Aucun chunk exploitable généré.")
    if output_path.exists():
        print("Index existant conservé : indexes/dev_chunks.pkl")
    else:
        print("Aucun index précédent disponible.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
