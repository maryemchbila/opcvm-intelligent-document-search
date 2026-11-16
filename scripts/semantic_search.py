"""Recherche semantique FAISS dans l'index technique de developpement."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.search.vector_index import VectorSearchIndex  # noqa: E402


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--top-k doit etre strictement positif")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Requete semantique")
    parser.add_argument("--top-k", type=_positive_integer, default=5)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    project_root: Path | None = None,
    index_factory: Callable[[list[dict[str, object]]], VectorSearchIndex] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    chunks_path = root / "indexes" / "dev_chunks.pkl"
    if not chunks_path.is_file():
        print(f"Erreur : index de chunks introuvable : {chunks_path}", file=sys.stderr)
        return 1

    try:
        with chunks_path.open("rb") as chunks_file:
            chunks = pickle.load(chunks_file)
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("L'index de chunks est vide ou invalide.")
        print("Chargement du modele et creation des embeddings...")
        factory = index_factory if index_factory is not None else VectorSearchIndex
        index = factory(chunks)
        print("Index FAISS pret.")
        results = index.search(args.query, top_k=args.top_k)
    except (OSError, pickle.PickleError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"Query : {args.query}")
    print(f"Resultats : {len(results)}")
    for result in results:
        print()
        print(f"[{result['rank']}]")
        print(f"Score : {result['score']:.6g}")
        print(f"Document : {result['source']}")
        print(f"Page : {result['page_number']}")
        print(f"OPCVM : {result['opcvm']}")
        print(f"Categorie : {result['category']}")
        print()
        print("Texte :")
        print(result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
