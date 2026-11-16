"""Normalise localement l'index de chunks existant, sans extraction PDF."""

from __future__ import annotations

import os
import pickle
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.extraction.chunker import (  # noqa: E402
    ChunkingStats,
    normalize_existing_chunks,
)


def _save_atomically(chunks: list[dict[str, object]], output_path: Path) -> None:
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as output_file:
            pickle.dump(chunks, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(*, project_root: Path | None = None) -> int:
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    chunks_path = root / "indexes" / "dev_chunks.pkl"
    backup_path = root / "indexes" / "dev_chunks.before_normalization.pkl"
    if not chunks_path.is_file():
        print(f"Erreur : index de chunks introuvable : {chunks_path}", file=sys.stderr)
        return 1

    try:
        with chunks_path.open("rb") as chunks_file:
            chunks = pickle.load(chunks_file)
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("L'index de chunks est vide ou invalide.")

        stats = ChunkingStats()
        normalized_chunks = normalize_existing_chunks(chunks, stats=stats)
        if not backup_path.exists():
            shutil.copy2(chunks_path, backup_path)
        _save_atomically(normalized_chunks, chunks_path)
    except (OSError, EOFError, pickle.PickleError, TypeError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"Chunks avant : {len(chunks)}")
    print(f"Micro-chunks fusionnés : {stats.micro_chunks_merged}")
    print(f"Micro-chunks ignorés : {stats.micro_chunks_ignored}")
    print(f"Chunks après : {len(normalized_chunks)}")
    print(f"Output : {chunks_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
