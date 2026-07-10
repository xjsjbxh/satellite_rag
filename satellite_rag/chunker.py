"""Document chunking utilities for RAG ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from satellite_rag.schemas import RagChunk


DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120


def load_text(path: str | Path) -> str:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return json.dumps(json.loads(source.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    return source.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    *,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks without introducing empty chunks."""

    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be >= 0 and smaller than chunk_chars")

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_chars)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def chunk_document(
    path: str | Path,
    *,
    doc_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RagChunk]:
    source = Path(path)
    text = load_text(source)
    resolved_doc_id = doc_id or _stable_id(str(source))
    chunks = []
    for index, content in enumerate(chunk_text(text, chunk_chars=chunk_chars, overlap=overlap), start=1):
        chunk_id = f"{resolved_doc_id}_chunk_{index:04d}"
        chunks.append(
            RagChunk(
                chunk_id=chunk_id,
                doc_id=resolved_doc_id,
                content=content,
                source_path=str(source),
                metadata=metadata or {},
            )
        )
    return chunks


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]

