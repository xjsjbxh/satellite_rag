"""Minimal data contracts for the RAG retrieval layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RagChunk(BaseModel):
    """Smallest persisted unit for retrieval."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    doc_id: str
    content: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result returned by vector, keyword, fusion, or rerank stages."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    doc_id: str
    content: str
    source_path: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    rank: int | None = None
    source: str = "unknown"


class SearchRequest(BaseModel):
    """Input to the hybrid retriever."""

    model_config = ConfigDict(extra="allow")

    query: str
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    vector_top_k: int | None = None
    bm25_top_k: int | None = None
    fusion_top_k: int | None = None
    rerank_top_k: int | None = None
    final_top_k: int | None = None

