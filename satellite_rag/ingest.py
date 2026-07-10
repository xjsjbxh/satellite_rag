"""Ingestion helpers for the RAG index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from satellite_rag.chunker import chunk_document
from satellite_rag.embeddings import DenseEmbeddingClient
from satellite_rag.keyword_store import KeywordStore
from satellite_rag.schemas import RagChunk
from satellite_rag.vector_store import VectorStore


class RagIngestor:
    """Write documents to both dense-vector and BM25 indexes."""

    def __init__(
        self,
        *,
        embedding_client: DenseEmbeddingClient,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    def ingest_chunks(self, chunks: list[RagChunk]) -> list[RagChunk]:
        vectors = self.embedding_client.embed_texts([chunk.content for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)
        self.keyword_store.upsert(chunks)
        return chunks

    def ingest_document(
        self,
        path: str | Path,
        *,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[RagChunk]:
        chunks = chunk_document(path, doc_id=doc_id, metadata=metadata)
        return self.ingest_chunks(chunks)

