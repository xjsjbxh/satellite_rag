"""Vector store adapters for Qdrant-backed dense retrieval."""

from __future__ import annotations

import math
import uuid
from typing import Any, Protocol

from satellite_rag.schemas import RagChunk, RetrievalResult


class VectorStore(Protocol):
    def upsert(self, chunks: list[RagChunk], vectors: list[list[float]]) -> None:
        """Persist chunks and their dense vectors."""

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return nearest chunks."""


class InMemoryVectorStore:
    """Small cosine-similarity store used for tests and local demos."""

    def __init__(self) -> None:
        self._items: list[tuple[RagChunk, list[float]]] = []

    def upsert(self, chunks: list[RagChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        by_id = {chunk.chunk_id: (chunk, vector) for chunk, vector in self._items}
        for chunk, vector in zip(chunks, vectors):
            by_id[chunk.chunk_id] = (chunk, vector)
        self._items = list(by_id.values())

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        results = []
        for chunk, vector in self._items:
            if not _metadata_matches(chunk.metadata, metadata_filter or {}):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    source_path=chunk.source_path,
                    metadata=chunk.metadata,
                    score=_cosine(query_vector, vector),
                    source="vector",
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(results[:top_k])]


class QdrantVectorStore:
    """Qdrant dense vector adapter.

    This wrapper assumes the collection already exists with a dense vector field.
    Collection creation is intentionally left to deployment scripts because model
    dimension and distance settings should be managed explicitly.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        collection: str,
        vector_name: str = "dense",
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client is required for QdrantVectorStore.") from exc
        if not url and not path:
            raise RuntimeError("QdrantVectorStore requires either url or path.")
        self.collection = collection
        self.vector_name = vector_name
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(path=path)

    def upsert(self, chunks: list[RagChunk], vectors: list[list[float]]) -> None:
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("qdrant-client models are required for Qdrant upsert.") from exc
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = chunk.model_dump()
            points.append(
                PointStruct(
                    id=_qdrant_point_id(chunk.chunk_id),
                    vector={self.vector_name: vector},
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_filter = _qdrant_filter(metadata_filter or {})
        if hasattr(self._client, "search"):
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=(self.vector_name, query_vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using=self.vector_name,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = getattr(response, "points", response)
        results = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            results.append(
                RetrievalResult(
                    chunk_id=str(payload.get("chunk_id") or hit.id),
                    doc_id=str(payload.get("doc_id") or ""),
                    content=str(payload.get("content") or ""),
                    source_path=str(payload.get("source_path") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                    score=float(hit.score or 0.0),
                    rank=index,
                    source="vector",
                )
            )
        return results



def _qdrant_point_id(chunk_id: str) -> str:
    """Return a deterministic UUID accepted by Qdrant point IDs."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"satdesign-reviewflow:{chunk_id}"))
def _metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        actual = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
            continue
        if actual != value:
            return False
    return True


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _qdrant_filter(metadata_filter: dict[str, Any]) -> Any:
    if not metadata_filter:
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("qdrant-client models are required for metadata filters.") from exc
    conditions = []
    for key, value in metadata_filter.items():
        field = f"metadata.{key}"
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                continue
            conditions.append(FieldCondition(key=field, match=MatchAny(any=values)))
        else:
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
    return Filter(
        must=conditions
    )

