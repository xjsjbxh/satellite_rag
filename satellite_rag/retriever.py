"""Dense-vector + BM25 hybrid retrieval."""

from __future__ import annotations

import os
from typing import Any

from satellite_rag.config import RagConfig
from satellite_rag.citations import format_retrieval_citation
from satellite_rag.embeddings import DenseEmbeddingClient
from satellite_rag.keyword_store import KeywordStore
from satellite_rag.reranker import IdentityReranker, Reranker
from satellite_rag.schemas import RetrievalResult, SearchRequest
from satellite_rag.vector_store import VectorStore


RRF_K = 60


class HybridRetriever:
    """Retrieve with dense vector search and BM25, then RRF fuse and rerank."""

    def __init__(
        self,
        *,
        embedding_client: DenseEmbeddingClient,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        reranker: Reranker | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.reranker = reranker or IdentityReranker()
        self.config = config or RagConfig()

    def search(self, request: SearchRequest) -> list[RetrievalResult]:
        vector_top_k = request.vector_top_k or self.config.vector_top_k
        bm25_top_k = request.bm25_top_k or self.config.bm25_top_k
        fusion_top_k = request.fusion_top_k or self.config.fusion_top_k
        rerank_top_k = request.rerank_top_k or self.config.rerank_top_k
        final_top_k = request.final_top_k or self.config.final_top_k

        query_vector = self.embedding_client.embed_query(request.query)
        vector_results = self.vector_store.search(
            query_vector,
            top_k=vector_top_k,
            metadata_filter=request.metadata_filter,
        )
        keyword_results = self.keyword_store.search(
            request.query,
            top_k=bm25_top_k,
            metadata_filter=request.metadata_filter,
        )
        fused = rrf_fuse(
            [vector_results, keyword_results],
            top_k=fusion_top_k,
            weights=[self.config.vector_weight, self.config.bm25_weight],
        )
        candidates = fused[:rerank_top_k]
        return self.reranker.rerank(request.query, candidates, top_k=final_top_k)


def rrf_fuse(
    result_sets: list[list[RetrievalResult]],
    *,
    top_k: int,
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Fuse ranked lists with reciprocal rank fusion."""

    if weights is None:
        weights = [1.0] * len(result_sets)
    if len(weights) != len(result_sets):
        raise ValueError("RRF weights must match the number of result sets.")

    by_chunk: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for results, weight in zip(result_sets, weights):
        if weight <= 0:
            continue
        for fallback_rank, result in enumerate(results, start=1):
            rank = result.rank or fallback_rank
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (k + rank)
            by_chunk.setdefault(result.chunk_id, result)
            sources.setdefault(result.chunk_id, set()).add(result.source)

    fused = [
        by_chunk[chunk_id].model_copy(
            update={
                "score": score,
                "source": "+".join(sorted(sources.get(chunk_id, {"fusion"}))),
            }
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: item.score, reverse=True)
    return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(fused[:top_k])]


def build_phase_query(
    *,
    phase: str,
    phase_title: str,
    required_inputs: list[str],
    phase_inputs: dict[str, Any],
) -> str:
    input_keys = ", ".join(sorted(phase_inputs.keys())) if phase_inputs else "no provided inputs"
    required = ", ".join(required_inputs)
    return f"{phase} {phase_title}\nrequired_inputs: {required}\nprovided_inputs: {input_keys}"


def retrieve_phase_evidence(
    *,
    phase: str,
    phase_title: str,
    required_inputs: list[str],
    phase_inputs: dict[str, Any],
    metadata_filter: dict[str, Any] | None = None,
    retriever: HybridRetriever | None = None,
    config: RagConfig | None = None,
) -> list[dict[str, Any]]:
    """Retrieve phase evidence when RAG is enabled.

    The default path is intentionally conservative: if RAG is disabled or no
    retriever is injected, it returns an empty list. This prevents tests and
    local demos from making accidental external service calls.
    """

    runtime_config = config or RagConfig.from_env()
    if retriever is None:
        if not runtime_config.enabled:
            return []
        from satellite_rag.factory import get_default_retriever

        retriever = get_default_retriever()

    query = build_phase_query(
        phase=phase,
        phase_title=phase_title,
        required_inputs=required_inputs,
        phase_inputs=phase_inputs,
    )
    request = SearchRequest(
        query=query,
        metadata_filter=metadata_filter or {},
        vector_top_k=runtime_config.vector_top_k,
        bm25_top_k=runtime_config.bm25_top_k,
        fusion_top_k=runtime_config.fusion_top_k,
        rerank_top_k=runtime_config.rerank_top_k,
        final_top_k=runtime_config.final_top_k,
    )
    rows = []
    for index, item in enumerate(retriever.search(request), start=1):
        row = item.model_dump()
        row["citation"] = format_retrieval_citation(item, index)
        rows.append(row)
    return rows

