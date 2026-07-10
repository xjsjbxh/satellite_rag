"""Factories for real RAG runtime components.

The default tests still use in-memory adapters. Production/demo runs can switch
providers through .env without changing workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from satellite_rag.config import RagConfig
from satellite_rag.embeddings import (
    BgeM3EmbeddingClient,
    DenseEmbeddingClient,
    HashDenseEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from satellite_rag.ingest import RagIngestor
from satellite_rag.keyword_store import ElasticsearchKeywordStore, KeywordStore, LocalBM25KeywordStore
from satellite_rag.reranker import BgeRerankerClient, IdentityReranker, OpenAICompatibleRerankerClient, Reranker
from satellite_rag.retriever import HybridRetriever
from satellite_rag.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore


@dataclass
class RagRuntime:
    config: RagConfig
    embedding_client: DenseEmbeddingClient
    vector_store: VectorStore
    keyword_store: KeywordStore
    reranker: Reranker
    ingestor: RagIngestor
    retriever: HybridRetriever


def build_embedding_client(config: RagConfig | None = None) -> DenseEmbeddingClient:
    runtime_config = config or RagConfig.from_env()
    provider = _normalise(runtime_config.embedding_provider)
    if provider in {"bge_m3", "bge-m3", "bge", "flagembedding"}:
        return BgeM3EmbeddingClient(
            model_name=runtime_config.embedding_model,
            use_fp16=runtime_config.embedding_use_fp16,
        )
    if provider in {"ollama", "ollama_bge_m3", "ollama_bge"}:
        return OllamaEmbeddingClient(
            model_name=runtime_config.embedding_model,
            endpoint=runtime_config.embedding_endpoint,
            timeout=runtime_config.embedding_timeout,
            batch_size=runtime_config.embedding_batch_size,
        )
    if provider in {"litellm", "openai", "openai_compatible", "openai-compatible"}:
        auth_header = runtime_config.embedding_auth_header
        if provider == "litellm" and auth_header == "Authorization":
            auth_header = "x-litellm-api-key"
        return OpenAICompatibleEmbeddingClient(
            model_name=runtime_config.embedding_model,
            endpoint=runtime_config.embedding_endpoint or "",
            api_key=runtime_config.embedding_api_key,
            auth_header=auth_header,
            timeout=runtime_config.embedding_timeout,
            batch_size=runtime_config.embedding_batch_size,
        )
    if provider in {"hash", "test", "local_hash"}:
        return HashDenseEmbeddingClient(
            dimensions=runtime_config.hash_embedding_dimensions,
            model_name="test/hash-dense",
        )
    raise ValueError(f"Unsupported RAG_EMBEDDING_PROVIDER: {runtime_config.embedding_provider}")


def build_vector_store(config: RagConfig | None = None) -> VectorStore:
    runtime_config = config or RagConfig.from_env()
    provider = _normalise(runtime_config.vector_provider)
    if provider == "auto":
        provider = "qdrant" if runtime_config.qdrant_url or runtime_config.qdrant_path else "memory"
    if provider in {"memory", "in_memory", "local"}:
        return InMemoryVectorStore()
    if provider == "qdrant":
        if not runtime_config.qdrant_url and not runtime_config.qdrant_path:
            raise RuntimeError("RAG_VECTOR_PROVIDER=qdrant requires QDRANT_URL or QDRANT_PATH.")
        return QdrantVectorStore(
            url=runtime_config.qdrant_url,
            path=runtime_config.qdrant_path,
            api_key=runtime_config.qdrant_api_key,
            collection=runtime_config.qdrant_collection,
            vector_name=runtime_config.qdrant_vector_name,
        )
    raise ValueError(f"Unsupported RAG_VECTOR_PROVIDER: {runtime_config.vector_provider}")


def build_keyword_store(config: RagConfig | None = None) -> KeywordStore:
    runtime_config = config or RagConfig.from_env()
    provider = _normalise(runtime_config.keyword_provider)
    if provider in {"local", "bm25", "memory"}:
        return LocalBM25KeywordStore()
    if provider in {"elasticsearch", "es", "opensearch", "ik"}:
        if not runtime_config.keyword_url:
            raise RuntimeError("RAG_KEYWORD_PROVIDER=elasticsearch requires RAG_KEYWORD_URL.")
        return ElasticsearchKeywordStore(
            url=runtime_config.keyword_url,
            index=runtime_config.keyword_index,
            analyzer=runtime_config.keyword_analyzer,
            search_analyzer=runtime_config.keyword_search_analyzer,
            username=runtime_config.keyword_username,
            password=runtime_config.keyword_password,
            api_key=runtime_config.keyword_api_key,
            timeout=runtime_config.keyword_timeout,
        )
    raise ValueError(f"Unsupported RAG_KEYWORD_PROVIDER: {runtime_config.keyword_provider}")


def build_reranker(config: RagConfig | None = None) -> Reranker:
    runtime_config = config or RagConfig.from_env()
    provider = _normalise(runtime_config.rerank_provider)
    if provider in {"identity", "none", "disabled"}:
        return IdentityReranker()
    if provider in {"bge", "bge_m3", "flagembedding"}:
        return BgeRerankerClient(model_name=runtime_config.rerank_model)
    if provider in {"litellm", "openai", "openai_compatible", "openai_compatible_rerank", "qwen"}:
        auth_header = runtime_config.rerank_auth_header
        if provider == "litellm" and auth_header == "Authorization":
            auth_header = "x-litellm-api-key"
        return OpenAICompatibleRerankerClient(
            model_name=runtime_config.rerank_model,
            endpoint=runtime_config.rerank_endpoint or "",
            api_key=runtime_config.rerank_api_key,
            auth_header=auth_header,
            timeout=runtime_config.rerank_timeout,
        )
    raise ValueError(f"Unsupported RAG_RERANK_PROVIDER: {provider}")


def build_rag_runtime(config: RagConfig | None = None) -> RagRuntime:
    runtime_config = config or RagConfig.from_env()
    embedding_client = build_embedding_client(runtime_config)
    vector_store = build_vector_store(runtime_config)
    keyword_store = build_keyword_store(runtime_config)
    reranker = build_reranker(runtime_config)
    ingestor = RagIngestor(
        embedding_client=embedding_client,
        vector_store=vector_store,
        keyword_store=keyword_store,
    )
    retriever = HybridRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        keyword_store=keyword_store,
        reranker=reranker,
        config=runtime_config,
    )
    return RagRuntime(
        config=runtime_config,
        embedding_client=embedding_client,
        vector_store=vector_store,
        keyword_store=keyword_store,
        reranker=reranker,
        ingestor=ingestor,
        retriever=retriever,
    )


@lru_cache(maxsize=1)
def get_default_retriever() -> HybridRetriever:
    return build_rag_runtime(RagConfig.from_env()).retriever


def clear_rag_runtime_cache() -> None:
    get_default_retriever.cache_clear()


def _normalise(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")

