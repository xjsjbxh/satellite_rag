"""Configuration for dense-vector + BM25 RAG retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass

from satellite_rag.env import load_dotenv


@dataclass(frozen=True)
class RagConfig:
    """Runtime knobs for the first RAG implementation."""

    enabled: bool = False
    embedding_provider: str = "bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_endpoint: str | None = None
    embedding_api_key: str | None = None
    embedding_auth_header: str = "Authorization"
    embedding_batch_size: int = 16
    embedding_timeout: float = 120.0
    embedding_use_fp16: bool = True
    hash_embedding_dimensions: int = 64
    vector_provider: str = "auto"
    rerank_provider: str = "identity"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_endpoint: str | None = None
    rerank_api_key: str | None = None
    rerank_auth_header: str = "Authorization"
    rerank_timeout: float = 120.0
    vector_top_k: int = 100
    bm25_top_k: int = 100
    fusion_top_k: int = 60
    rerank_top_k: int = 60
    final_top_k: int = 8
    fusion_method: str = "rrf"
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "satdesign_chunks"
    qdrant_vector_name: str = "dense"
    keyword_provider: str = "local"
    keyword_url: str | None = None
    keyword_index: str = "satdesign_chunks"
    keyword_analyzer: str = "ik_max_word"
    keyword_search_analyzer: str = "ik_smart"
    keyword_username: str | None = None
    keyword_password: str | None = None
    keyword_api_key: str | None = None
    keyword_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "RagConfig":
        load_dotenv()
        return cls(
            enabled=_bool_env("RAG_ENABLED", False),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", cls.embedding_provider),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            embedding_endpoint=os.getenv("RAG_EMBEDDING_ENDPOINT")
            or os.getenv("LITELLM_EMBEDDING_ENDPOINT")
            or os.getenv("OLLAMA_EMBEDDING_URL"),
            embedding_api_key=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("LLM_API_KEY"),
            embedding_auth_header=os.getenv("RAG_EMBEDDING_AUTH_HEADER")
            or os.getenv("LITELLM_AUTH_HEADER")
            or cls.embedding_auth_header,
            embedding_batch_size=_int_env("RAG_EMBEDDING_BATCH_SIZE", cls.embedding_batch_size),
            embedding_timeout=_float_env("RAG_EMBEDDING_TIMEOUT", cls.embedding_timeout),
            embedding_use_fp16=_bool_env("RAG_EMBEDDING_USE_FP16", cls.embedding_use_fp16),
            hash_embedding_dimensions=_int_env("RAG_HASH_EMBEDDING_DIMENSIONS", cls.hash_embedding_dimensions),
            vector_provider=os.getenv("RAG_VECTOR_PROVIDER", cls.vector_provider),
            rerank_provider=os.getenv("RAG_RERANK_PROVIDER", cls.rerank_provider),
            rerank_model=os.getenv("RAG_RERANK_MODEL", cls.rerank_model),
            rerank_endpoint=os.getenv("RAG_RERANK_ENDPOINT")
            or os.getenv("LITELLM_RERANK_ENDPOINT")
            or _base_url_endpoint(os.getenv("LITELLM_BASE_URL"), "rerank"),
            rerank_api_key=os.getenv("RAG_RERANK_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("LLM_API_KEY"),
            rerank_auth_header=os.getenv("RAG_RERANK_AUTH_HEADER")
            or os.getenv("LITELLM_AUTH_HEADER")
            or cls.rerank_auth_header,
            rerank_timeout=_float_env("RAG_RERANK_TIMEOUT", cls.rerank_timeout),
            vector_top_k=_int_env("RAG_VECTOR_TOP_K", cls.vector_top_k),
            bm25_top_k=_int_env("RAG_BM25_TOP_K", cls.bm25_top_k),
            fusion_top_k=_int_env("RAG_FUSION_TOP_K", cls.fusion_top_k),
            rerank_top_k=_int_env("RAG_RERANK_TOP_K", cls.rerank_top_k),
            final_top_k=_int_env("RAG_FINAL_TOP_K", cls.final_top_k),
            fusion_method=os.getenv("RAG_FUSION_METHOD", cls.fusion_method),
            vector_weight=_float_env("RAG_VECTOR_WEIGHT", cls.vector_weight),
            bm25_weight=_float_env("RAG_BM25_WEIGHT", cls.bm25_weight),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_path=os.getenv("QDRANT_PATH"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", cls.qdrant_collection),
            qdrant_vector_name=os.getenv("QDRANT_VECTOR_NAME", cls.qdrant_vector_name),
            keyword_provider=os.getenv("RAG_KEYWORD_PROVIDER", cls.keyword_provider),
            keyword_url=os.getenv("RAG_KEYWORD_URL"),
            keyword_index=os.getenv("RAG_KEYWORD_INDEX", cls.keyword_index),
            keyword_analyzer=os.getenv("RAG_KEYWORD_ANALYZER", cls.keyword_analyzer),
            keyword_search_analyzer=os.getenv("RAG_KEYWORD_SEARCH_ANALYZER", cls.keyword_search_analyzer),
            keyword_username=os.getenv("RAG_KEYWORD_USERNAME"),
            keyword_password=os.getenv("RAG_KEYWORD_PASSWORD"),
            keyword_api_key=os.getenv("RAG_KEYWORD_API_KEY"),
            keyword_timeout=_float_env("RAG_KEYWORD_TIMEOUT", cls.keyword_timeout),
        )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _base_url_endpoint(base_url: str | None, path: str) -> str | None:
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

