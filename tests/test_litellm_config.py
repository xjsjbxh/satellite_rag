from __future__ import annotations

from satellite_rag.config import RagConfig
from satellite_rag.embeddings import OpenAICompatibleEmbeddingClient
from satellite_rag.factory import build_embedding_client


def test_litellm_embedding_provider_uses_openai_compatible_client(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "litellm")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "company-embedding")
    monkeypatch.setenv("RAG_EMBEDDING_ENDPOINT", "http://api.example.test/v1/embeddings")
    monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "secret")
    monkeypatch.setenv("RAG_EMBEDDING_AUTH_HEADER", "x-litellm-api-key")

    client = build_embedding_client(RagConfig.from_env())

    assert isinstance(client, OpenAICompatibleEmbeddingClient)
    assert client.model_name == "company-embedding"
    assert client.endpoint == "http://api.example.test/v1/embeddings"
    assert client.api_key == "secret"
    assert client.auth_header == "x-litellm-api-key"
