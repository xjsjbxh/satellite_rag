"""Dense embedding clients.

The production target is BAAI/bge-m3. A deterministic hash embedder is kept for
tests and offline demos where heavy model dependencies are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Protocol


class DenseEmbeddingClient(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""

    def embed_query(self, text: str) -> list[float]:
        """Return a dense query vector."""


class HashDenseEmbeddingClient:
    """Small deterministic embedder used for local tests.

    It is not a semantic model. It only gives stable vectors so the rest of the
    retrieval pipeline can be tested without downloading BGE-M3.
    """

    def __init__(self, dimensions: int = 64, model_name: str = "test/hash-dense") -> None:
        self.dimensions = dimensions
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0 for _ in range(self.dimensions)]
        for token in _tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BgeM3EmbeddingClient:
    """BGE-M3 dense embedding adapter based on FlagEmbedding.

    Install optional dependency before using:
        pip install FlagEmbedding
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("FlagEmbedding is required for BAAI/bge-m3 embeddings.") from exc
        self.model_name = model_name
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        encoded = self._model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vectors = encoded["dense_vecs"]
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OllamaEmbeddingClient:
    """Ollama embedding adapter for local OpenAI-compatible deployments."""

    def __init__(
        self,
        model_name: str = "bge-m3",
        endpoint: str | None = None,
        timeout: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint or _default_ollama_embedding_endpoint()
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts}
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        data = _urlopen_json_with_retries(request, timeout=self.timeout)

        if isinstance(data.get("embeddings"), list):
            embeddings = data["embeddings"]
            return [list(map(float, vector)) for vector in embeddings]

        # OpenAI-compatible /v1/embeddings shape.
        if isinstance(data.get("data"), list):
            ordered = sorted(data["data"], key=lambda item: int(item.get("index", 0)))
            return [list(map(float, item["embedding"])) for item in ordered]

        # Legacy Ollama /api/embeddings shape for a single input.
        if isinstance(data.get("embedding"), list):
            return [list(map(float, data["embedding"]))]

        raise RuntimeError(f"Ollama embedding response did not contain embeddings: {data.keys()}")


class OpenAICompatibleEmbeddingClient:
    """Embedding client for LiteLLM or any OpenAI-compatible /v1/embeddings API."""

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        *,
        api_key: str | None = None,
        auth_header: str = "Authorization",
        timeout: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        if not endpoint:
            raise RuntimeError("OpenAI-compatible embeddings require RAG_EMBEDDING_ENDPOINT.")
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.auth_header = auth_header or "Authorization"
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            if self.auth_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        data = _urlopen_json_with_retries(request, timeout=self.timeout)
        if not isinstance(data.get("data"), list):
            raise RuntimeError(f"Embedding response did not contain OpenAI-compatible data: {data.keys()}")
        ordered = sorted(data["data"], key=lambda item: int(item.get("index", 0)))
        return [list(map(float, item["embedding"])) for item in ordered]


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    current = []
    tokens = []
    for char in normalized:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _default_ollama_embedding_endpoint() -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host}/api/embed"


def _urlopen_json_with_retries(request: urllib.request.Request, *, timeout: float) -> dict:
    attempts = max(1, _int_env("RAG_HTTP_RETRIES", 4))
    delay = max(0.0, _float_env("RAG_HTTP_RETRY_DELAY", 1.0))
    retry_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in retry_statuses and attempt < attempts:
                time.sleep(delay * attempt)
                continue
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {request.full_url} failed with HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, OSError) as exc:
            if attempt < attempts:
                time.sleep(delay * attempt)
                continue
            raise RuntimeError(f"POST {request.full_url} failed: {exc}") from exc
    raise RuntimeError(f"POST {request.full_url} failed after {attempts} attempts.")


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

