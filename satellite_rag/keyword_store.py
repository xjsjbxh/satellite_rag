"""BM25 keyword retrieval adapters.

For production, configure Elasticsearch/OpenSearch with IK Analyzer on the
content field. The local BM25 store is a dependency-free fallback for tests.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Protocol

from satellite_rag.schemas import RagChunk, RetrievalResult


class KeywordStore(Protocol):
    def upsert(self, chunks: list[RagChunk]) -> None:
        """Persist chunks for keyword search."""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run BM25-like keyword retrieval."""


class LocalBM25KeywordStore:
    """Small BM25 implementation for tests and local demos."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, RagChunk] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_freqs: Counter[str] = Counter()
        self._lengths: dict[str, int] = {}

    def upsert(self, chunks: list[RagChunk]) -> None:
        for chunk in chunks:
            if chunk.chunk_id in self._term_freqs:
                for token in self._term_freqs[chunk.chunk_id]:
                    self._doc_freqs[token] -= 1
            tokens = _tokenize(chunk.content)
            terms = Counter(tokens)
            self._chunks[chunk.chunk_id] = chunk
            self._term_freqs[chunk.chunk_id] = terms
            self._lengths[chunk.chunk_id] = len(tokens)
            for token in terms:
                self._doc_freqs[token] += 1

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        avg_len = (sum(self._lengths.values()) / len(self._lengths)) if self._lengths else 0.0
        scores = []
        for chunk_id, chunk in self._chunks.items():
            if not _metadata_matches(chunk.metadata, metadata_filter or {}):
                continue
            score = sum(self._score_term(term, chunk_id, avg_len) for term in query_terms)
            if score > 0:
                scores.append((score, chunk))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                source_path=chunk.source_path,
                metadata=chunk.metadata,
                score=float(score),
                rank=index + 1,
                source="bm25",
            )
            for index, (score, chunk) in enumerate(scores[:top_k])
        ]

    def _score_term(self, term: str, chunk_id: str, avg_len: float) -> float:
        freq = self._term_freqs[chunk_id].get(term, 0)
        if freq == 0:
            return 0.0
        total_docs = max(1, len(self._chunks))
        doc_freq = max(1, self._doc_freqs.get(term, 0))
        idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        doc_len = self._lengths.get(chunk_id, 0)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (avg_len or 1.0))
        return idf * (freq * (self.k1 + 1)) / denominator


class ElasticsearchKeywordStore:
    """Elasticsearch/OpenSearch BM25 client.

    Use an index configured with IK Analyzer for Chinese keyword recall.
    """

    def __init__(
        self,
        *,
        url: str,
        index: str,
        analyzer: str = "ik_max_word",
        search_analyzer: str = "ik_smart",
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.index = index
        self.analyzer = analyzer
        self.search_analyzer = search_analyzer
        self.username = username
        self.password = password
        self.api_key = api_key
        self.timeout = timeout

    def ping(self) -> bool:
        try:
            self._request("GET", "/")
            return True
        except RuntimeError:
            return False

    def index_exists(self) -> bool:
        request = urllib.request.Request(f"{self.url}/{self.index}", method="HEAD", headers=self._headers())
        try:
            urllib.request.urlopen(request, timeout=self.timeout).read()
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise RuntimeError(f"keyword index check failed: {exc.code}: {exc.read().decode('utf-8', 'ignore')}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"keyword service is unreachable: {exc}") from exc

    def create_index(self, *, if_not_exists: bool = True) -> dict[str, Any]:
        if if_not_exists and self.index_exists():
            return {"acknowledged": True, "already_exists": True, "index": self.index}
        return self._request("PUT", f"/{self.index}", self.index_settings_payload())

    def refresh(self) -> dict[str, Any]:
        return self._request("POST", f"/{self.index}/_refresh")

    def upsert(self, chunks: list[RagChunk]) -> None:
        for chunk in chunks:
            self._request("PUT", f"/{self.index}/_doc/{chunk.chunk_id}", chunk.model_dump())

    def search(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        must = [{"match": {"content": query}}]
        for key, value in (metadata_filter or {}).items():
            if isinstance(value, (list, tuple, set)):
                values = list(value)
                field = f"metadata.{key}.keyword" if any(isinstance(item, str) for item in values) else f"metadata.{key}"
                must.append({"terms": {field: values}})
            else:
                field = f"metadata.{key}.keyword" if isinstance(value, str) else f"metadata.{key}"
                must.append({"term": {field: value}})
        payload = {"size": top_k, "query": {"bool": {"must": must}}}
        response = self._request("POST", f"/{self.index}/_search", payload)
        results = []
        for index, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = hit.get("_source", {})
            results.append(
                RetrievalResult(
                    chunk_id=str(source.get("chunk_id") or hit.get("_id")),
                    doc_id=str(source.get("doc_id") or ""),
                    content=str(source.get("content") or ""),
                    source_path=str(source.get("source_path") or ""),
                    metadata=dict(source.get("metadata") or {}),
                    score=float(hit.get("_score") or 0.0),
                    rank=index,
                    source="bm25",
                )
            )
        return results

    def index_settings_payload(self) -> dict[str, Any]:
        return {
            "settings": {
                "analysis": {},
            },
            "mappings": {
                "dynamic": True,
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source_path": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": self.analyzer,
                        "search_analyzer": self.search_analyzer,
                    },
                    "metadata": {"type": "object", "enabled": True},
                },
            },
        }

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = self._headers()
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.url}{path}", data=body, method=method, headers=headers)
        try:
            raw = urllib.request.urlopen(request, timeout=self.timeout).read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(f"keyword request failed: {method} {path} -> {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"keyword service is unreachable: {exc}") from exc
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        elif self.username and self.password:
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "."}):
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current.clear()
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
    if current:
        tokens.append("".join(current))
    # Add simple CJK bigrams to improve local fallback recall.
    cjk_chars = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return tokens


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

