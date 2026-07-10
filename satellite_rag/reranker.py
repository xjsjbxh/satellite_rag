"""Reranker adapters for final RAG candidate ordering."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from satellite_rag.schemas import RetrievalResult


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        """Return reranked results."""


class IdentityReranker:
    """No-op reranker for tests and disabled deployments."""

    model_name = "identity"

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(results[:top_k])]


class LexicalOverlapReranker:
    """Dependency-free reranker useful for local tests."""

    model_name = "test/lexical-overlap"

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        query_terms = set(_tokenize(query))
        rescored = []
        for result in results:
            terms = set(_tokenize(result.content))
            overlap = len(query_terms & terms)
            score = float(overlap) + result.score * 0.01
            rescored.append(result.model_copy(update={"score": score, "source": f"{result.source}+rerank"}))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(rescored[:top_k])]


class OpenAICompatibleRerankerClient:
    """Reranker client for LiteLLM or compatible /v1/rerank APIs."""

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        *,
        api_key: str | None = None,
        auth_header: str = "Authorization",
        timeout: float = 120.0,
    ) -> None:
        if not endpoint:
            raise RuntimeError("OpenAI-compatible reranking requires RAG_RERANK_ENDPOINT.")
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.auth_header = auth_header or "Authorization"
        self.timeout = timeout

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        if not results:
            return []
        ranked = self._rerank_remote(query, results, top_k=len(results))
        if not ranked:
            return IdentityReranker().rerank(query, results, top_k=top_k)

        score_by_index: dict[int, float] = {}
        remote_rank_by_index: dict[int, int] = {}
        seen: set[int] = set()
        for remote_rank, (item_index, score) in enumerate(ranked, start=1):
            if item_index < 0 or item_index >= len(results) or item_index in seen:
                continue
            seen.add(item_index)
            score_by_index[item_index] = float(score)
            remote_rank_by_index[item_index] = remote_rank

        if not score_by_index:
            return IdentityReranker().rerank(query, results, top_k=top_k)

        fallback_rerank_score = min(score_by_index.values())
        rerank_scores = [score_by_index.get(index, fallback_rerank_score) for index in range(len(results))]
        retrieval_scores = [item.score for item in results]
        rerank_norms = _minmax_normalize(rerank_scores)
        retrieval_norms = _minmax_normalize(retrieval_scores)
        rerank_weight, retrieval_weight = _score_blend_weights()

        rescored: list[tuple[RetrievalResult, float, float, int, int]] = []
        for item_index, item in enumerate(results):
            rerank_norm = rerank_norms[item_index]
            retrieval_norm = retrieval_norms[item_index]
            blended_score = rerank_weight * rerank_norm + retrieval_weight * retrieval_norm
            remote_rank = remote_rank_by_index.get(item_index, len(results) + 1)
            original_rank = item.rank or item_index + 1
            updated = item.model_copy(
                update={
                    "score": blended_score,
                    "source": f"{item.source}+rerank_fused",
                    "retrieval_score": item.score,
                    "retrieval_score_norm": retrieval_norm,
                    "rerank_score": rerank_scores[item_index],
                    "rerank_score_norm": rerank_norm,
                }
            )
            rescored.append((updated, blended_score, rerank_norm, -remote_rank, -original_rank))

        rescored.sort(key=lambda row: (row[1], row[2], row[3], row[4]), reverse=True)
        return [row[0].model_copy(update={"rank": index + 1}) for index, row in enumerate(rescored[:top_k])]

    def _rerank_remote(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [item.content for item in results],
            "top_n": min(max(1, top_k), len(results)),
            "return_documents": False,
        }
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
        return _parse_rerank_response(data)


class BgeRerankerClient:
    """BAAI BGE cross-encoder reranker based on transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("BGE reranking requires torch and transformers.") from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=_looks_like_local_path(model_name))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=_looks_like_local_path(model_name),
        )
        self._model.to(self._device)
        if use_fp16 and self._device == "cuda":
            self._model.half()
        self._model.eval()

    def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int) -> list[RetrievalResult]:
        if not results:
            return []
        scores = self._score_pairs(query, [item.content for item in results])
        rescored = [
            item.model_copy(update={"score": float(score), "source": f"{item.source}+rerank"})
            for item, score in zip(results, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return [item.model_copy(update={"rank": index + 1}) for index, item in enumerate(rescored[:top_k])]

    def _score_pairs(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        torch = self._torch
        for start in range(0, len(passages), self.batch_size):
            batch_passages = passages[start : start + self.batch_size]
            encoded = self._tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits
                values = torch.sigmoid(logits.reshape(-1)).detach().float().cpu().tolist()
            scores.extend(float(value) for value in values)
        return scores


def _tokenize(text: str) -> list[str]:
    return [part for part in text.lower().replace("\n", " ").split(" ") if part]


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    spread = max_score - min_score
    if abs(spread) < 1e-12:
        return [0.5] * len(scores)
    return [(score - min_score) / spread for score in scores]


def _score_blend_weights() -> tuple[float, float]:
    rerank_weight = _float_env("RAG_RERANK_SCORE_WEIGHT", 0.65)
    retrieval_weight = _float_env("RAG_RETRIEVAL_SCORE_WEIGHT", 0.35)
    total = rerank_weight + retrieval_weight
    if total <= 0:
        return 0.65, 0.35
    return rerank_weight / total, retrieval_weight / total


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _urlopen_json_with_retries(request: urllib.request.Request, *, timeout: float) -> Any:
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


def _parse_rerank_response(data: Any) -> list[tuple[int, float]]:
    rows = data
    if isinstance(data, dict):
        rows = data.get("results") or data.get("data") or data.get("rankings") or []
    if not isinstance(rows, list):
        raise RuntimeError(f"Rerank response did not contain a results list: {type(data).__name__}")

    parsed: list[tuple[int, float]] = []
    for fallback_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        index_value = row.get("index", row.get("document_index", row.get("doc_index", fallback_index)))
        score_value = row.get(
            "relevance_score",
            row.get("score", row.get("rank_score", row.get("similarity", row.get("relevance")))),
        )
        try:
            item_index = int(index_value)
        except (TypeError, ValueError):
            item_index = fallback_index
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = float(len(rows) - fallback_index)
        parsed.append((item_index, score))

    if not parsed:
        raise RuntimeError(f"Rerank response had no parseable result rows: {data}")
    return parsed


def _looks_like_local_path(model_name: str) -> bool:
    return Path(model_name).expanduser().exists()

