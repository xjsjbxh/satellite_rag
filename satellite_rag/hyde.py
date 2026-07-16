"""HyDE (Hypothetical Document Embeddings) for improved recall.

The HyDE approach:
1. Generate a "hypothetical document" that would answer the query
2. Embed that hypothetical document (not the query itself)
3. Retrieve documents similar to the hypothetical one

This bridges the vocabulary gap between short queries and full-length documents.
When an LLM is unavailable, a template-based fallback expands the query into a
document-like text.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from satellite_rag.retriever import HybridRetriever, rrf_fuse
from satellite_rag.schemas import RetrievalResult, SearchRequest


# ---------------------------------------------------------------------------
#  Hypothetical document generators
# ---------------------------------------------------------------------------

class HydeGenerator(Protocol):
    """Generates a hypothetical document from a query."""

    def generate(self, query: str, **kwargs: Any) -> str:
        """Return a hypothetical document string that would answer the query."""


class TemplateHydeGenerator:
    """Template-based HyDE generator — no LLM needed.

    Takes the query and wraps it in a document-like template, expanding key
    terms with domain vocabulary.  This is a weak approximation of real HyDE
    but is useful for integration testing.
    """

    DOMAIN_EXPANSION: dict[str, list[str]] = {
        "thermal": ["temperature", "heat", "thermal control", "cooling", "heating"],
        "battery": ["cell", "power", "capacity", "energy storage", "charge discharge"],
        "test": ["verification", "qualification", "inspection", "measurement", "analysis"],
        "safety": ["hazard", "risk", "protection", "fault", "reliability"],
        "data": ["telemetry", "information", "signal", "communication", "transmission"],
        "satellite": ["spacecraft", "space vehicle", "orbit", "platform", "space system"],
        "design": ["architecture", "configuration", "layout", "specification", "requirement"],
    }

    def __init__(self, expansion_scale: int = 3) -> None:
        self.expansion_scale = max(1, expansion_scale)

    def generate(self, query: str, **kwargs: Any) -> str:
        """Generate a hypothetical document by repeating and expanding the query."""
        expanded: list[str] = [query]

        # Add domain expansions for matched terms
        tokens = re.findall(r"[A-Za-z]+|[一-鿿]", query.lower())
        matched_expansions: list[str] = []
        for token in tokens:
            if token in self.DOMAIN_EXPANSION:
                matched_expansions.extend(self.DOMAIN_EXPANSION[token])

        if matched_expansions:
            expanded.append(" ".join(matched_expansions[:6]))

        # Repeat the query with different phrasings
        for i in range(self.expansion_scale - 1):
            expanded.append(query)

        hypo_doc = ". ".join(expanded)
        # Pad to a reasonable document length to improve embedding quality
        while len(hypo_doc) < 200:
            hypo_doc = f"{hypo_doc} {query}"
        return hypo_doc


class LlmHydeGenerator:
    """Uses an LLM to generate a realistic hypothetical document.

    Requires an ``openai``-compatible client.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 300,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, **kwargs: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("LlmHydeGenerator requires: pip install openai") from None

        client = OpenAI(api_key=self.api_key or "", base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an aerospace engineer writing a technical document. "
                        "Given a search query, write a concise paragraph (100-200 words) "
                        "that a technical document about this topic would contain. "
                        "Use formal technical language with specific terminology. "
                        "Do not mention that this is a hypothetical document."
                    ),
                },
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
        return (response.choices[0].message.content or query).strip()


# ---------------------------------------------------------------------------
#  HyDE retriever
# ---------------------------------------------------------------------------

class HydeRetriever:
    """Wraps a ``HybridRetriever`` with HyDE-based retrieval.

    Strategy:
    1. Generate a hypothetical document from the query
    2. Embed the hypothetical document for vector search (the "HyDE" step)
    3. Also run BM25 with the original query
    4. Fuse both result sets via RRF

    This gives the best of both worlds: HyDE for dense vector matching *and*
    the original query for keyword matching.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: HydeGenerator | None = None,
        rrf_k: int = 60,
        hyde_weight: float = 1.0,
        original_weight: float = 1.0,
    ) -> None:
        self._retriever = retriever
        self._generator = generator or TemplateHydeGenerator()
        self._rrf_k = rrf_k
        self._hyde_weight = hyde_weight
        self._original_weight = original_weight

    def search(self, request: SearchRequest) -> list[RetrievalResult]:
        """Search using HyDE + original query, then RRF fuse."""
        hypo_doc = self._generator.generate(request.query)

        # --- Run HyDE vector search (use the hypothetical document as query) ---
        hyde_request = SearchRequest(
            query=hypo_doc,
            vector_top_k=request.vector_top_k,
            bm25_top_k=1,            # minimal BM25 for the HyDE path
            fusion_top_k=request.fusion_top_k,
            rerank_top_k=request.rerank_top_k,
            final_top_k=request.final_top_k,
        )
        hyde_results = self._retriever.search(hyde_request)
        for r in hyde_results:
            r.source = f"{r.source}+hyde"

        # --- Run original query BM25 search ---
        original_request = SearchRequest(
            query=request.query,
            vector_top_k=1,                       # minimal vector for original
            bm25_top_k=request.bm25_top_k,
            fusion_top_k=request.fusion_top_k,
            rerank_top_k=request.rerank_top_k,
            final_top_k=request.final_top_k,
        )
        original_results = self._retriever.search(original_request)

        # --- RRF fuse ---
        all_sets: list[list[RetrievalResult]] = []
        if hyde_results:
            all_sets.append(hyde_results)
        if original_results:
            all_sets.append(original_results)

        if not all_sets:
            return []
        if len(all_sets) == 1:
            return all_sets[0]

        total = max(len(s) for s in all_sets)
        return rrf_fuse(
            all_sets,
            top_k=total,
            k=self._rrf_k,
            weights=[self._hyde_weight, self._original_weight],
        )
