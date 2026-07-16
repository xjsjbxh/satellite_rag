"""Query expansion and multi-query retrieval for improved recall.

Provides:
- ``QueryRewriter`` — generates multiple query variants from a single query.
- ``MultiQueryRetriever`` — wraps ``HybridRetriever``, runs each variant, and
  fuses results via RRF.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from satellite_rag.retriever import HybridRetriever, rrf_fuse
from satellite_rag.schemas import RetrievalResult, SearchRequest


# ---------------------------------------------------------------------------
#  Aerospace-domain synonym dictionary  –  free, no LLM needed
# ---------------------------------------------------------------------------

AEROSPACE_SYNONYMS: dict[str, list[str]] = {
    # English → English synonyms / variants
    "acceptance": ["acceptance", "accept"],
    "qualification": ["qualification", "qualify"],
    "verification": ["verification", "verify", "check", "inspection"],
    "test": ["test", "testing", "trial", "examination"],
    "thermal": ["thermal", "temperature", "heat", "thermo"],
    "battery": ["battery", "cell", "batteries"],
    "capacity": ["capacity", "capability", "performance"],
    "outgassing": ["outgassing", "offgassing", "gas emission", "volatile"],
    "materials": ["materials", "material", "substance"],
    "safety": ["safety", "safe", "hazard", "risk"],
    "payload": ["payload", "payloads", "cargo"],
    "hazard": ["hazard", "hazardous", "danger", "risk"],
    "data": ["data", "information", "telemetry", "bus"],
    "bus": ["bus", "data bus", "communication bus", "CAN"],
    "satellite": ["satellite", "satellites", "spacecraft", "space", "space system"],
    "design": ["design", "designing", "architecture"],
    "analysis": ["analysis", "analyses", "simulation", "calculation"],
    "requirement": ["requirement", "requirements", "specification"],
    "control": ["control", "controlling", "regulation"],
    "temperature": ["temperature", "thermal", "temp"],
    "cycling": ["cycling", "cycle", "cycles", "thermal cycling"],
    "vibration": ["vibration", "vibrations", "shake", "mechanical"],
    "vacuum": ["vacuum", "vacuum chamber", "space environment"],
    # Chinese → Chinese synonyms / variants
    "热控": ["热控", "热控制", "温度控制", "热设计", "thermal control"],
    "被动": ["被动", "无源", "passive"],
    "主动": ["主动", "有源", "active"],
    "卫星": ["卫星", "航天器", "航天", "spacecraft"],
    "高温": ["高温", "热", "high temperature", "heat"],
    "低温": ["低温", "冷", "low temperature", "cold"],
    "试验": ["试验", "测试", "实验", "test", "testing"],
    "温度": ["温度", "温度变化", "thermal", "temperature"],
    "总线": ["总线", "数据总线", "通信总线", "CAN", "bus"],
    "数据": ["数据", "信息", "遥测", "data", "telemetry"],
    "材料": ["材料", "材质", "物料", "materials"],
    "出气": ["出气", "放气", "释气", "outgassing", "offgassing"],
    "要求": ["要求", "需求", "规范", "requirement", "specification"],
    "设计": ["设计", "设计方法", "design"],
    "冲击": ["冲击", "冲击试验", "shock", "shock test"],
    "方法": ["方法", "方式", "办法", "method", "approach"],
    "贮存": ["贮存", "存储", "储存", "storage"],
    "工作": ["工作", "运行", "operate", "operation"],
    "装备": ["装备", "设备", "产品", "产品", "equipment"],
}

# Standard ID → full name mapping for cross-lingual expansion
STANDARD_NAME_MAP: dict[str, str] = {
    "ECSS": "ECSS European Cooperation for Space Standardization",
    "GJB": "GJB 国军标 国家军用标准",
    "NASA": "NASA National Aeronautics and Space Administration",
}

STOPWORD_SET = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "shall", "should", "may", "might", "of", "in", "on", "at",
    "to", "for", "from", "with", "by", "about", "into", "through",
    "and", "or", "but", "not", "no", "how", "what", "which", "who",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "your",
})


# ---------------------------------------------------------------------------
#  QueryRewriter
# ---------------------------------------------------------------------------

class QueryRewriter(Protocol):
    """Protocol for query rewriters that produce multiple query variants."""

    def rewrite(self, query: str, **kwargs: Any) -> list[str]:
        """Return a list of query strings to search with (including the original)."""


class SynonymQueryRewriter:
    """Expands a query using a domain synonym dictionary.

    Strategy: for matched terms, **append** synonym variants after the
    original term rather than replacing it.  This increases token coverage
    for both dense (hash) and keyword (BM25) retrievers without losing
    the original signal.
    """

    def __init__(
        self,
        synonym_dict: dict[str, list[str]] | None = None,
        max_variants: int = 5,
    ) -> None:
        self.synonym_dict = synonym_dict or AEROSPACE_SYNONYMS
        self.max_variants = max_variants

    def rewrite(self, query: str, **kwargs: Any) -> list[str]:
        """Generate up to ``max_variants`` query variants.

        Variant 0 is the original query.  Subsequent variants add synonym
        terms for different matched tokens.
        """
        variants: list[str] = [query]

        tokens = self._tokenize(query)
        # Find which tokens we have synonyms for
        matched_indices: list[tuple[int, str, list[str]]] = []
        for i, token in enumerate(tokens):
            lower = token.lower()
            if lower in self.synonym_dict:
                matches = self.synonym_dict[lower]
                if any(m.lower() != lower for m in matches):
                    matched_indices.append((i, token, matches))

        if not matched_indices:
            return variants

        # Generate variants by appending synoyms for different terms
        for idx, original, matches in matched_indices:
            for syn in matches:
                if syn.lower() != original.lower() and syn not in tokens:
                    # Append the synonym after the query rather than replacing
                    variant = f"{query} {syn}"
                    if variant not in variants:
                        variants.append(variant)
                        if len(variants) >= self.max_variants:
                            return variants

        return variants[: self.max_variants]

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer that preserves CJK characters."""
        tokens = []
        for match in re.finditer(r"[A-Za-z0-9._-]+|[一-鿿]", text):
            tokens.append(match.group(0))
        return tokens


class LlmQueryRewriter:
    """Uses an LLM to generate diverse query variants.

    Requires an ``openai``-compatible client.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        num_variants: int = 4,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.num_variants = num_variants

    def rewrite(self, query: str, **kwargs: Any) -> list[str]:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("LlmQueryRewriter requires: pip install openai") from None

        client = OpenAI(
            api_key=self.api_key or "",
            base_url=self.base_url,
        )
        prompt = (
            f"You are an aerospace domain search expert. "
            f"Generate {self.num_variants} search queries that cover different "
            f"aspects, synonyms, and languages (English / Chinese) of the "
            f"original query. Return one query per line, no numbering, no preamble.\n\n"
            f"Original query: {query}"
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        variants = [query]
        for line in (response.choices[0].message.content or "").splitlines():
            line = line.strip().strip('"').strip("'").strip("0123456789.- ")
            if line and line not in variants:
                variants.append(line)
        return variants


# ---------------------------------------------------------------------------
#  SubQueryRewriter  –  decompose compound queries into sub-queries
# ---------------------------------------------------------------------------

class SubQueryRewriter:
    """Decompose a compound query into sub-queries by splitting on conjunctions.

    Also generates overlapping CJK bigram expansions so that sub-queries that
    *are* present in documents have a better chance to surface when using
    token-level (hash) embeddings.

    For production use with BGE-M3, combine this with ``SynonymQueryRewriter``
    or ``LlmQueryRewriter`` for richer query coverage.
    """

    CJK_SPLIT = re.compile(r"[的与和及、,， ]+")
    EN_SPLIT = re.compile(r"\s+(?:and|or|&)\s*|[\s,;]\s*", re.IGNORECASE)

    def rewrite(self, query: str, **kwargs: Any) -> list[str]:
        """Return the original query plus sub-queries and CJK bigram expansions."""
        variants: list[str] = [query]

        # Sub-query split
        if re.search(r"[一-鿿]", query):
            parts = [p for p in self.CJK_SPLIT.split(query) if len(p) >= 2]
        else:
            parts = [p for p in self.EN_SPLIT.split(query) if len(p) >= 2]
        for p in parts:
            if p not in variants:
                variants.append(p)

        # CJK bigram expansion  –  helps token-level matching
        cjk = [c for c in query if "一" <= c <= "鿿"]
        if len(cjk) >= 3:
            bigrams_str = " ".join(a + b for a, b in zip(cjk, cjk[1:]))
            ext = f"{query} {bigrams_str}"
            if ext not in variants:
                variants.append(ext)

        return variants[:6]


# ---------------------------------------------------------------------------
#  MultiQueryRetriever
# ---------------------------------------------------------------------------

class MultiQueryRetriever:
    """Wraps a ``HybridRetriever`` and runs multiple query variants.

    Each variant is searched independently; results are fused via RRF so
    that relevant documents surfaced by *any* variant get a high combined
    rank.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        rewriter: QueryRewriter | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._retriever = retriever
        self._rewriter = rewriter or SynonymQueryRewriter()
        self._rrf_k = rrf_k

    def search(self, request: SearchRequest) -> list[RetrievalResult]:
        variants = self._rewriter.rewrite(request.query)
        all_result_sets: list[list[RetrievalResult]] = []

        for variant in variants:
            variant_request = request.model_copy(update={"query": variant})
            results = self._retriever.search(variant_request)
            if results:
                all_result_sets.append(results)

        if not all_result_sets:
            return []
        if len(all_result_sets) == 1:
            return all_result_sets[0]

        # Determine top_k from the original request (or default)
        total_candidates = max(
            request.final_top_k or 20,
            len(all_result_sets[0]),
        )
        return rrf_fuse(
            all_result_sets,
            top_k=total_candidates,
            k=self._rrf_k,
        )

    @property
    def embedding_client(self):
        return self._retriever.embedding_client

    @embedding_client.setter
    def embedding_client(self, value):
        self._retriever.embedding_client = value

    @property
    def vector_store(self):
        return self._retriever.vector_store

    @property
    def keyword_store(self):
        return self._retriever.keyword_store
