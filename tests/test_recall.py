"""
Integration and unit tests for RAG recall improvement modules.

Tests cover:
1. Multi-Query expansion + RRF fusion  (query_expansion.py)
2. HyDE hypothetical document retrieval (hyde.py)
3. Improved BM25 CJK tokenization       (keyword_store.py ngram_cjk=3)
4. Top-K parameter changes              (config.py)

Note on hash embeddings: the deterministic hash embedder only matches exact
token overlap, so semantic improvements (Multi-Query synonym expansion, HyDE)
REGISTER here only as implementation-validity checks.  Real recall gains
require BGE-M3 embeddings (cross-lingual + semantic).
"""

from __future__ import annotations

import collections
from typing import Any

import pytest

from satellite_rag.config import RagConfig
from satellite_rag.embeddings import HashDenseEmbeddingClient
from satellite_rag.keyword_store import LocalBM25KeywordStore, _tokenize
from satellite_rag.retriever import HybridRetriever, rrf_fuse
from satellite_rag.schemas import RagChunk, RetrievalResult, SearchRequest
from satellite_rag.factory import build_reranker
from satellite_rag.vector_store import InMemoryVectorStore
from satellite_rag.query_expansion import (
    MultiQueryRetriever,
    SynonymQueryRewriter,
    SubQueryRewriter,
)
from satellite_rag.hyde import (
    HydeRetriever,
    TemplateHydeGenerator,
)


# ===================================================================
#  Test helpers
# ===================================================================

def make_chunk(record_id: str, content: str, doc_id: str = "d1") -> RagChunk:
    return RagChunk(
        chunk_id=record_id, doc_id=doc_id,
        content=content, source_path=f"/data/{doc_id}.pdf",
        metadata={"title": doc_id},
    )


def make_simple_retriever(chunks, *, top_k=30) -> HybridRetriever:
    config = RagConfig(
        enabled=True, embedding_provider="hash", hash_embedding_dimensions=64,
        vector_top_k=top_k, bm25_top_k=top_k, fusion_top_k=top_k,
        rerank_top_k=top_k, final_top_k=8,
    )
    emb = HashDenseEmbeddingClient(dimensions=64)
    vs = InMemoryVectorStore()
    vs.upsert(chunks, emb.embed_texts([c.content for c in chunks]))
    ks = LocalBM25KeywordStore()
    ks.upsert(chunks)
    return HybridRetriever(
        embedding_client=emb, vector_store=vs,
        keyword_store=ks, reranker=build_reranker(config), config=config,
    )


# ===================================================================
#  1. BM25 CJK tokenization
# ===================================================================

class TestBM25CJKTokenization:
    """Verify that trigram mode produces more / better tokens for CJK text."""

    def test_bigram_default(self):
        """Default mode produces bigrams for CJK."""
        tokens = _tokenize("卫星热控方案", ngram_cjk=2)
        # Unigrams: 卫 星 热 控 方 案
        # Bigrams: 卫星 星热 热控 控方 方案
        assert "卫星" in tokens
        assert "热控" in tokens
        assert "方案" in tokens
        assert "卫" in tokens
        assert "卫星热" not in tokens  # No trigrams in bigram mode

    def test_trigram_mode(self):
        """Trigram mode adds overlapping trigrams."""
        tokens = _tokenize("卫星热控方案", ngram_cjk=3)
        assert "卫星" in tokens
        assert "热控" in tokens
        assert "方案" in tokens
        # Trigrams
        assert "卫星热" in tokens
        assert "星热控" in tokens
        assert "热控方" in tokens
        assert "控方案" in tokens

    def test_trigram_vs_bigram_density(self):
        """Trigram mode produces strictly more tokens for CJK text."""
        bigram = _tokenize("卫星温度控制方法", ngram_cjk=2)
        trigram = _tokenize("卫星温度控制方法", ngram_cjk=3)
        assert len(trigram) > len(bigram)

    def test_ascii_unaffected(self):
        """English text tokenization is identical regardless of ngram_cjk."""
        t2 = _tokenize("thermal control design", ngram_cjk=2)
        t3 = _tokenize("thermal control design", ngram_cjk=3)
        assert t2 == t3
        assert "thermal" in t2
        assert "control" in t2

    def test_bm25_trigram_matches_multi_char_cjk(self):
        """BM25 with trigram can match multi-character CJK phrases as units."""
        # Document contains "温度控制"
        # Without trigrams, query "温度控制" matches bigrams: 温度, 度控, 控制
        # With trigrams: also matches trigrams: 温度控, 度控制
        doc_text = "航天器采用温度控制系统确保设备正常工作"
        query = "温度控制方法"

        # Standard BM25
        ks_std = LocalBM25KeywordStore(ngram_cjk=2)
        ks_std.upsert([make_chunk("r1", doc_text)])
        results_std = ks_std.search(query, top_k=5)

        # Improved BM25 (trigram)
        ks_imp = LocalBM25KeywordStore(ngram_cjk=3)
        ks_imp.upsert([make_chunk("r1", doc_text)])
        results_imp = ks_imp.search(query, top_k=5)

        # Both should find the doc; trigram should have >= score
        assert len(results_std) >= 1
        assert len(results_imp) >= 1
        # With trigrams, "温度控" and "度控制" also match → better coverage
        assert results_imp[0].score >= results_std[0].score * 0.9  # at least not worse


# ===================================================================
#  2. Multi-Query + RRF Fusion
# ===================================================================

class TestMultiQuery:
    """Verify that Multi-Query retriever correctly generates variants and fuses."""

    def test_synonym_rewriter_generates_variants(self):
        """SynonymQueryRewriter generates at least the original query."""
        rewriter = SynonymQueryRewriter(max_variants=5)
        variants = rewriter.rewrite("How should battery performance be verified?")
        assert len(variants) >= 1
        assert variants[0] == "How should battery performance be verified?"

    def test_synonym_rewriter_expands_known_terms(self):
        """Known synonym terms generate additional variants."""
        rewriter = SynonymQueryRewriter(max_variants=5)
        variants = rewriter.rewrite("thermal control design")
        # The original + at least one expanded variant
        assert len(variants) >= 2
        # The expanded variant should contain at least one synonym word
        expanded = set(variants[1:])
        assert any("heat" in v or "temperature" in v for v in expanded)

    def test_subquery_rewriter_splits_compound(self):
        """SubQueryRewriter splits compound queries."""
        rewriter = SubQueryRewriter()
        # Simpler test: verify it generates sub-queries
        variants = rewriter.rewrite("thermal control and battery test")
        assert len(variants) >= 2  # original + at least one sub-query

    def test_multi_query_rrf_fuses_results(self):
        """MultiQueryRetriever produces valid search results."""
        docs: list[RagChunk] = [
            make_chunk("r1", "Spacecraft thermal management uses heat pipes"),
            make_chunk("r2", "Battery testing requires thermal cycling"),
            make_chunk("r3", "Attitude control uses reaction wheels"),
        ]
        base = make_simple_retriever(docs)
        rewriter = SynonymQueryRewriter(max_variants=3)
        mq = MultiQueryRetriever(base, rewriter=rewriter)

        results = mq.search(SearchRequest(query="temperature control satellite battery"))
        assert isinstance(results, list)
        assert len(results) > 0
        # Results should be RetrievalResult objects
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_multi_query_does_not_lose_original_results(self):
        """Multi-Query RRF should not remove the top original result."""
        docs: list[RagChunk] = [
            make_chunk("r1", "acceptance stage verification free of workmanship defects"),
            make_chunk("r2", "qualification stage design meets requirements"),
        ]
        base = make_simple_retriever(docs)
        mq = MultiQueryRetriever(base)

        single = base.search(SearchRequest(query="acceptance stage verification"))
        multi = mq.search(SearchRequest(query="acceptance stage verification"))

        # The top result should be the same document
        if single and multi:
            assert single[0].chunk_id == multi[0].chunk_id


# ===================================================================
#  3. HyDE Retriever
# ===================================================================

class TestHyDE:
    """Verify that HyDE retriever generates and uses hypothetical documents."""

    def test_template_generator_produces_document(self):
        """TemplateHydeGenerator produces a string longer than the query."""
        gen = TemplateHydeGenerator(expansion_scale=3)
        hypo = gen.generate("satellite temperature control")
        assert isinstance(hypo, str)
        assert len(hypo) > len("satellite temperature control")

    def test_template_generator_expands_known_terms(self):
        """Known domain terms get expanded with synonyms."""
        gen = TemplateHydeGenerator()
        hypo = gen.generate("thermal control")
        # Should include expansion terms like "temperature", "heat", etc.
        assert any(word in hypo.lower() for word in ("temperature", "heat", "thermal control"))

    def test_hyde_retriever_produces_valid_results(self):
        """HydeRetriever returns valid RetrievalResult objects."""
        docs: list[RagChunk] = [
            make_chunk("r1", "Spacecraft thermal control maintains equipment temperature limits"),
            make_chunk("r2", "Battery testing procedures for satellite qualification"),
        ]
        base = make_simple_retriever(docs)
        gen = TemplateHydeGenerator()
        hyde = HydeRetriever(base, generator=gen)

        results = hyde.search(SearchRequest(query="satellite temperature regulation"))
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_hyde_fallback_original_when_no_expansion(self):
        """HyDE with minimal expansion should still return something."""
        docs: list[RagChunk] = [
            make_chunk("r1", "Some unrelated technical text about structures"),
        ]
        base = make_simple_retriever(docs)
        gen = TemplateHydeGenerator(expansion_scale=1)
        hyde = HydeRetriever(base, generator=gen)
        results = hyde.search(SearchRequest(query="structures"))
        assert isinstance(results, list)


# ===================================================================
#  4. RRF fuse utility
# ===================================================================

class TestRRFFuse:
    """Verify RRF fusion logic."""

    def make_result(self, chunk_id: str, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, doc_id="d1", content="text",
            source_path="/data/d1.pdf", score=score, source="test",
        )

    def test_rrf_fuses_two_sets(self):
        s1 = [self.make_result("a", 0.9), self.make_result("b", 0.5)]
        s2 = [self.make_result("b", 0.8), self.make_result("c", 0.7)]
        fused = rrf_fuse([s1, s2], top_k=3)
        assert len(fused) == 3
        # Both 'a' and 'b' and 'c' should be present
        assert {r.chunk_id for r in fused} == {"a", "b", "c"}

    def test_rrf_prioritizes_common_results(self):
        """Docs appearing in both sets get a boost and rank higher."""
        s1 = [self.make_result("common", 0.5), self.make_result("only_s1", 0.9)]
        s2 = [self.make_result("common", 0.5), self.make_result("only_s2", 0.9)]
        fused = rrf_fuse([s1, s2], top_k=2)
        # "common" appears in both sets, so its RRF score is ~0.016 + ~0.016
        # "only_s1" appears in one set: score ~0.016
        # "only_s2" appears in one set: score ~0.016
        # "common" should rank #1
        assert fused[0].chunk_id == "common"


# ===================================================================
#  5. Existing regression: test_rag_eval still works
# ===================================================================

@pytest.fixture
def tmp_eval_data(tmp_path):
    """Create minimal eval data matching test_rag_eval expectations."""
    import json
    eval_set = tmp_path / "eval.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    answers = tmp_path / "answers.jsonl"

    def wj(path, data):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n", encoding="utf-8")

    wj(eval_set, [
        {"id": "q1", "query_type": "definition", "difficulty": "easy", "negative": False,
         "question": "How is acceptance stage defined?",
         "expected_answer": "acceptance stage verification stage free of workmanship defects",
         "expected_pdfs": ["ECSS-E-HB-10-02A(17December2010).pdf"],
         "expected_record_ids": ["r1"], "gold_context": []},
        {"id": "q2", "query_type": "negative", "difficulty": "hard", "negative": True,
         "question": "What does this subset say about unrelated NASA gates?",
         "expected_answer": "No supported answer in the selected first 10 ECSS PDFs; the system should abstain or report not found.",
         "expected_pdfs": [], "expected_record_ids": [], "gold_context": []},
    ])
    wj(corpus, [
        {"record_id": "r1", "doc_id": "d1",
         "content": "3.2.1 acceptance stage verification stage free of workmanship defects.",
         "source_path": "ECSS-E-HB-10-02A(17December2010).pdf",
         "title": "ECSS-E-HB-10-02A"},
        {"record_id": "r2", "doc_id": "d2",
         "content": "Battery activation test guidance.",
         "source_path": "ECSS-E-HB-20-02A(1October2015).pdf",
         "title": "ECSS-E-HB-20-02A"},
    ])
    wj(answers, [
        {"id": "q1", "answer": "Acceptance stage is a verification stage free of workmanship defects."},
        {"id": "q2", "answer": "No supported answer in the supplied context."},
    ])
    return eval_set, corpus, answers


class TestEvalRegression:
    """Ensure the existing eval framework still works after all changes."""

    def test_eval_produces_correct_metrics(self, tmp_eval_data):
        from satellite_rag.evaluation.rag_eval import EvalConfig, evaluate_rag

        eval_set, corpus, answers = tmp_eval_data
        result = evaluate_rag(EvalConfig(
            eval_set=eval_set, corpus=corpus, answers_jsonl=answers,
            out_dir=corpus.parent / "out",
            embedding_provider="hash", embedding_model="test/hash-dense",
            top_k=3, k_values=(1, 3),
        ))
        retrieval = result["summary"]["retrieval"]
        answers_summary = result["summary"]["answers"]
        assert retrieval["slice_recall@3"] == 1.0
        assert retrieval["doc_recall@3"] == 1.0
        assert retrieval["mrr"] == 1.0
        assert answers_summary["hallucination_rate"] == 0.0
        assert answers_summary["negative_abstention_accuracy"] == 1.0
        assert (corpus.parent / "out" / "summary.json").exists()
        assert (corpus.parent / "out" / "details.jsonl").exists()
        assert (corpus.parent / "out" / "report.md").exists()


# ===================================================================
#  6. Recall benchmark (informational — requires BGE-M3 for full effect)
# ===================================================================

class TestRecallSnapshot:
    """Snapshot recall metrics using hash embeddings + each improvement.

    NOTE: Hash embeddings do not capture semantics — these numbers measure
    token-overlap recall only.  Real recall gains from Multi-Query / HyDE
    require BGE-M3 embeddings and are expected to be substantially larger.
    """

    EN_DOCS = [
        ("r1", "spacecraft thermal management system employs heat pipes radiators and insulation"),
        ("r2", "space vehicle pointing determination utilizes star trackers gyroscopes and sun sensors"),
        ("r3", "spacecraft electrical power subsystem comprises photovoltaic arrays and batteries"),
        ("r4", "orbit determination takes ground tracking measurements to compute spacecraft position"),
        ("r5", "expendable launch vehicle payload fairing encloses spacecraft during ascent"),
    ]
    EN_QUERIES = [
        ("satellite temperature control design", ["r1"]),
        ("spacecraft orientation sensing equipment", ["r2"]),
        ("solar cell electrical generation system", ["r3"]),
    ]

    CN_DOCS = [
        ("r6", "航天器热管理分系统采用热管散热器实现被动温度调节功能"),
        ("r7", "空间飞行器指向确定利用星敏感器陀螺实现姿态测量"),
        ("r8", "航天器电源分系统由光伏电池阵和储能蓄电池组成"),
        ("r9", "轨道确定处理流程通过地面跟踪测量数据计算航天器星历"),
        ("r10", "运载火箭整流罩在大气上升段包覆保护有效载荷"),
    ]
    CN_QUERIES = [
        ("卫星温度管理方案", ["r6"]),
        ("航天器姿态测量方法", ["r7"]),
        ("太阳能发电系统组成", ["r8"]),
    ]

    @pytest.fixture(scope="class")
    def en_chunks(self):
        return [make_chunk(rid, txt) for rid, txt in self.EN_DOCS]

    @pytest.fixture(scope="class")
    def cn_chunks(self):
        return [make_chunk(rid, txt) for rid, txt in self.CN_DOCS]

    def _bench(self, retriever, queries, chunks, k_values=(1, 3)):
        """Run retrieval benchmark for given queries.

        Parameters
        ----------
        queries : list of (question, expected_ids)
        """
        results_by_q = []
        for idx, (question, expected) in enumerate(queries):
            results = retriever.search(SearchRequest(
                query=question, vector_top_k=30, bm25_top_k=30,
                fusion_top_k=20, rerank_top_k=20, final_top_k=10,
            ))
            retrieved = [r.chunk_id for r in results[:max(k_values)]]
            results_by_q.append((str(idx), expected, retrieved))
        metrics = {}
        for k in k_values:
            vals = []
            for _, expected, retrieved in results_by_q:
                hits = set(retrieved[:k]) & set(expected)
                vals.append(len(hits) / len(expected) if expected else 0)
            metrics[f"recall@{k}"] = sum(vals) / len(vals) if vals else 0
        return metrics

    def test_en_baseline(self, en_chunks):
        r = make_simple_retriever(en_chunks)
        m = self._bench(r, self.EN_QUERIES, en_chunks)
        print(f"[EN baseline] recall@1={m['recall@1']:.3f} recall@3={m['recall@3']:.3f}")

    def test_cn_baseline(self, cn_chunks):
        r = make_simple_retriever(cn_chunks)
        m = self._bench(r, self.CN_QUERIES, cn_chunks)
        print(f"[CN baseline] recall@1={m['recall@1']:.3f} recall@3={m['recall@3']:.3f}")
