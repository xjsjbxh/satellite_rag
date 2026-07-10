"""Automated RAG evaluation for ECSS retrieval sets.

The retrieval metrics are ID-based when the eval set provides gold
``expected_record_ids``. Answer groundedness is deterministic: it checks whether
answer sentences are supported by retrieved evidence text.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from satellite_rag import RagConfig, SearchRequest, build_embedding_client, build_reranker
from satellite_rag.citations import format_retrieval_citation
from satellite_rag.keyword_store import LocalBM25KeywordStore
from satellite_rag.query_routing import metadata_filter_from_query
from satellite_rag.retriever import HybridRetriever
from satellite_rag.schemas import RagChunk, RetrievalResult
from satellite_rag.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore


DEFAULT_K_VALUES = (1, 3, 5, 10)
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}
ABSTAIN_PATTERNS = (
    "not found",
    "no supported answer",
    "insufficient information",
    "not provided",
    "cannot determine",
    "not in the selected",
    "not in these",
    "无法找到",
    "没有找到",
    "资料中没有",
    "无法确定",
)


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    expected_answer: str
    expected_pdfs: list[str]
    expected_record_ids: list[str]
    query_type: str = "unknown"
    difficulty: str = "unknown"
    negative: bool = False
    gold_context: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EvalConfig:
    eval_set: Path
    out_dir: Path
    corpus: Path | None = None
    answers_jsonl: Path | None = None
    qdrant_url: str | None = None
    qdrant_path: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "satdesign_chunks"
    vector_name: str = "dense"
    embedding_provider: str = "hash"
    embedding_model: str = "test/hash-dense"
    embedding_endpoint: str | None = None
    embedding_api_key: str | None = None
    embedding_auth_header: str = "Authorization"
    embedding_batch_size: int = 16
    embedding_timeout: float = 120.0
    hash_dimensions: int = 64
    top_k: int = 10
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    limit: int | None = None
    corpus_limit: int | None = None
    restrict_corpus_to_eval_pdfs: bool = True
    route_query_metadata: bool = False
    vector_overfetch: int = 200
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    rerank_provider: str = "identity"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_endpoint: str | None = None
    rerank_api_key: str | None = None
    rerank_auth_header: str = "Authorization"
    rerank_timeout: float = 120.0
    rerank_top_k: int | None = None
    generate_answers: bool = False
    answer_model: str | None = None
    answer_base_url: str | None = None
    answer_api_key: str | None = None
    answer_auth_header: str | None = None
    answer_max_tokens: int = 900
    answer_temperature: float = 0.0
    support_threshold: float = 0.55


def evaluate_rag(config: EvalConfig) -> dict[str, Any]:
    started_at = time.time()
    items = load_eval_items(config.eval_set)
    if config.limit is not None:
        items = items[: config.limit]
    if not items:
        raise ValueError(f"eval set is empty: {config.eval_set}")

    answers_by_id = load_answers(config.answers_jsonl) if config.answers_jsonl else {}
    retriever, corpus_chunks = build_eval_retriever(config, items)
    allowed_chunk_ids = {chunk.chunk_id for chunk in corpus_chunks} if corpus_chunks and config.restrict_corpus_to_eval_pdfs else None
    max_k = max(config.top_k, *(config.k_values or DEFAULT_K_VALUES))
    rerank_candidate_k = config.rerank_top_k or config.vector_overfetch
    candidate_k = max(max_k, config.vector_overfetch, rerank_candidate_k)

    rows = []
    for item in items:
        metadata_filter = metadata_filter_from_query(item.question) if config.route_query_metadata else {}
        raw_results = retriever.search(
            SearchRequest(
                query=item.question,
                metadata_filter=metadata_filter,
                vector_top_k=candidate_k,
                bm25_top_k=candidate_k,
                fusion_top_k=candidate_k,
                rerank_top_k=max(max_k, rerank_candidate_k),
                final_top_k=candidate_k,
            )
        )
        results = post_filter_results(raw_results, allowed_chunk_ids=allowed_chunk_ids, top_k=max_k)
        answer = answers_by_id.get(item.id)
        if config.generate_answers:
            answer = generate_grounded_answer(config, item, results)
        rows.append(score_item(item, results, answer=answer, k_values=config.k_values, support_threshold=config.support_threshold))

    summary = summarize_rows(rows, k_values=config.k_values)
    result = {
        "schema_version": "satdesign.rag_eval.v1",
        "eval_set": str(config.eval_set),
        "corpus": str(config.corpus) if config.corpus else None,
        "items": len(items),
        "config": {
            "top_k": config.top_k,
            "k_values": list(config.k_values),
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "qdrant_url": config.qdrant_url,
            "qdrant_path": config.qdrant_path,
            "collection": config.collection,
            "vector_name": config.vector_name,
            "restrict_corpus_to_eval_pdfs": config.restrict_corpus_to_eval_pdfs,
            "route_query_metadata": config.route_query_metadata,
            "vector_weight": config.vector_weight,
            "bm25_weight": config.bm25_weight,
            "rerank_provider": config.rerank_provider,
            "rerank_model": config.rerank_model,
            "rerank_endpoint": config.rerank_endpoint,
            "rerank_top_k": config.rerank_top_k,
            "generate_answers": config.generate_answers,
            "answers_jsonl": str(config.answers_jsonl) if config.answers_jsonl else None,
        },
        "summary": summary,
        "rows": rows,
        "duration_seconds": round(time.time() - started_at, 3),
    }
    write_eval_outputs(config.out_dir, result)
    return result


def load_eval_items(path: Path) -> list[EvalItem]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [_item_from_mapping(row) for row in csv.DictReader(handle)]
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            items.append(_item_from_mapping(payload))
    return items


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            item_id = str(payload.get("id") or payload.get("eval_id") or "").strip()
            answer = str(payload.get("answer") or payload.get("generated_answer") or "").strip()
            if not item_id:
                raise ValueError(f"{path}:{line_number}: answer row is missing id")
            answers[item_id] = answer
    return answers


def build_eval_retriever(config: EvalConfig, items: list[EvalItem]) -> tuple[HybridRetriever, list[RagChunk]]:
    allowed_pdfs = {pdf for item in items for pdf in item.expected_pdfs} if config.restrict_corpus_to_eval_pdfs else set()
    chunks = load_corpus_chunks(config.corpus, allowed_pdfs=allowed_pdfs, limit=config.corpus_limit) if config.corpus else []

    rag_config = RagConfig(
        enabled=True,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_endpoint=config.embedding_endpoint,
        embedding_api_key=config.embedding_api_key,
        embedding_auth_header=config.embedding_auth_header,
        embedding_batch_size=config.embedding_batch_size,
        embedding_timeout=config.embedding_timeout,
        hash_embedding_dimensions=config.hash_dimensions,
        vector_provider="qdrant" if config.qdrant_url or config.qdrant_path else "memory",
        qdrant_url=config.qdrant_url,
        qdrant_path=config.qdrant_path,
        qdrant_api_key=config.qdrant_api_key,
        qdrant_collection=config.collection,
        qdrant_vector_name=config.vector_name,
        keyword_provider="local",
        final_top_k=config.top_k,
        vector_top_k=config.vector_overfetch,
        bm25_top_k=config.vector_overfetch,
        fusion_top_k=config.vector_overfetch,
        vector_weight=config.vector_weight,
        bm25_weight=config.bm25_weight,
        rerank_provider=config.rerank_provider,
        rerank_model=config.rerank_model,
        rerank_endpoint=config.rerank_endpoint,
        rerank_api_key=config.rerank_api_key,
        rerank_auth_header=config.rerank_auth_header,
        rerank_timeout=config.rerank_timeout,
        rerank_top_k=config.rerank_top_k or config.vector_overfetch,
    )
    embedding = build_embedding_client(rag_config)
    keyword_store = LocalBM25KeywordStore()
    if chunks:
        keyword_store.upsert(chunks)

    if config.qdrant_url or config.qdrant_path:
        vector_store: VectorStore = QdrantVectorStore(
            url=config.qdrant_url,
            path=config.qdrant_path,
            api_key=config.qdrant_api_key,
            collection=config.collection,
            vector_name=config.vector_name,
        )
    else:
        vector_store = InMemoryVectorStore()
        if chunks:
            vectors = embedding.embed_texts([chunk.content for chunk in chunks])
            vector_store.upsert(chunks, vectors)

    retriever = HybridRetriever(
        embedding_client=embedding,
        vector_store=vector_store,
        keyword_store=keyword_store,
        reranker=build_reranker(rag_config),
        config=rag_config,
    )
    return retriever, chunks


def load_corpus_chunks(path: Path, *, allowed_pdfs: set[str] | None = None, limit: int | None = None) -> list[RagChunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if allowed_pdfs and source_pdf_name(record) not in allowed_pdfs:
                continue
            chunks.append(corpus_record_to_chunk(record))
            if limit is not None and len(chunks) >= limit:
                break
    return chunks


def corpus_record_to_chunk(record: dict[str, Any]) -> RagChunk:
    metadata = dict(record.get("metadata") or {})
    for key in (
        "title",
        "source_type",
        "authority_level",
        "record_kind",
        "relative_path",
        "page",
        "page_start",
        "page_end",
        "section",
        "section_title",
        "standard_id",
        "part_index",
        "schema_version",
        "generated_at",
    ):
        value = record.get(key)
        if value is not None:
            metadata.setdefault(key, value)
    record_id = str(record.get("record_id") or record.get("chunk_id") or "")
    if record_id:
        metadata.setdefault("record_id", record_id)
    return RagChunk(
        chunk_id=record_id or str(record["doc_id"]),
        doc_id=str(record["doc_id"]),
        content=str(record.get("content") or ""),
        source_path=str(record.get("source_path") or ""),
        metadata=metadata,
    )


def post_filter_results(
    results: list[RetrievalResult],
    *,
    allowed_chunk_ids: set[str] | None,
    top_k: int,
) -> list[RetrievalResult]:
    filtered = []
    seen = set()
    for result in results:
        if allowed_chunk_ids is not None and result.chunk_id not in allowed_chunk_ids:
            continue
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        filtered.append(result.model_copy(update={"rank": len(filtered) + 1}))
        if len(filtered) >= top_k:
            break
    return filtered


def score_item(
    item: EvalItem,
    results: list[RetrievalResult],
    *,
    answer: str | None,
    k_values: tuple[int, ...],
    support_threshold: float,
) -> dict[str, Any]:
    expected_ids = set(item.expected_record_ids)
    expected_pdfs = set(item.expected_pdfs)
    retrieved_ids = [result_id(result) for result in results]
    retrieved_pdfs = [result_pdf_name(result) for result in results]

    row: dict[str, Any] = {
        "id": item.id,
        "query_type": item.query_type,
        "difficulty": item.difficulty,
        "negative": item.negative,
        "question": item.question,
        "expected_record_ids": item.expected_record_ids,
        "expected_pdfs": item.expected_pdfs,
        "top_results": [result_to_row(result) for result in results],
    }
    for k in k_values:
        top_ids = set(retrieved_ids[:k])
        top_pdfs = set(retrieved_pdfs[:k])
        row[f"slice_recall@{k}"] = recall_fraction(expected_ids, top_ids) if expected_ids else None
        row[f"slice_all_hit@{k}"] = bool(expected_ids and expected_ids.issubset(top_ids))
        row[f"doc_recall@{k}"] = recall_fraction(expected_pdfs, top_pdfs) if expected_pdfs else None
        row[f"doc_all_hit@{k}"] = bool(expected_pdfs and expected_pdfs.issubset(top_pdfs))
        row[f"source_precision@{k}"] = precision_fraction(expected_pdfs, retrieved_pdfs[:k]) if expected_pdfs else None
        row[f"gold_answer_coverage@{k}"] = answer_coverage(item.expected_answer, "\n".join(r.content for r in results[:k])) if not item.negative else None

    row["first_gold_rank"] = first_hit_rank(retrieved_ids, expected_ids)
    row["mrr"] = (1.0 / row["first_gold_rank"]) if row["first_gold_rank"] else 0.0
    row["answer"] = answer
    if answer is not None:
        row.update(score_answer(item, answer, results, support_threshold=support_threshold))
    return row


def score_answer(
    item: EvalItem,
    answer: str,
    results: list[RetrievalResult],
    *,
    support_threshold: float,
) -> dict[str, Any]:
    evidence_text = "\n".join(result.content for result in results)
    sentences = [sentence for sentence in split_sentences(answer) if meaningful_sentence(sentence)]
    support_rows = []
    unsupported = 0
    for sentence in sentences:
        score = support_score(sentence, evidence_text)
        critical_ok = critical_terms_supported(sentence, evidence_text)
        supported = score >= support_threshold and critical_ok
        if not supported and not is_abstention(answer):
            unsupported += 1
        support_rows.append(
            {
                "sentence": sentence,
                "support_score": round(score, 4),
                "critical_terms_supported": critical_ok,
                "supported": supported,
            }
        )

    factual_count = len(sentences)
    hallucination_rate = (unsupported / factual_count) if factual_count else 0.0
    abstained = is_abstention(answer)
    return {
        "answer_token_f1": token_f1(answer, item.expected_answer) if not item.negative else None,
        "answer_expected_recall": answer_coverage(item.expected_answer, answer) if not item.negative else None,
        "abstained": abstained,
        "negative_abstention_correct": abstained if item.negative else None,
        "answer_sentence_count": factual_count,
        "unsupported_sentence_count": unsupported,
        "hallucination_rate": hallucination_rate,
        "grounded_sentence_rate": 1.0 - hallucination_rate if factual_count else 1.0,
        "sentence_support": support_rows,
    }


def summarize_rows(rows: list[dict[str, Any]], *, k_values: tuple[int, ...]) -> dict[str, Any]:
    positive = [row for row in rows if not row["negative"]]
    negative = [row for row in rows if row["negative"]]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row["query_type"]), []).append(row)

    retrieval: dict[str, Any] = {"positive_items": len(positive), "negative_items": len(negative)}
    for k in k_values:
        retrieval[f"slice_recall@{k}"] = mean_defined(row.get(f"slice_recall@{k}") for row in positive)
        retrieval[f"slice_all_hit_rate@{k}"] = mean_bool(row.get(f"slice_all_hit@{k}") for row in positive)
        retrieval[f"doc_recall@{k}"] = mean_defined(row.get(f"doc_recall@{k}") for row in positive)
        retrieval[f"doc_all_hit_rate@{k}"] = mean_bool(row.get(f"doc_all_hit@{k}") for row in positive)
        retrieval[f"source_precision@{k}"] = mean_defined(row.get(f"source_precision@{k}") for row in positive)
        retrieval[f"gold_answer_coverage@{k}"] = mean_defined(row.get(f"gold_answer_coverage@{k}") for row in positive)
    retrieval["mrr"] = mean_defined(row.get("mrr") for row in positive)
    retrieval["first_gold_rank_median"] = median_defined(row.get("first_gold_rank") for row in positive)

    answer_rows = [row for row in rows if row.get("answer") is not None]
    answer_summary: dict[str, Any] | None = None
    if answer_rows:
        answer_summary = {
            "answered_items": len(answer_rows),
            "answer_token_f1": mean_defined(row.get("answer_token_f1") for row in answer_rows if not row["negative"]),
            "answer_expected_recall": mean_defined(row.get("answer_expected_recall") for row in answer_rows if not row["negative"]),
            "hallucination_rate": mean_defined(row.get("hallucination_rate") for row in answer_rows),
            "grounded_sentence_rate": mean_defined(row.get("grounded_sentence_rate") for row in answer_rows),
            "negative_abstention_accuracy": mean_defined(row.get("negative_abstention_correct") for row in answer_rows if row["negative"]),
        }

    return {
        "items": len(rows),
        "query_types": dict(Counter(str(row["query_type"]) for row in rows)),
        "retrieval": retrieval,
        "answers": answer_summary,
        "by_query_type": {query_type: summarize_group(group, k_values=k_values) for query_type, group in sorted(by_type.items())},
    }


def summarize_group(rows: list[dict[str, Any]], *, k_values: tuple[int, ...]) -> dict[str, Any]:
    positive = [row for row in rows if not row["negative"]]
    data: dict[str, Any] = {"items": len(rows), "positive_items": len(positive)}
    for k in k_values:
        data[f"slice_recall@{k}"] = mean_defined(row.get(f"slice_recall@{k}") for row in positive)
        data[f"doc_recall@{k}"] = mean_defined(row.get(f"doc_recall@{k}") for row in positive)
        data[f"gold_answer_coverage@{k}"] = mean_defined(row.get(f"gold_answer_coverage@{k}") for row in positive)
    data["mrr"] = mean_defined(row.get("mrr") for row in positive)
    return data


def render_rag_eval_markdown(result: dict[str, Any]) -> str:
    retrieval = result["summary"]["retrieval"]
    answers = result["summary"].get("answers")
    k_values = result["config"]["k_values"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Eval set: `{result['eval_set']}`",
        f"- Corpus: `{result.get('corpus')}`",
        f"- Items: {result['items']}",
        f"- Duration: {result['duration_seconds']}s",
        "",
        "## Retrieval",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MRR | {fmt(retrieval.get('mrr'))} |",
        f"| First gold rank median | {fmt(retrieval.get('first_gold_rank_median'))} |",
    ]
    for k in k_values:
        lines.extend(
            [
                f"| Slice recall@{k} | {fmt(retrieval.get(f'slice_recall@{k}'))} |",
                f"| Slice all-hit@{k} | {fmt(retrieval.get(f'slice_all_hit_rate@{k}'))} |",
                f"| Doc recall@{k} | {fmt(retrieval.get(f'doc_recall@{k}'))} |",
                f"| Gold answer coverage@{k} | {fmt(retrieval.get(f'gold_answer_coverage@{k}'))} |",
            ]
        )
    if answers:
        lines.extend(
            [
                "",
                "## Answer Groundedness",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Answered items | {answers['answered_items']} |",
                f"| Answer token F1 | {fmt(answers.get('answer_token_f1'))} |",
                f"| Answer expected recall | {fmt(answers.get('answer_expected_recall'))} |",
                f"| Hallucination sentence rate | {fmt(answers.get('hallucination_rate'))} |",
                f"| Grounded sentence rate | {fmt(answers.get('grounded_sentence_rate'))} |",
                f"| Negative abstention accuracy | {fmt(answers.get('negative_abstention_accuracy'))} |",
            ]
        )
    lines.extend(["", "## Query Type Breakdown", "", "| Query type | Items | MRR | Slice recall@10 | Coverage@10 |", "| --- | ---: | ---: | ---: | ---: |"])
    for query_type, data in result["summary"]["by_query_type"].items():
        lines.append(
            f"| {query_type} | {data['items']} | {fmt(data.get('mrr'))} | "
            f"{fmt(data.get('slice_recall@10'))} | {fmt(data.get('gold_answer_coverage@10'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_eval_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {key: value for key, value in result.items() if key != "rows"}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_rag_eval_markdown(result), encoding="utf-8")


def generate_grounded_answer(config: EvalConfig, item: EvalItem, results: list[RetrievalResult]) -> str:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Answer generation requires: pip install openai") from exc
    api_key = config.answer_api_key
    if not api_key:
        raise RuntimeError("--generate-answers requires --answer-api-key or ANSWER_LLM_API_KEY/LLM_API_KEY.")
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60}
    if config.answer_base_url:
        client_kwargs["base_url"] = config.answer_base_url
    if config.answer_auth_header:
        client_kwargs["default_headers"] = {config.answer_auth_header: api_key}
    client = OpenAI(**client_kwargs)
    context = "\n\n".join(
        f"{format_retrieval_citation(result, index)}\n{result.content}"
        for index, result in enumerate(results, start=1)
    )
    response = client.chat.completions.create(
        model=config.answer_model or "gpt-4o-mini",
        temperature=config.answer_temperature,
        max_tokens=config.answer_max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied context. If the answer is not supported, say "
                    "'No supported answer in the supplied context.' Keep citations in the supplied "
                    "format, for example: [1] file.pdf, section 1 Title, pp.1-2, chunk_id=abc."
                ),
            },
            {"role": "user", "content": f"Question:\n{item.question}\n\nContext:\n{context}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _item_from_mapping(payload: dict[str, Any]) -> EvalItem:
    return EvalItem(
        id=str(payload.get("id") or "").strip(),
        query_type=str(payload.get("query_type") or "unknown"),
        difficulty=str(payload.get("difficulty") or "unknown"),
        negative=parse_bool(payload.get("negative")),
        question=str(payload.get("question") or "").strip(),
        expected_answer=str(payload.get("expected_answer") or "").strip(),
        expected_pdfs=as_list(payload.get("expected_pdfs")),
        expected_record_ids=as_list(payload.get("expected_record_ids")),
        gold_context=list(payload.get("gold_context") or []),
    )


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    parts = re.split(r"\s*[|;]\s*", text)
    return [part.strip() for part in parts if part.strip()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def source_pdf_name(record: dict[str, Any]) -> str:
    candidates = [
        record.get("pdf"),
        record.get("filename"),
        record.get("title"),
        Path(str(record.get("source_path") or "")).name,
        Path(str(record.get("relative_path") or "")).name,
    ]
    metadata = record.get("metadata") or {}
    candidates.extend([metadata.get("pdf"), metadata.get("filename"), metadata.get("title")])
    return first_pdf_name(candidates)


def result_pdf_name(result: RetrievalResult) -> str:
    metadata = result.metadata or {}
    return first_pdf_name(
        [
            metadata.get("pdf"),
            metadata.get("filename"),
            metadata.get("title"),
            Path(result.source_path).name,
            metadata.get("relative_path"),
        ]
    )


def first_pdf_name(candidates: Iterable[Any]) -> str:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        match = re.search(r"([^\\/]+\.pdf)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def result_id(result: RetrievalResult) -> str:
    return str(result.metadata.get("record_id") or result.chunk_id or "").strip()


def result_to_row(result: RetrievalResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "citation": format_retrieval_citation(result, result.rank),
        "chunk_id": result.chunk_id,
        "record_id": result_id(result),
        "doc_id": result.doc_id,
        "pdf": result_pdf_name(result),
        "score": result.score,
        "source": result.source,
        "source_path": result.source_path,
        "metadata": result.metadata,
        "content_preview": normalize_space(result.content)[:360],
    }


def recall_fraction(expected: set[str], observed: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & observed) / len(expected)


def precision_fraction(expected_pdfs: set[str], observed_pdfs: list[str]) -> float:
    if not observed_pdfs:
        return 0.0
    return sum(1 for pdf in observed_pdfs if pdf in expected_pdfs) / len(observed_pdfs)


def first_hit_rank(retrieved_ids: list[str], expected_ids: set[str]) -> int | None:
    if not expected_ids:
        return None
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected_ids:
            return index
    return None


def answer_coverage(expected_answer: str, text: str) -> float:
    expected = content_tokens(expected_answer)
    if not expected:
        return 0.0
    observed = Counter(content_tokens(text))
    expected_counts = Counter(expected)
    overlap = sum(min(count, observed.get(token, 0)) for token, count in expected_counts.items())
    return overlap / sum(expected_counts.values())


def token_f1(left: str, right: str) -> float:
    left_counts = Counter(content_tokens(left))
    right_counts = Counter(content_tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    overlap = sum(min(count, right_counts.get(token, 0)) for token, count in left_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_counts.values())
    recall = overlap / sum(right_counts.values())
    return 2 * precision * recall / (precision + recall)


def support_score(sentence: str, evidence_text: str) -> float:
    sentence_tokens = content_tokens(sentence)
    if not sentence_tokens:
        return 1.0
    evidence_counts = Counter(content_tokens(evidence_text))
    sentence_counts = Counter(sentence_tokens)
    overlap = sum(min(count, evidence_counts.get(token, 0)) for token, count in sentence_counts.items())
    return overlap / sum(sentence_counts.values())


def critical_terms_supported(sentence: str, evidence_text: str) -> bool:
    evidence_normalized = normalize_text(evidence_text)
    for term in critical_terms(sentence):
        if normalize_text(term) not in evidence_normalized:
            return False
    return True


def critical_terms(text: str) -> list[str]:
    terms = []
    terms.extend(re.findall(r"\bECSS-[A-Z]-[A-Z]{1,2}-\d{2}(?:-\d{2})?[A-Z]?\b", text, flags=re.IGNORECASE))
    terms.extend(re.findall(r"\b\d+(?:\.\d+){1,5}\b", text))
    terms.extend(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|V|A|W|Hz|kHz|MHz|GHz|mN|N|kg|mm|cm|m|s|ms)\b", text, flags=re.IGNORECASE))
    return terms


def content_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]", normalize_text(text))
    return [
        token
        for token in raw_tokens
        if (token not in DEFAULT_STOPWORDS and len(token) > 1) or ("\u4e00" <= token <= "\u9fff")
    ]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [normalize_space(part) for part in parts if normalize_space(part)]


def meaningful_sentence(sentence: str) -> bool:
    return bool(content_tokens(sentence)) and len(sentence.strip()) >= 4


def is_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(pattern in normalized for pattern in ABSTAIN_PATTERNS)


def mean_defined(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float, bool)) and not math.isnan(float(value))]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def mean_bool(values: Iterable[Any]) -> float | None:
    bools = [bool(value) for value in values if value is not None]
    if not bools:
        return None
    return sum(1 for value in bools if value) / len(bools)


def median_defined(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and value]
    if not numbers:
        return None
    return statistics.median(numbers)


def normalize_text(text: str) -> str:
    return normalize_space(str(text).lower())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

