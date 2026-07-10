r"""Run automated retrieval and grounded-answer evaluation.

Example:
    python scripts/evaluate_rag.py --eval-set runs/eval_sets/ecss_first10_120.jsonl \
      --corpus C:\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl \
      --embedding-provider ollama --embedding-model bge-m3 \
      --out runs/eval/ecss_first10_ollama
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satellite_rag.evaluation.rag_eval import DEFAULT_K_VALUES, EvalConfig, evaluate_rag
from satellite_rag.env import load_dotenv


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    config = EvalConfig(
        eval_set=Path(args.eval_set),
        out_dir=Path(args.out),
        corpus=Path(args.corpus) if args.corpus else None,
        answers_jsonl=Path(args.answers_jsonl) if args.answers_jsonl else None,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        qdrant_api_key=args.qdrant_api_key,
        collection=args.collection,
        vector_name=args.vector_name,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_endpoint=args.embedding_endpoint,
        embedding_api_key=args.embedding_api_key,
        embedding_auth_header=args.embedding_auth_header,
        embedding_batch_size=args.embedding_batch_size,
        embedding_timeout=args.embedding_timeout,
        hash_dimensions=args.hash_dimensions,
        top_k=args.top_k,
        k_values=tuple(args.k_values),
        limit=args.limit,
        corpus_limit=args.corpus_limit,
        restrict_corpus_to_eval_pdfs=not args.no_restrict_corpus_to_eval_pdfs,
        route_query_metadata=args.route_query_metadata,
        vector_overfetch=args.vector_overfetch,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        rerank_provider=args.rerank_provider,
        rerank_model=args.rerank_model,
        rerank_endpoint=args.rerank_endpoint,
        rerank_api_key=args.rerank_api_key,
        rerank_auth_header=args.rerank_auth_header,
        rerank_timeout=args.rerank_timeout,
        rerank_top_k=args.rerank_top_k,
        generate_answers=args.generate_answers,
        answer_model=args.answer_model,
        answer_base_url=args.answer_base_url,
        answer_api_key=args.answer_api_key,
        answer_auth_header=args.answer_auth_header,
        answer_max_tokens=args.answer_max_tokens,
        answer_temperature=args.answer_temperature,
        support_threshold=args.support_threshold,
    )
    result = evaluate_rag(config)
    print(
        json.dumps(
            {
                "out": str(config.out_dir),
                "items": result["items"],
                "retrieval": result["summary"]["retrieval"],
                "answers": result["summary"].get("answers"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="runs/eval_sets/ecss_first10_120.jsonl")
    parser.add_argument("--corpus", default=os.getenv("RAG_CORPUS_PATH"))
    parser.add_argument("--out", default="runs/eval/ecss_first10")
    parser.add_argument("--answers-jsonl", help="Optional JSONL with {id, answer}.")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-path", default=os.getenv("QDRANT_PATH"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "satdesign_chunks"))
    parser.add_argument("--vector-name", default=os.getenv("QDRANT_VECTOR_NAME", "dense"))
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "bge_m3", "ollama", "litellm", "openai_compatible"],
        default=os.getenv("RAG_EMBEDDING_PROVIDER", "hash"),
    )
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "test/hash-dense"))
    parser.add_argument(
        "--embedding-endpoint",
        default=os.getenv("RAG_EMBEDDING_ENDPOINT") or os.getenv("LITELLM_EMBEDDING_ENDPOINT") or os.getenv("OLLAMA_EMBEDDING_URL"),
    )
    parser.add_argument("--embedding-api-key", default=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--embedding-auth-header", default=os.getenv("RAG_EMBEDDING_AUTH_HEADER") or os.getenv("LITELLM_AUTH_HEADER") or "x-litellm-api-key")
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16")))
    parser.add_argument("--embedding-timeout", type=float, default=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "120")))
    parser.add_argument("--hash-dimensions", type=int, default=int(os.getenv("RAG_HASH_EMBEDDING_DIMENSIONS", "64")))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--limit", type=int, help="Limit eval items for smoke tests.")
    parser.add_argument("--corpus-limit", type=int, help="Limit loaded corpus chunks for smoke tests.")
    parser.add_argument("--no-restrict-corpus-to-eval-pdfs", action="store_true")
    parser.add_argument("--route-query-metadata", action="store_true")
    parser.add_argument("--vector-overfetch", type=int, default=200)
    parser.add_argument("--vector-weight", type=float, default=float(os.getenv("RAG_VECTOR_WEIGHT", "1.0")))
    parser.add_argument("--bm25-weight", type=float, default=float(os.getenv("RAG_BM25_WEIGHT", "1.0")))
    parser.add_argument(
        "--rerank-provider",
        choices=["identity", "none", "disabled", "bge", "bge_m3", "flagembedding", "litellm", "openai_compatible", "qwen"],
        default=os.getenv("RAG_RERANK_PROVIDER", "identity"),
    )
    parser.add_argument("--rerank-model", default=os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
    parser.add_argument(
        "--rerank-endpoint",
        default=os.getenv("RAG_RERANK_ENDPOINT")
        or os.getenv("LITELLM_RERANK_ENDPOINT")
        or _base_url_endpoint(os.getenv("LITELLM_BASE_URL"), "rerank"),
    )
    parser.add_argument("--rerank-api-key", default=os.getenv("RAG_RERANK_API_KEY") or os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--rerank-auth-header", default=os.getenv("RAG_RERANK_AUTH_HEADER") or os.getenv("LITELLM_AUTH_HEADER") or "x-litellm-api-key")
    parser.add_argument("--rerank-timeout", type=float, default=float(os.getenv("RAG_RERANK_TIMEOUT", "120")))
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=int(os.getenv("RAG_RERANK_TOP_K")) if os.getenv("RAG_RERANK_TOP_K") else None,
        help="Number of fused candidates to send to the reranker. Defaults to --vector-overfetch.",
    )
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--answer-model", default=os.getenv("ANSWER_LLM_MODEL") or os.getenv("LLM_MODEL"))
    parser.add_argument("--answer-base-url", default=os.getenv("ANSWER_LLM_BASE_URL") or os.getenv("LLM_BASE_URL"))
    parser.add_argument("--answer-api-key", default=os.getenv("ANSWER_LLM_API_KEY") or os.getenv("LLM_API_KEY"))
    parser.add_argument("--answer-auth-header", default=os.getenv("ANSWER_LLM_AUTH_HEADER") or os.getenv("LITELLM_AUTH_HEADER"))
    parser.add_argument("--answer-max-tokens", type=int, default=int(os.getenv("ANSWER_LLM_MAX_TOKENS", "900")))
    parser.add_argument("--answer-temperature", type=float, default=float(os.getenv("ANSWER_LLM_TEMPERATURE", "0")))
    parser.add_argument("--support-threshold", type=float, default=0.55)
    return parser.parse_args()


def _base_url_endpoint(base_url: str | None, path: str) -> str | None:
    if not base_url:
        return None
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())

