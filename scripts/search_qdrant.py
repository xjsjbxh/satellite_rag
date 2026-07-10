"""Search a Qdrant collection with the configured embedding provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satellite_rag import RagConfig, build_embedding_client
from satellite_rag.citations import format_retrieval_citation
from satellite_rag.env import load_dotenv
from satellite_rag.vector_store import QdrantVectorStore


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    config = RagConfig(
        enabled=True,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_endpoint=args.embedding_endpoint,
        embedding_api_key=args.embedding_api_key,
        embedding_auth_header=args.embedding_auth_header,
        embedding_batch_size=args.embedding_batch_size,
        embedding_timeout=args.embedding_timeout,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection=args.collection,
        qdrant_vector_name=args.vector_name,
    )
    embedding = build_embedding_client(config)
    vector = embedding.embed_query(args.query)
    store = QdrantVectorStore(
        url=args.qdrant_url,
        path=args.qdrant_path,
        api_key=args.qdrant_api_key,
        collection=args.collection,
        vector_name=args.vector_name,
    )
    results = store.search(vector, top_k=args.top_k)
    rows = []
    for index, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": index,
                "score": result.score,
                "citation": format_retrieval_citation(result, index),
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "source_path": result.source_path,
                "metadata": result.metadata,
                "content_preview": result.content[: args.preview_chars],
            }
        )
    print(json.dumps({"query": args.query, "results": rows}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-path", default=os.getenv("QDRANT_PATH"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "satdesign_chunks"))
    parser.add_argument("--vector-name", default=os.getenv("QDRANT_VECTOR_NAME", "dense"))
    parser.add_argument("--embedding-provider", default=os.getenv("RAG_EMBEDDING_PROVIDER", "litellm"))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", ""))
    parser.add_argument("--embedding-endpoint", default=os.getenv("RAG_EMBEDDING_ENDPOINT") or os.getenv("LITELLM_EMBEDDING_ENDPOINT"))
    parser.add_argument("--embedding-api-key", default=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--embedding-auth-header", default=os.getenv("RAG_EMBEDDING_AUTH_HEADER") or os.getenv("LITELLM_AUTH_HEADER") or "x-litellm-api-key")
    parser.add_argument("--embedding-batch-size", type=int, default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16")))
    parser.add_argument("--embedding-timeout", type=float, default=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "120")))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--preview-chars", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
