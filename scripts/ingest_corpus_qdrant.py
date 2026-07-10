"""Ingest a prepared Phase-1 corpus JSONL into Qdrant.

The script is intentionally separate from the app CLI because it handles a
prepared corpus file, not raw documents. It writes payloads in the same shape as
the project's RagChunk model so the existing QdrantVectorStore can read them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satellite_rag.embeddings import (
    BgeM3EmbeddingClient,
    HashDenseEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from satellite_rag.citations import format_retrieval_citation
from satellite_rag.env import load_dotenv
from satellite_rag.schemas import RagChunk
from satellite_rag.vector_store import _qdrant_point_id


DEFAULT_CORPUS = r"C:\satdesign-rag-data\public_data_mvp\corpus.jsonl"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    started_at = time.time()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")

    embedding = build_embedding(args)
    vector_size = len(embedding.embed_query(args.verify_query or "satellite"))

    client = build_client(args)
    ensure_collection(
        client,
        collection=args.collection,
        vector_name=args.vector_name,
        vector_size=vector_size,
        recreate=args.recreate,
    )

    ingested = 0
    skipped = 0
    last_progress_at = time.time()
    for records in batched(
        iter_corpus_records(corpus_path, limit=args.limit, skip_records=args.skip_records),
        args.batch_size,
    ):
        chunks = []
        for record in records:
            chunk = corpus_record_to_chunk(record)
            if chunk.content.strip():
                chunks.append(chunk)
            else:
                skipped += 1
        if not chunks:
            continue
        vectors = embedding.embed_texts([chunk.content for chunk in chunks])
        points = [
            models.PointStruct(
                id=_qdrant_point_id(chunk.chunk_id),
                vector={args.vector_name: vector},
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        ingested += len(points)
        now = time.time()
        if args.progress and now - last_progress_at >= args.progress_interval:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "ingested_points_this_run": ingested,
                        "collection_count": client.count(collection_name=args.collection, exact=False).count,
                        "elapsed_seconds": round(now - started_at, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_progress_at = now

    count = client.count(collection_name=args.collection, exact=True).count
    verify_hits = verify_search(
        client,
        collection=args.collection,
        vector_name=args.vector_name,
        query_vector=embedding.embed_query(args.verify_query),
        top_k=args.verify_top_k,
    )

    summary = {
        "corpus": str(corpus_path),
        "database": args.url or str(Path(args.qdrant_path).resolve()),
        "collection": args.collection,
        "vector_name": args.vector_name,
        "embedding_provider": args.embedding_provider,
        "embedding_model": getattr(embedding, "model_name", args.embedding_provider),
        "vector_size": vector_size,
        "ingested_points": ingested,
        "skipped_empty_content": skipped,
        "qdrant_count": count,
        "verify_query": args.verify_query,
        "verify_hits": verify_hits,
        "duration_seconds": round(time.time() - started_at, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    client.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.getenv("RAG_CORPUS_PATH", DEFAULT_CORPUS))
    parser.add_argument("--url", default=os.getenv("QDRANT_URL"), help="Qdrant server URL. If omitted, --qdrant-path is used.")
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--qdrant-path", default="runs/qdrant_public_data_mvp")
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "satdesign_chunks"))
    parser.add_argument("--vector-name", default=os.getenv("QDRANT_VECTOR_NAME", "dense"))
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "bge_m3", "ollama", "litellm", "openai_compatible"],
        default=os.getenv("RAG_EMBEDDING_PROVIDER", "hash"),
    )
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument(
        "--embedding-endpoint",
        default=os.getenv("RAG_EMBEDDING_ENDPOINT") or os.getenv("LITELLM_EMBEDDING_ENDPOINT") or os.getenv("OLLAMA_EMBEDDING_URL"),
    )
    parser.add_argument("--embedding-api-key", default=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LITELLM_API_KEY"))
    parser.add_argument("--embedding-auth-header", default=os.getenv("RAG_EMBEDDING_AUTH_HEADER") or "x-litellm-api-key")
    parser.add_argument("--embedding-timeout", type=float, default=float(os.getenv("RAG_EMBEDDING_TIMEOUT", "120")))
    parser.add_argument("--hash-dimensions", type=int, default=int(os.getenv("RAG_HASH_EMBEDDING_DIMENSIONS", "64")))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--skip-records", type=int, default=0, help="Skip this many corpus records before ingesting.")
    parser.add_argument("--progress", action="store_true", help="Print periodic JSON progress lines during ingestion.")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="Seconds between progress lines.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection before ingesting.")
    parser.add_argument("--verify-query", default="satellite power margin requirement")
    parser.add_argument("--verify-top-k", type=int, default=5)
    return parser.parse_args()


def build_embedding(
    args: argparse.Namespace,
) -> HashDenseEmbeddingClient | BgeM3EmbeddingClient | OllamaEmbeddingClient | OpenAICompatibleEmbeddingClient:
    if args.embedding_provider == "hash":
        return HashDenseEmbeddingClient(dimensions=args.hash_dimensions)
    if args.embedding_provider == "ollama":
        return OllamaEmbeddingClient(model_name=args.embedding_model, endpoint=args.embedding_endpoint, timeout=args.embedding_timeout)
    if args.embedding_provider in {"litellm", "openai_compatible"}:
        return OpenAICompatibleEmbeddingClient(
            model_name=args.embedding_model,
            endpoint=args.embedding_endpoint or "",
            api_key=args.embedding_api_key,
            auth_header=args.embedding_auth_header,
            timeout=args.embedding_timeout,
            batch_size=args.batch_size,
        )
    return BgeM3EmbeddingClient(model_name=args.embedding_model)


def build_client(args: argparse.Namespace) -> QdrantClient:
    if args.url:
        return QdrantClient(url=args.url, api_key=args.api_key)
    path = Path(args.qdrant_path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def ensure_collection(
    client: QdrantClient,
    *,
    collection: str,
    vector_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                vector_name: models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
        )


def iter_corpus_records(path: Path, *, limit: int | None = None, skip_records: int = 0) -> Iterable[dict[str, Any]]:
    emitted = 0
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            seen += 1
            if seen <= skip_records:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


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
    metadata.setdefault("record_id", str(record["record_id"]))

    return RagChunk(
        chunk_id=str(record["record_id"]),
        doc_id=str(record["doc_id"]),
        content=str(record.get("content") or ""),
        source_path=str(record.get("source_path") or ""),
        metadata=metadata,
    )


def batched(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def verify_search(
    client: QdrantClient,
    *,
    collection: str,
    vector_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        using=vector_name,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": round(float(point.score or 0.0), 6),
            "citation": format_retrieval_citation(point.payload or {}, index),
            "chunk_id": (point.payload or {}).get("chunk_id"),
            "doc_id": (point.payload or {}).get("doc_id"),
            "title": ((point.payload or {}).get("metadata") or {}).get("title"),
            "source_type": ((point.payload or {}).get("metadata") or {}).get("source_type"),
            "source_path": (point.payload or {}).get("source_path"),
            "content_preview": str((point.payload or {}).get("content") or "")[:240],
        }
        for index, point in enumerate(response.points, start=1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())

