"""Command line entry points for satellite_rag."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from satellite_rag.config import RagConfig
from satellite_rag.factory import build_embedding_client
from satellite_rag.prepare import PrepareOptions, prepare_corpus
from satellite_rag.scan import ScanOptions, scan_public_data_root


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


def cmd_scan(args: argparse.Namespace) -> int:
    result = scan_public_data_root(
        ScanOptions(
            root=Path(args.root),
            out_dir=Path(args.out),
            profile=args.profile,
            hash_mode=args.hash_mode,
            content_hash_max_mb=args.content_hash_max_mb,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    routes = set(args.routes.split(",")) if args.routes else None
    result = prepare_corpus(
        PrepareOptions(
            manifest_path=Path(args.manifest),
            out_dir=Path(args.out),
            action=args.action,
            routes=routes,
            max_record_chars=args.max_record_chars,
            limit=args.limit,
            pdf_chunk_mode=args.pdf_chunk_mode,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_embed_test(args: argparse.Namespace) -> int:
    config = RagConfig.from_env()
    if args.provider:
        config = config.__class__(
            **{
                **config.__dict__,
                "embedding_provider": args.provider,
                "embedding_model": args.model or config.embedding_model,
                "embedding_endpoint": args.endpoint or config.embedding_endpoint,
            }
        )
    embedding = build_embedding_client(config)
    vector = embedding.embed_query(args.text)
    print(
        json.dumps(
            {
                "provider": config.embedding_provider,
                "model": embedding.model_name,
                "dimension": len(vector),
                "preview": vector[:8],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satellite_rag")
    subparsers = parser.add_subparsers(dest="command")

    scan = subparsers.add_parser("scan", help="Scan a source library and build manifest.jsonl.")
    scan.add_argument("--root", required=True)
    scan.add_argument("--out", required=True)
    scan.add_argument("--profile", default="public_data_mvp")
    scan.add_argument("--hash-mode", choices=["fingerprint", "content", "none"], default="fingerprint")
    scan.add_argument("--content-hash-max-mb", type=int, default=64)
    scan.set_defaults(func=cmd_scan)

    prepare = subparsers.add_parser("prepare", help="Prepare corpus.jsonl from a manifest.")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--action", default="include")
    prepare.add_argument("--routes", default="markdown,jsonl_records,pdf_text_first")
    prepare.add_argument("--max-record-chars", type=int, default=6000)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--pdf-chunk-mode", choices=["page", "clause"], default="page")
    prepare.set_defaults(func=cmd_prepare)

    embed = subparsers.add_parser("embed-test", help="Embed one text and report vector dimension.")
    embed.add_argument("--text", default="Satellite power margin requirement.")
    embed.add_argument("--provider")
    embed.add_argument("--model")
    embed.add_argument("--endpoint")
    embed.set_defaults(func=cmd_embed_test)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
