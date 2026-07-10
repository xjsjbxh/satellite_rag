"""Check a LiteLLM/OpenAI-compatible proxy before using it in RAG.

Examples:
    python scripts/check_litellm.py
    python scripts/check_litellm.py --embedding --rerank
    python scripts/check_litellm.py --chat --model qwen-plus
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satellite_rag.env import load_dotenv


DEFAULT_BASE_URL = "http://api.opearlai.com:30096/v1"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key or os.getenv("LITELLM_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Provide --api-key or set LITELLM_API_KEY.")

    print(json.dumps({"models": request_json("GET", f"{base_url}/models", api_key=api_key, header=args.auth_header)}, ensure_ascii=False, indent=2))

    if args.chat:
        chat = request_json(
            "POST",
            f"{base_url}/chat/completions",
            api_key=api_key,
            header=args.auth_header,
            payload={
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0,
            },
        )
        print(json.dumps({"chat": chat}, ensure_ascii=False, indent=2))

    if args.embedding:
        emb = request_json(
            "POST",
            f"{base_url}/embeddings",
            api_key=api_key,
            header=args.auth_header,
            payload={"model": args.embedding_model, "input": [args.prompt]},
        )
        vector = emb.get("data", [{}])[0].get("embedding", [])
        print(json.dumps({"embedding_dimension": len(vector), "embedding_model": args.embedding_model}, ensure_ascii=False, indent=2))

    if args.rerank:
        rerank = request_json(
            "POST",
            args.rerank_endpoint,
            api_key=api_key,
            header=args.auth_header,
            payload={
                "model": args.rerank_model,
                "query": args.prompt,
                "documents": [
                    "A satellite RAG system retrieves and cites relevant engineering evidence.",
                    "Thermal vacuum testing verifies spacecraft behavior under space-like temperature and pressure.",
                ],
                "top_n": 2,
                "return_documents": False,
            },
        )
        print(json.dumps({"rerank_model": args.rerank_model, "rerank": rerank}, ensure_ascii=False, indent=2))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", nargs="?", default=None)
    parser.add_argument("--auth-header", default=os.getenv("LITELLM_AUTH_HEADER", "x-litellm-api-key"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", ""))
    parser.add_argument("--rerank-model", default=os.getenv("RAG_RERANK_MODEL", ""))
    parser.add_argument(
        "--rerank-endpoint",
        default=os.getenv("RAG_RERANK_ENDPOINT")
        or os.getenv("LITELLM_RERANK_ENDPOINT")
        or _base_url_endpoint(os.getenv("LITELLM_BASE_URL", DEFAULT_BASE_URL), "rerank"),
    )
    parser.add_argument("--prompt", default="Briefly introduce a satellite knowledge-base RAG system.")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    return parser.parse_args()


def request_json(method: str, url: str, *, api_key: str, header: str, payload: dict[str, Any] | None = None) -> Any:
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if header.lower() == "authorization":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers[header] = api_key
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc


def _base_url_endpoint(base_url: str | None, path: str) -> str:
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/{path.lstrip('/')}"


if __name__ == "__main__":
    raise SystemExit(main())
