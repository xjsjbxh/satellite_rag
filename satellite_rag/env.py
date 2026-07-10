"""Small .env loader used by satellite_rag scripts.

This intentionally avoids an extra python-dotenv dependency. It supports the
simple KEY=value format used by this project and does not override variables
already present in the process environment unless requested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load environment variables from .env.

    If ``path`` is omitted, this searches the current working directory and its
    parents, then the installed project root. If no real .env file exists, it
    falls back to .env.example so local template-based setups still work.
    """

    env_path = Path(path) if path else find_dotenv()
    if env_path and not env_path.exists():
        env_path = _fallback_example_path(env_path)
    if not env_path or not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def find_dotenv() -> Path | None:
    cwd = Path.cwd().resolve()
    project_root = Path(__file__).resolve().parents[1]
    roots = [*_walk_parents(cwd), project_root]
    candidates = [root / ".env" for root in roots]
    candidates.extend(root / ".env.example" for root in roots)
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _fallback_example_path(path: Path) -> Path | None:
    if path.name != ".env":
        return None
    fallback = path.with_name(".env.example")
    return fallback if fallback.exists() else None


def _walk_parents(path: Path) -> Iterable[Path]:
    yield path
    yield from path.parents


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _strip_inline_comment(value: str) -> str:
    if not value or value[0] in {'"', "'"}:
        return value
    marker = " #"
    if marker in value:
        return value.split(marker, 1)[0].rstrip()
    return value
