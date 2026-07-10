"""Citation formatting for retrieved RAG evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


def format_retrieval_citation(result: Any, index: int | None = None) -> str:
    """Return a stable, user-facing citation for a retrieval result.

    Format:
        [1] ECSS-E-HB-10-02A.pdf, section 5.2.1 Acceptance stage, pp.12-13, chunk_id=...
    """

    metadata = _metadata(result)
    pdf = _pdf_name(result, metadata)
    section = _string(_first(_value(metadata, "section"), _value(result, "section")))
    section_title = _string(_first(_value(metadata, "section_title"), _value(result, "section_title")))
    chunk_id = _string(_first(_value(result, "chunk_id"), _value(metadata, "record_id"), _value(result, "record_id")))

    parts = [pdf]
    parts.append(_section_label(section, section_title))
    parts.append(_page_label(result, metadata))
    parts.append(f"chunk_id={chunk_id or 'n/a'}")

    prefix = f"[{index}] " if index is not None else ""
    return prefix + ", ".join(parts)


def _metadata(result: Any) -> Mapping[str, Any]:
    metadata = _value(result, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _pdf_name(result: Any, metadata: Mapping[str, Any]) -> str:
    candidates = [
        _value(metadata, "pdf"),
        _value(metadata, "filename"),
        Path(str(_value(result, "source_path") or "")).name,
        Path(str(_value(metadata, "relative_path") or "")).name,
        _value(metadata, "title"),
        _value(result, "doc_id"),
    ]
    for candidate in candidates:
        text = _string(candidate)
        if not text:
            continue
        match = re.search(r"([^\\/]+\.pdf)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        if "." in text or text:
            return text
    return "unknown_source"


def _section_label(section: str, title: str) -> str:
    if section and title:
        return f"section {section} {title}"
    if section:
        return f"section {section}"
    if title:
        return f"section {title}"
    return "section n/a"


def _page_label(result: Any, metadata: Mapping[str, Any]) -> str:
    start = _string(_first(_value(metadata, "page_start"), _value(result, "page_start"), _value(metadata, "page"), _value(result, "page")))
    end = _string(_first(_value(metadata, "page_end"), _value(result, "page_end"), _value(metadata, "page"), _value(result, "page")))
    if not start and not end:
        return "pp.n/a"
    if not end:
        end = start
    if not start:
        start = end
    if start == end:
        return f"p.{start}"
    return f"pp.{start}-{end}"


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""

