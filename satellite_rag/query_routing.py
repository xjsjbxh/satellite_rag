"""Query-derived routing helpers for metadata-aware retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


PDF_NAME_RE = re.compile(
    r"\b(?P<pdf>ECSS-[A-Z]-[A-Z]{1,3}-\d{2}(?:-\d{2})?[A-Z]?\([^)]*\)\.pdf)\b",
    flags=re.IGNORECASE,
)
STANDARD_ID_RE = re.compile(
    r"\b(?P<standard>ECSS-[A-Z]-[A-Z]{1,3}-\d{2}(?:-\d{2})?[A-Z]?)\b",
    flags=re.IGNORECASE,
)


def metadata_filter_from_query(query: str) -> dict[str, Any]:
    """Infer a conservative metadata filter from explicit source mentions."""

    titles = sorted({_pdf_title(match.group("pdf")) for match in PDF_NAME_RE.finditer(query)})
    if titles:
        return {"title": _single_or_many(titles)}

    standard_ids = sorted({match.group("standard").upper() for match in STANDARD_ID_RE.finditer(query)})
    if standard_ids:
        return {"standard_id": _single_or_many(standard_ids)}

    return {}


def _pdf_title(pdf_name: str) -> str:
    return Path(pdf_name).stem


def _single_or_many(values: list[str]) -> str | list[str]:
    return values[0] if len(values) == 1 else values

