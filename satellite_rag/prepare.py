"""Phase-1 corpus preparation from a Phase-0 RAG manifest."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


DEFAULT_MAX_RECORD_CHARS = 6000
DEFAULT_ROUTES = {"markdown", "jsonl_records", "pdf_text_first"}
SUPPORTED_ROUTES = DEFAULT_ROUTES | {"json_document"}
PdfChunkMode = Literal["page", "clause"]

NASA_LESSON_FIELDS = [
    "Subject",
    "Abstract",
    "Driving Event",
    "Lesson(s) Learned",
    "Recommendation(s)",
    "Evidence of Recurrence Control Effectiveness",
]


@dataclass(frozen=True)
class PrepareOptions:
    manifest_path: Path
    out_dir: Path
    action: str = "include"
    routes: set[str] | None = None
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS
    limit: int | None = None
    pdf_chunk_mode: PdfChunkMode = "page"


@dataclass(frozen=True)
class ClauseHeading:
    section: str
    title: str
    page: int
    line_index: int
    body_start_line_index: int
    level: int


@dataclass(frozen=True)
class PageLine:
    page: int
    text: str


@dataclass(frozen=True)
class PdfCleanProfile:
    repeated_edge_lines: frozenset[str]
    page_count: int


@dataclass(frozen=True)
class StandardClause:
    section: str
    title: str
    page_start: int
    page_end: int
    content: str


def prepare_corpus(options: PrepareOptions) -> dict[str, Any]:
    """Prepare a normalized JSONL corpus from selected manifest records."""

    if not options.manifest_path.exists():
        raise FileNotFoundError(f"RAG manifest does not exist: {options.manifest_path}")
    if options.max_record_chars <= 0:
        raise ValueError("max_record_chars must be positive")

    options.out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = options.out_dir / "corpus.jsonl"
    failures_path = options.out_dir / "parse_failures.jsonl"
    report_json_path = options.out_dir / "parse_report.json"
    report_md_path = options.out_dir / "parse_report.md"

    generated_at = datetime.now(timezone.utc).isoformat()
    selected_routes = options.routes or DEFAULT_ROUTES
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    selected_files = 0
    parsed_files = 0
    record_count = 0

    with corpus_path.open("w", encoding="utf-8") as corpus_handle, failures_path.open("w", encoding="utf-8") as failure_handle:
        for manifest_record in iter_manifest(options.manifest_path):
            if manifest_record.get("mvp_action") != options.action:
                continue
            route = str(manifest_record.get("parse_route") or "")
            if route not in selected_routes:
                continue
            if options.limit is not None and selected_files >= options.limit:
                break

            selected_files += 1
            route_counts[route] += 1
            try:
                records = list(
                    parse_manifest_record(
                        manifest_record,
                        generated_at=generated_at,
                        max_record_chars=options.max_record_chars,
                        pdf_chunk_mode=options.pdf_chunk_mode,
                    )
                )
                if not records:
                    raise ParseFailure("empty_content", "No text records were produced.")
                for record in records:
                    corpus_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    corpus_handle.write("\n")
                    source_counts[record["source_type"]] += 1
                parsed_files += 1
                record_count += len(records)
                counts["parsed"] += 1
            except ParseFailure as exc:
                failures[exc.reason] += 1
                counts["failed"] += 1
                failure_handle.write(json.dumps(build_failure(manifest_record, exc.reason, exc.message, generated_at), ensure_ascii=False, sort_keys=True))
                failure_handle.write("\n")
            except Exception as exc:  # noqa: BLE001 - preparation should report per-file failures and continue
                failures["unexpected_error"] += 1
                counts["failed"] += 1
                failure_handle.write(
                    json.dumps(
                        build_failure(manifest_record, "unexpected_error", f"{type(exc).__name__}: {exc}", generated_at),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                failure_handle.write("\n")

    report = {
        "schema_version": "satdesign.rag_prepare_report.v1",
        "generated_at": generated_at,
        "manifest_path": str(options.manifest_path),
        "out_dir": str(options.out_dir),
        "corpus_path": str(corpus_path),
        "failures_path": str(failures_path),
        "selected_action": options.action,
        "selected_routes": sorted(selected_routes),
        "pdf_chunk_mode": options.pdf_chunk_mode,
        "selected_files": selected_files,
        "parsed_files": parsed_files,
        "failed_files": counts["failed"],
        "records": record_count,
        "routes": dict(sorted(route_counts.items())),
        "source_types": dict(sorted(source_counts.items())),
        "failures": dict(sorted(failures.items())),
    }
    write_json(report_json_path, report)
    report_md_path.write_text(render_prepare_report(report), encoding="utf-8")
    return report | {"report_json_path": str(report_json_path), "report_md_path": str(report_md_path)}


def parse_manifest_record(
    manifest_record: dict[str, Any],
    *,
    generated_at: str,
    max_record_chars: int,
    pdf_chunk_mode: PdfChunkMode = "page",
) -> Iterable[dict[str, Any]]:
    route = str(manifest_record.get("parse_route") or "")
    if route == "markdown":
        yield from parse_markdown_record(manifest_record, generated_at=generated_at, max_record_chars=max_record_chars)
        return
    if route == "jsonl_records":
        yield from parse_jsonl_records(manifest_record, generated_at=generated_at, max_record_chars=max_record_chars)
        return
    if route == "json_document":
        yield from parse_json_document(manifest_record, generated_at=generated_at, max_record_chars=max_record_chars)
        return
    if route == "pdf_text_first":
        yield from parse_pdf_record(
            manifest_record,
            generated_at=generated_at,
            max_record_chars=max_record_chars,
            chunk_mode=pdf_chunk_mode,
        )
        return
    raise ParseFailure("unsupported_route", f"Route is not supported in Phase 1: {route}")


def parse_markdown_record(manifest_record: dict[str, Any], *, generated_at: str, max_record_chars: int) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    text = read_text_fallback(source_path)
    frontmatter, body = split_frontmatter(text)
    cleaned = clean_markdown(body)
    title = frontmatter.get("title") or first_markdown_heading(cleaned) or source_path.stem
    for index, section in enumerate(split_text_sections(cleaned, max_record_chars=max_record_chars), start=1):
        yield build_corpus_record(
            manifest_record,
            generated_at=generated_at,
            title=title,
            content=section,
            record_kind="markdown_section",
            part_index=index,
            extra_metadata={"frontmatter": frontmatter},
        )


def parse_jsonl_records(manifest_record: dict[str, Any], *, generated_at: str, max_record_chars: int) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    produced = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ParseFailure("invalid_jsonl", f"Line {line_number}: {exc}") from exc
            title = str(payload.get("Subject") or payload.get("title") or f"JSONL record {line_number}")
            content = nasa_lesson_content(payload) if is_nasa_lesson(manifest_record, payload) else json.dumps(payload, ensure_ascii=False, indent=2)
            for part_index, part in enumerate(split_long_text(normalize_text(content), max_record_chars=max_record_chars), start=1):
                yield build_corpus_record(
                    manifest_record,
                    generated_at=generated_at,
                    title=title,
                    content=part,
                    record_kind="jsonl_record",
                    part_index=part_index,
                    source_record=line_number,
                    extra_metadata={"jsonl_line": line_number, **jsonl_metadata(payload)},
                )
            produced += 1
    if produced == 0:
        raise ParseFailure("empty_jsonl", "JSONL file contains no records.")


def parse_json_document(manifest_record: dict[str, Any], *, generated_at: str, max_record_chars: int) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    payload = json.loads(read_text_fallback(source_path))
    title = str(payload.get("title") if isinstance(payload, dict) else source_path.stem)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for index, part in enumerate(split_text_sections(text, max_record_chars=max_record_chars), start=1):
        yield build_corpus_record(
            manifest_record,
            generated_at=generated_at,
            title=title,
            content=part,
            record_kind="json_document",
            part_index=index,
        )


def parse_pdf_record(
    manifest_record: dict[str, Any],
    *,
    generated_at: str,
    max_record_chars: int,
    chunk_mode: PdfChunkMode = "page",
) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    pages = extract_pdf_pages(source_path)
    if chunk_mode == "clause":
        yield from parse_pdf_clause_record(
            manifest_record,
            pages=pages,
            generated_at=generated_at,
            max_record_chars=max_record_chars,
        )
        return

    produced = 0
    for page_number, page_text in pages:
        cleaned = normalize_text(page_text)
        if not cleaned:
            continue
        for part_index, part in enumerate(split_text_sections(cleaned, max_record_chars=max_record_chars), start=1):
            yield build_corpus_record(
                manifest_record,
                generated_at=generated_at,
                title=source_path.stem,
                content=part,
                record_kind="pdf_page",
                part_index=part_index,
                page=page_number,
                page_end=page_number,
                standard_id=infer_standard_id(source_path),
                extra_metadata={"pdf_page": page_number, "pdf_chunk_mode": "page"},
            )
            produced += 1
    if produced == 0:
        raise ParseFailure("needs_ocr_or_empty_pdf", "PDF text extraction produced no text.")


def parse_pdf_clause_record(
    manifest_record: dict[str, Any],
    *,
    pages: list[tuple[int, str]],
    generated_at: str,
    max_record_chars: int,
) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    standard_id = infer_standard_id(source_path)
    clauses = extract_standard_clauses(pages)
    if len(clauses) < 3:
        yield from parse_pdf_page_records(
            manifest_record,
            pages=pages,
            generated_at=generated_at,
            max_record_chars=max_record_chars,
            fallback_reason="clause_headings_not_found",
        )
        return

    produced = 0
    for clause_index, clause in enumerate(clauses, start=1):
        heading_text = normalize_text(f"{clause.section} {clause.title}")
        content = normalize_text(f"{heading_text}\n\n{clause.content}")
        for part_index, part in enumerate(split_long_text(content, max_record_chars=max_record_chars), start=1):
            if not part.startswith(heading_text):
                part = normalize_text(f"{heading_text}\n\n{part}")
            yield build_corpus_record(
                manifest_record,
                generated_at=generated_at,
                title=source_path.stem,
                content=part,
                record_kind="standard_clause",
                part_index=part_index,
                page=clause.page_start,
                page_end=clause.page_end,
                section=clause.section,
                section_title=clause.title,
                standard_id=standard_id,
                extra_metadata={
                    "pdf_chunk_mode": "clause",
                    "page_start": clause.page_start,
                    "page_end": clause.page_end,
                    "section": clause.section,
                    "section_title": clause.title,
                    "standard_id": standard_id,
                    "clause_index": clause_index,
                },
            )
            produced += 1
    if produced == 0:
        raise ParseFailure("needs_ocr_or_empty_pdf", "PDF clause extraction produced no text.")


def parse_pdf_page_records(
    manifest_record: dict[str, Any],
    *,
    pages: list[tuple[int, str]],
    generated_at: str,
    max_record_chars: int,
    fallback_reason: str | None = None,
) -> Iterable[dict[str, Any]]:
    source_path = Path(str(manifest_record["source_path"]))
    clean_profile = build_pdf_clean_profile(pages)
    produced = 0
    for page_number, page_text in pages:
        cleaned = clean_pdf_page_text(page_number, page_text, clean_profile)
        if not cleaned:
            continue
        for part_index, part in enumerate(split_text_sections(cleaned, max_record_chars=max_record_chars), start=1):
            metadata = {"pdf_page": page_number, "pdf_chunk_mode": "page"}
            if fallback_reason:
                metadata["fallback_reason"] = fallback_reason
            yield build_corpus_record(
                manifest_record,
                generated_at=generated_at,
                title=source_path.stem,
                content=part,
                record_kind="pdf_page",
                part_index=part_index,
                page=page_number,
                page_end=page_number,
                standard_id=infer_standard_id(source_path),
                extra_metadata=metadata,
            )
            produced += 1
    if produced == 0:
        raise ParseFailure("needs_ocr_or_empty_pdf", "PDF text extraction produced no text.")


def extract_pdf_pages(source_path: Path) -> list[tuple[int, str]]:
    backend = load_pdf_backend()
    if backend == "pypdf":
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(str(source_path))
        return [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1)]
    if backend == "pdfplumber":
        import pdfplumber  # type: ignore[import-not-found]

        with pdfplumber.open(str(source_path)) as pdf:
            return [(index, page.extract_text() or "") for index, page in enumerate(pdf.pages, start=1)]
    raise ParseFailure("missing_pdf_backend", "Install pypdf or pdfplumber to extract PDF text.")


def extract_standard_clauses(pages: list[tuple[int, str]]) -> list[StandardClause]:
    clean_profile = build_pdf_clean_profile(pages)
    lines = flatten_pdf_lines(pages, clean_profile=clean_profile)
    if not lines:
        return []

    toc_pages = detect_toc_pages(lines)
    headings: list[ClauseHeading] = []
    index = 0
    while index < len(lines):
        heading = detect_clause_heading(lines, index)
        if heading and heading.page not in toc_pages:
            headings.append(heading)
            index = max(index + 1, heading.body_start_line_index)
            continue
        index += 1

    headings = filter_clause_headings(dedupe_adjacent_headings(headings))
    if len(headings) < 3:
        return []

    clauses: list[StandardClause] = []
    for heading_index, heading in enumerate(headings):
        start = heading.body_start_line_index
        end = headings[heading_index + 1].line_index if heading_index + 1 < len(headings) else len(lines)
        body_lines = [
            line.text
            for line in lines[start:end]
            if is_content_line(line.text) and not is_pdf_noise_line(line.text, clean_profile=clean_profile)
        ]
        content = reflow_pdf_text(body_lines)
        if not content:
            continue
        if is_probably_spurious_single_number_clause(heading, content):
            continue
        page_end = lines[end - 1].page if end > start else heading.page
        clauses.append(
            StandardClause(
                section=heading.section,
                title=heading.title,
                page_start=heading.page,
                page_end=page_end,
                content=content,
            )
        )
    return clauses


def flatten_pdf_lines(pages: list[tuple[int, str]], *, clean_profile: PdfCleanProfile | None = None) -> list[PageLine]:
    flattened: list[PageLine] = []
    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = normalize_pdf_line(raw_line)
            if is_content_line(line) and not is_pdf_noise_line(line, clean_profile=clean_profile):
                flattened.append(PageLine(page=page_number, text=line))
    return flattened


def build_pdf_clean_profile(pages: list[tuple[int, str]]) -> PdfCleanProfile:
    page_count = len(pages)
    edge_counts: Counter[str] = Counter()
    for _, page_text in pages:
        page_lines = [normalize_pdf_line(line) for line in page_text.splitlines()]
        page_lines = [line for line in page_lines if is_content_line(line)]
        edge_lines = page_lines[:4] + page_lines[-4:]
        for line in edge_lines:
            if is_candidate_repeated_edge_line(line):
                edge_counts[line] += 1

    min_count = max(3, int(page_count * 0.15))
    repeated = frozenset(line for line, count in edge_counts.items() if count >= min_count)
    return PdfCleanProfile(repeated_edge_lines=repeated, page_count=page_count)


def clean_pdf_page_text(page_number: int, page_text: str, clean_profile: PdfCleanProfile) -> str:
    lines = []
    for raw_line in page_text.splitlines():
        line = normalize_pdf_line(raw_line)
        if is_content_line(line) and not is_pdf_noise_line(line, clean_profile=clean_profile):
            lines.append(line)
    return reflow_pdf_text(lines)


def is_candidate_repeated_edge_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4:
        return False
    if is_toc_line(stripped):
        return False
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if is_repeated_pdf_noise(stripped):
        return True
    if re.search(r"\b(?:ECSS|GJB|QJ|GB/T|NASA)\b", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", stripped):
        return True
    if len(stripped) <= 80 and not re.search(r"[.!?;:]\s*$", stripped):
        return True
    return False


def is_pdf_noise_line(line: str, *, clean_profile: PdfCleanProfile | None = None) -> bool:
    if clean_profile and line in clean_profile.repeated_edge_lines:
        return True
    return is_repeated_pdf_noise(line)


def detect_toc_pages(lines: list[PageLine]) -> set[int]:
    pages: dict[int, list[str]] = {}
    for line in lines:
        pages.setdefault(line.page, []).append(line.text)

    toc_pages: set[int] = set()
    for page, page_lines in pages.items():
        if not page_lines:
            continue
        toc_line_count = sum(1 for line in page_lines if is_toc_line(line) or looks_like_toc_title_with_page(line, current_page=page))
        has_contents_marker = any(re.fullmatch(r"(contents|table of contents)", line, flags=re.IGNORECASE) for line in page_lines)
        if toc_line_count >= 3 and (has_contents_marker or toc_line_count / len(page_lines) >= 0.12):
            toc_pages.add(page)
    return toc_pages


def detect_clause_heading(lines: list[PageLine], index: int) -> ClauseHeading | None:
    line = lines[index].text.strip()
    if not line or is_toc_line(line) or is_repeated_pdf_noise(line):
        return None

    split_heading = detect_split_number_heading(lines, index)
    if split_heading:
        return split_heading

    for pattern in (ANNEX_HEADING_RE, APPENDIX_SECTION_RE, NUMBERED_HEADING_RE):
        match = pattern.match(line)
        if not match:
            continue
        section = normalize_section(match.group("section"))
        title = normalize_heading_title(match.groupdict().get("title") or "")
        if is_annex_heading_section(section) and not is_valid_annex_heading_context(lines, index, title):
            continue
        body_start_line_index = index + 1
        if is_annex_heading_section(section):
            title, body_start_line_index = annex_title_and_body_start(lines, index, title)
        if is_valid_heading(section, title, line, page=lines[index].page):
            return ClauseHeading(
                section=section,
                title=title,
                page=lines[index].page,
                line_index=index,
                body_start_line_index=body_start_line_index,
                level=section_level(section),
            )
    return None


def detect_split_number_heading(lines: list[PageLine], index: int) -> ClauseHeading | None:
    line = lines[index].text.strip()
    if not re.fullmatch(r"\d{1,2}(?:\.\d+){0,5}|[A-Z]\.\d+(?:\.\d+){0,5}", line):
        return None
    if index + 1 >= len(lines):
        return None
    next_line = lines[index + 1].text.strip()
    if not next_line or is_toc_line(next_line) or is_repeated_pdf_noise(next_line):
        return None
    if len(next_line) > 120 or re.fullmatch(r"[\d .\-–—]+", next_line):
        return None
    if not contains_heading_text(next_line):
        return None
    return ClauseHeading(
        section=normalize_section(line),
        title=normalize_heading_title(next_line),
        page=lines[index].page,
        line_index=index,
        body_start_line_index=index + 2,
        level=section_level(line),
    )


def is_valid_annex_heading_context(lines: list[PageLine], index: int, title: str) -> bool:
    line = lines[index].text.strip()
    if not line.startswith(("Annex", "Appendix")) and not line.startswith("\u9644"):
        return False
    if title:
        return not looks_like_annex_reference_title(title)

    previous_line = previous_same_page_line(lines, index)
    next_line = next_same_page_line(lines, index)
    if previous_line and looks_like_annex_reference_continuation(previous_line):
        return False
    if next_line and re.match(r"^NOTE\s+The\s+quoted\s+annex", next_line, flags=re.IGNORECASE):
        return False
    return has_following_annex_title(lines, index)


def looks_like_annex_reference_title(title: str) -> bool:
    normalized = title.lower().strip(" .,:;锛屻€傦紱锛?")
    if not normalized:
        return False
    if re.match(
        r"^(?:of|in|for|from|to|by|with|states?|specifies?|provides?|contains?|describes?|"
        r"defines?|lists|is|are|was|were|shall|should|can|may)\b",
        normalized,
    ):
        return True
    if re.match(r"^\d", normalized):
        return True
    return False


def has_following_annex_title(lines: list[PageLine], index: int) -> bool:
    next_index = index + 1
    if next_index >= len(lines) or lines[next_index].page != lines[index].page:
        return False
    candidate = lines[next_index].text.strip()
    if re.fullmatch(r"\((?:informative|normative)\)", candidate, flags=re.IGNORECASE):
        title_index = next_index + 1
        return (
            title_index < len(lines)
            and lines[title_index].page == lines[index].page
            and is_probable_annex_title_line(lines[title_index].text.strip())
        )
    return is_probable_annex_title_line(candidate)


def annex_title_and_body_start(lines: list[PageLine], index: int, title: str) -> tuple[str, int]:
    if title and re.fullmatch(r"\((?:informative|normative)\)", title, flags=re.IGNORECASE):
        title_index = index + 1
        if title_index < len(lines) and lines[title_index].page == lines[index].page:
            candidate = lines[title_index].text.strip()
            if is_probable_annex_title_line(candidate):
                return normalize_heading_title(candidate), title_index + 1
        return title, index + 1
    if title:
        return title, index + 1
    next_index = index + 1
    if next_index >= len(lines) or lines[next_index].page != lines[index].page:
        return title, index + 1

    first_candidate = lines[next_index].text.strip()
    if re.fullmatch(r"\((?:informative|normative)\)", first_candidate, flags=re.IGNORECASE):
        title_index = next_index + 1
        if title_index < len(lines) and lines[title_index].page == lines[index].page:
            candidate = lines[title_index].text.strip()
            if is_probable_annex_title_line(candidate):
                return normalize_heading_title(candidate), title_index + 1
        return title, next_index + 1

    if is_probable_annex_title_line(first_candidate):
        return normalize_heading_title(first_candidate), next_index + 1
    return title, index + 1


def is_probable_annex_title_line(line: str) -> bool:
    if not line or is_toc_line(line) or is_repeated_pdf_noise(line):
        return False
    if len(line) > 120:
        return False
    if re.match(r"^(NOTE|Note:)\b", line):
        return False
    if re.match(r"^[a-z]\.", line) or re.match(r"^\d{1,2}[.)]\s+", line):
        return False
    if detect_inline_numbered_section(line):
        return False
    return contains_heading_text(line)


def detect_inline_numbered_section(line: str) -> bool:
    return bool(NUMBERED_HEADING_RE.match(line) or APPENDIX_SECTION_RE.match(line))


def previous_same_page_line(lines: list[PageLine], index: int) -> str:
    if index <= 0 or lines[index - 1].page != lines[index].page:
        return ""
    return lines[index - 1].text.strip()


def next_same_page_line(lines: list[PageLine], index: int) -> str:
    if index + 1 >= len(lines) or lines[index + 1].page != lines[index].page:
        return ""
    return lines[index + 1].text.strip()


def looks_like_annex_reference_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or is_repeated_pdf_noise(stripped):
        return False
    lowered = stripped.lower()
    if re.search(
        r"(?:\b(?:in|of|to|from|by|per|see)|shown in|given in|provided in|defined in|specified in|"
        r"accordance with|conformance with|conformity with|according to)\s*$",
        lowered,
    ):
        return True
    if len(stripped) > 24 and contains_heading_text(stripped) and not re.search(r"[.!?:;。！？：；]\s*$", stripped):
        return True
    return False


def dedupe_adjacent_headings(headings: list[ClauseHeading]) -> list[ClauseHeading]:
    deduped: list[ClauseHeading] = []
    for heading in headings:
        if deduped and heading.section == deduped[-1].section and heading.page == deduped[-1].page:
            continue
        deduped.append(heading)
    return deduped


def filter_clause_headings(headings: list[ClauseHeading]) -> list[ClauseHeading]:
    filtered: list[ClauseHeading] = []
    current_top_level: int | None = None
    last_single_integer_top: int | None = None
    in_annex_context = False
    for index, heading in enumerate(headings):
        if is_single_integer_section(heading.section):
            number = int(heading.section)
            if in_annex_context:
                continue
            if not is_trusted_single_integer_heading(heading.section, heading.title):
                continue
            if last_single_integer_top is None:
                if number != 1:
                    continue
            elif number != last_single_integer_top + 1:
                continue

        filtered.append(heading)
        numeric_top = numeric_top_level(heading.section)
        if numeric_top is not None:
            current_top_level = numeric_top
            in_annex_context = False
            if is_single_integer_section(heading.section):
                last_single_integer_top = numeric_top
        elif is_annex_heading_section(heading.section):
            current_top_level = None
            last_single_integer_top = None
            in_annex_context = True
        elif re.match(r"^(Annex|Appendix|附)", heading.section, flags=re.IGNORECASE):
            current_top_level = None
    return filtered


def has_child_heading(headings: list[ClauseHeading], index: int, section: str) -> bool:
    if not is_single_integer_section(section):
        return False
    current = int(section)
    prefix = f"{section}."
    for later in headings[index + 1 :]:
        if later.section.startswith(prefix):
            return True
        later_top = numeric_top_level(later.section)
        if later_top is not None and later_top != current:
            return False
        if re.match(r"^(Annex|Appendix|附)", later.section, flags=re.IGNORECASE):
            return False
    return False


def is_single_integer_section(section: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}", section))


def numeric_top_level(section: str) -> int | None:
    match = re.match(r"^(\d{1,2})(?:\.|$)", section)
    if not match:
        return None
    return int(match.group(1))


def is_annex_heading_section(section: str) -> bool:
    return section.lower().startswith(("annex", "appendix")) or section.startswith("\u9644")


def is_trusted_single_integer_heading(section: str, title: str) -> bool:
    if not is_single_integer_section(section):
        return False
    number = int(section)
    normalized = title.lower().strip(" .,:;锛屻€傦紱锛?")
    if number == 1:
        return normalized.startswith(("scope", "introduction")) or "\u8303\u56f4" in normalized or "\u5f15\u8a00" in normalized
    if number == 2:
        return "reference" in normalized or "applicable" in normalized or "\u5f15\u7528" in normalized or "\u89c4\u8303\u6027" in normalized
    if number == 3:
        return (
            "term" in normalized
            or "definition" in normalized
            or "abbreviat" in normalized
            or "symbol" in normalized
            or "\u672f\u8bed" in normalized
            or "\u5b9a\u4e49" in normalized
            or "\u7f29\u7565" in normalized
            or "\u7b26\u53f7" in normalized
        )
    return False


def is_canonical_top_heading(title: str) -> bool:
    normalized = title.lower().strip(" .,:;，。；：")
    canonical_keywords = {
        "scope",
        "reference",
        "applicable",
        "terms",
        "definition",
        "abbreviated",
        "symbol",
        "introduction",
        "overview",
        "general",
        "requirement",
        "verification",
        "validation",
        "test",
        "management",
        "process",
        "architecture",
        "functionality",
        "design",
        "analysis",
        "safety",
        "quality",
        "范围",
        "引用",
        "规范性",
        "术语",
        "定义",
        "缩略",
        "符号",
        "概述",
        "总则",
        "一般",
        "要求",
        "验证",
        "确认",
        "试验",
        "检验",
        "管理",
        "过程",
        "设计",
        "分析",
        "安全",
        "质量",
    }
    return any(keyword in normalized for keyword in canonical_keywords)


NUMBERED_HEADING_RE = re.compile(
    r"^(?P<section>(?:\d{1,2}(?:\.\d+){0,5}|[A-Z]\.\d+(?:\.\d+){0,5}))"
    r"(?:[.)、．]|\s+)+(?P<title>[^.。…·\d][^\n]{0,160})$"
)
ANNEX_HEADING_RE = re.compile(
    r"^(?P<section>(?:Annex|Appendix)\s+[A-Z]|附\s*录\s*[A-ZＡ-Ｚ]|附录\s*[A-ZＡ-Ｚ])"
    r"(?:\s+|[：:、．.])*(?P<title>[^\n]{0,160})$",
    re.IGNORECASE,
)
APPENDIX_SECTION_RE = re.compile(r"^(?P<section>[A-Z]\.\d+(?:\.\d+){0,5})(?:\s+)+(?P<title>[^\n]{1,160})$")


def normalize_pdf_symbols(text: str) -> str:
    text = text.translate(
        {
            ord("\u00ad"): None,
            ord("\u2010"): "-",
            ord("\u2011"): "-",
            ord("\u2012"): "-",
            ord("\u2013"): "-",
            ord("\u2014"): "-",
            ord("\u2212"): "-",
            ord("\u2043"): "-",
            ord("\u00b7"): "-",
            ord("\u2022"): "-",
            ord("\uf0b7"): "-",
            ord("\uf0fc"): "-",
        }
    )
    text = re.sub(r"[\u2000-\u200b\u202f\u205f\u3000]", " ", text)
    return text


def normalize_pdf_line(line: str) -> str:
    line = html.unescape(line)
    line = line.replace("\u00a0", " ")
    line = normalize_pdf_symbols(line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def reflow_pdf_text(lines: list[str]) -> str:
    segments: list[str] = []
    current = ""
    current_structural = False

    for raw_line in lines:
        line = normalize_pdf_line(raw_line)
        if not line:
            continue
        structural = is_structural_pdf_line(line)
        if not current:
            current = line
            current_structural = structural
            continue
        if structural:
            segments.append(current.strip())
            current = line
            current_structural = True
            continue
        if should_merge_pdf_line(current, line, current_structural=current_structural):
            current = merge_pdf_lines(current, line)
        else:
            segments.append(current.strip())
            current = line
            current_structural = False

    if current:
        segments.append(current.strip())
    return normalize_text("\n".join(segment for segment in segments if segment.strip()))


def should_merge_pdf_line(previous: str, line: str, *, current_structural: bool) -> bool:
    if looks_like_standard_id_line(previous) and re.match(r"^[A-Za-z]", line):
        return True
    if is_heading_like_pdf_line(previous):
        return False
    if current_structural and is_list_or_note_line(previous):
        return True
    if is_table_like_line(previous) or is_table_like_line(line):
        return False
    if re.search(r"[:;,]\s*$", previous):
        return True
    if re.search(r"[.!?。！？]\s*$", previous):
        return False
    if re.match(r"^[a-z),;]", line):
        return True
    if len(previous) >= 35 and len(line) >= 12:
        return True
    return False


def merge_pdf_lines(previous: str, line: str) -> str:
    if re.search(r"[A-Za-z]-$", previous):
        return previous[:-1] + line
    return f"{previous} {line}"


def is_structural_pdf_line(line: str) -> bool:
    if is_heading_like_pdf_line(line):
        return True
    if is_list_or_note_line(line):
        return True
    return False


def is_heading_like_pdf_line(line: str) -> bool:
    if detect_inline_numbered_section(line):
        return True
    if re.match(r"^(Annex|Appendix)\s+[A-Z]\b", line, flags=re.IGNORECASE):
        return True
    if re.match(r"^(Table|Figure)\s+[A-Z0-9.-]+", line, flags=re.IGNORECASE):
        return True
    return False


def is_list_or_note_line(line: str) -> bool:
    if re.match(r"^(?:[-*]|\(?[a-z]\)|[a-z]\.|\d{1,2}[.)])\s+", line):
        return True
    if re.match(r"^(NOTE|Note:|CAUTION|WARNING)\b", line):
        return True
    return False


def is_table_like_line(line: str) -> bool:
    tokens = line.split()
    if len(tokens) < 5:
        return False
    numeric_or_short = sum(1 for token in tokens if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?(?:[-/]\d+(?:[.,]\d+)?)?%?|[A-Z]{1,4}|[-]+", token))
    if numeric_or_short / len(tokens) >= 0.6:
        return True
    if sum(1 for token in tokens if token.lower() in {"yes", "no", "n/a"}) >= 4:
        return True
    return False


def looks_like_standard_id_line(line: str) -> bool:
    return bool(re.fullmatch(r"(?:ECSS|GJB|QJ|GB/T|NASA)[- A-Z0-9./]+(?:Rev\.\d+)?", line.strip(), flags=re.IGNORECASE))


def is_content_line(line: str) -> bool:
    if not line:
        return False
    if re.fullmatch(r"\d{1,4}", line):
        return False
    return True


def is_toc_line(line: str) -> bool:
    return bool(re.search(r"(\.{3,}|(?:\.\s*){6,}|…{2,})\s*\d+\s*$", line))


def is_repeated_pdf_noise(line: str) -> bool:
    stripped = line.strip()
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", stripped):
        return True
    if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", stripped):
        return True
    if stripped in {"Space engineering", "Space product assurance", "Space management", "Requirements & Standards Division"}:
        return True
    return False


def normalize_section(section: str) -> str:
    return re.sub(r"\s+", " ", section.replace("．", ".")).strip()


def normalize_heading_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .。:：-–—")
    return title


def is_valid_heading(section: str, title: str, full_line: str, *, page: int | None = None) -> bool:
    if not section:
        return False
    if is_toc_line(full_line):
        return False
    if section.count(".") > 5:
        return False
    if re.fullmatch(r"\d{3,}", section):
        return False
    if re.fullmatch(r"\d{1,2}", section):
        try:
            if int(section) > 20:
                return False
        except ValueError:
            pass
    if title and len(title) > 140:
        return False
    if title and page is not None and looks_like_toc_title_with_page(title, current_page=page):
        return False
    if re.fullmatch(r"\d{1,2}", section):
        if len(title) > 70:
            return False
        if title and not re.match(r"^[A-Za-z\u4e00-\u9fff]", title):
            return False
        if re.match(r"^[a-z]", title):
            return False
    if title and not contains_heading_text(title):
        return False
    if not title and not re.match(r"^(Annex|Appendix|附)", section, flags=re.IGNORECASE):
        return False
    return True


def is_probably_spurious_single_number_clause(heading: ClauseHeading, content: str) -> bool:
    if not re.fullmatch(r"\d{1,2}", heading.section):
        return False
    if len(content) >= 120:
        return False
    title = heading.title.lower()
    if title.rstrip(",，") == "requirement text":
        return True
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", content)
    if tokens and sum(1 for token in tokens if token.lower() == "no") / len(tokens) > 0.6:
        return True
    good_keywords = {
        "scope",
        "reference",
        "terms",
        "definition",
        "abbreviated",
        "introduction",
        "general",
        "requirement",
        "verification",
        "validation",
        "test",
        "applicability",
        "overview",
        "purpose",
        "process",
        "design",
        "safety",
        "quality",
        "management",
        "范围",
        "引用",
        "术语",
        "定义",
        "缩略",
        "要求",
        "试验",
        "检验",
        "验证",
        "设计",
        "管理",
    }
    return not any(keyword in title for keyword in good_keywords)


def looks_like_toc_title_with_page(title: str, *, current_page: int) -> bool:
    match = re.search(r"\s(\d{1,3})$", title.strip())
    if not match:
        return False
    try:
        referenced_page = int(match.group(1))
    except ValueError:
        return False
    return referenced_page != current_page


def contains_heading_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))


def section_level(section: str) -> int:
    if re.match(r"^(Annex|Appendix|附)", section, flags=re.IGNORECASE):
        return 1
    return section.count(".") + 1


def infer_standard_id(source_path: Path) -> str:
    stem = source_path.stem
    stem = re.sub(r"^\[OCR\]_", "", stem, flags=re.IGNORECASE)
    patterns = [
        r"(ECSS-[A-Z]-[A-Z]+-\d+(?:-\d+)?[A-Z]?)",
        r"(ECSS-[A-Z]-[A-Z]+-\d+(?:-\d+)?[A-Z]?(?:[-_ ]?Rev\.?\s*\d+)?)",
        r"(GJB(?:[- ]?Z)?\s*[\w.]+(?:-\d{4})?)",
        r"(QJ\s*[\w.]+(?:-\d{4})?)",
        r"(GB/?T\s*[\w.]+(?:-\d{4})?)",
        r"(GBT\s*[\w.]+(?:-\d{4})?)",
        r"(NASA[- ][A-Z0-9-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return normalize_text(match.group(1)).replace("_", " ")
    return stem.split(" ")[0]


def load_pdf_backend() -> str:
    try:
        import pypdf  # noqa: F401

        return "pypdf"
    except ImportError:
        pass
    try:
        import pdfplumber  # noqa: F401

        return "pdfplumber"
    except ImportError:
        return ""


def build_corpus_record(
    manifest_record: dict[str, Any],
    *,
    generated_at: str,
    title: str,
    content: str,
    record_kind: str,
    part_index: int,
    source_record: int | None = None,
    page: int | None = None,
    page_end: int | None = None,
    section: str | None = None,
    section_title: str | None = None,
    standard_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_text(content)
    if not normalized:
        raise ParseFailure("empty_record", "Record content is empty after cleanup.")
    relative_path = str(manifest_record.get("relative_path") or "")
    source_type = source_type_for(manifest_record)
    doc_id = stable_id(f"{manifest_record.get('hash')}|{relative_path}")
    record_key = f"{relative_path}|{record_kind}|{source_record or ''}|{page or ''}|{part_index}|{normalized[:80]}"
    metadata = {
        "top_level": manifest_record.get("top_level"),
        "relative_path": relative_path,
        "mvp_reason": manifest_record.get("mvp_reason"),
        "parse_route": manifest_record.get("parse_route"),
        "hash": manifest_record.get("hash"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "schema_version": "satdesign.rag_corpus.v1",
        "generated_at": generated_at,
        "doc_id": doc_id,
        "record_id": stable_id(record_key),
        "title": normalize_text(title) or str(manifest_record.get("filename") or ""),
        "content": normalized,
        "source_path": manifest_record.get("source_path"),
        "relative_path": relative_path,
        "source_type": source_type,
        "authority_level": authority_level_for(source_type),
        "record_kind": record_kind,
        "part_index": part_index,
        "source_record": source_record,
        "page": page,
        "page_start": page,
        "page_end": page_end,
        "section": section,
        "section_title": section_title,
        "standard_id": standard_id,
        "metadata": metadata,
    }


def source_type_for(manifest_record: dict[str, Any]) -> str:
    reason = str(manifest_record.get("mvp_reason") or "")
    if reason == "strong_evidence":
        return "standard"
    if reason == "lesson_evidence":
        return "lesson"
    if reason == "wiki_auxiliary":
        return "wiki"
    if reason == "article_auxiliary":
        return "article"
    return "source"


def authority_level_for(source_type: str) -> str:
    if source_type == "standard":
        return "standard"
    if source_type == "lesson":
        return "lesson"
    return "auxiliary"


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, parts[2]


def clean_markdown(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|tr|li|h\d|table|tbody|thead)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def split_text_sections(text: str, *, max_record_chars: int) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    heading_sections: list[str] = []
    current: list[str] = []
    for line in cleaned.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line) and current:
            heading_sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        heading_sections.append("\n".join(current).strip())
    if not heading_sections:
        heading_sections = [cleaned]

    parts: list[str] = []
    for section in heading_sections:
        parts.extend(split_long_text(section, max_record_chars=max_record_chars))
    return [part for part in parts if part]


def split_long_text(text: str, *, max_record_chars: int) -> list[str]:
    if len(text) <= max_record_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_record_chars:
            if current:
                available = max_record_chars - len(current) - 2
                if available > 120:
                    chunks.append(f"{current}\n\n{paragraph[:available]}".strip())
                    paragraph = paragraph[available:]
                else:
                    chunks.append(current.strip())
                current = ""
            chunks.extend(paragraph[start : start + max_record_chars].strip() for start in range(0, len(paragraph), max_record_chars) if paragraph[start : start + max_record_chars].strip())
            continue
        if current and len(current) + len(paragraph) + 2 > max_record_chars:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current.strip())
    return chunks


def nasa_lesson_content(payload: dict[str, Any]) -> str:
    lines = []
    for field in NASA_LESSON_FIELDS:
        value = payload.get(field)
        if value and str(value).strip() and str(value).strip().lower() != "none":
            lines.append(f"## {field}\n{value}")
    return "\n\n".join(lines)


def jsonl_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in [
        "Lesson Number",
        "Lesson Date",
        "Submitting Organization",
        "Program/Project Phase",
        "Mission Directorate(s)",
        "Topic(s)",
        "source_id",
    ]:
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def is_nasa_lesson(manifest_record: dict[str, Any], payload: dict[str, Any]) -> bool:
    return manifest_record.get("mvp_reason") == "lesson_evidence" or "Lesson Number" in payload


def read_text_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseFailure("invalid_manifest", f"Line {line_number}: {exc}") from exc


def build_failure(manifest_record: dict[str, Any], reason: str, message: str, generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": "satdesign.rag_parse_failure.v1",
        "generated_at": generated_at,
        "reason": reason,
        "message": message,
        "source_path": manifest_record.get("source_path"),
        "relative_path": manifest_record.get("relative_path"),
        "parse_route": manifest_record.get("parse_route"),
        "mvp_reason": manifest_record.get("mvp_reason"),
    }


def render_prepare_report(report: dict[str, Any]) -> str:
    lines = [
        "# RAG MVP Parse Report",
        "",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Output: `{report['out_dir']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Selected files: {report['selected_files']}",
        f"- Parsed files: {report['parsed_files']}",
        f"- Failed files: {report['failed_files']}",
        f"- Corpus records: {report['records']}",
        "",
        "## Source Types",
        "",
        "| Type | Records |",
        "| --- | ---: |",
    ]
    for key, value in report["source_types"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failures", "", "| Reason | Files |", "| --- | ---: |"])
    for key, value in report["failures"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


class ParseFailure(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

