"""Phase-0 source library scanning for RAG corpus preparation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


PUBLIC_DATA_MVP_PROFILE = "public_data_mvp"

STRONG_EVIDENCE_DIRS = {
    "ECSS标准",
    "ECSS手册",
    "GJB",
    "ISO标准",
    "QJ标准",
    "GB和GBT国标",
    "NASA标准",
    "美国航天中心标准和手册",
    "美军标",
}

DEFERRED_TOP_LEVEL_DIRS = {
    "中文教材": "textbook_deferred",
    "英文教材": "textbook_deferred",
}

ARTICLE_ENGINEERING_DIRS = {
    "仿真优化工匠",
    "复材工艺笔记",
    "宇航抗辐射",
    "航天质量那点事儿",
    "邱工笔记",
    "模态空间",
}

ARTICLE_ENGINEERING_KEYWORDS = {
    "结构",
    "热控",
    "电源",
    "接口",
    "可靠",
    "抗辐",
    "辐射",
    "仿真",
    "模态",
    "振动",
    "复材",
    "材料",
    "工艺",
    "测试",
    "试验",
    "质量",
    "轨道",
    "总线",
    "电子",
    "FPGA",
    "COTS",
}

TEXT_EXTENSIONS = {".md", ".txt", ".json", ".jsonl"}
PDF_EXTENSIONS = {".pdf"}
OFFICE_EXTENSIONS = {".docx", ".doc", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}

HashMode = Literal["fingerprint", "content", "none"]


@dataclass(frozen=True)
class ScanOptions:
    root: Path
    out_dir: Path
    profile: str = PUBLIC_DATA_MVP_PROFILE
    hash_mode: HashMode = "fingerprint"
    content_hash_max_mb: int = 64


def scan_public_data_root(options: ScanOptions) -> dict[str, Any]:
    """Scan a source library and write manifest + summary artifacts."""

    root = options.root
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"RAG source root does not exist or is not a directory: {root}")

    generated_at = datetime.now(timezone.utc).isoformat()
    records = list(iter_manifest_records(root, generated_at=generated_at, options=options))
    summary = build_scan_summary(records, root=root, generated_at=generated_at, profile=options.profile)

    options.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = options.out_dir / "manifest.jsonl"
    summary_json_path = options.out_dir / "scan_summary.json"
    summary_md_path = options.out_dir / "scan_summary.md"

    write_jsonl(manifest_path, records)
    write_json(summary_json_path, summary)
    summary_md_path.write_text(render_summary_markdown(summary), encoding="utf-8")

    return {
        "root": str(root),
        "profile": options.profile,
        "generated_at": generated_at,
        "manifest_path": str(manifest_path),
        "summary_json_path": str(summary_json_path),
        "summary_md_path": str(summary_md_path),
        "files": summary["totals"]["files"],
        "size_bytes": summary["totals"]["size_bytes"],
        "mvp_actions": summary["mvp_actions"],
    }


def iter_manifest_records(root: Path, *, generated_at: str, options: ScanOptions) -> Iterable[dict[str, Any]]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        stat = path.stat()
        relative_path = path.relative_to(root)
        top_level = _top_level(relative_path)
        extension = path.suffix.lower()
        policy = classify_mvp_policy(relative_path, extension)
        record = {
            "schema_version": "satdesign.rag_manifest.v1",
            "generated_at": generated_at,
            "profile": options.profile,
            "source_root": str(root),
            "source_path": str(path),
            "relative_path": relative_path.as_posix(),
            "top_level": top_level,
            "filename": path.name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "hash": file_hash(path, stat.st_size, stat.st_mtime_ns, mode=options.hash_mode, max_mb=options.content_hash_max_mb),
            "parse_route": parse_route_for(extension),
            **policy,
        }
        yield record


def classify_mvp_policy(relative_path: Path, extension: str) -> dict[str, str]:
    top_level = _top_level(relative_path)
    rel_text = relative_path.as_posix()

    if extension in ARCHIVE_EXTENSIONS:
        return _policy("defer", "archive_deferred", "Archives are unpacked only after source selection.")
    if extension in IMAGE_EXTENSIONS:
        return _policy("exclude", "image_reference_only", "Images are retained as references; full OCR is outside MVP.")
    if extension == ".py":
        return _policy("exclude", "tooling_file", "Source utility files are not corpus content.")

    if top_level in STRONG_EVIDENCE_DIRS:
        if extension in PDF_EXTENSIONS | OFFICE_EXTENSIONS | TEXT_EXTENSIONS:
            return _policy("include", "strong_evidence", "First-batch standard or handbook source.")
        return _policy("defer", "unsupported_standard_extension", "Standard source exists but extension needs a parser decision.")

    if top_level == "NASA经验教训库liss_merged_lessons_filtered.jsonl" and extension == ".jsonl":
        return _policy("include", "lesson_evidence", "NASA lesson record source.")

    if top_level == "卫星百科数据":
        if extension == ".md":
            return _policy("include", "wiki_auxiliary", "Satellite wiki Markdown source.")
        if extension in {".xlsx", ".xls"}:
            return _policy("defer", "wiki_metadata_table", "Spreadsheet may enrich metadata after MVP text ingestion.")
        return _policy("defer", "wiki_non_markdown", "MVP ingests wiki Markdown first.")

    if top_level == "公众号文章":
        if extension == ".md" and is_engineering_article(relative_path):
            return _policy("include", "article_auxiliary", "Engineering article Markdown source.")
        if extension == ".md":
            return _policy("defer", "article_non_engineering_or_uncertain", "Article Markdown is outside the first engineering whitelist.")
        if extension == ".html":
            return _policy("defer", "article_html_fallback", "HTML is used only when Markdown is missing.")
        if extension == ".pdf":
            return _policy("defer", "article_pdf_deferred", "Article PDFs need separate cleanup.")
        return _policy("exclude", "article_attachment", "Article attachments are not ingested in MVP.")

    if top_level in DEFERRED_TOP_LEVEL_DIRS:
        return _policy("defer", DEFERRED_TOP_LEVEL_DIRS[top_level], "Directory is intentionally outside the first MVP batch.")

    if extension in {".xlsx", ".xls"}:
        return _policy("defer", "metadata_spreadsheet", "Spreadsheet is reserved for metadata enrichment.")

    return _policy("exclude", "not_in_mvp_profile", "Source is outside the public-data MVP profile.")


def is_engineering_article(relative_path: Path) -> bool:
    parts = list(relative_path.parts)
    publisher = parts[1] if len(parts) > 1 else ""
    rel_text = relative_path.as_posix()
    return publisher in ARTICLE_ENGINEERING_DIRS or any(keyword in rel_text for keyword in ARTICLE_ENGINEERING_KEYWORDS)


def parse_route_for(extension: str) -> str:
    if extension == ".pdf":
        return "pdf_text_first"
    if extension == ".md":
        return "markdown"
    if extension == ".html":
        return "html_fallback"
    if extension == ".jsonl":
        return "jsonl_records"
    if extension == ".json":
        return "json_document"
    if extension in {".docx", ".doc"}:
        return "word_document"
    if extension in {".xlsx", ".xls"}:
        return "spreadsheet_metadata"
    if extension in IMAGE_EXTENSIONS:
        return "image_reference_only"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive_deferred"
    return "unsupported"


def file_hash(path: Path, size: int, mtime_ns: int, *, mode: HashMode, max_mb: int) -> str:
    if mode == "none":
        return ""
    if mode == "fingerprint":
        digest = hashlib.sha1(f"{path.name}|{size}|{mtime_ns}".encode("utf-8")).hexdigest()
        return f"fingerprint:{digest}"
    if mode == "content":
        max_bytes = max_mb * 1024 * 1024
        if size > max_bytes:
            digest = hashlib.sha1(f"{path.name}|{size}|{mtime_ns}".encode("utf-8")).hexdigest()
            return f"fingerprint:{digest}"
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return f"sha1:{digest.hexdigest()}"
    raise ValueError(f"Unsupported hash mode: {mode}")


def build_scan_summary(records: list[dict[str, Any]], *, root: Path, generated_at: str, profile: str) -> dict[str, Any]:
    total_size = sum(int(record["size_bytes"]) for record in records)
    by_extension = Counter(record["extension"] or "<none>" for record in records)
    by_action = Counter(record["mvp_action"] for record in records)
    by_parse_route = Counter(record["parse_route"] for record in records)

    dir_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "size_bytes": 0, "extensions": Counter(), "actions": Counter()})
    for record in records:
        stats = dir_stats[record["top_level"]]
        stats["files"] += 1
        stats["size_bytes"] += int(record["size_bytes"])
        stats["extensions"][record["extension"] or "<none>"] += 1
        stats["actions"][record["mvp_action"]] += 1

    directories = []
    for name, stats in sorted(dir_stats.items(), key=lambda item: item[1]["size_bytes"], reverse=True):
        directories.append(
            {
                "name": name,
                "files": stats["files"],
                "size_bytes": stats["size_bytes"],
                "size_mb": round(stats["size_bytes"] / 1024 / 1024, 2),
                "extensions": _top_counter(stats["extensions"]),
                "actions": dict(sorted(stats["actions"].items())),
            }
        )

    return {
        "schema_version": "satdesign.rag_scan_summary.v1",
        "root": str(root),
        "profile": profile,
        "generated_at": generated_at,
        "totals": {
            "files": len(records),
            "size_bytes": total_size,
            "size_gb": round(total_size / 1024 / 1024 / 1024, 2),
        },
        "mvp_actions": dict(sorted(by_action.items())),
        "parse_routes": dict(sorted(by_parse_route.items())),
        "extensions": _top_counter(by_extension, limit=20),
        "directories": directories,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    lines = [
        "# RAG MVP Scan Summary",
        "",
        f"- Source root: `{summary['root']}`",
        f"- Profile: `{summary['profile']}`",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Total files: {totals['files']}",
        f"- Total size: {totals['size_gb']} GB",
        "",
        "## MVP Actions",
        "",
        "| Action | Files |",
        "| --- | ---: |",
    ]
    for action, count in summary["mvp_actions"].items():
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## Top-Level Directories",
            "",
            "| Directory | Files | Size MB | Actions | Top Extensions |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in summary["directories"]:
        actions = ", ".join(f"{key}:{value}" for key, value in item["actions"].items())
        extensions = ", ".join(f"{key}:{value}" for key, value in item["extensions"].items())
        lines.append(f"| {item['name']} | {item['files']} | {item['size_mb']} | {actions} | {extensions} |")

    lines.extend(
        [
            "",
            "## Parse Routes",
            "",
            "| Route | Files |",
            "| --- | ---: |",
        ]
    )
    for route, count in summary["parse_routes"].items():
        lines.append(f"| {route} | {count} |")

    lines.extend(
        [
            "",
            "## Phase-0 Notes",
            "",
            "- This scan does not parse document bodies or ingest vectors.",
            "- `include` means selected for the first MVP preparation pass.",
            "- `defer` means intentionally kept out of MVP ingestion but available for later phases.",
            "- `exclude` means not corpus text for the MVP profile.",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _policy(action: str, reason: str, note: str) -> dict[str, str]:
    return {"mvp_action": action, "mvp_reason": reason, "mvp_note": note}


def _top_level(relative_path: Path) -> str:
    return relative_path.parts[0] if relative_path.parts else "."


def _top_counter(counter: Counter[str], *, limit: int = 8) -> dict[str, int]:
    return dict(counter.most_common(limit))

