"""Generate a retrieval evaluation set for the first 10 ECSS PDFs.

The output is meant for comparing retrieval systems. Questions are generated
from parsed clause records, and each non-negative question carries one or more
gold chunks from the corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = r"C:\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl"
DEFAULT_PDF_ROOT = r"C:\baidunetdiskdownload\公开数据\ECSS手册"
DEFAULT_OUT_DIR = "runs/eval_sets"

QUERY_COUNTS = {
    "definition": 15,
    "abbreviation": 10,
    "clause_locator": 20,
    "requirement_process": 20,
    "parameter_table": 15,
    "method_explanation": 20,
    "cross_doc_compare": 10,
    "scenario": 5,
    "negative": 5,
}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = first_pdfs(Path(args.pdf_root), args.pdf_count)
    records = load_records(Path(args.corpus), set(pdfs))
    by_pdf = defaultdict(list)
    for record in records:
        by_pdf[record["pdf"]].append(record)

    questions = build_questions(pdfs, by_pdf)
    if len(questions) != sum(QUERY_COUNTS.values()):
        raise RuntimeError(f"expected {sum(QUERY_COUNTS.values())} questions, got {len(questions)}")

    base = out_dir / "ecss_first10_120"
    write_jsonl(base.with_suffix(".jsonl"), questions)
    write_csv(base.with_suffix(".csv"), questions)
    write_summary(base.with_suffix(".md"), questions, pdfs, by_pdf)
    print(
        json.dumps(
            {
                "jsonl": str(base.with_suffix(".jsonl")),
                "csv": str(base.with_suffix(".csv")),
                "summary": str(base.with_suffix(".md")),
                "questions": len(questions),
                "query_type_counts": Counter(item["query_type"] for item in questions),
                "pdfs": pdfs,
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--pdf-root", default=DEFAULT_PDF_ROOT)
    parser.add_argument("--pdf-count", type=int, default=10)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def first_pdfs(root: Path, count: int) -> list[str]:
    return [path.name for path in sorted(root.glob("*.pdf"), key=lambda item: item.name)[:count]]


def load_records(corpus: Path, pdfs: set[str]) -> list[dict[str, Any]]:
    records = []
    with corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            pdf = os.path.basename(raw.get("source_path") or "")
            if pdf not in pdfs:
                continue
            metadata = raw.get("metadata") or {}
            content = normalize_space(raw.get("content") or "")
            if not content:
                continue
            record = {
                "record_id": str(raw.get("record_id")),
                "doc_id": str(raw.get("doc_id")),
                "pdf": pdf,
                "title": raw.get("title") or metadata.get("title") or Path(pdf).stem,
                "source_path": raw.get("source_path") or "",
                "section": raw.get("section") or metadata.get("section"),
                "section_title": raw.get("section_title") or metadata.get("section_title") or "",
                "page_start": raw.get("page_start") or raw.get("page"),
                "page_end": raw.get("page_end") or raw.get("page"),
                "record_kind": raw.get("record_kind") or metadata.get("record_kind"),
                "content": content,
            }
            records.append(record)
    return records


def build_questions(pdfs: list[str], by_pdf: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    used: set[str] = set()
    used_contexts: set[tuple[str, str, str]] = set()

    definition_pool = [r for pdf in pdfs for r in by_pdf[pdf] if is_definition_record(r)]
    abbreviation_pool = [r for pdf in pdfs for r in by_pdf[pdf] if is_abbreviation_record(r)]
    locator_pool = [r for pdf in pdfs for r in by_pdf[pdf] if good_title(r)]
    requirement_pool = [r for pdf in pdfs for r in by_pdf[pdf] if keyword_record(r, REQUIREMENT_WORDS)]
    parameter_pool = [r for pdf in pdfs for r in by_pdf[pdf] if has_parameter_signal(r)]
    method_pool = [r for pdf in pdfs for r in by_pdf[pdf] if keyword_record(r, METHOD_WORDS)]

    for r in balanced_pick(definition_pool, QUERY_COUNTS["definition"], used, used_contexts):
        term = clean_term(r["section_title"])
        items.append(make_item("definition", "easy", f'In {r["pdf"]}, how is "{term}" defined?', [r]))

    for r in balanced_pick(abbreviation_pool, QUERY_COUNTS["abbreviation"], used, used_contexts):
        items.append(
            make_item(
                "abbreviation",
                "easy",
                f'Which abbreviated terms or symbols are listed in section {label(r)} of {r["pdf"]}?',
                [r],
            )
        )

    for r in balanced_pick(locator_pool, QUERY_COUNTS["clause_locator"], used, used_contexts):
        items.append(
            make_item(
                "clause_locator",
                "easy",
                f'Which section of {r["pdf"]} discusses "{clean_term(r["section_title"])}"?',
                [r],
                expected_answer=f'{r["pdf"]}, section {label(r)}.',
            )
        )

    for r in balanced_pick(requirement_pool, QUERY_COUNTS["requirement_process"], used, used_contexts):
        items.append(
            make_item(
                "requirement_process",
                "medium",
                f'According to {r["pdf"]}, what does section {label(r)} say about {topic_phrase(r)}?',
                [r],
            )
        )

    for r in balanced_pick(parameter_pool, QUERY_COUNTS["parameter_table"], used, used_contexts):
        items.append(
            make_item(
                "parameter_table",
                "medium",
                f'Find the ECSS clause in {r["pdf"]} that provides numerical, table, unit, or threshold information about {topic_phrase(r)}.',
                [r],
            )
        )

    for r in balanced_pick(method_pool, QUERY_COUNTS["method_explanation"], used, used_contexts):
        items.append(
            make_item(
                "method_explanation",
                "medium",
                f'What method, test, analysis, or verification guidance is given in {r["pdf"]} section {label(r)}?',
                [r],
            )
        )

    items.extend(cross_doc_questions(pdfs, by_pdf, QUERY_COUNTS["cross_doc_compare"]))
    items.extend(scenario_questions(pdfs, by_pdf, QUERY_COUNTS["scenario"]))
    items.extend(negative_questions(QUERY_COUNTS["negative"]))

    for index, item in enumerate(items, start=1):
        item["id"] = f"ecss10_eval_{index:03d}"

    duplicated = [question for question, count in Counter(item["question"] for item in items).items() if count > 1]
    if duplicated:
        raise RuntimeError(f"duplicate questions generated: {duplicated[:3]}")
    return items


def balanced_pick(
    pool: list[dict[str, Any]],
    count: int,
    used: set[str],
    used_contexts: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    selected = []
    by_pdf = defaultdict(list)
    for record in pool:
        if record["record_id"] in used or record_context_key(record) in used_contexts:
            continue
        by_pdf[record["pdf"]].append(record)

    pdfs = sorted(by_pdf)
    cursor = 0
    while len(selected) < count and any(by_pdf.values()):
        pdf = pdfs[cursor % len(pdfs)]
        cursor += 1
        while by_pdf[pdf]:
            record = by_pdf[pdf].pop(0)
            context_key = record_context_key(record)
            if record["record_id"] not in used and context_key not in used_contexts:
                used.add(record["record_id"])
                used_contexts.add(context_key)
                selected.append(record)
                break
        if cursor > len(pdfs) * 500:
            break
    if len(selected) < count:
        raise RuntimeError(f"not enough records selected: {len(selected)} / {count}")
    return selected


def record_context_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["pdf"],
        str(record.get("section") or ""),
        clean_term(record.get("section_title") or "").lower(),
    )


def make_item(
    query_type: str,
    difficulty: str,
    question: str,
    gold_records: list[dict[str, Any]],
    *,
    expected_answer: str | None = None,
) -> dict[str, Any]:
    gold = [gold_context(record) for record in gold_records]
    if expected_answer is None:
        expected_answer = " ".join(g["answer_span"] for g in gold)
    return {
        "id": "",
        "question": question,
        "query_type": query_type,
        "difficulty": difficulty,
        "expected_answer": expected_answer,
        "expected_pdfs": [record["pdf"] for record in gold_records],
        "expected_record_ids": [record["record_id"] for record in gold_records],
        "gold_context": gold,
        "negative": False,
    }


def gold_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "doc_id": record["doc_id"],
        "pdf": record["pdf"],
        "source_path": record["source_path"],
        "section": record["section"],
        "section_title": record["section_title"],
        "page_start": record["page_start"],
        "page_end": record["page_end"],
        "answer_span": excerpt(record["content"]),
    }


def cross_doc_questions(pdfs: list[str], by_pdf: dict[str, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    pairs = [
        (pdfs[0], "verification", pdfs[1], "test programme"),
        (pdfs[0], "analysis", pdfs[2], "radiation"),
        (pdfs[2], "radiation", pdfs[9], "fault"),
        (pdfs[4], "multipactor", pdfs[6], "electrical"),
        (pdfs[7], "EMC", pdfs[6], "bonding"),
        (pdfs[5], "battery", pdfs[4], "power"),
        (pdfs[3], "TRL", pdfs[0], "qualification"),
        (pdfs[8], "interface", pdfs[1], "EGSE"),
        (pdfs[9], "ASIC", pdfs[2], "single event"),
        (pdfs[7], "conducted", pdfs[7], "radiated"),
        (pdfs[6], "grounding", pdfs[7], "LISN"),
        (pdfs[1], "AIT", pdfs[0], "verification"),
        (pdfs[4], "power", pdfs[5], "battery"),
        (pdfs[9], "memory", pdfs[2], "radiation"),
        (pdfs[3], "technology", pdfs[1], "qualification"),
    ]
    questions = []
    for left_pdf, left_kw, right_pdf, right_kw in pairs[:count]:
        left = best_record(by_pdf[left_pdf], left_kw)
        right = best_record(by_pdf[right_pdf], right_kw)
        questions.append(
            make_item(
                "cross_doc_compare",
                "hard",
                f'Compare the guidance related to "{left_kw}" in {left_pdf} with "{right_kw}" in {right_pdf}. Which clauses should be retrieved?',
                [left, right],
                expected_answer=f'Retrieve {left_pdf} section {label(left)} and {right_pdf} section {label(right)}.',
            )
        )
    return questions


def scenario_questions(pdfs: list[str], by_pdf: dict[str, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    scenarios = [
        (pdfs[0], "verification matrix", "I need evidence for building a satellite verification matrix and selecting verification methods."),
        (pdfs[1], "AIT Plan", "I am planning assembly, integration, and test activities for a spacecraft."),
        (pdfs[2], "single event", "I need radiation guidance for single event effects on spacecraft electronics."),
        (pdfs[3], "TRL", "I need to assess technology readiness level for a space technology item."),
        (pdfs[4], "multipactor", "I need to verify a high-power RF component against multipactor risk."),
        (pdfs[5], "battery", "I need battery management or ageing guidance for a spacecraft power subsystem."),
        (pdfs[6], "electrical bonding", "I need electrical bonding, grounding, or insulation guidance for spacecraft equipment."),
        (pdfs[7], "EMC test", "I need electromagnetic compatibility test setup guidance."),
        (pdfs[8], "interface", "I need spacecraft interface engineering guidance for an equipment interface."),
        (pdfs[9], "memory", "I need ASIC or memory fault tolerance guidance under radiation."),
    ]
    questions = []
    for pdf, keyword, prompt in scenarios[:count]:
        record = best_record(by_pdf[pdf], keyword)
        questions.append(
            make_item(
                "scenario",
                "hard",
                f"{prompt} Which ECSS handbook section should the retriever return?",
                [record],
                expected_answer=f'{pdf}, section {label(record)}: {record["section_title"]}.',
            )
        )
    return questions


def negative_questions(count: int) -> list[dict[str, Any]]:
    prompts = [
        "What does this 10-PDF ECSS subset say about NASA NPR 7120.5 project life-cycle gate reviews?",
        "Which clause in these 10 ECSS handbooks defines CubeSat deployer rail dimensions for a 3U CubeSat?",
        "What does this subset say about ITU frequency filing procedures for Ka-band satellite networks?",
        "Which section in these 10 PDFs specifies CCSDS packet telemetry primary header bit layout?",
        "Where do these 10 ECSS handbooks define orbital debris mitigation passivation rules after mission disposal?",
    ]
    items = []
    for prompt in prompts[:count]:
        items.append(
            {
                "id": "",
                "question": prompt,
                "query_type": "negative",
                "difficulty": "hard",
                "expected_answer": "No supported answer in the selected first 10 ECSS PDFs; the system should abstain or report not found.",
                "expected_pdfs": [],
                "expected_record_ids": [],
                "gold_context": [],
                "negative": True,
            }
        )
    return items


REQUIREMENT_WORDS = {
    "shall",
    "should",
    "requirement",
    "requirements",
    "process",
    "plan",
    "programme",
    "management",
    "verification",
    "qualification",
    "acceptance",
}
METHOD_WORDS = {
    "method",
    "test",
    "analysis",
    "inspection",
    "verification",
    "measurement",
    "simulation",
    "assessment",
    "review",
}


def is_definition_record(record: dict[str, Any]) -> bool:
    section = str(record.get("section") or "")
    title = (record.get("section_title") or "").lower()
    if title.startswith("terms "):
        return False
    return section.startswith("3.") and 2 <= len(title) <= 80 and not title.startswith("abbreviated")


def is_abbreviation_record(record: dict[str, Any]) -> bool:
    text = f'{record.get("section_title", "")} {record.get("content", "")}'.lower()
    return "abbreviated" in text or "abbreviation" in text or "symbols" in text


def good_title(record: dict[str, Any]) -> bool:
    title = record.get("section_title") or ""
    return bool(title) and 4 <= len(title) <= 90 and not title.lower().startswith("terms from")


def keyword_record(record: dict[str, Any], words: set[str]) -> bool:
    text = f'{record.get("section_title", "")} {record.get("content", "")}'.lower()
    return any(word.lower() in text for word in words)


def has_parameter_signal(record: dict[str, Any]) -> bool:
    text = record["content"]
    lower = text.lower()
    return (
        bool(re.search(r"\b\d+(?:[.,]\d+)?\s*(?:v|a|hz|khz|mhz|ghz|db|w|kw|kg|mm|cm|m|rad|gy|ev|kev|mev|%)\b", lower))
        or "table" in lower
        or "figure" in lower
        or "threshold" in lower
        or "limit" in lower
    )


def best_record(records: list[dict[str, Any]], keyword: str) -> dict[str, Any]:
    keyword_lower = keyword.lower()
    scored = []
    for record in records:
        title = (record.get("section_title") or "").lower()
        content = record["content"].lower()
        score = 0
        score += 10 * title.count(keyword_lower)
        score += content.count(keyword_lower)
        score += 1 if record.get("section") else 0
        scored.append((score, len(record["content"]), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][2]
    return records[len(records) // 2]


def label(record: dict[str, Any]) -> str:
    section = record.get("section")
    title = record.get("section_title")
    if section and title:
        return f"{section} {title}"
    return str(section or title or record["record_id"])


def topic_phrase(record: dict[str, Any]) -> str:
    title = clean_term(record.get("section_title") or "")
    if title:
        return title
    return excerpt(record["content"], 80)


def clean_term(value: str) -> str:
    return normalize_space(value).strip(" .:-")


def excerpt(text: str, limit: int = 320) -> str:
    normalized = normalize_space(text)
    return normalized if len(normalized) <= limit else normalized[: limit - 1].rstrip() + "…"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "query_type",
                "difficulty",
                "negative",
                "question",
                "expected_answer",
                "expected_pdfs",
                "expected_record_ids",
                "primary_section",
                "primary_page_start",
            ],
        )
        writer.writeheader()
        for record in records:
            primary = record["gold_context"][0] if record["gold_context"] else {}
            writer.writerow(
                {
                    "id": record["id"],
                    "query_type": record["query_type"],
                    "difficulty": record["difficulty"],
                    "negative": record["negative"],
                    "question": record["question"],
                    "expected_answer": record["expected_answer"],
                    "expected_pdfs": ";".join(record["expected_pdfs"]),
                    "expected_record_ids": ";".join(record["expected_record_ids"]),
                    "primary_section": primary.get("section"),
                    "primary_page_start": primary.get("page_start"),
                }
            )


def write_summary(path: Path, records: list[dict[str, Any]], pdfs: list[str], by_pdf: dict[str, list[dict[str, Any]]]) -> None:
    counts = Counter(item["query_type"] for item in records)
    pdf_counts = Counter(pdf for item in records for pdf in item["expected_pdfs"])
    lines = [
        "# ECSS First-10 PDF Retrieval Evaluation Set",
        "",
        f"Questions: {len(records)}",
        "",
        "## Query Types",
        "",
    ]
    lines.extend(f"- {key}: {counts[key]}" for key in sorted(counts))
    lines.extend(["", "## Source PDFs", ""])
    for pdf in pdfs:
        lines.append(f"- {pdf}: {len(by_pdf[pdf])} parsed clauses, {pdf_counts[pdf]} gold references")
    lines.extend(["", "## Sample", ""])
    for item in records[:12]:
        lines.append(f"- `{item['id']}` [{item['query_type']}] {item['question']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

