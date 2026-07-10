from __future__ import annotations

import json
from pathlib import Path

from satellite_rag.evaluation.rag_eval import EvalConfig, evaluate_rag


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def test_rag_eval_reports_slice_recall_and_answer_groundedness(tmp_path: Path):
    eval_set = tmp_path / "eval.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    answers = tmp_path / "answers.jsonl"
    write_jsonl(
        eval_set,
        [
            {
                "id": "q1",
                "query_type": "definition",
                "difficulty": "easy",
                "negative": False,
                "question": "How is acceptance stage defined?",
                "expected_answer": "acceptance stage verification stage free of workmanship defects",
                "expected_pdfs": ["ECSS-E-HB-10-02A(17December2010).pdf"],
                "expected_record_ids": ["r1"],
                "gold_context": [],
            },
            {
                "id": "q2",
                "query_type": "negative",
                "difficulty": "hard",
                "negative": True,
                "question": "What does this subset say about unrelated NASA gates?",
                "expected_answer": "No supported answer in the selected first 10 ECSS PDFs; the system should abstain or report not found.",
                "expected_pdfs": [],
                "expected_record_ids": [],
                "gold_context": [],
            },
        ],
    )
    write_jsonl(
        corpus,
        [
            {
                "record_id": "r1",
                "doc_id": "d1",
                "content": "3.2.1 acceptance stage verification stage free of workmanship defects.",
                "source_path": r"C:\data\ECSS-E-HB-10-02A(17December2010).pdf",
                "title": "ECSS-E-HB-10-02A(17December2010).pdf",
            },
            {
                "record_id": "r2",
                "doc_id": "d2",
                "content": "Battery activation test guidance.",
                "source_path": r"C:\data\ECSS-E-HB-20-02A(1October2015).pdf",
                "title": "ECSS-E-HB-20-02A(1October2015).pdf",
            },
        ],
    )
    write_jsonl(
        answers,
        [
            {"id": "q1", "answer": "Acceptance stage is a verification stage free of workmanship defects."},
            {"id": "q2", "answer": "No supported answer in the supplied context."},
        ],
    )

    result = evaluate_rag(
        EvalConfig(
            eval_set=eval_set,
            corpus=corpus,
            answers_jsonl=answers,
            out_dir=tmp_path / "out",
            embedding_provider="hash",
            embedding_model="test/hash-dense",
            top_k=3,
            k_values=(1, 3),
        )
    )

    retrieval = result["summary"]["retrieval"]
    answers_summary = result["summary"]["answers"]
    assert retrieval["slice_recall@3"] == 1.0
    assert retrieval["doc_recall@3"] == 1.0
    assert retrieval["mrr"] == 1.0
    assert answers_summary["hallucination_rate"] == 0.0
    assert answers_summary["negative_abstention_accuracy"] == 1.0
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "details.jsonl").exists()
    assert (tmp_path / "out" / "report.md").exists()

