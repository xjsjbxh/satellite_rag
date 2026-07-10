from __future__ import annotations

import json
from pathlib import Path

from satellite_rag.scan import ScanOptions, scan_public_data_root


def test_rag_scan_writes_manifest_and_summary(tmp_path: Path):
    root = tmp_path / "public-data"
    (root / "ECSS标准").mkdir(parents=True)
    (root / "ECSS标准" / "ECSS-E-ST-10.pdf").write_text("standard", encoding="utf-8")
    (root / "ISO标准").mkdir(parents=True)
    (root / "ISO标准" / "ISO-24113.pdf").write_text("iso", encoding="utf-8")
    (root / "美国航天中心标准和手册").mkdir(parents=True)
    (root / "美国航天中心标准和手册" / "GSFC-STD-7000.pdf").write_text("center", encoding="utf-8")
    (root / "美军标").mkdir(parents=True)
    (root / "美军标" / "MIL-STD-1540.pdf").write_text("mil", encoding="utf-8")
    (root / "NASA经验教训库liss_merged_lessons_filtered.jsonl").write_text(
        '{"Lesson Number":"1","Subject":"Solder defect"}\n',
        encoding="utf-8",
    )
    (root / "公众号文章" / "仿真优化工匠").mkdir(parents=True)
    (root / "公众号文章" / "仿真优化工匠" / "模态分析.md").write_text("modal", encoding="utf-8")
    (root / "公众号文章" / "仿真优化工匠" / "模态分析.html").write_text("<p>modal</p>", encoding="utf-8")
    (root / "中文教材").mkdir(parents=True)
    (root / "中文教材" / "book.pdf").write_text("book", encoding="utf-8")
    (root / "公众号文章" / "仿真优化工匠" / "图片.jpg").write_text("image", encoding="utf-8")

    result = scan_public_data_root(ScanOptions(root=root, out_dir=tmp_path / "out"))

    manifest_path = Path(result["manifest_path"])
    summary_path = Path(result["summary_json_path"])
    assert manifest_path.exists()
    assert summary_path.exists()

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    by_name = {record["filename"]: record for record in records}

    assert by_name["ECSS-E-ST-10.pdf"]["mvp_action"] == "include"
    assert by_name["ISO-24113.pdf"]["mvp_reason"] == "strong_evidence"
    assert by_name["GSFC-STD-7000.pdf"]["mvp_reason"] == "strong_evidence"
    assert by_name["MIL-STD-1540.pdf"]["mvp_reason"] == "strong_evidence"
    assert by_name["NASA经验教训库liss_merged_lessons_filtered.jsonl"]["mvp_reason"] == "lesson_evidence"
    assert by_name["模态分析.md"]["mvp_action"] == "include"
    assert by_name["模态分析.html"]["mvp_reason"] == "article_html_fallback"
    assert by_name["book.pdf"]["mvp_action"] == "defer"
    assert by_name["图片.jpg"]["mvp_action"] == "exclude"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["totals"]["files"] == 9
    assert summary["mvp_actions"]["include"] == 6
    assert (tmp_path / "out" / "scan_summary.md").exists()

