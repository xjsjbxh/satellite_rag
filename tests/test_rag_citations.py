from __future__ import annotations

from satellite_rag.citations import format_retrieval_citation
from satellite_rag.schemas import RetrievalResult


def test_format_retrieval_citation_uses_clause_provenance() -> None:
    result = RetrievalResult(
        chunk_id="8ec9a00bfd09b319",
        doc_id="doc-1",
        content="Acceptance stage evidence.",
        source_path=r"C:\data\ECSS-E-HB-10-02A.pdf",
        metadata={
            "section": "5.2.1",
            "section_title": "Acceptance stage",
            "page_start": 12,
            "page_end": 13,
        },
    )

    citation = format_retrieval_citation(result, 1)

    assert citation == (
        "[1] ECSS-E-HB-10-02A.pdf, section 5.2.1 Acceptance stage, "
        "pp.12-13, chunk_id=8ec9a00bfd09b319"
    )


def test_format_retrieval_citation_accepts_payload_dict() -> None:
    citation = format_retrieval_citation(
        {
            "chunk_id": "r1",
            "doc_id": "d1",
            "source_path": r"C:\data\ECSS-E-HB-10-02A.pdf",
            "metadata": {"section": "3.2.1", "section_title": "acceptance stage", "page": 9},
        },
        2,
    )

    assert citation == "[2] ECSS-E-HB-10-02A.pdf, section 3.2.1 acceptance stage, p.9, chunk_id=r1"

