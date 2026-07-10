# RAG MVP Scan Summary

- Source root: `C:\baidunetdiskdownload\公开数据`
- Profile: `public_data_mvp`
- Generated at: `2026-07-09T03:28:39.612602+00:00`
- Total files: 34698
- Total size: 56.57 GB

## MVP Actions

| Action | Files |
| --- | ---: |
| defer | 1689 |
| exclude | 19739 |
| include | 13270 |

## Top-Level Directories

| Directory | Files | Size MB | Actions | Top Extensions |
| --- | ---: | ---: | --- | --- |
| 中文教材 | 322 | 27310.14 | defer:322 | .pdf:322 |
| GJB | 479 | 10965.85 | include:479 | .pdf:479 |
| QJ标准 | 324 | 6721.21 | exclude:1, include:323 | .pdf:323, .py:1 |
| ISO标准 | 202 | 4706.71 | include:202 | .pdf:202 |
| 英文教材 | 106 | 2550.39 | defer:106 | .pdf:106 |
| GB和GBT国标 | 80 | 2378.48 | include:80 | .pdf:80 |
| 公众号文章 | 21796 | 2096.27 | defer:1255, exclude:19737, include:804 | .jpg:19640, .html:1139, .md:808, .pdf:112, .docx:97 |
| 美国航天中心标准和手册 | 232 | 384.42 | include:232 | .pdf:232 |
| ECSS手册 | 61 | 369.06 | include:61 | .pdf:58, .xls:1, .doc:1, .docx:1 |
| NASA标准 | 82 | 186.77 | include:82 | .pdf:82 |
| ECSS标准 | 151 | 152.37 | defer:3, include:148 | .pdf:144, .xlsx:4, .zip:3 |
| 卫星百科数据 | 10856 | 85.15 | defer:1, exclude:1, include:10854 | .md:10854, .xlsx:1, .py:1 |
| 美军标 | 4 | 13.52 | include:4 | .pdf:4 |
| NASA经验教训库liss_merged_lessons_filtered.jsonl | 1 | 7.91 | include:1 | .jsonl:1 |
| GJB目录.xlsx | 1 | 0.36 | defer:1 | .xlsx:1 |
| QJ标准2次评估.xlsx | 1 | 0.24 | defer:1 | .xlsx:1 |

## Parse Routes

| Route | Files |
| --- | ---: |
| archive_deferred | 3 |
| html_fallback | 1139 |
| image_reference_only | 19640 |
| jsonl_records | 1 |
| markdown | 11662 |
| pdf_text_first | 2144 |
| spreadsheet_metadata | 8 |
| unsupported | 2 |
| word_document | 99 |

## Phase-0 Notes

- This scan does not parse document bodies or ingest vectors.
- `include` means selected for the first MVP preparation pass.
- `defer` means intentionally kept out of MVP ingestion but available for later phases.
- `exclude` means not corpus text for the MVP profile.
