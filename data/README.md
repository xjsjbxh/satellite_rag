# satellite_rag 数据目录说明

本目录用于放置本机可用的 RAG 语料数据，默认不提交到 Git。

当前数据已复制到：

```text
C:\satellite_rag\data\satdesign-rag-data
```

目录说明：

| 目录 | 用途 | 大小 |
| --- | --- | ---: |
| `public_data_mvp` | 原始扫描/初始语料输出 | 约 104 MB |
| `public_data_mvp_pdf_clause_audit50` | 当前推荐主语料，50 个 PDF，条款/术语切片 | 约 63 MB |
| `public_data_mvp_pdf_clause_smoke` | 小规模 clause smoke 测试语料 | 约 4 MB |
| `public_data_mvp_pdf_smoke` | 小规模 page smoke 测试语料 | 约 4 MB |

推荐主 corpus：

```text
data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl
```

当前主 corpus 状态：

```text
records: 14959
selected_files: 50
parsed_files: 50
failed_files: 0
```

如果团队成员通过 Git 获取项目，需要另外同步本目录数据；如果通过压缩包或共享盘获取完整 `C:\satellite_rag`，则数据已经在项目内。

