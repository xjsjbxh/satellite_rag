# satellite_rag 当前系统设计

本文档说明当前 RAG 系统的数据结构、检索链路、模型接入和评测口径，方便团队理解系统为什么这样设计。

## 1. 设计目标

`satellite_rag` 是从卫星设计工作流中抽离出来的独立 RAG 项目，目标是：

- 管理卫星设计标准、手册和参考资料的解析语料。
- 使用 Qdrant 建立本地向量数据库。
- 接入公司 LiteLLM API 调用 embedding、reranker 和可选回答模型。
- 支持可追溯引用，返回文件、章节、页码和切片 ID。
- 支持自动化评测，用指标比较不同检索策略。

当前重点不是生成漂亮回答，而是先把“检索是否召回正确证据”测准。

## 2. 项目结构

```text
C:\satellite_rag
├─ satellite_rag\
│  ├─ scan.py              # 扫描原始资料，生成 manifest
│  ├─ prepare.py           # PDF/文本解析与切片
│  ├─ chunker.py           # 切片辅助逻辑
│  ├─ embeddings.py        # embedding 适配器
│  ├─ vector_store.py      # Qdrant / memory 向量库适配
│  ├─ keyword_store.py     # 本地 BM25 / ES 关键词检索
│  ├─ retriever.py         # 混合检索与 RRF 融合
│  ├─ reranker.py          # reranker 与分数融合
│  ├─ citations.py         # 引用格式
│  ├─ evaluation\          # 自动评测
│  └─ factory.py           # 根据 .env 创建运行时组件
├─ scripts\
│  ├─ check_litellm.py
│  ├─ ingest_corpus_qdrant.py
│  ├─ search_qdrant.py
│  └─ evaluate_rag.py
├─ data\
│  └─ satdesign-rag-data\
├─ docs\
├─ .env.example
└─ README.md
```

## 3. 数据层设计

### 3.1 corpus.jsonl

主语料文件：

```text
C:\satellite_rag\data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl
```

每一行是一条可检索记录，核心字段包括：

```text
record_id
chunk_id
doc_id
content
source_path
metadata
```

`metadata` 中常见字段：

```text
title
relative_path
source_type
authority_level
record_kind
section
section_title
page_start
page_end
standard_id
schema_version
generated_at
hash
mvp_reason
```

### 3.2 ID 含义

| 字段 | 用途 |
| --- | --- |
| `doc_id` | 文档级 ID，用于识别同一份 PDF/资料 |
| `record_id` | 解析记录 ID，评测时常用它作为 gold evidence |
| `chunk_id` | 检索切片 ID，便于定位、排查和审计 |

当前实现中 `record_id` 和 `chunk_id` 有时一致，有时不一致。评测优先使用 `record_id`，引用展示使用 `chunk_id`。

### 3.3 metadata 是否参与向量化

当前向量主要由 `content` 生成。`metadata` 不直接作为向量内容写入 embedding，但会作为 Qdrant payload 保存，用于：

- 生成引用。
- 过滤和路由。
- 评测命中率。
- 后续做文档级或章节级预检索。

## 4. 当前检索链路

```text
用户问题
  ↓
Qwen3-Embedding-8B 生成查询向量
  ↓
Qdrant dense vector top_k
  ↓
本地 BM25 top_k
  ↓
RRF 融合向量检索和关键词检索
  ↓
Qwen3-Reranker-8B 对候选切片重排
  ↓
reranker 分数 + 原始检索分数融合
  ↓
返回 top N 证据和引用
```

### 4.1 向量检索

当前 Qdrant collection：

```text
collection: satellite_rag
vector_name: dense
vector_size: 4096
distance: Cosine
```

`4096` 是 `Qwen3-Embedding-8B` 的实际向量维度。

### 4.2 关键词检索

当前默认使用本地 BM25。它不需要额外服务，适合快速验证，但中文和专业符号处理能力有限。后续如果要提升中文、缩写、标准号、表格参数检索，可以考虑接入 Elasticsearch/OpenSearch。

### 4.3 RRF 融合

向量检索和 BM25 检索先通过 Reciprocal Rank Fusion 融合：

```text
score += weight / (60 + rank)
```

默认：

```text
RAG_VECTOR_WEIGHT=1.0
RAG_BM25_WEIGHT=1.0
```

### 4.4 reranker 分数融合

之前的纯 reranker 做法会用 reranker 分数直接覆盖原始检索分数，导致 Qwen reranker 有时把短 glossary 或词面相似切片排得过高。

当前改为：

```text
final_score =
  0.65 * normalized_rerank_score
+ 0.35 * normalized_retrieval_score
```

可通过环境变量调整：

```text
RAG_RERANK_SCORE_WEIGHT=0.65
RAG_RETRIEVAL_SCORE_WEIGHT=0.35
```

同口径评测表明，在不提供 PDF 名、全库检索情况下，融合比纯 reranker 更稳。

## 5. 引用格式

当前目标引用格式：

```text
[1] 文件名, section xxx, pp.x-y, chunk_id=xxx
```

示例：

```text
[1] ECSS-E-HB-10-02A.pdf, section 5.2.1 Acceptance stage, pp.12-13, chunk_id=8ec9a00bfd09b319
```

说明：

- `pp.` 表示页码范围。
- 方括号 `[1]` 是当前回答中的引用序号，会随排序变化。
- `chunk_id` 对普通用户不是必需的，但对评测和排查很有用。

## 6. 评测设计

评测集：

```text
C:\021-AI-Workflow-for-Satellite\satellite-agent\runs\eval_sets\ecss_first10_120.jsonl
```

题型覆盖：

- definition
- abbreviation
- clause_locator
- requirement_process
- parameter_table
- method_explanation
- cross_doc_compare
- scenario
- negative

主要指标：

| 指标 | 含义 |
| --- | --- |
| `slice_recall@k` | top k 是否召回 gold 切片 |
| `doc_recall@k` | top k 是否召回 gold 文档 |
| `source_precision@k` | top k 中来源文档正确比例 |
| `gold_answer_coverage@k` | top k 内容覆盖参考答案关键词的程度 |
| `MRR` | 第一个 gold 切片排名越靠前越好 |

真实使用口径应使用：

```text
--no-restrict-corpus-to-eval-pdfs
```

因为真实用户通常不会告诉系统 PDF 名。

## 7. 当前已知效果

在 120 题评测中，不提供 PDF 名、全 50 PDF 检索：

| 方法 | slice_recall@10 | doc_recall@10 | MRR |
| --- | ---: | ---: | ---: |
| 纯 Qwen reranker | 0.4522 | 0.7522 | 0.2003 |
| Qwen reranker + 原始检索分融合 | 0.6304 | 0.7826 | 0.2993 |

结论：

- 分数融合显著提升真实全库检索效果。
- 仅增大 rerank 候选数不一定更好，k50 曾引入更多噪声。
- 下一步更值得做文档/章节级预检索、query type 路由和 record_kind 权重。

## 8. 后续优化方向

建议优先级：

1. 导出 `rerank_score`、`retrieval_score`、`fused_score` 到评测详情，便于逐题排查。
2. 做文档/章节级预检索，不依赖用户给 PDF 名。
3. 做 query type 分类，对 glossary、standard_clause、table 等 record_kind 动态加权。
4. 改进中文、缩写、标准号的 BM25 分词。
5. 为表格和长 annex 增加专门切片策略。

## 9. 安全和工程约束

- `.env` 不能提交。
- `data/` 默认不提交，避免把大文件放入 Git。
- 公司 API 当前是 HTTP，建议只在可信网络或 VPN 下使用。
- Qdrant 本地数据建议持久化到 `C:\qdrant_storage`。
- 重建 collection 前确认是否需要备份。

