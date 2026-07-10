# satellite_rag 团队操作指南

本文档用于团队成员在本机快速跑通卫星设计 RAG 系统。默认项目目录为 `C:\satellite_rag`。

## 1. 当前项目内容

项目已经包含：

- RAG 代码：`C:\satellite_rag\satellite_rag`
- 脚本入口：`C:\satellite_rag\scripts`
- 示例与评测：`C:\satellite_rag\tests`
- 语料数据：`C:\satellite_rag\data\satdesign-rag-data`
- 配置模板：`C:\satellite_rag\.env.example`

当前主语料推荐使用：

```powershell
C:\satellite_rag\data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl
```

这份语料包含 50 个 PDF 的解析结果，共 14959 条记录，`parse_failures.jsonl` 为空。`ECSS-E-HB-32-20_Part8A(20March2011)` 已按 glossary term 重新切片。

## 2. 安装 Python 环境

```powershell
cd C:\satellite_rag
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
```

如果本机没有 `py`，用实际 Python 路径替换即可。建议 Python 版本为 3.10 或以上。

如果需要重新解析 PDF，再安装 PDF 可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pdf]"
```

## 3. 配置公司 LiteLLM API

复制配置模板：

```powershell
Copy-Item .env.example .env
```

打开 `C:\satellite_rag\.env`，至少填写：

```text
LITELLM_API_KEY=你的公司token
```

当前推荐模型配置：

```text
RAG_EMBEDDING_PROVIDER=litellm
RAG_EMBEDDING_ENDPOINT=http://api.opearlai.com:30096/v1/embeddings
RAG_EMBEDDING_MODEL=Qwen3-Embedding-8B
RAG_EMBEDDING_BATCH_SIZE=16

RAG_RERANK_PROVIDER=litellm
RAG_RERANK_ENDPOINT=http://api.opearlai.com:30096/v1/rerank
RAG_RERANK_MODEL=Qwen3-Reranker-8B
RAG_RERANK_TOP_K=20

QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=satellite_rag
QDRANT_VECTOR_NAME=dense
```

`RAG_EMBEDDING_API_KEY` 和 `RAG_RERANK_API_KEY` 可以留空，代码会自动回退使用 `LITELLM_API_KEY`。

不要把真实 `.env` 提交到 Git。

## 4. 启动本地 Qdrant

如果已经安装 Docker Desktop，可以运行：

```powershell
docker run -d --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v C:\qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

如果容器已经存在：

```powershell
docker start qdrant
```

检查服务：

```powershell
Invoke-RestMethod http://127.0.0.1:6333/collections
```

Qdrant 网页：

```text
http://127.0.0.1:6333/dashboard
```

当前 collection 约定：

```text
collection: satellite_rag
vector_name: dense
vector_size: 4096
distance: Cosine
```

`4096` 来自 `Qwen3-Embedding-8B` 的真实向量维度。

## 5. 检查 LiteLLM 模型权限

```powershell
cd C:\satellite_rag
.\.venv\Scripts\python.exe scripts\check_litellm.py --embedding --rerank
```

如果返回 403，通常是公司 token 没有对应模型权限。需要让公司开通：

- `Qwen3-Embedding-8B`
- `Qwen3-Reranker-8B`

如果偶发 502 或连接重置，可以重跑。当前代码已经对 429、5xx 和连接重置做了有限重试。

## 6. 将语料写入 Qdrant

首次入库或需要重建 collection 时：

```powershell
cd C:\satellite_rag
.\.venv\Scripts\python.exe scripts\ingest_corpus_qdrant.py `
  --corpus data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl `
  --url http://127.0.0.1:6333 `
  --collection satellite_rag `
  --vector-name dense `
  --embedding-provider litellm `
  --embedding-model Qwen3-Embedding-8B `
  --batch-size 16 `
  --recreate
```

说明：

- `--recreate` 会删除并重建同名 collection，谨慎使用。
- `--batch-size 16` 表示每次请求 embedding API 的文本条数。
- 入库期间会调用公司 embedding API，速度取决于公司服务延迟和限流。

## 7. 快速检索检查

`search_qdrant.py` 是向量检索 smoke test，用来确认 Qdrant 和 embedding 正常：

```powershell
cd C:\satellite_rag
.\.venv\Scripts\python.exe scripts\search_qdrant.py `
  "How is acceptance stage defined?" `
  --qdrant-url http://127.0.0.1:6333 `
  --collection satellite_rag `
  --embedding-provider litellm `
  --embedding-model Qwen3-Embedding-8B `
  --top-k 8
```

返回结果会包含类似引用：

```text
[1] ECSS-E-HB-10-02A.pdf, section 5.2.1 Acceptance stage, pp.12-13, chunk_id=...
```

注意：这个脚本只做 dense vector 检索，不代表完整混合检索效果。

## 8. 跑 120 题评测

真实使用时用户不会提供 PDF 名，因此推荐评测口径是不限制到测试 PDF：

```powershell
cd C:\satellite_rag
$env:RAG_HTTP_RETRIES="6"
$env:RAG_HTTP_RETRY_DELAY="2"

.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --eval-set C:\021-AI-Workflow-for-Satellite\satellite-agent\runs\eval_sets\ecss_first10_120.jsonl `
  --corpus data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl `
  --out runs\eval\ecss_first10_qwen_k20_blend `
  --qdrant-url http://127.0.0.1:6333 `
  --collection satellite_rag `
  --embedding-provider litellm `
  --embedding-model Qwen3-Embedding-8B `
  --rerank-provider litellm `
  --rerank-model Qwen3-Reranker-8B `
  --rerank-top-k 20 `
  --no-restrict-corpus-to-eval-pdfs
```

当前一次完整评测大约 8 分钟，取决于公司 API 稳定性。

最近一次同口径结果：

| 配置 | slice_recall@10 | doc_recall@10 | MRR | gold_coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| 纯 reranker 排序 | 0.4522 | 0.7522 | 0.2003 | 0.6326 |
| reranker + 原始检索分融合 | 0.6304 | 0.7826 | 0.2993 | 0.6845 |

结果文件：

- `summary.json`：总体指标
- `details.jsonl`：逐题详情
- `report.md`：简要报告

## 9. 在代码里调用检索

```python
from satellite_rag import SearchRequest
from satellite_rag.factory import build_rag_runtime

runtime = build_rag_runtime()
results = runtime.retriever.search(
    SearchRequest(
        query="How is acceptance stage defined?",
        final_top_k=8,
    )
)

for item in results:
    print(item.rank, item.score, item.chunk_id, item.metadata)
```

默认完整检索链路是：Qwen embedding + Qdrant 向量检索 + 本地 BM25 + RRF 融合 + Qwen reranker + 分数融合。

## 10. 常见问题

### 10.1 Qdrant 网页打不开

先确认容器是否运行：

```powershell
docker ps
```

再访问：

```text
http://127.0.0.1:6333/dashboard
```

### 10.2 API 返回 403

说明 token 没有模型权限。把报错截图给公司，确认 token 是否允许访问 `Qwen3-Embedding-8B` 和 `Qwen3-Reranker-8B`。

### 10.3 API 返回 502 或连接被重置

这是公司服务或网络临时问题。当前代码会重试，仍失败时可以稍后重跑。

### 10.4 为什么 collection 里点数少于 corpus 行数

当前 corpus 是 14959 行。Qdrant 点数可能略少，是因为个别记录使用了重复 ID，Qdrant 会按相同 point id 覆盖。

### 10.5 是否需要把 `chunk_id` 返回给最终用户

面向普通用户可以不显示 `chunk_id`。面向评测、排查、审计时建议保留，方便定位具体切片。

