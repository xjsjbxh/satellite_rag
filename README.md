# satellite_rag

Standalone satellite-design RAG toolkit extracted from `satellite-agent`.

## 中文团队入口

当前本机项目已经包含语料数据：

```text
C:\satellite_rag\data\satdesign-rag-data
```

团队成员建议先阅读：

- `docs\TEAM_GUIDE_CN.md`：从安装、配置、Qdrant、入库到评测的完整操作指南。
- `docs\RAG_SYSTEM_DESIGN_CN.md`：当前 RAG 系统设计、检索链路、评测口径和后续优化方向。
- `data\README.md`：项目内数据目录说明。

当前推荐主语料：

```text
data\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl
```

当前推荐 Qdrant collection：

```text
collection=satellite_rag
vector_name=dense
vector_size=4096
distance=Cosine
```

It supports:

- source scanning and corpus preparation
- PDF clause/page chunking with provenance fields
- Qdrant dense-vector storage
- local BM25 fusion
- retrieval evaluation
- LiteLLM / OpenAI-compatible embeddings, reranking, and answer generation

## 1. Install

```powershell
cd C:\satellite_rag
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If `py` is not available, use your Python executable directly.

Optional PDF parsing:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pdf]"
```

## 2. Configure LiteLLM

Copy `.env.example` to `.env` and fill in model names and keys. Scripts load
`C:\satellite_rag\.env` automatically, so you do not need to pass `--api-key`
or manually set `$env:*` every time.

You can still set variables in PowerShell to override `.env`:

```powershell
$env:LITELLM_BASE_URL="http://api.opearlai.com:30096/v1"
$env:LITELLM_API_KEY="your_company_key"
$env:LITELLM_AUTH_HEADER="x-litellm-api-key"

$env:RAG_EMBEDDING_PROVIDER="litellm"
$env:RAG_EMBEDDING_ENDPOINT="http://api.opearlai.com:30096/v1/embeddings"
$env:RAG_EMBEDDING_API_KEY=$env:LITELLM_API_KEY
$env:RAG_EMBEDDING_AUTH_HEADER="x-litellm-api-key"
$env:RAG_EMBEDDING_MODEL="Qwen3-Embedding-8B"

$env:RAG_RERANK_PROVIDER="litellm"
$env:RAG_RERANK_ENDPOINT="http://api.opearlai.com:30096/v1/rerank"
$env:RAG_RERANK_API_KEY=$env:LITELLM_API_KEY
$env:RAG_RERANK_AUTH_HEADER="x-litellm-api-key"
$env:RAG_RERANK_MODEL="Qwen3-Reranker-8B"

$env:ANSWER_LLM_BASE_URL="http://api.opearlai.com:30096/v1"
$env:ANSWER_LLM_API_KEY=$env:LITELLM_API_KEY
$env:ANSWER_LLM_AUTH_HEADER="x-litellm-api-key"
$env:ANSWER_LLM_MODEL="chat_model_name"
```

First check what models are available:

```powershell
.\.venv\Scripts\python.exe scripts\check_litellm.py --api-key $env:LITELLM_API_KEY
```

Check embedding and reranker calls:

```powershell
.\.venv\Scripts\python.exe scripts\check_litellm.py `
  --api-key $env:LITELLM_API_KEY `
  --model chat_model_name `
  --embedding-model Qwen3-Embedding-8B `
  --rerank-model Qwen3-Reranker-8B `
  --chat `
  --embedding `
  --rerank
```

## 3. Prepare Corpus

Scan source data:

```powershell
.\.venv\Scripts\python.exe -m satellite_rag scan `
  --root C:\baidunetdiskdownload\公开数据 `
  --out C:\satdesign-rag-data\public_data_mvp
```

Prepare a clause-level corpus:

```powershell
.\.venv\Scripts\python.exe -m satellite_rag prepare `
  --manifest C:\satdesign-rag-data\public_data_mvp\manifest.jsonl `
  --out C:\satdesign-rag-data\public_data_mvp_pdf_clause `
  --routes pdf_text_first `
  --pdf-chunk-mode clause
```

You can also reuse existing corpus files, for example:

```powershell
C:\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl
```

Each corpus row should contain fields such as:

```text
record_id, doc_id, content, source_path, page_start, page_end,
section, section_title, standard_id, source_type, authority_level
```

## 4. Ingest to Qdrant

```powershell
.\.venv\Scripts\python.exe scripts\ingest_corpus_qdrant.py `
  --corpus C:\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl `
  --url http://localhost:6333 `
  --collection satellite_agent_test `
  --embedding-provider litellm `
  --embedding-endpoint http://api.opearlai.com:30096/v1/embeddings `
  --embedding-api-key $env:LITELLM_API_KEY `
  --embedding-auth-header x-litellm-api-key `
  --embedding-model Qwen3-Embedding-8B `
  --batch-size 16 `
  --recreate
```

## 5. Search

```powershell
.\.venv\Scripts\python.exe scripts\search_qdrant.py `
  "In ECSS-E-HB-10-02A, how is acceptance stage defined?" `
  --qdrant-url http://localhost:6333 `
  --collection satellite_agent_test `
  --embedding-provider litellm `
  --embedding-endpoint http://api.opearlai.com:30096/v1/embeddings `
  --embedding-api-key $env:LITELLM_API_KEY `
  --embedding-model Qwen3-Embedding-8B
```

Results include stable citations:

```text
[1] ECSS-E-HB-10-02A.pdf, section 5.2.1 Acceptance stage, pp.12-13, chunk_id=...
```

## 6. Evaluate

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --eval-set runs\eval_sets\ecss_first10_120.jsonl `
  --corpus C:\satdesign-rag-data\public_data_mvp_pdf_clause_audit50\corpus.jsonl `
  --qdrant-url http://localhost:6333 `
  --collection satellite_agent_test `
  --embedding-provider litellm `
  --embedding-endpoint http://api.opearlai.com:30096/v1/embeddings `
  --embedding-api-key $env:LITELLM_API_KEY `
  --embedding-model Qwen3-Embedding-8B `
  --rerank-provider litellm `
  --rerank-endpoint http://api.opearlai.com:30096/v1/rerank `
  --rerank-api-key $env:LITELLM_API_KEY `
  --rerank-model Qwen3-Reranker-8B `
  --out runs\eval\satellite_agent_test_litellm
```

For answer generation through LiteLLM, add:

```powershell
  --generate-answers `
  --answer-base-url http://api.opearlai.com:30096/v1 `
  --answer-api-key $env:LITELLM_API_KEY `
  --answer-auth-header x-litellm-api-key `
  --answer-model chat_model_name
```

## Notes

- Do not commit real API keys.
- The company endpoint currently uses HTTP. Use it only on trusted network/VPN, or ask for HTTPS.
- `chunk_id` is stable for evaluation/debugging. User-facing answers can show citation without exposing `chunk_id`.
