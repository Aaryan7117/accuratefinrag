# Enterprise RAG System

A high-accuracy, enterprise-grade Retrieval-Augmented Generation system for SEC 10-K filings, financial reports, and structured PDF documents.

## Features

- **Zero Hallucination**: All answers grounded in document context with citations
- **Hybrid Chunking**: Structure-aware + semantic chunking in a single pass  
- **Multi-Store Architecture**: Vector DB (FAISS), Knowledge Graph (Neo4j), Table Store (SQLite), BM25 Index
- **Mandatory Reranking**: Cross-encoder reranking for accurate retrieval
- **Cloud LLM**: Uses OpenRouter API (LLaMA-3.3-70B) - no local GPU required for LLM

## Quick Start

### 1. Install Dependencies

```powershell
cd c:\Users\lenovo\Downloads\finrag
pip install -r requirements.txt
```

### 2. Configure Environment

Create/edit `.env` file:

```env
# Model configuration  
RAG_MODEL_DEVICE=cpu/cuda(if you have min 4gb vram)

# LLM Provider (uses OpenRouter cloud API)
RAG_MODEL_LLM_PROVIDER=openrouter
RAG_MODEL_OPENROUTER_API_KEY=your_api_key_here
RAG_MODEL_OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Neo4j (optional - falls back to memory mode if unavailable)
RAG_STORE_NEO4J_URI=bolt://localhost:7687
RAG_STORE_NEO4J_USER=neo4j
RAG_STORE_NEO4J_PASSWORD=your_password
```

Get your OpenRouter API key at: https://openrouter.ai/keys

### 3. Start the Server

```powershell
python -m api.main
```

Open http://127.0.0.1:8000 in your browser.

### 4. Upload and Query

1. Upload a PDF document using the sidebar
2. Wait for ingestion to complete (~30-120 seconds)
3. Ask questions in the chat interface
4. Get grounded answers with citations

## Architecture

```
finrag/
├── config/
│   └── settings.py           # Configuration management
├── src/
│   ├── document/             # PDF parsing, chunking, table extraction
│   ├── stores/               # FAISS, Neo4j, SQLite, BM25
│   ├── retrieval/            # Query processing, hybrid retrieval, reranking
│   ├── generation/           # LLM engine, response building
│   └── pipeline/             # Ingestion and query orchestrators
├── api/
│   └── main.py               # FastAPI backend
├── frontend/
│   └── index.html            # Web UI
└── data/                     # Uploads, indexes, databases
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload and ingest PDF |
| `/query` | POST | RAG query with citations |
| `/documents` | GET | List ingested documents |
| `/health` | GET | System health check |

## Models & Requirements

| Component | Model | Runs On |
|-----------|-------|---------|
| Embeddings | bge-large-en-v1.5 | Local (CPU/GPU) |
| Reranker | bge-reranker-large | Local (CPU/GPU) |
| LLM | LLaMA-3.3-70B-Instruct | **OpenRouter Cloud** |

### CPU vs GPU

- **CPU (default)**: Works on any system, ~10 sec per query
- **GPU**: Requires CUDA PyTorch, ~3-5 sec per query

To enable GPU (requires Python 3.12 and CUDA PyTorch):
```env
RAG_MODEL_DEVICE=cuda
```

## Troubleshooting

### "Torch not compiled with CUDA enabled"
Your PyTorch doesn't have GPU support. Either:
- Use `RAG_MODEL_DEVICE=cpu` (recommended)
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Queries fail after uploading second document
This was a bug that has been fixed. The query pipeline now resets after each document upload.

### Cover page content not indexed
Fixed. Orphaned elements (cover pages, TOC) are now included in chunks.

## License

MIT License
