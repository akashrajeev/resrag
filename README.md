# ResRAG

A small, local-first PDF question-answering app with a restrained ChatGPT-style interface.

## What it does

Upload one PDF, then ask questions in a normal conversation. ResRAG retrieves evidence from the document, reranks the strongest passages, and asks the configured LLM to answer only from that evidence. Answers include page citations and the UI lets you inspect the passages used.

## Architecture

```text
PDF
 │
 ├─ page-aware text extraction ──┐
 ├─ native table extraction      ├─> structure-aware chunks
 └─ optional Tesseract OCR ─────┘
                    │
          ┌─────────┴─────────┐
          │                   │
       BM25              dense embeddings
          │                   │
          └─────────┬─────────┘
                    │
             Reciprocal Rank Fusion
                    │
          cross-encoder reranking
                    │
             grounded LLM answer
                    │
          page-level source passages
```

The retrieval choice is intentional. A 2026 benchmark on 23,088 queries across 7,318 mixed text-and-table documents found that hybrid retrieval followed by neural reranking was the strongest two-stage configuration, while BM25 remained especially useful for precise and numerical queries. T²-RAGBench also reported Hybrid BM25 as the strongest approach in its text-and-table setting. These findings support keeping both lexical and semantic signals instead of relying on vector search alone.

Native PDF tables are extracted with PyMuPDF's `Page.find_tables()` and stored as Markdown so row/column relationships survive into the retrieval context. Scanned pages can use the optional PyMuPDF OCR API backed by Tesseract.

## LLM providers

ResRAG uses the OpenAI Python client for generation and can switch providers without changing the retrieval pipeline. The sidebar has a provider selector and model field.

### OpenAI

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

### OpenRouter

OpenRouter exposes an OpenAI-compatible API at `https://openrouter.ai/api/v1`, so the same OpenAI client can be used with an OpenRouter key and model slug.

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
```

Optional attribution headers are supported with `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME`.

### Groq

Groq exposes an OpenAI-compatible API at `https://api.groq.com/openai/v1`, so ResRAG can use Groq without a separate generation code path.

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=...
```

A generic `LLM_MODEL` can be used as a fallback when a provider-specific model variable is not set.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

streamlit run app.py
```

The requirements include matching PyTorch/torchvision versions because Streamlit's module watcher can inspect optional Transformers vision modules on Windows. This is not used by the RAG pipeline itself, but keeping the dependency installed prevents the repeated `ModuleNotFoundError: torchvision` watcher errors.

The embedding and reranker models are downloaded from Hugging Face on first use.

### OCR note

`OCR_ENABLED=1` enables a fallback for pages with little or no extractable text. PyMuPDF delegates OCR to Tesseract, so Tesseract must also be installed on the host machine and available on `PATH`. OCR is intentionally optional because native PDF text extraction is faster and more reliable whenever the source PDF already contains text.

## Test

```bash
pytest -q
```

## Research references

- Akarsu, Karaman & Mierbach (2026), *From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents*.
- Strich et al. (2025), *T²-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation*.
- Yu, Jian & Chen (2025), *TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning*.
- Ru et al. (2024), *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*.

## Scope

The current product intentionally focuses on one active PDF at a time. Evaluation covers retrieval recall/MRR, citation correctness, answer faithfulness, latency, and memory use on ordinary, table-heavy, and scanned PDFs.
