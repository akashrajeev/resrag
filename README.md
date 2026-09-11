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

Set at minimum:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Optional:

```text
OPENAI_BASE_URL=https://...
OCR_ENABLED=1
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

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

The current product intentionally focuses on one active PDF at a time. The next evaluation phase should measure retrieval recall/MRR, citation correctness, answer faithfulness, latency, and memory use on a small curated set of ordinary, table-heavy, and scanned PDFs.
