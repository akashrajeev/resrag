# ResRAG

A small, local-first PDF question-answering app with a restrained ChatGPT-style interface.

## What it does

Upload one PDF, then ask questions in a normal conversation. ResRAG gets the document chat-ready with a lightweight text/BM25 index first, then upgrades it in the background to full hybrid retrieval. Answers include page citations and the UI lets you inspect the passages used.

## Architecture

```text
PDF upload
   │
   ├─ FAST PATH ──► text extraction ─► BM25 index ─► CHAT READY
   │                                      │
   │                                      └─ immediate questions
   │
   └─ BACKGROUND ─► tables + OCR + structural parsing
                    │
                    └─ dense embeddings
                         │
              child-level BM25 + dense retrieval
                         │
                    RRF fusion
                         │
             parent/group reconstruction
                         │
          long PDFs: page-level coarse routing
                         │
                 cross-encoder reranking
                         │
                  grounded LLM answer
```

The design separates time-to-first-answer from time-to-full-index. This follows the demand-side/deferred-ingestion direction explored by recent document QA work: lightweight metadata and lexical indexing can locate relevant content first, while expensive understanding is performed only when it is available or needed.

For long documents, ResRAG adds page-level retrieval between document-wide child retrieval and reranking. Recent 2026 long-document work reports that page-level retrieval can improve within-document recall, while HiKEY and H-RAG show that hierarchical parent/child retrieval helps recover fragmented evidence across long documents.

## Retrieval

The full index uses hybrid BM25 + dense retrieval, reciprocal-rank fusion, document-discovered parent/group reconstruction, and cross-encoder reranking. Contextual structural metadata is indexed with child chunks so generic questions can match headings even when the PDF uses unusual section names.

Native PDF tables are extracted with PyMuPDF and stored as Markdown. Scanned pages can use the optional Tesseract-backed OCR fallback.

## LLM providers

ResRAG uses the OpenAI Python client for generation and can switch providers without changing the retrieval pipeline.

### OpenAI

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

### OpenRouter

```text
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
```

### Groq

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=...
```

## Progressive ingestion

On upload, ResRAG performs a lightweight text extraction and builds a BM25 index before starting the expensive full indexing task. Small documents can be answered directly from their complete extracted text when they are below `FAST_FULL_TEXT_WORDS`. Larger documents use BM25 for immediate answers while dense embeddings, tables, OCR, structural groups, and reranking are prepared in the background.

The document is identified by SHA-256, so re-uploading the same PDF within a running process can reuse the existing indexing job.

## Long-document retrieval

For documents at or above `LONG_DOCUMENT_PAGES`, the full index adds a page-level representation built from precomputed child embeddings plus a page BM25 index. Queries retrieve a small set of relevant pages and adjacent pages, then recover the strongest child passages from those pages before the normal reranking stage. This gives the system an intermediate page-recall layer without another query embedding pass.

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

The requirements include matching PyTorch/torchvision versions because Streamlit's module watcher can inspect optional Transformers vision modules on Windows. The embedding and reranker models are downloaded from Hugging Face on first use of the full index.

## Test

```bash
pytest -q
```

## Research references

- Shin et al. (2026), *HiKEY: Hierarchical Multimodal Retrieval for Open-Domain Document Question Answering*.
- Elchafei et al. (2026), *H-RAG at SemEval-2026 Task 8: Hierarchical Parent–Child Retrieval for Multi-Turn RAG Conversations*.
- Chen et al. (2026), *Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering*.
- Kobeissi & Langlais (2026), *Decomposing Retrieval Failures in RAG for Long-Document Financial Question Answering*.
- Xu (2026), *Index Light, Reason Deep: Deferred Visual Ingestion for Visual-Dense Document Question Answering*.
- Lu et al. (2026), *HiChunk: Evaluating and Enhancing Retrieval Augmented Generation with Hierarchical Chunking*.
- Anthropic (2024), *Contextual Retrieval*.

## Scope

The current product intentionally focuses on one active PDF at a time. Evaluation covers retrieval recall/MRR, citation correctness, answer faithfulness, latency, and memory use on ordinary, table-heavy, scanned, and long documents.
