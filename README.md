# ResRAG

A small, local-first PDF question-answering app with a ChatGPT-style interface.

## Architecture

PDF → page-aware extraction → paragraph-aware chunks → parallel BM25 + dense retrieval → Reciprocal Rank Fusion → cross-encoder reranking → grounded LLM answer with page citations.

The retrieval stack follows current evidence favoring sparse+dense hybrid retrieval and second-stage neural reranking for heterogeneous document QA. The implementation deliberately stays simple enough to run locally while preserving clear seams for stronger parsers, table-aware extraction, and evaluation later.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set these environment variables for generation:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=...
# Optional for OpenAI-compatible providers:
# OPENAI_BASE_URL=https://...
```

The embedding and reranker models are downloaded from Hugging Face on first use.

## Research references

- Akarsu, Karaman & Mierbach (2026), *From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents*. The benchmark reports strong gains from a two-stage hybrid retrieval + neural reranking pipeline and shows BM25 remains important for precise/numerical document questions.
- Meng & Ilvovsky (2026), *Sifei at SemEval-2026 Task 8: Hybrid Retrieval and Query Rewriting for Multi-Turn RAG*. A training-free dense+sparse retrieval + cross-encoder reranking pipeline ranked 3rd of 38 teams on retrieval.
- Shrestha & Aryal (2026), *Howard University-AI4PC at SemEval-2026 Task 8*. Uses dense-lexical fusion and cross-encoder reranking in a multi-turn RAG system.
- Ru et al. (2024), *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*. Provides fine-grained retrieval and generation diagnostics.

## Notes

The current UI is intended for a single active PDF at a time. It keeps the document state in Streamlit session state, exposes page-level citations, and lets the user inspect the retrieved passages behind each answer.
