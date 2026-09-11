# Low-Latency RAG Design

ResRAG is intended to feel interactive rather than like a batch document-analysis tool. The key constraint is that "under one second" must be treated as an end-to-end latency target, not a guarantee: network round trips, provider queueing, model prefill, and output length are outside the local retrieval process.

## Research-driven changes

### 1. Adaptive reranking instead of reranking every query

Cross-encoders improve ranking quality but add inference latency. Recent 2026 work on cascade reranking reports large latency reductions by applying expensive reranking only to the cases that need it. ResRAG therefore uses an `auto` mode: when BM25 and dense retrieval independently agree on the top result, the cross-encoder is skipped; disagreement triggers reranking on a small candidate set.

This preserves the high-quality path for ambiguous queries while making the common path cheaper.

### 2. Parallel lexical and dense retrieval

BM25 scoring and query embedding are independent operations. ResRAG now overlaps them with a two-worker thread pool instead of waiting for one before starting the other.

### 3. Smaller candidate and evidence budgets

The default request now uses 12 dense candidates, 12 sparse candidates, and 4 final evidence passages. Reranking, when needed, is limited to 8 candidates. Evidence is truncated to 1,400 characters per passage before generation.

The goal is to reduce model prefill and generation work without removing page-aware grounding.

### 4. Faster local inference backends

Sentence Transformers currently supports ONNX/OpenVINO backends and quantization. Their published efficiency benchmarks show that CPU ONNX/OpenVINO int8 can materially accelerate short-text inference, while warning that the best backend depends on model and hardware. ResRAG exposes `EMBEDDING_BACKEND=onnx` as an optional optimization rather than forcing it on every installation.

### 5. Stream generation instead of waiting for the entire answer

The UI now consumes streamed chat-completion tokens. This does not reduce total generation time by itself, but it lowers time-to-first-visible-token and makes the application feel much more responsive.

### 6. Provider-aware low-latency routing

Groq currently lists `openai/gpt-oss-20b` at roughly 1,000 tokens/sec. OpenRouter supports routing by latency. ResRAG therefore uses `openai/gpt-oss-20b` as the Groq default and requests OpenRouter latency-prioritized routing.

## What we are deliberately not doing yet

### Always-on heavy reranking

Keeping `BAAI/bge-reranker-v2-m3` on every request defeats the latency goal. The model remains available for difficult/ambiguous queries and for evaluation mode.

### Large query-rewriting LLMs

Query rewriting can improve multi-turn retrieval, but invoking another model before retrieval adds a second network/inference step. The current follow-up resolver is deterministic and local. A small local rewriter can be evaluated later if real multi-turn tests show a measurable retrieval benefit.

### Full-document CAG for every PDF

Cache-Augmented Generation can eliminate runtime retrieval for small, fixed knowledge bases, and recent work such as CacheNotes/RAGCache demonstrates substantial latency improvements through offline compression or KV reuse. For ResRAG, making CAG the default would weaken the central retrieval experiment and can increase sensitivity to prompt size. It is better treated as an explicit small-document mode after baseline RAG measurements are available.

### Semantic response caching as a correctness shortcut

Semantic caching can make repeated or near-duplicate questions extremely fast, but an overly permissive similarity threshold can return a stale or wrong answer. We will add it only after we have a benchmark for cache precision/recall.

## Target latency budget

For a warm local session on a normal laptop, the initial engineering target is:

- PDF index already built.
- Retrieval: ideally <150 ms.
- Time to first visible generation token: ideally <500 ms when using a low-latency hosted model and a warm connection.
- Total response: around or below 1 second for short answers is a goal, not a universal guarantee.

To tune this honestly, set `SHOW_LATENCY=1`. The UI reports retrieval, LLM, total request time, and whether adaptive reranking ran.

## Recommended fast configuration

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b

RERANK_MODE=auto
RETRIEVAL_DENSE_K=12
RETRIEVAL_SPARSE_K=12
RETRIEVAL_FINAL_K=4
RERANK_CANDIDATE_K=8
MAX_OUTPUT_TOKENS=192
SHOW_LATENCY=1
```

For CPU-only embedding optimization, install the Sentence Transformers ONNX extra and set:

```text
EMBEDDING_BACKEND=onnx
```

Always benchmark the backend on the machine that will actually run ResRAG before making it the default.

## Sources

- Novais et al. (2026), *Optimizing Efficiency in Multi-Stage Semantic Re-ranking Architectures*, ACL PROPOR 2026.
- Farhan & Liebeskind (2026), *JCT at SemEval-2026 Task 8: Resource-Efficient Multi-Turn RAG via Nano-LLM Rewriting and Hybrid Reranking*.
- Wen et al. (2026), *SpecCache: Speculative KV Cache Reuse for Efficient RAG Serving*, ACL 2026.
- Corallo et al. (2026), *CacheNotes: Task-Aware Key-Value Cache Compression for Reasoning-Intensive Knowledge Tasks*, EACL 2026.
- Jin et al. (2024), *RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation*.
