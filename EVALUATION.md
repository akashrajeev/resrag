# ResRAG evaluation plan

ResRAG is designed to be tested as a retrieval system first and an answer-generation system second.

## Retrieval

Create a small labeled set of questions for each test PDF. For every question, record one or more relevant chunk IDs. Report Recall@k and MRR using `src.evaluation`.

Recommended first checkpoints:

- Recall@5 and Recall@10 for the hybrid retriever before reranking.
- MRR@10 after reranking.
- Separate results for ordinary prose, exact-number questions, table questions, and scanned-page questions.

## Citations

For each generated answer, record the expected evidence pages and the pages cited by the answer. Report citation precision and citation recall using `src.evaluation`.

## Answer quality

For a human-labeled set, score whether the answer is supported by the retrieved evidence, whether it answers the question directly, and whether numerical/table claims preserve the source meaning. RAGChecker-style fine-grained diagnostics are a useful reference for this stage.

## Latency

Measure:

1. PDF ingestion and indexing time.
2. Retrieval time.
3. Reranking time.
4. LLM generation time.
5. End-to-end question latency.

Keep these measurements separate so retrieval regressions are distinguishable from model/API latency.
