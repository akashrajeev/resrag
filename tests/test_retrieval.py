import numpy as np

from src.resrag import Chunk, HybridIndex


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            value = 1.0 if "python" in text.lower() else 0.2
            vectors.append([value, 1.0 - value])
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, show_progress_bar=False):
        return np.asarray([2.0 if "python" in text.lower() else 0.1 for _, text in pairs], dtype=np.float32)


def test_forced_reranking_uses_cross_encoder():
    chunks = [
        Chunk(0, 1, "Python is used for data analysis."),
        Chunk(1, 2, "The office is located in Berlin."),
        Chunk(2, 3, "Python supports scientific computing."),
    ]
    index = HybridIndex("fake", "fake", embedder=FakeEmbedder(), reranker=FakeReranker())
    index.build(chunks)
    results = index.retrieve("python", dense_k=3, sparse_k=3, final_k=2, rerank_mode="on")
    assert len(results) == 2
    assert results[0]["chunk"].page in {1, 3}
    assert "rerank_score" in results[0]
    assert index.last_rerank_mode == "on"


def test_auto_mode_skips_reranking_when_retrievers_agree():
    chunks = [
        Chunk(0, 1, "Python is used for data analysis."),
        Chunk(1, 2, "Berlin is the office location."),
        Chunk(2, 3, "Cooking recipes are unrelated."),
    ]
    index = HybridIndex("fake", "fake", embedder=FakeEmbedder(), reranker=FakeReranker())
    index.build(chunks)
    results = index.retrieve("python", dense_k=3, sparse_k=3, final_k=2, rerank_mode="auto")
    assert len(results) == 2
    assert index.last_rerank_mode == "off"
    assert "rerank_score" not in results[0]


def test_empty_query_returns_no_results():
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build([Chunk(0, 1, "Some content")])
    assert index.retrieve("   ") == []
    assert index.retrieve("content", final_k=0) == []
