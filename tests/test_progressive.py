import numpy as np

from src.progressive import FastLexicalIndex, LongDocumentHybridIndex
from src.resrag import Chunk


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "target" in lower or "revenue" in lower:
                vectors.append([1.0, 0.0])
            elif "other" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)


def test_fast_lexical_index_returns_all_small_document_content():
    chunks = [
        Chunk(0, 1, "Alpha item one."),
        Chunk(1, 1, "Alpha item two."),
        Chunk(2, 1, "Alpha item three."),
    ]
    index = FastLexicalIndex(chunks, full_text_word_limit=100)
    results = index.retrieve("what are the items?", final_k=1)
    assert len(results) == 3


def test_fast_lexical_index_limits_large_documents():
    chunks = [Chunk(i, i + 1, f"page content {i}") for i in range(20)]
    index = FastLexicalIndex(chunks, full_text_word_limit=10)
    results = index.retrieve("page content", final_k=5)
    assert len(results) == 5


def test_long_document_builds_page_level_index():
    chunks = [
        Chunk(0, 1, "Other content one."),
        Chunk(1, 2, "Target revenue appears here."),
        Chunk(2, 3, "Other content three."),
        Chunk(3, 4, "Other content four."),
    ]
    index = LongDocumentHybridIndex("fake", embedder=FakeEmbedder(), long_document_pages=2)
    index.build(chunks)
    assert index.is_long_document
    assert index.page_bm25 is not None
    assert index.page_centroids is not None


def test_long_document_retrieval_keeps_target_evidence():
    chunks = [
        Chunk(0, 1, "Other content one."),
        Chunk(1, 2, "Target revenue is 42 million."),
        Chunk(2, 3, "Other content three."),
        Chunk(3, 4, "Other content four."),
    ]
    index = LongDocumentHybridIndex("fake", embedder=FakeEmbedder(), long_document_pages=2)
    index.build(chunks)
    results = index.retrieve("What was the target revenue?", dense_k=4, sparse_k=4, final_k=2, rerank_mode="off")
    assert any("Target revenue" in item["chunk"].text for item in results)
