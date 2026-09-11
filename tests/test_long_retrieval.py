import numpy as np

from src.long_retrieval import PageFirstLongDocumentIndex
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


def test_page_first_index_routes_to_relevant_page():
    chunks = [
        Chunk(0, 1, "Other discussion."),
        Chunk(1, 2, "Target revenue was 42 million."),
        Chunk(2, 3, "Other discussion."),
        Chunk(3, 4, "Other discussion."),
    ]
    index = PageFirstLongDocumentIndex("fake", embedder=FakeEmbedder(), long_document_pages=2)
    index.build(chunks)

    assert index.is_long_document
    assert index.page_centroids is not None
    assert index.page_bm25 is not None

    results = index.retrieve("What was the target revenue?", dense_k=4, sparse_k=4, final_k=1, rerank_mode="off")
    assert len(results) == 1
    assert results[0]["chunk"].page == 2
