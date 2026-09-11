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


def test_generic_group_coverage_retrieves_all_children():
    chunks = [
        Chunk(0, 1, "Alpha item one.", section="Selected Work"),
        Chunk(1, 1, "Alpha item two.", section="Selected Work"),
        Chunk(2, 1, "Alpha item three.", section="Selected Work"),
        Chunk(3, 1, "Technical capability one.", section="Technical Capabilities"),
    ]
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build(chunks)
    results = index.retrieve("What are the main selected work items?", dense_k=4, sparse_k=4, final_k=4)
    work_texts = [item["chunk"].text for item in results if item["chunk"].section == "Selected Work"]
    assert index.last_query_profile == "coverage"
    assert len(work_texts) == 3


def test_coverage_can_fall_back_to_semantic_group_selection():
    chunks = [
        Chunk(0, 1, "A collection of research findings and experiments.", section="Research Notes"),
        Chunk(1, 2, "Implementation details and architecture decisions.", section="Engineering Record"),
        Chunk(2, 3, "Lessons learned and conclusions.", section="Closing Thoughts"),
    ]
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build(chunks)
    results = index.retrieve("Give an overview of this document", dense_k=3, sparse_k=3, final_k=3)
    sections = {item["chunk"].section for item in results}
    assert index.last_query_profile == "coverage"
    assert len(sections) >= 2


def test_page_groups_work_when_no_heading_is_recovered():
    chunks = [
        Chunk(0, 1, "Page one item A."),
        Chunk(1, 1, "Page one item B."),
        Chunk(2, 2, "Page two item A."),
        Chunk(3, 2, "Page two item B."),
    ]
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build(chunks)
    results = index.retrieve("What are the main items?", dense_k=4, sparse_k=4, final_k=4)
    assert index.last_query_profile == "coverage"
    assert {item["chunk"].page for item in results} == {1, 2}


def test_empty_query_returns_no_results():
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build([Chunk(0, 1, "Some content")])
    assert index.retrieve("   ") == []
    assert index.retrieve("content", final_k=0) == []
