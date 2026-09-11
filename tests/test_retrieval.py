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


def test_targeted_coverage_retrieves_all_chunks_from_matching_section():
    chunks = [
        Chunk(0, 1, "YatraMind: AI-powered metro operations system.", section="Projects"),
        Chunk(1, 1, "AgenticCommerce: autonomous commerce workflow.", section="Projects"),
        Chunk(2, 1, "ResRAG: retrieval augmented generation system.", section="Projects"),
        Chunk(3, 1, "Python, C++, FastAPI, Docker and REST APIs.", section="Skills"),
    ]
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build(chunks)
    results = index.retrieve("What are the main projects?", dense_k=4, sparse_k=4, final_k=4)
    project_texts = [item["chunk"].text for item in results if item["chunk"].section == "Projects"]
    assert index.last_query_profile == "targeted"
    assert len(project_texts) == 3
    assert all(name in " ".join(project_texts) for name in ("YatraMind", "AgenticCommerce", "ResRAG"))


def test_overview_coverage_spreads_across_sections():
    chunks = [
        Chunk(0, 1, "Candidate profile and software engineering student.", section="Summary"),
        Chunk(1, 1, "Python, C++, FastAPI and Docker.", section="Skills"),
        Chunk(2, 1, "Army Institute of Technology.", section="Education"),
        Chunk(3, 1, "YatraMind AI-powered metro operations system.", section="Projects"),
    ]
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build(chunks)
    results = index.retrieve("What is this document about?", dense_k=4, sparse_k=4, final_k=4)
    sections = {item["chunk"].section for item in results}
    assert index.last_query_profile == "overview"
    assert {"Summary", "Skills", "Education", "Projects"}.issubset(sections)


def test_empty_query_returns_no_results():
    index = HybridIndex("fake", embedder=FakeEmbedder())
    index.build([Chunk(0, 1, "Some content")])
    assert index.retrieve("   ") == []
    assert index.retrieve("content", final_k=0) == []
