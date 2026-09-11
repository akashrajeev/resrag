import numpy as np

from src.resrag import Chunk
from src.universal_retrieval import UniversalHybridIndex


class SemanticEmbedder:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "selected work" in lower or "yatra" in lower or "commerce" in lower or "resrag" in lower:
                vectors.append([1.0, 0.0])
            elif "technical capabilities" in lower or "python" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.4, 0.4])
        return np.asarray(vectors, dtype=np.float32)


def test_arbitrary_group_heading_is_searchable_and_siblings_are_recovered():
    chunks = [
        Chunk(0, 1, "YatraMind builds metro operations tooling.", section="Selected Work"),
        Chunk(1, 1, "AgenticCommerce automates commerce workflows.", section="Selected Work"),
        Chunk(2, 1, "ResRAG provides grounded document retrieval.", section="Selected Work"),
        Chunk(3, 1, "Python, C++, FastAPI and Docker.", section="Technical Capabilities"),
    ]
    index = UniversalHybridIndex("fake", embedder=SemanticEmbedder())
    index.build(chunks)

    results = index.retrieve("What are the projects?", dense_k=4, sparse_k=4, final_k=4)
    text = " ".join(item["chunk"].text for item in results)

    assert index.last_query_profile == "coverage"
    assert all(name in text for name in ("YatraMind", "AgenticCommerce", "ResRAG"))
    assert sum(item["chunk"].section == "Selected Work" for item in results) >= 3


def test_focus_query_remains_small_and_fast_path_friendly():
    chunks = [
        Chunk(0, 1, "Revenue was $42M in 2025.", section="Financial Results"),
        Chunk(1, 2, "The company launched a new product.", section="Announcements"),
        Chunk(2, 3, "Employees numbered 120.", section="Company Data"),
    ]
    index = UniversalHybridIndex("fake", embedder=SemanticEmbedder())
    index.build(chunks)
    results = index.retrieve("What was revenue in 2025?", dense_k=3, sparse_k=3, final_k=3)

    assert index.last_query_profile == "focused"
    assert len(results) <= 3
