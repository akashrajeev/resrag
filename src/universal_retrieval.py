from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .resrag import Chunk
from .text_utils import tokenize


# Query-side breadth detection is domain-neutral: it describes the information
# shape requested by the user, not the type of document being searched.
_BROAD_QUERY_RE = re.compile(
    r"^(?:what are|which are|what kinds|what types|list|name|summarize|overview|"
    r"give me an overview|give an overview|describe|compare|how many)\b",
    re.IGNORECASE,
)
_BROAD_WORDS = {
    "all", "every", "main", "key", "major", "primary", "overview",
    "summary", "highlights", "various", "different", "multiple",
}


class UniversalHybridIndex:
    """Fast hybrid retrieval with document-discovered parent/child structure.

    Each child chunk is indexed together with its local structural context
    (detected section name + page). Retrieval happens at child level, then
    parent/sibling evidence is reconstructed before optional reranking.
    No document-domain vocabulary is required.
    """

    def __init__(
        self,
        embedding_model: str,
        reranker_model: str | None = None,
        *,
        embedder: SentenceTransformer | None = None,
        reranker: CrossEncoder | None = None,
        rerank_candidate_k: int | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedder = embedder or SentenceTransformer(embedding_model)
        self.reranker = reranker or (CrossEncoder(reranker_model) if reranker_model else None)
        self.rerank_candidate_k = (
            int(rerank_candidate_k)
            if rerank_candidate_k is not None
            else int(os.getenv("RERANK_CANDIDATE_K", "8"))
        )
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.retrieval_texts: list[str] = []
        self.group_to_ids: dict[str, list[int]] = {}
        self.group_names: list[str] = []
        self.group_centroids: np.ndarray | None = None
        self.last_latency: dict[str, float] = {}
        self.last_rerank_mode = "off"
        self.last_query_profile = "focused"
        self.last_evidence_groups: int = 0

    @staticmethod
    def _group_key(chunk: Chunk) -> str:
        """Use the document's own discovered section, else a page fallback."""
        return chunk.section.strip() or f"__page_{chunk.page}"

    @staticmethod
    def _contextual_text(chunk: Chunk) -> str:
        """Keep structural metadata in the searchable representation.

        This fixes a subtle failure mode in hierarchical RAG: headings may be
        removed from body chunks during parsing, making exact queries for a
        section impossible to retrieve. The UI still displays only chunk.text.
        """
        prefix: list[str] = []
        if chunk.section.strip():
            prefix.append(f"Section: {chunk.section.strip()}")
        prefix.append(f"Page: {chunk.page}")
        prefix.append(chunk.kind)
        return " | ".join(prefix) + "\n" + chunk.text

    @staticmethod
    def _query_profile(query: str) -> str:
        normalized = " ".join(query.lower().split())
        tokens = set(tokenize(normalized))
        if _BROAD_QUERY_RE.search(normalized):
            return "coverage"
        if tokens & _BROAD_WORDS:
            return "coverage"
        if re.search(r"\b(?:has|have|includes|contains|consists of)\b", normalized):
            return "coverage"
        # Plural noun phrasing often asks for several evidence units. This is
        # intentionally generic rather than tied to any document category.
        words = normalized.rstrip("?").split()
        if words and re.search(r"s$", words[-1]) and len(words) <= 8:
            return "coverage"
        return "focused"

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> list[int]:
        k = min(max(1, k), len(scores))
        if k >= len(scores):
            return np.argsort(-scores).tolist()
        partition = np.argpartition(scores, -k)[-k:]
        return partition[np.argsort(-scores[partition])].tolist()

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        start = perf_counter()
        self.chunks = chunks
        self.retrieval_texts = [self._contextual_text(chunk) for chunk in chunks]

        groups: dict[str, list[int]] = {}
        for idx, chunk in enumerate(chunks):
            groups.setdefault(self._group_key(chunk), []).append(idx)
        self.group_to_ids = groups
        self.group_names = list(groups)

        self.embeddings = self.embedder.encode(
            self.retrieval_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([tokenize(text) for text in self.retrieval_texts])

        centroids: list[np.ndarray] = []
        for group in self.group_names:
            ids = self.group_to_ids[group]
            centroid = self.embeddings[ids].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids.append(centroid / norm if norm else centroid)
        self.group_centroids = np.asarray(centroids, dtype=np.float32)
        self.last_latency = {"index_build_ms": (perf_counter() - start) * 1000.0}

    def _expand_groups(
        self,
        candidate_ids: list[int],
        fused: dict[int, float],
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        broad: bool,
    ) -> tuple[list[int], int]:
        dense_lookup = {idx: rank for rank, idx in enumerate(self._top_k(dense_scores, len(dense_scores)), start=1)}
        sparse_lookup = {idx: rank for rank, idx in enumerate(self._top_k(sparse_scores, len(sparse_scores)), start=1)}

        def score(idx: int) -> float:
            return (
                fused.get(idx, 0.0)
                + 0.02 / (60 + dense_lookup.get(idx, 100000))
                + 0.02 / (60 + sparse_lookup.get(idx, 100000))
            )

        # Seed parents from independent child retrieval results.
        selected_groups: list[str] = []
        for idx in candidate_ids:
            group = self._group_key(self.chunks[idx])
            if group not in selected_groups:
                selected_groups.append(group)
            if len(selected_groups) >= (6 if broad else 3):
                break

        expanded: list[int] = []
        for group in selected_groups:
            ids = sorted(self.group_to_ids[group], key=score, reverse=True)
            limit = len(ids) if broad else min(len(ids), 3)
            expanded.extend(ids[:limit])

        # Also add physical neighbors for chunks in ungrouped/page groups;
        # this recovers continuation text without requiring headings.
        if not broad:
            for idx in candidate_ids[:3]:
                page = self.chunks[idx].page
                nearby = [j for j, chunk in enumerate(self.chunks) if chunk.page == page]
                if idx in nearby:
                    position = nearby.index(idx)
                    expanded.extend(nearby[max(0, position - 1) : position + 2])

        ordered = list(dict.fromkeys(expanded + candidate_ids))

        # For coverage queries, diversify by parent first so the evidence bundle
        # does not get dominated by a single highly similar child chunk.
        if broad:
            diverse: list[int] = []
            seen_groups: set[str] = set()
            for idx in sorted(ordered, key=score, reverse=True):
                group = self._group_key(self.chunks[idx])
                if group not in seen_groups:
                    diverse.append(idx)
                    seen_groups.add(group)
            for idx in sorted(ordered, key=score, reverse=True):
                if idx not in diverse:
                    diverse.append(idx)
            ordered = diverse

        target_k = max(6 if broad else 4, int(os.getenv("RETRIEVAL_FINAL_K", "4")))
        return ordered, target_k

    def retrieve(
        self,
        query: str,
        dense_k: int = 16,
        sparse_k: int = 16,
        final_k: int = 4,
        *,
        rerank_mode: str = "auto",
    ) -> list[dict[str, Any]]:
        if not query.strip() or final_k <= 0:
            self.last_latency = {}
            return []
        if rerank_mode not in {"auto", "on", "off"}:
            raise ValueError("rerank_mode must be one of: auto, on, off")
        if self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        start = perf_counter()
        self.last_query_profile = self._query_profile(query)
        broad = self.last_query_profile == "coverage"
        dense_k = min(max(dense_k, 1), len(self.chunks))
        sparse_k = min(max(sparse_k, 1), len(self.chunks))

        def dense_query():
            return self.embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

        def sparse_query():
            return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(dense_query)
            sparse_future = pool.submit(sparse_query)
            q = dense_future.result()
            sparse_scores = sparse_future.result()

        dense_scores = self.embeddings @ q
        dense_rank = self._top_k(dense_scores, dense_k)
        sparse_rank = self._top_k(sparse_scores, sparse_k)

        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(sparse_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)

        child_candidates = [idx for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        expanded_ids, effective_k = self._expand_groups(
            child_candidates,
            fused,
            dense_scores,
            sparse_scores,
            broad,
        )

        candidates: list[dict[str, Any]] = []
        candidate_window = max(self.rerank_candidate_k, effective_k * 2, 12 if broad else 8)
        for idx in expanded_ids[:candidate_window]:
            candidates.append(
                {
                    "chunk": self.chunks[idx],
                    "hybrid_score": fused.get(idx, 0.0),
                    "dense_score": float(dense_scores[idx]),
                    "bm25_score": float(sparse_scores[idx]),
                }
            )

        should_rerank = bool(
            self.reranker
            and len(candidates) > effective_k
            and rerank_mode != "off"
            and (rerank_mode == "on" or not broad)
        )
        rerank_start = perf_counter()
        if should_rerank:
            rerank_candidates = candidates[: min(self.rerank_candidate_k, len(candidates))]
            scores = self.reranker.predict(
                [(query, item["chunk"].text) for item in rerank_candidates],
                show_progress_bar=False,
            )
            for item, score_value in zip(rerank_candidates, scores, strict=True):
                item["rerank_score"] = float(score_value)
            rerank_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
            candidates = rerank_candidates + candidates[len(rerank_candidates):]
            self.last_rerank_mode = "on"
        else:
            self.last_rerank_mode = "off"
        rerank_ms = (perf_counter() - rerank_start) * 1000.0

        # For broad questions, select a compact but coverage-oriented bundle.
        selected: list[dict[str, Any]] = []
        seen_groups: set[str] = set()
        if broad:
            for item in candidates:
                group = self._group_key(item["chunk"])
                if group not in seen_groups or len(selected) < min(effective_k, 2):
                    selected.append(item)
                    seen_groups.add(group)
                if len(selected) >= effective_k:
                    break
            if len(selected) < effective_k:
                selected.extend(item for item in candidates if item not in selected)
                selected = selected[:effective_k]
        else:
            selected = candidates[:effective_k]

        self.last_evidence_groups = len({self._group_key(item["chunk"]) for item in selected})
        self.last_latency = {
            "query_ms": (perf_counter() - start) * 1000.0,
            "rerank_ms": rerank_ms,
        }
        return selected
