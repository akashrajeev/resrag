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


# Domain-neutral query breadth detection. It describes the shape of the
# requested answer rather than assuming anything about the PDF's subject.
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
    """Low-latency hybrid retrieval with document-discovered parent/child groups.

    Child chunks are indexed with their own structural context (section/page).
    At query time, dense + BM25 retrieval finds precise evidence, while a
    precomputed group embedding identifies the relevant parent context. The
    engine then reconstructs sibling evidence from that parent. No document
    type or domain taxonomy is embedded in the algorithm.
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
        self.last_evidence_groups = 0

    @staticmethod
    def _group_key(chunk: Chunk) -> str:
        """Use a section discovered from the PDF; otherwise use a page group."""
        return chunk.section.strip() or f"__page_{chunk.page}"

    @staticmethod
    def _contextual_text(chunk: Chunk) -> str:
        """Preserve structural context for both dense and lexical retrieval.

        This is a deterministic version of contextual retrieval: section/page
        information is prepended during indexing, while the original chunk is
        still shown to the user and sent to the generator.
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

    def _group_rank(
        self,
        query: str,
        query_embedding: np.ndarray,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        fused: dict[int, float],
    ) -> list[str]:
        """Rank discovered groups by semantic fit + strongest child evidence.

        Exact overlap with a group's actual discovered heading is only a small
        boost. Otherwise the group centroid and child relevance drive the rank,
        allowing arbitrary section names such as "Selected Work" or "Methodology".
        """
        if not self.group_names or self.group_centroids is None:
            return []

        group_semantic = self.group_centroids @ query_embedding
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, str]] = []
        for group_index, group in enumerate(self.group_names):
            ids = self.group_to_ids[group]
            child_dense = max((float(dense_scores[idx]) for idx in ids), default=0.0)
            child_fused = max((fused.get(idx, 0.0) for idx in ids), default=0.0)
            group_tokens = set(tokenize(group.replace("__page_", "")))
            lexical_overlap = len(query_tokens & group_tokens) / max(len(query_tokens), 1)
            score = (
                0.58 * float(group_semantic[group_index])
                + 0.24 * child_dense
                + 0.14 * child_fused
                + 0.04 * lexical_overlap
            )
            scored.append((score, group))
        scored.sort(reverse=True)
        return [group for _, group in scored]

    def _select_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        candidate_ids: list[int],
        fused: dict[int, float],
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        broad: bool,
        effective_k: int,
    ) -> list[int]:
        group_rank = self._group_rank(query, query_embedding, dense_scores, sparse_scores, fused)
        if not group_rank:
            return candidate_ids[: max(effective_k, 4)]

        candidate_set = set(candidate_ids)
        selected_groups: list[str] = []
        if broad:
            # A strong top-group match gets most of the evidence budget. This
            # is what turns "what are the ...?" into parent-level coverage while
            # still allowing overview questions to span multiple groups.
            top_group = group_rank[0]
            top_group_tokens = set(tokenize(top_group.replace("__page_", "")))
            query_tokens = set(tokenize(query))
            exact_group_match = bool(query_tokens & top_group_tokens)
            selected_groups.append(top_group)
            if not exact_group_match:
                selected_groups.extend(group_rank[1:3])
            else:
                selected_groups.extend(group_rank[1:2])
        else:
            selected_groups = group_rank[:3]

        expanded: list[int] = []
        for group in selected_groups:
            ids = self.group_to_ids[group]
            ranked = sorted(
                ids,
                key=lambda idx: (
                    fused.get(idx, 0.0),
                    float(dense_scores[idx]),
                    float(sparse_scores[idx]),
                ),
                reverse=True,
            )
            if broad:
                limit = len(ranked)
            else:
                limit = min(len(ranked), 3)
            expanded.extend(ranked[:limit])

        expanded.extend(candidate_ids)
        expanded = list(dict.fromkeys(expanded))

        if broad:
            # Allocate about 70% of the final bundle to the best semantic
            # parent and the remainder across secondary groups.
            primary = self.group_to_ids[group_rank[0]]
            primary_ranked = sorted(
                [idx for idx in expanded if idx in primary],
                key=lambda idx: (fused.get(idx, 0.0), float(dense_scores[idx])),
                reverse=True,
            )
            primary_budget = min(len(primary_ranked), max(2, int(np.ceil(effective_k * 0.7))))
            ordered = primary_ranked[:primary_budget]
            remaining = [idx for idx in expanded if idx not in ordered]
            ordered.extend(remaining)
            return ordered

        return sorted(
            expanded,
            key=lambda idx: (fused.get(idx, 0.0), float(dense_scores[idx]), float(sparse_scores[idx])),
            reverse=True,
        )

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
            self.last_query_profile = "focused"
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
        effective_k = max(final_k, 6 if broad else 4)
        expanded = self._select_candidates(
            query,
            q,
            child_candidates,
            fused,
            dense_scores,
            sparse_scores,
            broad,
            effective_k,
        )

        candidate_window = max(self.rerank_candidate_k, effective_k * 2, 12 if broad else 8)
        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": fused.get(idx, 0.0),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
            }
            for idx in expanded[:candidate_window]
        ]

        # Rerank coverage queries too, but only over the compact reconstructed
        # parent bundle. This recovers quality without reranking the full index.
        should_rerank = bool(
            self.reranker
            and len(candidates) > effective_k
            and rerank_mode != "off"
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

        group_rank = self._group_rank(query, q, dense_scores, sparse_scores, fused)
        selected: list[dict[str, Any]] = []
        if broad and group_rank:
            primary_group = group_rank[0]
            primary_items = [
                item for item in candidates if self._group_key(item["chunk"]) == primary_group
            ]
            secondary_items = [
                item for item in candidates if self._group_key(item["chunk"]) != primary_group
            ]
            # Preserve most of the evidence from the best parent group.
            primary_budget = min(len(primary_items), max(3, int(np.ceil(effective_k * 0.7))))
            selected.extend(primary_items[:primary_budget])
            selected.extend(secondary_items[: max(0, effective_k - len(selected))])
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
