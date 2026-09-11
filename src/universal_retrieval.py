from __future__ import annotations

import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .resrag import Chunk
from .text_utils import tokenize


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
    """Quality-first, document-agnostic hybrid retrieval.

    Dense + BM25 preserve high first-stage recall. The PDF's own discovered
    section/page groups add parent/child and neighbor evidence. Reranking is
    retained for quality over a bounded candidate pool. Query caching and
    parallel independent retrieval are the primary latency optimizations.
    """

    def __init__(
        self,
        embedding_model: str,
        reranker_model: str | None = None,
        *,
        embedder: SentenceTransformer | None = None,
        reranker: CrossEncoder | None = None,
        rerank_candidate_k: int | None = None,
        cache_size: int | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedder = embedder or SentenceTransformer(embedding_model)
        self.reranker = reranker or (CrossEncoder(reranker_model) if reranker_model else None)
        self.rerank_candidate_k = int(rerank_candidate_k or os.getenv("RERANK_CANDIDATE_K", "24"))
        self.cache_size = int(cache_size or os.getenv("RETRIEVAL_CACHE_SIZE", "64"))
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.retrieval_texts: list[str] = []
        self.group_to_ids: dict[str, list[int]] = {}
        self.group_names: list[str] = []
        self.group_centroids: np.ndarray | None = None
        self.query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.result_cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()
        self.last_latency: dict[str, float] = {}
        self.last_rerank_mode = "off"
        self.last_query_profile = "focused"
        self.last_evidence_groups = 0

    @staticmethod
    def _group_key(chunk: Chunk) -> str:
        return chunk.section.strip() or f"__page_{chunk.page}"

    @staticmethod
    def _contextual_text(chunk: Chunk) -> str:
        metadata = [f"Page: {chunk.page}", f"Type: {chunk.kind}"]
        if chunk.section.strip():
            metadata.insert(0, f"Section: {chunk.section.strip()}")
        return " | ".join(metadata) + "\n" + chunk.text

    @staticmethod
    def _query_profile(query: str) -> str:
        normalized = " ".join(query.lower().split())
        if _BROAD_QUERY_RE.search(normalized):
            return "coverage"
        tokens = set(tokenize(normalized))
        if tokens & _BROAD_WORDS:
            return "coverage"
        if re.search(r"\b(?:has|have|includes|contains|consists of)\b", normalized):
            return "coverage"
        words = normalized.rstrip("?").split()
        if words and len(words) <= 8 and words[-1].endswith("s"):
            return "coverage"
        return "focused"

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> list[int]:
        if len(scores) == 0:
            return []
        k = min(max(1, k), len(scores))
        if k >= len(scores):
            return np.argsort(-scores).tolist()
        partition = np.argpartition(scores, -k)[-k:]
        return partition[np.argsort(-scores[partition])].tolist()

    def _get_query_embedding(self, query: str) -> tuple[np.ndarray, bool]:
        key = " ".join(query.lower().split())
        cached = self.query_embedding_cache.get(key)
        if cached is not None:
            self.query_embedding_cache.move_to_end(key)
            return cached, True
        vector = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].astype(np.float32)
        self.query_embedding_cache[key] = vector
        self.query_embedding_cache.move_to_end(key)
        while len(self.query_embedding_cache) > self.cache_size:
            self.query_embedding_cache.popitem(last=False)
        return vector, False

    def _get_cached_result(self, key: tuple) -> list[dict[str, Any]] | None:
        value = self.result_cache.get(key)
        if value is not None:
            self.result_cache.move_to_end(key)
            return list(value)
        return None

    def _put_cached_result(self, key: tuple, value: list[dict[str, Any]]) -> None:
        self.result_cache[key] = list(value)
        self.result_cache.move_to_end(key)
        while len(self.result_cache) > self.cache_size:
            self.result_cache.popitem(last=False)

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        start = perf_counter()
        self.chunks = chunks
        self.retrieval_texts = [self._contextual_text(c) for c in chunks]
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
            centroid = self.embeddings[self.group_to_ids[group]].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids.append(centroid / norm if norm else centroid)
        self.group_centroids = np.asarray(centroids, dtype=np.float32)
        self.query_embedding_cache.clear()
        self.result_cache.clear()
        self.last_latency = {"index_build_ms": (perf_counter() - start) * 1000.0}

    def _rank_groups(self, query_embedding: np.ndarray, dense_scores: np.ndarray, fused: dict[int, float], query: str) -> list[str]:
        if not self.group_names or self.group_centroids is None:
            return []
        semantic = self.group_centroids @ query_embedding
        query_tokens = set(tokenize(query))
        ranked: list[tuple[float, str]] = []
        for group_index, group in enumerate(self.group_names):
            ids = self.group_to_ids[group]
            best_dense = max((float(dense_scores[i]) for i in ids), default=0.0)
            best_fused = max((fused.get(i, 0.0) for i in ids), default=0.0)
            group_tokens = set(tokenize(group.replace("__page_", "")))
            lexical = len(query_tokens & group_tokens) / max(1, len(query_tokens))
            score = 0.62 * float(semantic[group_index]) + 0.25 * best_dense + 0.09 * best_fused + 0.04 * lexical
            ranked.append((score, group))
        ranked.sort(reverse=True)
        return [group for _, group in ranked]

    def _expand(self, child_candidates: list[int], query_embedding: np.ndarray, dense_scores: np.ndarray, sparse_scores: np.ndarray, fused: dict[int, float], query: str, broad: bool) -> list[int]:
        groups = self._rank_groups(query_embedding, dense_scores, fused, query)
        expanded = list(child_candidates)
        for group in groups[:4 if broad else 3]:
            ids = sorted(
                self.group_to_ids[group],
                key=lambda i: (fused.get(i, 0.0), float(dense_scores[i]), float(sparse_scores[i])),
                reverse=True,
            )
            expanded.extend(ids[: min(len(ids), 12 if broad else 4)])
        for idx in child_candidates[:8]:
            group = self._group_key(self.chunks[idx])
            ids = self.group_to_ids.get(group, [])
            try:
                pos = ids.index(idx)
            except ValueError:
                continue
            expanded.extend(ids[max(0, pos - 1): pos + 2])
        return list(dict.fromkeys(expanded))

    def retrieve(self, query: str, dense_k: int = 32, sparse_k: int = 32, final_k: int = 8, *, rerank_mode: str = "auto") -> list[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned or final_k <= 0:
            self.last_latency = {}
            self.last_rerank_mode = "off"
            self.last_query_profile = "focused"
            return []
        if rerank_mode not in {"auto", "on", "off"}:
            raise ValueError("rerank_mode must be one of: auto, on, off")
        if self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        profile = self._query_profile(cleaned)
        self.last_query_profile = profile
        broad = profile == "coverage"
        cache_key = (" ".join(cleaned.lower().split()), dense_k, sparse_k, final_k, rerank_mode)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            self.last_latency = {"cache_hit": 0.0}
            return cached

        start = perf_counter()
        dense_k = min(max(1, dense_k), len(self.chunks))
        sparse_k = min(max(1, sparse_k), len(self.chunks))
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._get_query_embedding, cleaned)
            sparse_future = pool.submit(lambda: np.asarray(self.bm25.get_scores(tokenize(cleaned)), dtype=np.float32))
            q, query_cache_hit = dense_future.result()
            sparse_scores = sparse_future.result()

        dense_scores = self.embeddings @ q
        dense_rank = self._top_k(dense_scores, dense_k)
        sparse_rank = self._top_k(sparse_scores, sparse_k)
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(sparse_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)

        small_threshold = max(8, int(os.getenv("SMALL_DOC_CHUNKS", "16")))
        if broad and len(self.chunks) <= small_threshold:
            candidate_ids = list(range(len(self.chunks)))
            effective_k = len(self.chunks)
        else:
            effective_k = min(len(self.chunks), max(final_k, 10 if broad else final_k))
            child_candidates = [idx for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
            expanded = self._expand(child_candidates, q, dense_scores, sparse_scores, fused, cleaned, broad)
            candidate_ids = expanded[: max(32 if broad else 24, self.rerank_candidate_k, effective_k * 3)]

        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": fused.get(idx, 0.0),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
            }
            for idx in candidate_ids
        ]

        # The earlier aggressive agreement-based reranker skip is deliberately
        # gone. Quality is favored; only the candidate set is bounded.
        should_rerank = bool(
            self.reranker
            and len(candidates) > effective_k
            and rerank_mode != "off"
            and not (broad and len(self.chunks) <= small_threshold)
        )
        rerank_start = perf_counter()
        if should_rerank:
            rerank_candidates = candidates[: min(self.rerank_candidate_k, len(candidates))]
            scores = self.reranker.predict(
                [(cleaned, item["chunk"].text) for item in rerank_candidates],
                show_progress_bar=False,
            )
            for item, score in zip(rerank_candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            rerank_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
            candidates = rerank_candidates + candidates[len(rerank_candidates):]
            self.last_rerank_mode = "on"
        else:
            self.last_rerank_mode = "off"
        rerank_ms = (perf_counter() - rerank_start) * 1000.0

        if broad and len(self.chunks) <= small_threshold:
            selected = candidates
        elif broad:
            groups = self._rank_groups(q, dense_scores, fused, cleaned)
            selected = []
            if groups:
                primary_group = groups[0]
                primary = [item for item in candidates if self._group_key(item["chunk"]) == primary_group]
                secondary = [item for item in candidates if self._group_key(item["chunk"]) != primary_group]
                primary_budget = min(len(primary), max(4, int(np.ceil(effective_k * 0.70))))
                selected.extend(primary[:primary_budget])
                selected.extend(secondary[: max(0, effective_k - len(selected))])
            else:
                selected = candidates[:effective_k]
            if len(selected) < effective_k:
                selected.extend(item for item in candidates if item not in selected)
                selected = selected[:effective_k]
        else:
            selected = candidates[:effective_k]

        self.last_evidence_groups = len({self._group_key(item["chunk"]) for item in selected})
        self.last_latency = {
            "query_ms": (perf_counter() - start) * 1000.0,
            "rerank_ms": rerank_ms,
            "query_embedding_cache_hit": 1.0 if query_cache_hit else 0.0,
            "cache_hit": 0.0,
        }
        self._put_cached_result(cache_key, selected)
        return list(selected)
