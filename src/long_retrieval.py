from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .progressive import LongDocumentHybridIndex
from .text_utils import tokenize


class PageFirstLongDocumentIndex(LongDocumentHybridIndex):
    """Long-document index that routes queries to pages before child scoring.

    Unlike the generic child-first retriever, this avoids multiplying the query
    embedding by every chunk vector for long PDFs. A query is first routed to a
    small page set with page-level dense + BM25 fusion, then only chunks in
    those pages are scored, expanded, and reranked.
    """

    def retrieve(
        self,
        query: str,
        dense_k: int = 32,
        sparse_k: int = 32,
        final_k: int = 8,
        *,
        rerank_mode: str = "auto",
    ) -> list[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned or final_k <= 0:
            self.last_latency = {}
            self.last_rerank_mode = "off"
            return []
        if not self.is_long_document or self.page_centroids is None or self.page_bm25 is None:
            return super().retrieve(cleaned, dense_k=dense_k, sparse_k=sparse_k, final_k=final_k, rerank_mode=rerank_mode)
        if rerank_mode not in {"auto", "on", "off"}:
            raise ValueError("rerank_mode must be one of: auto, on, off")

        profile = self._query_profile(cleaned)
        self.last_query_profile = profile
        broad = profile == "coverage"
        cache_key = ("page-first", " ".join(cleaned.lower().split()), dense_k, sparse_k, final_k, rerank_mode)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            self.last_latency = {"cache_hit": 0.0}
            return cached

        start = perf_counter()
        q, query_cache_hit = self._get_query_embedding(cleaned)

        page_dense = self.page_centroids @ q
        page_sparse = np.asarray(self.page_bm25.get_scores(tokenize(cleaned)), dtype=np.float32)
        page_dense_rank = self._top_k(page_dense, min(16 if broad else 10, len(self.page_names)))
        page_sparse_rank = self._top_k(page_sparse, min(16 if broad else 10, len(self.page_names)))

        page_fused: dict[int, float] = {}
        for rank, idx in enumerate(page_dense_rank, start=1):
            page_fused[idx] = page_fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(page_sparse_rank, start=1):
            page_fused[idx] = page_fused.get(idx, 0.0) + 1.0 / (60.0 + rank)

        ranked_page_ids = [idx for idx, _ in sorted(page_fused.items(), key=lambda item: item[1], reverse=True)]
        page_count = 12 if broad else 8
        top_page_numbers = [self.page_names[i] for i in ranked_page_ids[:page_count]]
        selected_page_numbers = set(top_page_numbers)
        neighbor_radius = 1 if broad else 0
        for page in top_page_numbers:
            for neighbor in range(page - neighbor_radius, page + neighbor_radius + 1):
                if neighbor in self.page_to_ids:
                    selected_page_numbers.add(neighbor)

        candidate_ids = [
            idx
            for page in sorted(selected_page_numbers)
            for idx in self.page_to_ids.get(page, [])
        ]
        if not candidate_ids:
            return []

        candidate_ids = list(dict.fromkeys(candidate_ids))
        max_child_candidates = max(64, final_k * 8, self.rerank_candidate_k * 2)
        if len(candidate_ids) > max_child_candidates:
            # Keep broad page coverage, but cap the child scoring work.
            scored_pages = []
            page_lookup = {page: i for i, page in enumerate(self.page_names)}
            for page in selected_page_numbers:
                page_idx = page_lookup[page]
                scored_pages.append((page_fused.get(page_idx, 0.0), page))
            scored_pages.sort(reverse=True)
            ordered_pages = [page for _, page in scored_pages]
            candidate_ids = [
                idx
                for page in ordered_pages
                for idx in self.page_to_ids[page]
            ][:max_child_candidates]

        dense_matrix = self.embeddings[candidate_ids]
        local_dense = dense_matrix @ q
        global_sparse = np.asarray(self.bm25.get_scores(tokenize(cleaned)), dtype=np.float32)
        local_sparse = global_sparse[candidate_ids]

        local_dense_rank = self._top_k(local_dense, min(dense_k, len(candidate_ids)))
        local_sparse_rank = self._top_k(local_sparse, min(sparse_k, len(candidate_ids)))
        local_fused: dict[int, float] = {}
        for rank, local_idx in enumerate(local_dense_rank, start=1):
            local_fused[candidate_ids[local_idx]] = local_fused.get(candidate_ids[local_idx], 0.0) + 1.0 / (60.0 + rank)
        for rank, local_idx in enumerate(local_sparse_rank, start=1):
            local_fused[candidate_ids[local_idx]] = local_fused.get(candidate_ids[local_idx], 0.0) + 1.0 / (60.0 + rank)

        ordered_ids = [idx for idx, _ in sorted(local_fused.items(), key=lambda item: item[1], reverse=True)]
        candidate_window = max(final_k * 3, self.rerank_candidate_k, 24 if broad else 16)
        ordered_ids = ordered_ids[:candidate_window]
        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": local_fused.get(idx, 0.0),
                "dense_score": float(self.embeddings[idx] @ q),
                "bm25_score": float(global_sparse[idx]),
            }
            for idx in ordered_ids
        ]

        should_rerank = bool(self.reranker and candidates and len(candidates) > final_k and rerank_mode != "off")
        rerank_start = perf_counter()
        if should_rerank:
            pool = candidates[: min(self.rerank_candidate_k, len(candidates))]
            scores = self.reranker.predict([(cleaned, item["chunk"].text) for item in pool], show_progress_bar=False)
            for item, score in zip(pool, scores, strict=True):
                item["rerank_score"] = float(score)
            pool.sort(key=lambda item: item["rerank_score"], reverse=True)
            candidates = pool + candidates[len(pool):]
            self.last_rerank_mode = "on"
        else:
            self.last_rerank_mode = "off"

        selected = candidates[:final_k]
        self.last_evidence_groups = len({self._group_key(item["chunk"]) for item in selected})
        self.last_latency = {
            "query_ms": (perf_counter() - start) * 1000.0,
            "rerank_ms": (perf_counter() - rerank_start) * 1000.0,
            "query_embedding_cache_hit": 1.0 if query_cache_hit else 0.0,
            "cache_hit": 0.0,
            "long_path": 1.0,
            "candidate_chunks": float(len(candidate_ids)),
            "candidate_pages": float(len(selected_page_numbers)),
        }
        self._put_cached_result(cache_key, selected)
        return list(selected)
