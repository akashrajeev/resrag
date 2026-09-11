from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pymupdf as fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .text_utils import split_into_chunks, tokenize


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    page: int
    text: str
    kind: str = "text"
    table_id: int | None = None


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _rect_contains(outer: fitz.Rect, inner: fitz.Rect, tolerance: float = 2.0) -> bool:
    return (
        outer.x0 - tolerance <= inner.x0
        and outer.y0 - tolerance <= inner.y0
        and outer.x1 + tolerance >= inner.x1
        and outer.y1 + tolerance >= inner.y1
    )


def _extract_page_text_without_tables(page: fitz.Page, table_rects: list[fitz.Rect]) -> str:
    blocks = page.get_text("blocks", sort=True)
    parts: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        rect = fitz.Rect(block[:4])
        text = block[4]
        if any(_rect_contains(table_rect, rect) for table_rect in table_rects):
            continue
        cleaned = _clean_text(text)
        if cleaned:
            parts.append(cleaned)
    return "\n\n".join(parts)


def _find_tables(page: fitz.Page):
    """Try strict line-based detection first, then text-based detection."""
    for strategy in ("lines_strict", "text"):
        try:
            found = list(page.find_tables(strategy=strategy).tables)
        except Exception:
            found = []
        if found:
            return found
    return []


def _ocr_page_text(page: fitz.Page) -> str:
    """Best-effort OCR fallback using the Tesseract-backed PyMuPDF API."""
    try:
        text_page = page.get_textpage_ocr(language="eng", dpi=180, full=True)
        return _clean_text(page.get_text("text", textpage=text_page))
    except Exception:
        return ""


def extract_pdf(
    path: str | Path,
    max_words: int = 220,
    overlap_words: int = 40,
    enable_ocr: bool | None = None,
) -> list[Chunk]:
    """Extract page-aware text and native PDF tables into citation-friendly chunks."""
    chunks: list[Chunk] = []
    ocr_enabled = enable_ocr if enable_ocr is not None else os.getenv("OCR_ENABLED", "0") == "1"

    with fitz.open(Path(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            tables = _find_tables(page)
            table_rects = [fitz.Rect(table.bbox) for table in tables]
            page_text = _extract_page_text_without_tables(page, table_rects)
            use_ocr = len(page_text.split()) < 8 and ocr_enabled
            if use_ocr:
                page_text = _ocr_page_text(page)

            chunk_kind = "ocr" if use_ocr and page_text else "text"
            for part in split_into_chunks(page_text, max_words, overlap_words):
                chunks.append(Chunk(len(chunks), page_number, part, kind=chunk_kind))

            for table_index, table in enumerate(tables, start=1):
                try:
                    markdown = _clean_text(table.to_markdown())
                except Exception:
                    rows = table.extract()
                    markdown = _clean_text(
                        "\n".join(" | ".join((cell or "").strip() for cell in row) for row in rows)
                    )
                if markdown:
                    chunks.append(
                        Chunk(
                            len(chunks),
                            page_number,
                            f"Table {table_index} on page {page_number}:\n{markdown}",
                            kind="table",
                            table_id=table_index,
                        )
                    )

    if not chunks and ocr_enabled:
        with fitz.open(Path(path)) as doc:
            for page_number, page in enumerate(doc, start=1):
                ocr_text = _ocr_page_text(page)
                for part in split_into_chunks(ocr_text, max_words, overlap_words):
                    chunks.append(Chunk(len(chunks), page_number, part, kind="ocr"))

    return chunks


class HybridIndex:
    """Dense + BM25 hybrid index with adaptive cross-encoder reranking."""

    def __init__(
        self,
        embedding_model: str,
        reranker_model: str | None = None,
        *,
        embedder: SentenceTransformer | None = None,
        reranker: CrossEncoder | None = None,
        rerank_skip_margin: float | None = None,
        rerank_candidate_k: int | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedder = embedder or SentenceTransformer(embedding_model)
        self.reranker = reranker or (CrossEncoder(reranker_model) if reranker_model else None)
        self.rerank_skip_margin = (
            float(rerank_skip_margin)
            if rerank_skip_margin is not None
            else float(os.getenv("RERANK_SKIP_MARGIN", "0.12"))
        )
        self.rerank_candidate_k = (
            int(rerank_candidate_k)
            if rerank_candidate_k is not None
            else int(os.getenv("RERANK_CANDIDATE_K", "8"))
        )
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.last_latency: dict[str, float] = {}
        self.last_rerank_mode: str = "off"

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        t0 = perf_counter()
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.last_latency = {"index_build_ms": (perf_counter() - t0) * 1000.0}

    @staticmethod
    def _top_k(scores: np.ndarray, k: int) -> list[int]:
        if k >= len(scores):
            return np.argsort(-scores).tolist()
        partition = np.argpartition(scores, -k)[-k:]
        return partition[np.argsort(-scores[partition])].tolist()

    def _should_rerank(
        self,
        candidate_ids: list[int],
        dense_rank: list[int],
        sparse_rank: list[int],
        fused: dict[int, float],
        final_k: int,
        mode: str,
    ) -> bool:
        if mode == "on":
            return bool(self.reranker and len(candidate_ids) > final_k)
        if mode == "off" or not self.reranker or len(candidate_ids) <= final_k:
            return False
        best = candidate_ids[0]
        second = candidate_ids[1] if len(candidate_ids) > 1 else best
        if best == second:
            return False
        # Hybrid retrieval already gives us strong evidence. When both
        # independent retrievers agree on the same top result, avoid the
        # expensive cross-encoder. Rerank only disagreement/ambiguous cases.
        independent_agreement = bool(dense_rank and sparse_rank and dense_rank[0] == best and sparse_rank[0] == best)
        if independent_agreement:
            return False
        relative_margin = (fused[best] - fused[second]) / max(abs(fused[best]), 1e-9)
        return relative_margin < self.rerank_skip_margin

    def retrieve(
        self,
        query: str,
        dense_k: int = 12,
        sparse_k: int = 12,
        final_k: int = 4,
        *,
        rerank_mode: str = "auto",
    ) -> list[dict[str, Any]]:
        if not query.strip() or final_k <= 0:
            self.last_latency = {}
            self.last_rerank_mode = "off"
            return []
        if rerank_mode not in {"auto", "on", "off"}:
            raise ValueError("rerank_mode must be one of: auto, on, off")
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        total_start = perf_counter()
        dense_k = min(max(dense_k, 1), len(self.chunks))
        sparse_k = min(max(sparse_k, 1), len(self.chunks))
        final_k = min(max(final_k, 1), len(self.chunks))

        def encode_query():
            return self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        def score_sparse():
            return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        parallel_start = perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(encode_query)
            sparse_future = pool.submit(score_sparse)
            q = dense_future.result()
            sparse_scores = sparse_future.result()
        parallel_ms = (perf_counter() - parallel_start) * 1000.0

        search_start = perf_counter()
        dense_scores = self.embeddings @ q
        dense_rank = self._top_k(dense_scores, dense_k)
        sparse_rank = self._top_k(sparse_scores, sparse_k)
        search_ms = (perf_counter() - search_start) * 1000.0

        fusion_start = perf_counter()
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(sparse_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        candidate_ids = [idx for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        fusion_ms = (perf_counter() - fusion_start) * 1000.0

        rerank_start = perf_counter()
        should_rerank = self._should_rerank(
            candidate_ids,
            dense_rank,
            sparse_rank,
            fused,
            final_k,
            rerank_mode,
        )
        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": fused[idx],
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
            }
            for idx in candidate_ids[: max(final_k * 2, self.rerank_candidate_k)]
        ]

        if should_rerank and candidates:
            rerank_candidates = candidates[: min(self.rerank_candidate_k, len(candidates))]
            pairs = [(query, item["chunk"].text) for item in rerank_candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(rerank_candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            rerank_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
            candidates = rerank_candidates + candidates[len(rerank_candidates):]
            self.last_rerank_mode = "on"
        else:
            self.last_rerank_mode = "off"
        rerank_ms = (perf_counter() - rerank_start) * 1000.0

        self.last_latency = {
            "query_parallel_ms": parallel_ms,
            "search_ms": search_ms,
            "fusion_ms": fusion_ms,
            "rerank_ms": rerank_ms,
            "total_retrieval_ms": (perf_counter() - total_start) * 1000.0,
        }
        return candidates[:final_k]
