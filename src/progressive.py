from __future__ import annotations

import hashlib
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .resrag import Chunk, extract_pdf
from .text_utils import split_into_chunks, tokenize
from .universal_retrieval import UniversalHybridIndex


@lru_cache(maxsize=2)
def _load_embedder(model_name: str):
    backend = os.getenv("EMBEDDING_BACKEND", "torch").strip().lower()
    if backend == "onnx":
        return SentenceTransformer(model_name, backend="onnx")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    return CrossEncoder(model_name) if model_name else None


def document_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def fast_extract_pdf_bytes(
    pdf_bytes: bytes,
    *,
    max_words: int = 260,
    overlap_words: int = 45,
) -> list[Chunk]:
    """Fast text-only ingestion for immediate chat readiness.

    This intentionally skips table detection, OCR, and typography analysis.
    Those expensive enrichments happen in the background.
    """
    chunks: list[Chunk] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = "\n\n".join(
                str(block[4]).strip()
                for block in page.get_text("blocks", sort=True)
                if len(block) >= 5 and str(block[4]).strip()
            )
            for part in split_into_chunks(text, max_words, overlap_words):
                chunks.append(Chunk(len(chunks), page_number, part, kind="text"))
    return chunks


class FastLexicalIndex:
    """Immediate BM25 retrieval used while the full index is building."""

    def __init__(self, chunks: list[Chunk], full_text_word_limit: int | None = None):
        self.chunks = chunks
        self.word_count = sum(len(chunk.text.split()) for chunk in chunks)
        limit = int(full_text_word_limit or os.getenv("FAST_FULL_TEXT_WORDS", "5000"))
        self.full_text_word_limit = max(500, limit)
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks]) if chunks else None
        self.last_rerank_mode = "off"
        self.last_query_profile = "fast"
        self.last_evidence_groups = 0
        self.last_latency: dict[str, float] = {}

    def retrieve(self, query: str, *, final_k: int = 12, **_: Any) -> list[dict[str, Any]]:
        cleaned = query.strip()
        if not cleaned or not self.chunks or self.bm25 is None:
            return []
        if self.word_count <= self.full_text_word_limit:
            selected = self.chunks
        else:
            scores = np.asarray(self.bm25.get_scores(tokenize(cleaned)), dtype=np.float32)
            k = min(max(1, final_k), len(self.chunks))
            ids = np.argsort(-scores)[:k].tolist()
            selected = [self.chunks[i] for i in ids]
        self.last_evidence_groups = len({chunk.page for chunk in selected})
        self.last_latency = {"fast_retrieval_ms": 0.0}
        return [{"chunk": chunk, "hybrid_score": 0.0} for chunk in selected]


class LongDocumentHybridIndex(UniversalHybridIndex):
    """UniversalHybridIndex with page-level coarse routing for long PDFs."""

    def __init__(self, *args: Any, long_document_pages: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.long_document_pages = int(long_document_pages or os.getenv("LONG_DOCUMENT_PAGES", "80"))
        self.page_to_ids: dict[int, list[int]] = {}
        self.page_names: list[int] = []
        self.page_centroids: np.ndarray | None = None
        self.page_bm25: BM25Okapi | None = None
        self.page_texts: list[str] = []
        self.is_long_document = False

    def build(self, chunks: list[Chunk]) -> None:
        super().build(chunks)
        self.page_to_ids = {}
        for idx, chunk in enumerate(self.chunks):
            self.page_to_ids.setdefault(chunk.page, []).append(idx)
        self.page_names = sorted(self.page_to_ids)
        self.is_long_document = len(self.page_names) >= self.long_document_pages

        page_vectors: list[np.ndarray] = []
        self.page_texts = []
        for page in self.page_names:
            ids = self.page_to_ids[page]
            page_vectors.append(self.embeddings[ids].mean(axis=0))  # type: ignore[index]
            self.page_texts.append("\n".join(self.retrieval_texts[i] for i in ids))

        if page_vectors:
            matrix = np.asarray(page_vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.page_centroids = matrix / norms
            self.page_bm25 = BM25Okapi([tokenize(text) for text in self.page_texts])
        else:
            self.page_centroids = None
            self.page_bm25 = None

    def _expand(
        self,
        child_candidates: list[int],
        query_embedding: np.ndarray,
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        fused: dict[int, float],
        query: str,
        broad: bool,
    ) -> list[int]:
        expanded = list(
            super()._expand(
                child_candidates,
                query_embedding,
                dense_scores,
                sparse_scores,
                fused,
                query,
                broad,
            )
        )
        if not self.is_long_document or self.page_centroids is None or self.page_bm25 is None:
            return expanded

        page_dense = self.page_centroids @ query_embedding
        page_sparse = np.asarray(self.page_bm25.get_scores(tokenize(query)), dtype=np.float32)
        page_dense_rank = self._top_k(page_dense, min(12 if broad else 8, len(self.page_names)))
        page_sparse_rank = self._top_k(page_sparse, min(12 if broad else 8, len(self.page_names)))

        page_fused: dict[int, float] = {}
        for rank, idx in enumerate(page_dense_rank, start=1):
            page_fused[idx] = page_fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(page_sparse_rank, start=1):
            page_fused[idx] = page_fused.get(idx, 0.0) + 1.0 / (60.0 + rank)

        ranked_pages = [idx for idx, _ in sorted(page_fused.items(), key=lambda item: item[1], reverse=True)]
        top_pages = [self.page_names[idx] for idx in ranked_pages[: (10 if broad else 6)]]
        selected_pages = set(top_pages)
        for page in top_pages:
            selected_pages.update(p for p in (page - 1, page + 1) if p in self.page_to_ids)

        for page in sorted(selected_pages):
            ids = sorted(
                self.page_to_ids[page],
                key=lambda i: (fused.get(i, 0.0), float(dense_scores[i]), float(sparse_scores[i])),
                reverse=True,
            )
            expanded.extend(ids[: min(len(ids), 12 if broad else 8)])
        return list(dict.fromkeys(expanded))


@dataclass
class ProgressiveJob:
    digest: str
    fast_index: FastLexicalIndex
    future: Future
    full_index: UniversalHybridIndex | None = None
    error: str | None = None


class ProgressiveIndexManager:
    """Per-process document jobs with immediate lexical retrieval and lazy upgrade."""

    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="resrag-index")
        self.jobs: dict[str, ProgressiveJob] = {}

    def start(self, pdf_bytes: bytes, embedding_model: str, reranker_model: str | None) -> ProgressiveJob:
        digest = document_id(pdf_bytes)
        existing = self.jobs.get(digest)
        if existing is not None:
            self.refresh(digest)
            return existing

        chunks = fast_extract_pdf_bytes(pdf_bytes)
        fast_index = FastLexicalIndex(chunks)
        future = self.executor.submit(_build_full_index, pdf_bytes, embedding_model, reranker_model)
        job = ProgressiveJob(digest, fast_index, future)
        self.jobs[digest] = job
        return job

    def refresh(self, digest: str) -> ProgressiveJob | None:
        job = self.jobs.get(digest)
        if job is None or not job.future.done():
            return job
        if job.full_index is not None or job.error is not None:
            return job
        try:
            job.full_index = job.future.result()
        except Exception as exc:  # pragma: no cover - runtime-only failure path
            job.error = str(exc)
        return job

    def get(self, digest: str) -> ProgressiveJob | None:
        return self.refresh(digest)


def _build_full_index(
    pdf_bytes: bytes,
    embedding_model: str,
    reranker_model: str | None,
) -> UniversalHybridIndex:
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        Path(temp_path).write_bytes(pdf_bytes)
        chunks = extract_pdf(temp_path)
        # Dynamic import avoids a module cycle while keeping the page-first
        # subclass isolated from the generic ingestion manager.
        from .long_retrieval import PageFirstLongDocumentIndex

        index = PageFirstLongDocumentIndex(
            embedding_model,
            reranker_model or None,
            embedder=_load_embedder(embedding_model),
            reranker=_load_reranker(reranker_model) if reranker_model else None,
        )
        index.build(chunks)
        return index
    finally:
        Path(temp_path).unlink(missing_ok=True)
