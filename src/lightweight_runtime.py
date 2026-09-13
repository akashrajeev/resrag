from __future__ import annotations

import hashlib
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymupdf as fitz
from rank_bm25 import BM25Okapi

from .text_utils import split_into_chunks, tokenize


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    page: int
    text: str
    kind: str = "text"
    table_id: int | None = None
    section: str = ""


def document_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def fast_extract_pdf_bytes(
    pdf_bytes: bytes,
    *,
    max_words: int = 260,
    overlap_words: int = 45,
) -> list[Chunk]:
    """Parse only the text needed to make a document immediately queryable."""
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
    """Lightweight BM25-only index with no sentence-transformers/PyTorch import."""

    def __init__(self, chunks: list[Chunk], full_text_word_limit: int | None = None):
        self.chunks = chunks
        self.word_count = sum(len(chunk.text.split()) for chunk in chunks)
        limit = int(full_text_word_limit or os.getenv("FAST_FULL_TEXT_WORDS", "5000"))
        self.full_text_word_limit = max(1, limit)
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


@dataclass
class LightweightJob:
    digest: str
    fast_index: FastLexicalIndex
    future: Future | None
    full_index: Any | None = None
    error: str | None = None


class LightweightProgressiveIndexManager:
    """Immediate BM25 indexing with an optional heavyweight background upgrade."""

    def __init__(self, max_workers: int = 1):
        self.jobs: dict[str, LightweightJob] = {}
        self._background_enabled = os.getenv("RESRAG_BACKGROUND_FULL_INDEX", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.executor = (
            ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="resrag-index")
            if self._background_enabled
            else None
        )

    def start(self, pdf_bytes: bytes, embedding_model: str, reranker_model: str | None) -> LightweightJob:
        digest = document_id(pdf_bytes)
        existing = self.jobs.get(digest)
        if existing is not None:
            self.refresh(digest)
            return existing

        chunks = fast_extract_pdf_bytes(pdf_bytes)
        fast_index = FastLexicalIndex(chunks)
        future: Future | None = None
        if self._background_enabled and self.executor is not None:
            future = self.executor.submit(_build_full_index_lazy, pdf_bytes, embedding_model, reranker_model)
        job = LightweightJob(digest, fast_index, future)
        self.jobs[digest] = job
        return job

    def refresh(self, digest: str) -> LightweightJob | None:
        job = self.jobs.get(digest)
        if job is None or job.future is None or not job.future.done():
            return job
        if job.full_index is not None or job.error is not None:
            return job
        try:
            job.full_index = job.future.result()
        except Exception as exc:  # pragma: no cover - runtime-only failure path
            job.error = str(exc)
        return job

    def get(self, digest: str) -> LightweightJob | None:
        return self.refresh(digest)


def _build_full_index_lazy(
    pdf_bytes: bytes,
    embedding_model: str,
    reranker_model: str | None,
):
    """Load the heavyweight implementation only when explicitly enabled."""
    from .progressive import _build_full_index
    return _build_full_index(pdf_bytes, embedding_model, reranker_model)
