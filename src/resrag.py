from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .text_utils import split_into_chunks, tokenize


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    page: int
    text: str


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: str | Path, max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    """Extract text page-by-page and make citation-friendly chunks."""
    chunks: list[Chunk] = []
    with fitz.open(Path(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            for part in split_into_chunks(text, max_words, overlap_words):
                chunks.append(Chunk(len(chunks), page_number, part))
    return chunks


class HybridIndex:
    """Dense + BM25 hybrid index with optional cross-encoder reranking."""

    def __init__(
        self,
        embedding_model: str,
        reranker_model: str | None = None,
        *,
        embedder: SentenceTransformer | None = None,
        reranker: CrossEncoder | None = None,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.embedder = embedder or SentenceTransformer(embedding_model)
        self.reranker = reranker or (CrossEncoder(reranker_model) if reranker_model else None)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, dense_k: int = 12, sparse_k: int = 12, final_k: int = 6) -> list[dict[str, Any]]:
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense_scores = self.embeddings @ q
        dense_rank = np.argsort(-dense_scores)[:dense_k].tolist()
        sparse_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        sparse_rank = np.argsort(-sparse_scores)[:sparse_k].tolist()

        # RRF keeps sparse and dense score calibration independent.
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)
        for rank, idx in enumerate(sparse_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (60.0 + rank)

        candidate_ids = [idx for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)]
        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": fused[idx],
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
            }
            for idx in candidate_ids[: max(final_k * 3, 12)]
        ]

        if self.reranker and candidates:
            pairs = [(query, item["chunk"].text) for item in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return candidates[:final_k]
