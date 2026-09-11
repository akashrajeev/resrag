from __future__ import annotations

import os
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
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(
        self,
        query: str,
        dense_k: int = 16,
        sparse_k: int = 16,
        final_k: int = 6,
    ) -> list[dict[str, Any]]:
        if not query.strip() or final_k <= 0:
            return []
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        dense_k = min(max(dense_k, 1), len(self.chunks))
        sparse_k = min(max(sparse_k, 1), len(self.chunks))
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        dense_scores = self.embeddings @ q
        dense_rank = np.argsort(-dense_scores)[:dense_k].tolist()
        sparse_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        sparse_rank = np.argsort(-sparse_scores)[:sparse_k].tolist()

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
            for idx in candidate_ids[: max(final_k * 4, 16)]
        ]

        if self.reranker and candidates:
            pairs = [(query, item["chunk"].text) for item in candidates]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            for item, score in zip(candidates, scores, strict=True):
                item["rerank_score"] = float(score)
            candidates.sort(key=lambda item: item["rerank_score"], reverse=True)

        return candidates[:final_k]
