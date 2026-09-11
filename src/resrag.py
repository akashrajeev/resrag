from __future__ import annotations

import os
import re
from collections import defaultdict
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


COMMON_SECTION_NAMES = {
    "abstract", "achievements", "awards", "certifications", "conclusion",
    "contact", "education", "experience", "interests", "languages",
    "objective", "projects", "profile", "publications", "references",
    "skills", "summary", "technical skills", "work experience",
}

COVERAGE_CUES = {
    "all", "any", "main", "key", "major", "primary", "projects", "skills",
    "experience", "education", "achievements", "awards", "certifications",
    "sections", "overview", "summary", "highlights", "listed", "mention",
    "mentions", "include", "includes", "contain", "contains",
}


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    page: int
    text: str
    kind: str = "text"
    table_id: int | None = None
    section: str = ""


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


def _looks_like_heading(text: str, max_font_size: float, median_font_size: float, bold: bool) -> bool:
    normalized = text.strip().rstrip(":").lower()
    if normalized in COMMON_SECTION_NAMES:
        return True
    if not text or len(text) > 60 or len(text.split()) > 7:
        return False
    if any(mark in text for mark in ".!?;|"):
        return False
    alpha = [char for char in text if char.isalpha()]
    upper_ratio = (sum(char.isupper() for char in alpha) / len(alpha)) if alpha else 0.0
    strong_typography = max_font_size >= max(median_font_size * 1.2, median_font_size + 1.0)
    return (strong_typography or (bold and upper_ratio >= 0.75)) and len(text.split()) <= 5


def _extract_page_blocks_without_tables(
    page: fitz.Page,
    table_rects: list[fitz.Rect],
) -> list[tuple[str, str]]:
    data = page.get_text("dict", sort=True)
    blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]
    sizes: list[float] = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")
                if isinstance(size, (int, float)) and size > 0:
                    sizes.append(float(size))
    median_size = float(np.median(np.asarray(sizes, dtype=np.float32))) if sizes else 10.0

    parts: list[tuple[str, str]] = []
    current_section = ""
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        rect = fitz.Rect(bbox)
        if any(_rect_contains(table_rect, rect) for table_rect in table_rects):
            continue
        lines = block.get("lines", [])
        text = "\n".join(
            "".join(str(span.get("text", "")) for span in line.get("spans", []))
            for line in lines
        )
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        spans = [span for line in lines for span in line.get("spans", [])]
        max_size = max((float(span.get("size", median_size)) for span in spans), default=median_size)
        bold = any(bool(int(span.get("flags", 0)) & 16) for span in spans)
        if _looks_like_heading(cleaned, max_size, median_size, bold):
            current_section = cleaned.rstrip(":").strip()
            continue
        parts.append((cleaned, current_section))
    return parts


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
    """Extract page-aware, section-aware text and native PDF tables."""
    chunks: list[Chunk] = []
    ocr_enabled = enable_ocr if enable_ocr is not None else os.getenv("OCR_ENABLED", "0") == "1"

    with fitz.open(Path(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            tables = _find_tables(page)
            table_rects = [fitz.Rect(table.bbox) for table in tables]
            text_blocks = _extract_page_blocks_without_tables(page, table_rects)
            page_text = "\n\n".join(text for text, _ in text_blocks)
            use_ocr = len(page_text.split()) < 8 and ocr_enabled
            if use_ocr:
                page_text = _ocr_page_text(page)
                text_blocks = [(page_text, "")]

            if text_blocks:
                grouped: dict[str, list[str]] = defaultdict(list)
                order: list[str] = []
                for text, section in text_blocks:
                    key = section or ""
                    if key not in grouped:
                        order.append(key)
                    grouped[key].append(text)
                for section in order:
                    section_text = "\n\n".join(grouped[section])
                    kind = "ocr" if use_ocr and section_text else "text"
                    for part in split_into_chunks(section_text, max_words, overlap_words):
                        chunks.append(
                            Chunk(
                                len(chunks),
                                page_number,
                                part,
                                kind=kind,
                                section=section,
                            )
                        )

            for table_index, table in enumerate(tables, start=1):
                try:
                    markdown = _clean_text(table.to_markdown())
                except Exception:
                    rows = table.extract()
                    markdown = _clean_text(
                        "\n".join(" | ".join((cell or "").strip() for cell in row) for row in rows)
                    )
                if markdown:
                    table_section = text_blocks[-1][1] if text_blocks else ""
                    chunks.append(
                        Chunk(
                            len(chunks),
                            page_number,
                            f"Table {table_index} on page {page_number}:\n{markdown}",
                            kind="table",
                            table_id=table_index,
                            section=table_section,
                        )
                    )

    if not chunks and ocr_enabled:
        with fitz.open(Path(path)) as doc:
            for page_number, page in enumerate(doc, start=1):
                ocr_text = _ocr_page_text(page)
                for part in split_into_chunks(ocr_text, max_words, overlap_words):
                    chunks.append(Chunk(len(chunks), page_number, part, kind="ocr"))

    return chunks


def _section_key(section: str) -> str:
    return re.sub(r"\s+", " ", section.strip().lower())


def _coverage_profile(query: str, sections: dict[str, list[int]]) -> tuple[str, list[str]]:
    tokens = set(tokenize(query))
    broad = len(tokens & COVERAGE_CUES) > 0 or len(tokens) <= 6
    target_sections: list[str] = []
    for section in sections:
        section_tokens = set(tokenize(section))
        if section_tokens and section_tokens & tokens:
            target_sections.append(section)
    return ("targeted" if target_sections else "overview") if broad else "focused", target_sections


class HybridIndex:
    """Dense + BM25 retrieval with structure-aware coverage retrieval and adaptive reranking."""

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
        self.section_to_ids: dict[str, list[int]] = {}
        self.last_latency: dict[str, float] = {}
        self.last_rerank_mode: str = "off"
        self.last_query_profile: str = "focused"

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        t0 = perf_counter()
        self.chunks = chunks
        section_map: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            if chunk.section:
                section_map[chunk.section].append(index)
        self.section_to_ids = dict(section_map)
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
        independent_agreement = bool(dense_rank and sparse_rank and dense_rank[0] == best and sparse_rank[0] == best)
        if independent_agreement:
            return False
        relative_margin = (fused[best] - fused[second]) / max(abs(fused[best]), 1e-9)
        return relative_margin < self.rerank_skip_margin

    def _select_coverage_candidates(
        self,
        candidate_ids: list[int],
        fused: dict[int, float],
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        dense_rank: list[int],
        sparse_rank: list[int],
        profile: str,
        target_sections: list[str],
        final_k: int,
    ) -> tuple[list[int], int]:
        if profile == "focused":
            return candidate_ids, final_k

        dense_pos = {idx: rank for rank, idx in enumerate(dense_rank, start=1)}
        sparse_pos = {idx: rank for rank, idx in enumerate(sparse_rank, start=1)}

        def local_score(idx: int) -> float:
            score = fused.get(idx, 0.0)
            score += 0.03 / (60.0 + dense_pos.get(idx, 9999))
            score += 0.03 / (60.0 + sparse_pos.get(idx, 9999))
            return score + 0.001 * float(dense_scores[idx])

        expanded: list[int] = list(candidate_ids)
        if target_sections:
            for section in target_sections:
                expanded.extend(self.section_to_ids.get(section, []))
            target_k = max(final_k, int(os.getenv("COVERAGE_FINAL_K", "8")))
        else:
            # Overview queries should touch multiple sections instead of letting
            # one semantically dense section dominate the entire answer.
            section_scores: list[tuple[float, str]] = []
            for section, ids in self.section_to_ids.items():
                best = max((local_score(idx) for idx in ids), default=0.0)
                section_scores.append((best, section))
            section_scores.sort(reverse=True)
            selected_sections = [section for _, section in section_scores[: int(os.getenv("OVERVIEW_SECTION_K", "5"))]]
            for section in selected_sections:
                ranked = sorted(self.section_to_ids.get(section, []), key=local_score, reverse=True)
                expanded.extend(ranked[:2])
            target_k = max(final_k, int(os.getenv("COVERAGE_FINAL_K", "8")))

        deduped = list(dict.fromkeys(expanded))
        if target_sections:
            ordered: list[int] = []
            for section in target_sections:
                section_candidates = sorted(
                    [idx for idx in deduped if self.chunks[idx].section == section],
                    key=local_score,
                    reverse=True,
                )
                ordered.extend(section_candidates[:target_k])
            ordered.extend([idx for idx in deduped if idx not in ordered])
            return ordered, target_k

        ordered = []
        used_sections: set[str] = set()
        for idx in sorted(deduped, key=local_score, reverse=True):
            section = self.chunks[idx].section or f"__page_{self.chunks[idx].page}"
            if section not in used_sections:
                ordered.append(idx)
                used_sections.add(section)
        for idx in sorted(deduped, key=local_score, reverse=True):
            if idx not in ordered:
                ordered.append(idx)
        return ordered, target_k

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
            self.last_query_profile = "focused"
            return []
        if rerank_mode not in {"auto", "on", "off"}:
            raise ValueError("rerank_mode must be one of: auto, on, off")
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")

        total_start = perf_counter()
        dense_k = min(max(dense_k, 1), len(self.chunks))
        sparse_k = min(max(sparse_k, 1), len(self.chunks))
        final_k = min(max(final_k, 1), len(self.chunks))
        profile, target_sections = _coverage_profile(query, self.section_to_ids)
        self.last_query_profile = profile

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
        candidate_ids, effective_final_k = self._select_coverage_candidates(
            candidate_ids,
            fused,
            dense_scores,
            sparse_scores,
            dense_rank,
            sparse_rank,
            profile,
            target_sections,
            final_k,
        )
        fusion_ms = (perf_counter() - fusion_start) * 1000.0

        rerank_start = perf_counter()
        should_rerank = self._should_rerank(
            candidate_ids,
            dense_rank,
            sparse_rank,
            fused,
            effective_final_k,
            rerank_mode,
        )
        candidate_window = max(effective_final_k * 2, self.rerank_candidate_k)
        candidates = []
        for idx in candidate_ids[:candidate_window]:
            candidates.append(
                {
                    "chunk": self.chunks[idx],
                    "hybrid_score": fused.get(idx, 0.0),
                    "dense_score": float(dense_scores[idx]),
                    "bm25_score": float(sparse_scores[idx]),
                }
            )

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
        return candidates[:effective_final_k]
