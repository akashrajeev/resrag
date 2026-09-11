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


# Coverage intent is deliberately document-agnostic. No PDF section names are
# embedded here; section/group discovery comes entirely from the input PDF.
_BROAD_QUERY_RE = re.compile(
    r"^(?:what are|which are|what kinds|what types|list|name|summarize|overview|"
    r"give me an overview|give an overview|describe|compare|how many)\b",
    re.IGNORECASE,
)
_BROAD_QUERY_WORDS = {
    "all",
    "every",
    "main",
    "key",
    "major",
    "primary",
    "overview",
    "summary",
    "highlights",
    "various",
    "different",
    "multiple",
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


def _heading_score(
    text: str,
    spans: list[dict[str, Any]],
    median_font_size: float,
    page_width: float,
    vertical_gap: float,
    median_line_height: float,
) -> float:
    """Score a possible heading using typography/layout only.

    No vocabulary from resumes, papers, reports, or any other document type is
    used. This makes section discovery portable across arbitrary PDFs.
    """
    if not text or len(text) > 90 or len(text.split()) > 12:
        return -10.0
    if any(symbol in text for symbol in ("@", "http://", "https://", "www.")):
        return -10.0
    if len(re.findall(r"\d", text)) > max(6, len(text) // 4):
        return -5.0

    alpha = [char for char in text if char.isalpha()]
    if not alpha:
        return -10.0
    upper_ratio = sum(char.isupper() for char in alpha) / len(alpha)
    punctuation_ratio = len(re.findall(r"[.!?;|]", text)) / max(len(text), 1)

    max_size = max(
        (float(span.get("size", median_font_size)) for span in spans),
        default=median_font_size,
    )
    bold_ratio = sum(
        bool(int(span.get("flags", 0)) & 16) for span in spans
    ) / max(len(spans), 1)
    relative_size = max_size / max(median_font_size, 1e-6)

    numbered = bool(
        re.match(
            r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*|[IVXLCDM]+\.)[\)\.]?\s+",
            text.strip(),
            re.IGNORECASE,
        )
    )
    compact = len(text.split()) <= 7
    narrow = len(text) <= max(70, int(page_width * 0.42))
    whitespace_before = vertical_gap >= max(median_line_height * 1.25, 3.0)

    score = 0.0
    score += 2.4 if relative_size >= 1.18 else 0.0
    score += 1.5 if bold_ratio >= 0.5 else 0.0
    score += 1.0 if upper_ratio >= 0.78 else 0.0
    score += 1.3 if numbered else 0.0
    score += 0.6 if compact else -0.8
    score += 0.4 if narrow else -0.4
    score += 0.8 if whitespace_before else 0.0
    score -= 3.0 * punctuation_ratio
    return score


def _extract_page_blocks_without_tables(
    page: fitz.Page,
    table_rects: list[fitz.Rect],
    active_section: str = "",
) -> tuple[list[tuple[str, str]], str]:
    data = page.get_text("dict", sort=True)
    blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]

    sizes: list[float] = []
    line_heights: list[float] = []
    for block in blocks:
        for line in block.get("lines", []):
            bbox = line.get("bbox") or [0, 0, 0, 0]
            line_heights.append(max(0.0, float(bbox[3]) - float(bbox[1])))
            for span in line.get("spans", []):
                size = span.get("size")
                if isinstance(size, (int, float)) and size > 0:
                    sizes.append(float(size))
    median_size = float(np.median(np.asarray(sizes, dtype=np.float32))) if sizes else 10.0
    median_line_height = (
        float(np.median(np.asarray(line_heights, dtype=np.float32)))
        if line_heights
        else max(median_size * 1.2, 10.0)
    )

    parts: list[tuple[str, str]] = []
    current_section = active_section
    previous_y1: float | None = None

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
        vertical_gap = 0.0 if previous_y1 is None else max(0.0, rect.y0 - previous_y1)
        score = _heading_score(
            cleaned,
            spans,
            median_size,
            float(page.rect.width),
            vertical_gap,
            median_line_height,
        )

        # The first large text block often represents the document title.
        # Keep it as content unless it has clear heading-like spacing/numbering.
        is_first_block_title = not parts and previous_y1 is None and score >= 3.0 and vertical_gap == 0
        is_heading = score >= 3.2 and not is_first_block_title
        if is_heading:
            current_section = cleaned.rstrip(":").strip()
            previous_y1 = rect.y1
            continue

        parts.append((cleaned, current_section))
        previous_y1 = rect.y1

    return parts, current_section


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
    """Extract text/tables while recovering document structure generically."""
    chunks: list[Chunk] = []
    ocr_enabled = enable_ocr if enable_ocr is not None else os.getenv("OCR_ENABLED", "0") == "1"
    active_section = ""

    with fitz.open(Path(path)) as doc:
        for page_number, page in enumerate(doc, start=1):
            tables = _find_tables(page)
            table_rects = [fitz.Rect(table.bbox) for table in tables]
            text_blocks, active_section = _extract_page_blocks_without_tables(
                page,
                table_rects,
                active_section=active_section,
            )
            page_text = "\n\n".join(text for text, _ in text_blocks)
            use_ocr = len(page_text.split()) < 8 and ocr_enabled
            if use_ocr:
                page_text = _ocr_page_text(page)
                text_blocks = [(page_text, active_section)]

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
                        "\n".join(
                            " | ".join((cell or "").strip() for cell in row)
                            for row in rows
                        )
                    )
                if markdown:
                    chunks.append(
                        Chunk(
                            len(chunks),
                            page_number,
                            f"Table {table_index} on page {page_number}:\n{markdown}",
                            kind="table",
                            table_id=table_index,
                            section=active_section,
                        )
                    )

    if not chunks and ocr_enabled:
        with fitz.open(Path(path)) as doc:
            for page_number, page in enumerate(doc, start=1):
                ocr_text = _ocr_page_text(page)
                for part in split_into_chunks(ocr_text, max_words, overlap_words):
                    chunks.append(Chunk(len(chunks), page_number, part, kind="ocr"))

    return chunks


def _query_profile(query: str) -> str:
    normalized = " ".join(query.lower().split())
    tokens = set(tokenize(normalized))
    if _BROAD_QUERY_RE.search(normalized):
        return "coverage"
    if tokens & _BROAD_QUERY_WORDS:
        return "coverage"
    # Questions asking what something "has", "includes", or "contains" often
    # request a set of items rather than one fact. This is query-only logic and
    # does not depend on any particular PDF domain.
    if re.search(r"\b(?:has|have|includes|contains)\b", normalized):
        return "coverage"
    return "focused"


class HybridIndex:
    """Dense + BM25 hybrid retrieval with generic group-level coverage expansion."""

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
        self.group_to_ids: dict[str, list[int]] = {}
        self.group_centroids: np.ndarray | None = None
        self.group_names: list[str] = []
        self.last_latency: dict[str, float] = {}
        self.last_rerank_mode: str = "off"
        self.last_query_profile: str = "focused"

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No usable text or tables could be extracted from the PDF.")
        t0 = perf_counter()
        self.chunks = chunks

        group_map: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            # Section is discovered from the document's own layout. When no
            # reliable heading exists, page becomes a neutral fallback group.
            group = chunk.section.strip() or f"__page_{chunk.page}"
            group_map[group].append(index)
        self.group_to_ids = dict(group_map)
        self.section_to_ids = self.group_to_ids
        self.group_names = list(self.group_to_ids)

        self.embeddings = self.embedder.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

        centroids: list[np.ndarray] = []
        for group in self.group_names:
            ids = self.group_to_ids[group]
            centroid = self.embeddings[ids].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            centroids.append(centroid / norm if norm else centroid)
        self.group_centroids = np.asarray(centroids, dtype=np.float32)
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
        if (
            mode == "off"
            or self.last_query_profile == "coverage"
            or not self.reranker
            or len(candidate_ids) <= final_k
        ):
            return False
        best = candidate_ids[0]
        second = candidate_ids[1] if len(candidate_ids) > 1 else best
        if best == second:
            return False
        independent_agreement = bool(
            dense_rank and sparse_rank and dense_rank[0] == best and sparse_rank[0] == best
        )
        if independent_agreement:
            return False
        best_score = fused.get(best, 0.0)
        second_score = fused.get(second, 0.0)
        relative_margin = (best_score - second_score) / max(abs(best_score), 1e-9)
        return relative_margin < self.rerank_skip_margin

    def _select_coverage_candidates(
        self,
        candidate_ids: list[int],
        fused: dict[int, float],
        dense_scores: np.ndarray,
        sparse_scores: np.ndarray,
        query_vector: np.ndarray,
        profile: str,
        final_k: int,
    ) -> tuple[list[int], int]:
        if profile != "coverage" or self.group_centroids is None:
            return candidate_ids, final_k

        dense_pos = {idx: rank for rank, idx in enumerate(self._top_k(dense_scores, min(len(self.chunks), 32)), start=1)}
        sparse_pos = {idx: rank for rank, idx in enumerate(self._top_k(sparse_scores, min(len(self.chunks), 32)), start=1)}

        def local_score(idx: int) -> float:
            score = fused.get(idx, 0.0)
            score += 0.06 * float(dense_scores[idx])
            score += 0.02 / (60.0 + dense_pos.get(idx, 9999))
            score += 0.02 / (60.0 + sparse_pos.get(idx, 9999))
            return score

        group_scores = self.group_centroids @ query_vector
        ranked_group_ids = np.argsort(-group_scores).tolist()
        max_groups = max(1, min(int(os.getenv("COVERAGE_GROUP_K", "5")), len(ranked_group_ids)))
        selected_groups = [self.group_names[index] for index in ranked_group_ids[:max_groups]]

        # A broad question is about recall/coverage, not one perfect chunk.
        # Allocate the evidence budget across the best semantic groups instead
        # of letting a single high-scoring child crowd out all its siblings.
        target_k = max(final_k, int(os.getenv("COVERAGE_FINAL_K", "8")))
        slots = [target_k // len(selected_groups)] * len(selected_groups)
        for index in range(target_k % len(selected_groups)):
            slots[index] += 1

        ordered: list[int] = []
        for group, slot_count in zip(selected_groups, slots, strict=True):
            ids = sorted(self.group_to_ids[group], key=local_score, reverse=True)
            ordered.extend(ids[:slot_count])

        # Preserve any directly retrieved evidence that was not selected by
        # centroid grouping, useful for PDFs where layout grouping is noisy.
        for idx in candidate_ids:
            if idx not in ordered:
                ordered.append(idx)
            if len(ordered) >= target_k:
                break
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
        self.last_query_profile = _query_profile(query)

        def encode_query():
            return self.embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]

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
        candidate_ids = [
            idx
            for idx, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ]
        candidate_ids, effective_final_k = self._select_coverage_candidates(
            candidate_ids,
            fused,
            dense_scores,
            sparse_scores,
            q,
            self.last_query_profile,
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
        candidates = [
            {
                "chunk": self.chunks[idx],
                "hybrid_score": fused.get(idx, 0.0),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(sparse_scores[idx]),
            }
            for idx in candidate_ids[:candidate_window]
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
        return candidates[:effective_final_k]
