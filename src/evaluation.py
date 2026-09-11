from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Fraction of relevant chunks recovered in the first k results."""
    if not relevant_ids:
        return 0.0
    found = len(set(retrieved_ids[:k]) & relevant_ids)
    return found / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: set[int], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant result."""
    limit = len(retrieved_ids) if k is None else min(k, len(retrieved_ids))
    for position, chunk_id in enumerate(retrieved_ids[:limit], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def citation_precision(cited_pages: list[int], supported_pages: set[int]) -> float:
    """Fraction of cited pages that are actually supported by retrieved evidence."""
    unique_pages = list(dict.fromkeys(cited_pages))
    if not unique_pages:
        return 0.0
    return len(set(unique_pages) & supported_pages) / len(unique_pages)


def citation_recall(cited_pages: list[int], expected_pages: set[int]) -> float:
    """Fraction of expected evidence pages referenced by the answer."""
    if not expected_pages:
        return 0.0
    return len(set(cited_pages) & expected_pages) / len(expected_pages)


def mean(values: list[float]) -> float:
    """Stable mean helper for small evaluation reports."""
    return sum(values) / len(values) if values else math.nan
