from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Small deterministic tokenizer for BM25."""
    return re.findall(r"(?u)\b\w+\b", text.lower())


def split_into_chunks(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Split mostly on paragraph boundaries, with word-level fallback."""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current).strip())
        current = []
        current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            flush()
            for start in range(0, len(words), max_words - overlap_words):
                window = words[start : start + max_words]
                if window:
                    chunks.append(" ".join(window))
            continue
        if current_words and current_words + len(words) > max_words:
            flush()
        current.append(paragraph)
        current_words += len(words)
    flush()
    return chunks
