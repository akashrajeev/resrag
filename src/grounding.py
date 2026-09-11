from __future__ import annotations

import re
from dataclasses import dataclass

PAGE_CITATION_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited_pages: tuple[int, ...]
    valid_pages: tuple[int, ...]
    invalid_pages: tuple[int, ...]
    missing_citations: bool


def validate_citations(answer: str, source_pages: set[int]) -> CitationCheck:
    """Check that page citations in an answer point to retrieved evidence."""
    cited = tuple(dict.fromkeys(int(value) for value in PAGE_CITATION_RE.findall(answer)))
    valid = tuple(page for page in cited if page in source_pages)
    invalid = tuple(page for page in cited if page not in source_pages)
    return CitationCheck(
        cited_pages=cited,
        valid_pages=valid,
        invalid_pages=invalid,
        missing_citations=not bool(cited),
    )


def build_context(retrieved: list[dict], max_chars_per_source: int = 1000) -> str:
    """Build compact, structured evidence with section metadata for low-latency grounded generation."""
    sections: list[str] = []
    for index, item in enumerate(retrieved, start=1):
        chunk = item["chunk"]
        text = chunk.text.strip()
        if len(text) > max_chars_per_source:
            text = text[:max_chars_per_source].rstrip() + "…"
        section = getattr(chunk, "section", "") or "Unsectioned"
        sections.append(
            f"SOURCE {index}\n"
            f"Page: {chunk.page}\n"
            f"Section: {section}\n"
            f"Type: {chunk.kind}\n"
            f"Content:\n{text}"
        )
    return "\n\n---\n\n".join(sections)


def build_followup_query(question: str, history: list[dict], max_history_chars: int = 1200) -> str:
    """Resolve short follow-up questions using a small amount of recent context."""
    cleaned = question.strip()
    if not history:
        return cleaned

    previous_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"),
        "",
    )
    previous_user = previous_user.strip()
    if not previous_user:
        return cleaned

    followup = cleaned.lower()
    cues = (
        followup.startswith(("what about", "how about", "and ", "why ", "how ", "does ", "did ", "can ", "could ", "what does that", "what is that"))
        or len(cleaned.split()) <= 7
    )
    if not cues:
        return cleaned

    context = previous_user[:max_history_chars]
    return f"Previous question: {context}\nFollow-up question: {cleaned}"
