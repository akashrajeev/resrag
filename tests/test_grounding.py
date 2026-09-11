from src.grounding import build_context, build_followup_query, validate_citations
from src.resrag import Chunk


def test_citation_validation_accepts_only_retrieved_pages():
    result = validate_citations("The value is 42 [Page 3]. See also [Page 99].", {3, 7})
    assert result.cited_pages == (3, 99)
    assert result.valid_pages == (3,)
    assert result.invalid_pages == (99,)
    assert not result.missing_citations


def test_build_context_labels_sources():
    context = build_context([{"chunk": Chunk(0, 4, "A fact", "text")}])
    assert "SOURCE 1" in context
    assert "Page: 4" in context
    assert "A fact" in context


def test_short_followup_uses_previous_question():
    history = [{"role": "user", "content": "What revenue did the company report in 2025?"}]
    query = build_followup_query("What about 2024?", history)
    assert "Previous question:" in query
    assert "2025" in query
    assert "2024" in query
