from src.text_utils import split_into_chunks, tokenize


def test_tokenize_is_case_insensitive():
    assert tokenize("Revenue, Revenue! 2026") == ["revenue", "revenue", "2026"]


def test_chunking_keeps_overlap():
    text = " ".join(f"word{i}" for i in range(30))
    chunks = split_into_chunks(text, max_words=10, overlap_words=3)
    assert len(chunks) >= 4
    assert "word7" in chunks[0]
    assert "word7" in chunks[1]
