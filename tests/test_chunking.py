from src.text_utils import split_into_chunks, tokenize


def test_paragraph_chunking_respects_limit():
    text = "\n\n".join(["one two three", "four five six", "seven eight nine"])
    chunks = split_into_chunks(text, max_words=6, overlap_words=1)
    assert chunks == ["one two three\n\nfour five six", "seven eight nine"]


def test_tokenize_is_lowercase_and_deterministic():
    assert tokenize("Hello, WORLD 42!") == ["hello", "world", "42"]
