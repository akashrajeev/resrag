from pathlib import Path

import pymupdf as fitz

from src.resrag import extract_pdf


def test_extract_pdf_preserves_page_numbers(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a simple document about retrieval systems.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Second page contains another useful fact.")
    doc.save(pdf_path)
    doc.close()

    chunks = extract_pdf(pdf_path)
    assert chunks
    assert chunks[0].page == 1
    assert any(chunk.page == 2 for chunk in chunks)
    assert all(chunk.kind in {"text", "table", "ocr"} for chunk in chunks)
