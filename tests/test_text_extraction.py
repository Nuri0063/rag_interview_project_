from pathlib import Path

import pymupdf
import pytest

from app.ingestion.text_extraction import TextExtractionError, extract_text


def test_extracts_pdf_text_and_preserves_page_order(tmp_path: Path) -> None:
    path = tmp_path / "guide.pdf"
    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text((72, 72), "First page.")
        second_page = pdf.new_page()
        second_page.insert_text((72, 72), "Second page.")
        pdf.save(path)

    extracted = extract_text(path)

    assert [page_range.page_number for page_range in extracted.page_ranges] == [1, 2]
    assert [
        extracted.content[page_range.start : page_range.end].strip()
        for page_range in extracted.page_ranges
    ] == ["First page.", "Second page."]
    assert extracted.content.index("First page.") < extracted.content.index("Second page.")


def test_rejects_unsupported_source_file_type(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("Notlar", encoding="utf-8")

    with pytest.raises(TextExtractionError, match="Desteklenmeyen"):
        extract_text(path)


def test_rejects_invalid_pdf_source_file(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")

    with pytest.raises(TextExtractionError, match="PDF dosyası okunamadı"):
        extract_text(path)


def test_rejects_source_file_without_extractable_text(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path)

    with pytest.raises(TextExtractionError, match="metin çıkarılamadı"):
        extract_text(path)
