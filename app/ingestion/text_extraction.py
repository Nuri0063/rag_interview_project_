"""Extract normalized plain text from PDF source files."""

from pathlib import Path

import pymupdf

from app.models.extraction import ExtractedText, PageRange


class TextExtractionError(ValueError):
    """Raised when a source file cannot produce usable text."""


def extract_text(path: Path) -> ExtractedText:
    """Extract text and citation ranges from a PDF source file."""
    if not path.is_file():
        raise TextExtractionError(f"Kaynak dosya okunamadı: {path}")

    if path.suffix.lower() != ".pdf":
        raise TextExtractionError(f"Desteklenmeyen kaynak dosya türü: {path.suffix or '(uzantısız)'}")

    return _extract_pdf_text(path)


def _extract_pdf_text(path: Path) -> ExtractedText:
    try:
        with pymupdf.open(path) as pdf:
            pages = [page.get_text() for page in pdf]
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        raise TextExtractionError(f"PDF dosyası okunamadı: {path}") from error

    content, page_ranges = _combine_pages(pages)
    _ensure_content(content, path)
    return ExtractedText(content=content, page_ranges=page_ranges)


def _combine_pages(pages: list[str]) -> tuple[str, list[PageRange]]:
    content_parts: list[str] = []
    page_ranges: list[PageRange] = []
    position = 0

    for page_number, page_text in enumerate(pages, start=1):
        if content_parts:
            content_parts.append("\n")
            position += 1

        start = position
        content_parts.append(page_text)
        position += len(page_text)
        page_ranges.append(PageRange(page_number=page_number, start=start, end=position))

    return "".join(content_parts), page_ranges


def _ensure_content(content: str, path: Path) -> None:
    if not content.strip():
        raise TextExtractionError(f"Kaynak dosyadan metin çıkarılamadı: {path}")
