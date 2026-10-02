"""Create small, page-provenanced chunks from normalized documents."""

from dataclasses import dataclass
import re

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.extraction import PageRange


DEFAULT_MAX_WORDS = 60
_WORD_PATTERN = re.compile(r"\S+")
_PARAGRAPH_PATTERN = re.compile(r"\S[\s\S]*?(?=\n\s*\n|\Z)")


class ChunkingError(ValueError):
    """Raised when a document cannot be divided into source-provenanced chunks."""


@dataclass(frozen=True)
class _TextRange:
    start: int
    end: int
    word_count: int


def chunk_document(document: Document, max_words: int = DEFAULT_MAX_WORDS) -> list[Chunk]:
    """Split a document at paragraph and word boundaries without overlap.

    ``max_words`` is a conservative approximation for the future 128-token
    embedding limit. It intentionally does not use a model tokenizer.
    """
    if max_words < 1:
        raise ValueError("max_words en az 1 olmalıdır.")

    paragraphs = _paragraph_ranges(document.content)
    if not paragraphs:
        raise ChunkingError(f"Belge parçalanacak metin içermiyor: {document.id}")
    if not document.page_ranges:
        raise ChunkingError(f"Belge sayfa aralığı içermiyor: {document.id}")

    text_ranges = _combine_paragraphs(paragraphs, document.content, max_words)
    return [
        _create_chunk(document, text_range, position)
        for position, text_range in enumerate(text_ranges)
    ]


def _paragraph_ranges(content: str) -> list[_TextRange]:
    ranges: list[_TextRange] = []
    for match in _PARAGRAPH_PATTERN.finditer(content):
        start, end = _trim_range(content, match.start(), match.end())
        words = list(_WORD_PATTERN.finditer(content, start, end))
        if words:
            ranges.append(_TextRange(start=start, end=end, word_count=len(words)))
    return ranges


def _combine_paragraphs(
    paragraphs: list[_TextRange], content: str, max_words: int
) -> list[_TextRange]:
    chunks: list[_TextRange] = []
    current: _TextRange | None = None

    for paragraph in paragraphs:
        for text_range in _split_if_needed(paragraph, content, max_words):
            if current is None:
                current = text_range
            elif current.word_count + text_range.word_count <= max_words:
                current = _TextRange(
                    start=current.start,
                    end=text_range.end,
                    word_count=current.word_count + text_range.word_count,
                )
            else:
                chunks.append(current)
                current = text_range

    if current is not None:
        chunks.append(current)
    return chunks


def _split_if_needed(
    paragraph: _TextRange, content: str, max_words: int
) -> list[_TextRange]:
    if paragraph.word_count <= max_words:
        return [paragraph]

    words = list(_WORD_PATTERN.finditer(content, paragraph.start, paragraph.end))
    return [
        _TextRange(
            start=word_group[0].start(),
            end=word_group[-1].end(),
            word_count=len(word_group),
        )
        for word_group in (words[index : index + max_words] for index in range(0, len(words), max_words))
    ]


def _create_chunk(document: Document, text_range: _TextRange, position: int) -> Chunk:
    page_ranges = _intersecting_pages(text_range, document.page_ranges)
    if not page_ranges:
        raise ChunkingError(f"Parça için sayfa aralığı bulunamadı: {document.id}")

    return Chunk(
        id=f"{document.id}:{position}",
        document_id=document.id,
        content=document.content[text_range.start : text_range.end],
        position=position,
        page_start=page_ranges[0].page_number,
        page_end=page_ranges[-1].page_number,
    )


def _intersecting_pages(text_range: _TextRange, page_ranges: list[PageRange]) -> list[PageRange]:
    return [
        page_range
        for page_range in page_ranges
        if text_range.start < page_range.end and text_range.end > page_range.start
    ]


def _trim_range(content: str, start: int, end: int) -> tuple[int, int]:
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end
