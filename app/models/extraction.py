from pydantic import BaseModel


class PageRange(BaseModel):
    """The zero-based character range for one PDF page in extracted content."""

    page_number: int
    start: int
    end: int


class ExtractedText(BaseModel):
    """Plain text extracted from a PDF source file."""

    content: str
    page_ranges: list[PageRange]
