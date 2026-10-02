from datetime import date

from pydantic import BaseModel

from app.models.extraction import PageRange


class Document(BaseModel):
    """A normalized source document with full plain-text content.

    family_id identifies a version lineage. Versions are comparable only
    within the same family; this model does not compare them or select a
    current document. document_type is metadata, not a chunking strategy.
    """

    id: str
    family_id: str
    title: str
    document_type: str
    version: str | None = None
    published_at: date | None = None
    source_name: str
    content: str
    page_ranges: list[PageRange]
