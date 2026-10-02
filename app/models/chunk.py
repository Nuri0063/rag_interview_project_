from pydantic import BaseModel


class Chunk(BaseModel):
    """A piece of document text identified by its position."""

    id: str
    document_id: str
    content: str
    position: int
    page_start: int
    page_end: int
