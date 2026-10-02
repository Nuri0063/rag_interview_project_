from typing import Literal

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "conflict"]
    answer: str
    sources: list["Source"]


class Source(BaseModel):
    source_name: str
    page_start: int
    page_end: int
    document_id: str
    chunk_id: str
