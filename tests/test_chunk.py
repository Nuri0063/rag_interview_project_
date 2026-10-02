import pytest
from pydantic import ValidationError

from app.models.chunk import Chunk


def test_chunk_preserves_text_and_document_reference() -> None:
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Türkçe içerik.\nİkinci satır.",
        position=0,
        page_start=1,
        page_end=1,
    )

    assert chunk.id == "chunk-1"
    assert chunk.document_id == "doc-1"
    assert chunk.content == "Türkçe içerik.\nİkinci satır."
    assert chunk.position == 0
    assert chunk.page_start == 1
    assert chunk.page_end == 1


def test_chunk_rejects_non_integer_position() -> None:
    with pytest.raises(ValidationError) as error:
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            content="İçerik",
            position="ilk",
            page_start=1,
            page_end=1,
        )

    assert error.value.errors()[0]["loc"] == ("position",)


def test_chunk_requires_document_id() -> None:
    with pytest.raises(ValidationError) as error:
        Chunk(id="chunk-1", content="İçerik", position=0, page_start=1, page_end=1)

    assert error.value.errors()[0]["loc"] == ("document_id",)
