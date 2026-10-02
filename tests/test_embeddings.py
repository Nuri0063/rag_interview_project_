import math

import pytest

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.extraction import PageRange
from app.retrieval.embeddings import EmbeddingService, build_chunk_embedding_text


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


def _document() -> Document:
    return Document(
        id="return-policy-v2",
        family_id="return-policy-private",
        title="Return Process and Customer Requests",
        document_type="guide-private",
        version="9.9-private",
        published_at="2026-02-01",
        source_name="private-source.pdf",
        content="Source text.",
        page_ranges=[PageRange(page_number=1, start=0, end=12)],
    )


def _chunk() -> Chunk:
    return Chunk(
        id="return-policy-v2:0",
        document_id="return-policy-v2",
        content="Customers can create a return request after delivery.",
        position=0,
        page_start=1,
        page_end=1,
    )


def test_document_embedding_returns_a_normalized_vector(
    embedding_service: EmbeddingService,
) -> None:
    vector = embedding_service.embed_documents(["Return requests after delivery."])[0]

    assert vector
    assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0, rel_tol=1e-5)


def test_query_embedding_returns_a_normalized_vector(
    embedding_service: EmbeddingService,
) -> None:
    vector = embedding_service.embed_query("Ürünü kaç gün içinde iade edebilirim?")

    assert vector
    assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0, rel_tol=1e-5)


def test_multiple_texts_return_one_vector_per_text(embedding_service: EmbeddingService) -> None:
    vectors = embedding_service.embed_documents(["First text.", "Second text.", "Third text."])

    assert len(vectors) == 3
    assert len({len(vector) for vector in vectors}) == 1


def test_embedding_text_contains_only_title_and_chunk_content() -> None:
    document = _document()
    chunk = _chunk()

    text = build_chunk_embedding_text(document, chunk)

    assert text == "Return Process and Customer Requests\n\nCustomers can create a return request after delivery."
    assert "return-policy-private" not in text
    assert "9.9-private" not in text
    assert "2026-02-01" not in text
    assert "guide-private" not in text
    assert "private-source.pdf" not in text
    assert "page_start" not in text
