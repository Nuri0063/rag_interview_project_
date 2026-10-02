import pytest

from app.models.chunk import Chunk
from app.models.document import Document
from app.models.extraction import PageRange
from app.retrieval.embeddings import EmbeddingService, build_chunk_embedding_text
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedding_service() -> EmbeddingService:
    return EmbeddingService()


@pytest.fixture()
def retriever(tmp_path, embedding_service: EmbeddingService) -> Retriever:
    document = Document(
        id="return-policy-v2",
        family_id="return-policy",
        title="Return Process and Customer Requests",
        document_type="procedure",
        version="2.0",
        published_at="2026-02-01",
        source_name="return-process.pdf",
        content="Source document text.",
        page_ranges=[PageRange(page_number=1, start=0, end=21)],
    )
    chunks = [
        Chunk(
            id="return-policy-v2:0",
            document_id=document.id,
            content="Customers can create a return request after delivery.",
            position=0,
            page_start=1,
            page_end=1,
        ),
        Chunk(
            id="return-policy-v2:1",
            document_id=document.id,
            content="Approved refunds are returned to the original payment method.",
            position=1,
            page_start=2,
            page_end=2,
        ),
        Chunk(
            id="return-policy-v2:2",
            document_id=document.id,
            content="Customers can recover account access with password reset steps.",
            position=2,
            page_start=3,
            page_end=3,
        ),
    ]
    embeddings = embedding_service.embed_documents(
        [build_chunk_embedding_text(document, chunk) for chunk in chunks]
    )
    store = VectorStore(tmp_path / "chroma")
    store.add_chunks(document, chunks, embeddings)
    return Retriever(store, embedding_service=embedding_service)


def test_query_returns_results(retriever: Retriever) -> None:
    results = retriever.retrieve("Ürünü nasıl iade edebilirim?")

    assert results


def test_top_k_is_respected(retriever: Retriever) -> None:
    results = retriever.retrieve("İade talebi", top_k=2)

    assert len(results) == 2


def test_result_metadata_is_mapped_correctly(retriever: Retriever) -> None:
    results = retriever.retrieve("İade talebi", top_k=3)

    result = next(item for item in results if item.chunk_id == "return-policy-v2:0")
    assert result.document_id == "return-policy-v2"
    assert result.family_id == "return-policy"
    assert result.version == "2.0"
    assert result.published_at == "2026-02-01"
    assert result.document_type == "procedure"
    assert result.source_name == "return-process.pdf"
    assert result.position == 0


def test_results_preserve_chroma_ranking_order(retriever: Retriever) -> None:
    results = retriever.retrieve("İade talebi", top_k=3)

    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


def test_results_include_page_provenance(retriever: Retriever) -> None:
    results = retriever.retrieve("İade talebi", top_k=3)

    assert all(result.page_start >= 1 for result in results)
    assert all(result.page_end >= result.page_start for result in results)
