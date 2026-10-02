"""Semantic candidate retrieval from the existing ChromaDB collection."""

from pydantic import BaseModel

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


DEFAULT_TOP_K = 8


class RetrievalResult(BaseModel):
    """One raw semantic candidate returned by the vector store."""

    chunk_id: str
    content: str
    document_id: str
    family_id: str
    version: str | None
    published_at: str | None
    document_type: str
    source_name: str
    page_start: int
    page_end: int
    position: int
    score: float
    rerank_score: float | None = None


class Retriever:
    """Embed user queries and return raw semantic chunk candidates."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service or EmbeddingService()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Return the most similar chunks in Chroma's native ranking order."""
        if top_k < 1:
            raise ValueError("top_k en az 1 olmalıdır.")

        query_embedding = self.embedding_service.embed_query(query)
        response = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            _to_result(chunk_id, content, metadata, distance)
            for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances)
        ]


def _to_result(
    chunk_id: str,
    content: str,
    metadata: dict[str, str | int],
    distance: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        document_id=str(metadata["document_id"]),
        family_id=str(metadata["family_id"]),
        version=_optional_text(metadata["version"]),
        published_at=_optional_text(metadata["published_at"]),
        document_type=str(metadata["document_type"]),
        source_name=str(metadata["source_name"]),
        page_start=int(metadata["page_start"]),
        page_end=int(metadata["page_end"]),
        position=int(metadata["position"]),
        score=1 / (1 + distance),
    )


def _optional_text(value: str | int) -> str | None:
    return str(value) or None
