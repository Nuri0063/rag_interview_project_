"""Multilingual embedding support for document chunks and user queries."""

from sentence_transformers import SentenceTransformer

from app.models.chunk import Chunk
from app.models.document import Document


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    """Embed text with the project's multilingual sentence-transformer model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return normalized embeddings for document texts."""
        if not texts:
            return []

        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Return a normalized embedding for a user question without translation."""
        return self.embed_documents([query])[0]


def build_chunk_embedding_text(document: Document, chunk: Chunk) -> str:
    """Build the future semantic-search input for a document chunk."""
    return f"{document.title}\n\n{chunk.content}"
