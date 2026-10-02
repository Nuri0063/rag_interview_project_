"""Persistent ChromaDB storage for embedded document chunks."""

from datetime import date
from pathlib import Path

import chromadb

from app.models.chunk import Chunk
from app.models.document import Document


CHROMA_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION_NAME = "knowledge_chunks"


class VectorStore:
    """Store document chunks and their precomputed embeddings in ChromaDB."""

    def __init__(
        self,
        directory: Path = CHROMA_DIRECTORY,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.directory = directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add or update chunks belonging to one document."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk ve embedding sayıları eşit olmalıdır.")
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[_chunk_metadata(document, chunk) for chunk in chunks],
        )

    def clear(self) -> None:
        """Explicitly remove all records and recreate the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()


def _chunk_metadata(document: Document, chunk: Chunk) -> dict[str, str | int]:
    return {
        "document_id": document.id,
        "family_id": document.family_id,
        "version": document.version or "",
        "published_at": _date_value(document.published_at),
        "document_type": document.document_type,
        "source_name": document.source_name,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "position": chunk.position,
    }


def _date_value(value: date | None) -> str:
    return value.isoformat() if value is not None else ""
