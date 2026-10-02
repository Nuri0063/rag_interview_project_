"""Index the configured PDF corpus into the persistent vector store."""

from app.ingestion.chunking import chunk_document
from app.ingestion.document_ingestion import build_document
from app.ingestion.manifest import load_document_manifest
from app.retrieval.embeddings import EmbeddingService, build_chunk_embedding_text
from app.retrieval.vector_store import VectorStore


def index_corpus(vector_store: VectorStore, reset: bool = False) -> int:
    """Build, embed, and store every document chunk from the manifest."""
    if reset:
        vector_store.clear()

    embedding_service = EmbeddingService()
    for entry in load_document_manifest():
        document = build_document(entry)
        chunks = chunk_document(document)
        embedding_texts = [build_chunk_embedding_text(document, chunk) for chunk in chunks]
        embeddings = embedding_service.embed_documents(embedding_texts)
        vector_store.add_chunks(document, chunks, embeddings)

    return vector_store.count()
