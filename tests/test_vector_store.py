from datetime import date
from pathlib import Path

from app.ingestion.chunking import chunk_document
from app.ingestion.document_ingestion import build_document
from app.ingestion.manifest import load_document_manifest
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.extraction import PageRange
from app.retrieval.indexing import index_corpus
from app.retrieval.vector_store import VectorStore


def _document() -> Document:
    return Document(
        id="return-policy-v2",
        family_id="return-policy",
        title="Return Process and Customer Requests",
        document_type="procedure",
        version="2.0",
        published_at=date(2026, 2, 1),
        source_name="return-process.pdf",
        content="Chunk source text.",
        page_ranges=[PageRange(page_number=1, start=0, end=18)],
    )


def _chunk() -> Chunk:
    return Chunk(
        id="return-policy-v2:0",
        document_id="return-policy-v2",
        content="Chunk source text.",
        position=0,
        page_start=1,
        page_end=1,
    )


def test_adds_chunk_and_stores_its_metadata(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")
    document = _document()
    chunk = _chunk()

    store.add_chunks(document, [chunk], [[0.0, 1.0]])

    assert store.count() == 1
    stored = store.collection.get(ids=[chunk.id], include=["documents", "metadatas"])
    assert stored["documents"] == [chunk.content]
    assert stored["metadatas"] == [
        {
            "document_id": "return-policy-v2",
            "family_id": "return-policy",
            "version": "2.0",
            "published_at": "2026-02-01",
            "document_type": "procedure",
            "source_name": "return-process.pdf",
            "page_start": 1,
            "page_end": 1,
            "position": 0,
        }
    ]


def test_persistent_storage_can_be_reopened(tmp_path: Path) -> None:
    directory = tmp_path / "chroma"
    store = VectorStore(directory)
    store.add_chunks(_document(), [_chunk()], [[0.0, 1.0]])

    reopened_store = VectorStore(directory)

    assert reopened_store.count() == 1


def test_full_corpus_indexing_matches_chunker_output(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma")

    indexed_count = index_corpus(store, reset=True)
    expected_count = sum(
        len(chunk_document(build_document(entry))) for entry in load_document_manifest()
    )

    assert indexed_count == expected_count
    assert store.count() == expected_count
