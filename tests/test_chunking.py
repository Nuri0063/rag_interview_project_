from app.ingestion.chunking import DEFAULT_MAX_WORDS, chunk_document
from app.ingestion.document_ingestion import build_document
from app.ingestion.manifest import load_document_manifest
from app.models.document import Document
from app.models.extraction import PageRange


def _document(content: str, page_ranges: list[PageRange]) -> Document:
    return Document(
        id="doc-1",
        family_id="family-1",
        title="Test Document",
        document_type="guide",
        source_name="test.pdf",
        content=content,
        page_ranges=page_ranges,
    )


def test_short_text_produces_a_valid_chunk() -> None:
    content = "Kısa bir metin."
    document = _document(content, [PageRange(page_number=1, start=0, end=len(content))])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Kısa bir metin."
    assert chunks[0].position == 0
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_multiple_short_paragraphs_are_combined() -> None:
    content = "İlk kısa paragraf.\n\nİkinci kısa paragraf."
    document = _document(content, [PageRange(page_number=1, start=0, end=len(content))])

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == content


def test_long_paragraph_is_split_at_the_word_limit() -> None:
    content = " ".join(f"kelime{index}" for index in range(161))
    document = _document(content, [PageRange(page_number=1, start=0, end=len(content))])

    chunks = chunk_document(document)

    assert len(chunks) == 3
    assert all(len(chunk.content.split()) <= DEFAULT_MAX_WORDS for chunk in chunks)


def test_chunk_order_is_preserved() -> None:
    content = "Bir iki.\n\nÜç dört.\n\nBeş altı."
    document = _document(content, [PageRange(page_number=1, start=0, end=len(content))])

    chunks = chunk_document(document, max_words=2)

    assert [chunk.position for chunk in chunks] == [0, 1, 2]
    assert [chunk.content for chunk in chunks] == ["Bir iki.", "Üç dört.", "Beş altı."]


def test_chunks_receive_the_correct_page_numbers() -> None:
    content = "İlk sayfa.\n\nİkinci sayfa."
    second_page_start = content.index("İkinci")
    document = _document(
        content,
        [
            PageRange(page_number=1, start=0, end=len("İlk sayfa.")),
            PageRange(page_number=2, start=second_page_start, end=len(content)),
        ],
    )

    chunks = chunk_document(document, max_words=2)

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2)]


def test_chunk_spanning_two_pages_has_both_page_numbers() -> None:
    content = "İlk sayfa.\n\nİkinci sayfa."
    second_page_start = content.index("İkinci")
    document = _document(
        content,
        [
            PageRange(page_number=1, start=0, end=len("İlk sayfa.")),
            PageRange(page_number=2, start=second_page_start, end=len(content)),
        ],
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_all_corpus_documents_can_be_chunked() -> None:
    documents = [build_document(entry) for entry in load_document_manifest()]

    assert len(documents) == 9
    for document in documents:
        chunks = chunk_document(document)
        assert chunks
        assert all(chunk.document_id == document.id for chunk in chunks)
