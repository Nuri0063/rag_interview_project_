from datetime import date
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.document_ingestion import (
    PdfMetadataExtractionError,
    build_document,
    extract_pdf_metadata,
)
from app.ingestion.manifest import DocumentManifestEntry, load_document_manifest


def test_extracts_title_version_and_published_date() -> None:
    content = (
        "Nexa Retail | Return Process and Customer Requests | Page 1\n"
        "Version 2.0 | Published: February 1, 2026\n"
    )

    metadata = extract_pdf_metadata(content)

    assert metadata.title == "Return Process and Customer Requests"
    assert metadata.version == "2.0"
    assert metadata.published_at == date(2026, 2, 1)


@pytest.mark.parametrize(
    "content",
    [
        "Version 2.0 | Published: February 1, 2026\n",
        "Nexa Retail | Test Guide | Page 1\n",
        (
            "Nexa Retail | Test Guide | Page 1\n"
            "Version 2.0 | Published: February 31, 2026\n"
        ),
    ],
)
def test_missing_or_malformed_required_metadata_raises_clear_error(content: str) -> None:
    with pytest.raises(PdfMetadataExtractionError):
        extract_pdf_metadata(content)


def test_builds_document_from_manifest_entry_and_pdf(tmp_path: Path) -> None:
    source_name = "test-guide.pdf"
    source_path = tmp_path / source_name
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "Nexa Retail | Test Guide | Page 1\nVersion 1.0 | Published: June 18, 2024\nBody text.",
        )
        pdf.save(source_path)

    entry = DocumentManifestEntry(
        id="test-guide",
        source_name=source_name,
        family_id="test-guides",
        document_type="guide",
    )

    document = build_document(entry, documents_dir=tmp_path)

    assert document.id == "test-guide"
    assert document.family_id == "test-guides"
    assert document.document_type == "guide"
    assert document.title == "Test Guide"
    assert document.version == "1.0"
    assert document.published_at == date(2024, 6, 18)
    assert "Body text." in document.content
    assert document.page_ranges[0].page_number == 1


def test_all_manifest_documents_exist_extract_and_build() -> None:
    entries = load_document_manifest()
    documents_dir = Path(__file__).resolve().parents[1] / "documents"

    assert len(entries) == 9
    for entry in entries:
        assert (documents_dir / entry.source_name).is_file()
        document = build_document(entry, documents_dir=documents_dir)
        assert document.id == entry.id
        assert document.content
