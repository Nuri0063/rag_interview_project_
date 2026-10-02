"""Build normalized documents from manifest metadata and PDF text."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re

from app.ingestion.manifest import DocumentManifestEntry
from app.ingestion.text_extraction import extract_text
from app.models.document import Document


DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"
_TITLE_PATTERN = re.compile(
    r"^Nexa Retail\s*\|\s*(?P<title>.+?)\s*\|\s*Page\s+\d+\s*$",
    re.MULTILINE,
)
_VERSION_DATE_PATTERN = re.compile(
    r"^Version\s+(?P<version>\S+)\s*\|\s*Published:\s*"
    r"(?P<published_at>[A-Za-z]+\s+\d{1,2},\s+\d{4})\s*$",
    re.MULTILINE,
)


class PdfMetadataExtractionError(ValueError):
    """Raised when required PDF metadata is missing or malformed."""


@dataclass(frozen=True)
class PdfMetadata:
    """Metadata deterministically extracted from a PDF's text."""

    title: str
    version: str
    published_at: date


def extract_pdf_metadata(content: str) -> PdfMetadata:
    """Extract the title, version, and publication date from PDF text."""
    title_match = _TITLE_PATTERN.search(content)
    if title_match is None:
        raise PdfMetadataExtractionError("PDF başlık üst bilgisi bulunamadı veya geçersiz.")

    version_date_match = _VERSION_DATE_PATTERN.search(content)
    if version_date_match is None:
        raise PdfMetadataExtractionError("PDF sürüm ve yayın tarihi bilgisi bulunamadı veya geçersiz.")

    published_at_text = version_date_match.group("published_at")
    try:
        published_at = datetime.strptime(published_at_text, "%B %d, %Y").date()
    except ValueError as error:
        raise PdfMetadataExtractionError("PDF yayın tarihi geçersiz.") from error

    return PdfMetadata(
        title=title_match.group("title").strip(),
        version=version_date_match.group("version"),
        published_at=published_at,
    )


def build_document(
    entry: DocumentManifestEntry, documents_dir: Path = DOCUMENTS_DIR
) -> Document:
    """Create a normalized Document from one manifest entry and its PDF source."""
    extracted_text = extract_text(documents_dir / entry.source_name)
    metadata = extract_pdf_metadata(extracted_text.content)

    return Document(
        id=entry.id,
        family_id=entry.family_id,
        title=metadata.title,
        document_type=entry.document_type,
        version=metadata.version,
        published_at=metadata.published_at,
        source_name=entry.source_name,
        content=extracted_text.content,
        page_ranges=extracted_text.page_ranges,
    )
