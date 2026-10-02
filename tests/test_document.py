from datetime import date

import pytest
from pydantic import ValidationError

from app.models.document import Document
from app.models.extraction import PageRange


@pytest.fixture
def document_data() -> dict[str, object]:
    return {
        "id": "doc-1",
        "family_id": "kurulum-kilavuzu",
        "title": "Kurulum Kılavuzu",
        "document_type": "technical_guide",
        "source_name": "kurulum.pdf",
        "content": "Önce bağlantıyı kontrol edin.\nArdından kurulumu başlatın.",
        "page_ranges": [{"page_number": 1, "start": 0, "end": 57}],
    }


def test_document_preserves_plain_text_and_metadata(
    document_data: dict[str, object],
) -> None:
    document = Document(**document_data)

    assert document.model_dump(exclude_none=True) == document_data
    assert document.version is None
    assert document.published_at is None
    assert document.page_ranges == [PageRange(page_number=1, start=0, end=57)]


def test_document_accepts_version_and_iso_date(
    document_data: dict[str, object],
) -> None:
    document = Document(**document_data, version="v2", published_at="2026-09-01")

    assert document.version == "v2"
    assert document.published_at == date(2026, 9, 1)


def test_document_accepts_explicit_nulls(document_data: dict[str, object]) -> None:
    document = Document(**document_data, version=None, published_at=None)

    assert document.version is None
    assert document.published_at is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("published_at", "2026-02-30"), ("content", ["metin"]), ("version", 2)],
)
def test_document_rejects_invalid_values(
    document_data: dict[str, object], field: str, value: object
) -> None:
    invalid_data: dict[str, object] = {**document_data, field: value}

    with pytest.raises(ValidationError) as error:
        Document.model_validate(invalid_data)

    assert error.value.errors()[0]["loc"] == (field,)


def test_document_requires_family_id(document_data: dict[str, object]) -> None:
    document_data.pop("family_id")

    with pytest.raises(ValidationError) as error:
        Document(**document_data)

    assert error.value.errors()[0]["loc"] == ("family_id",)
