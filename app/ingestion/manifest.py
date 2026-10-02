"""Load document-management metadata from the YAML manifest."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "document_manifest.yaml"
REQUIRED_FIELDS = frozenset({"id", "source_name", "family_id", "document_type"})


class DocumentManifestError(ValueError):
    """Raised when a document manifest cannot be read or is invalid."""


@dataclass(frozen=True)
class DocumentManifestEntry:
    """Document-management metadata supplied outside source document text."""

    id: str
    source_name: str
    family_id: str
    document_type: str


def load_document_manifest(path: Path = MANIFEST_PATH) -> list[DocumentManifestEntry]:
    """Read and validate document entries from a YAML manifest."""
    try:
        with path.open(encoding="utf-8") as manifest_file:
            raw_manifest: Any = yaml.safe_load(manifest_file)
    except OSError as error:
        raise DocumentManifestError(f"Manifest dosyası okunamadı: {path}") from error
    except yaml.YAMLError as error:
        raise DocumentManifestError(f"Manifest YAML biçimi geçersiz: {path}") from error

    if not isinstance(raw_manifest, dict):
        raise DocumentManifestError("Manifest kökü bir eşleme olmalıdır.")

    documents = raw_manifest.get("documents")
    if not isinstance(documents, list):
        raise DocumentManifestError("Manifest içindeki 'documents' alanı bir liste olmalıdır.")

    return [_to_manifest_entry(document, index) for index, document in enumerate(documents)]


def _to_manifest_entry(raw_entry: Any, index: int) -> DocumentManifestEntry:
    if not isinstance(raw_entry, dict):
        raise DocumentManifestError(f"documents[{index}] bir eşleme olmalıdır.")

    entry_fields = set(raw_entry)
    if entry_fields != REQUIRED_FIELDS:
        missing = REQUIRED_FIELDS - entry_fields
        extra = entry_fields - REQUIRED_FIELDS
        details: list[str] = []
        if missing:
            details.append(f"eksik alanlar: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"izin verilmeyen alanlar: {', '.join(sorted(extra))}")
        raise DocumentManifestError(f"documents[{index}] geçersiz; {'; '.join(details)}.")

    if not all(isinstance(value, str) for value in raw_entry.values()):
        raise DocumentManifestError(f"documents[{index}] içindeki tüm alanlar metin olmalıdır.")

    return DocumentManifestEntry(**raw_entry)
