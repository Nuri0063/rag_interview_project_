from pathlib import Path

import pytest

from app.ingestion.manifest import DocumentManifestError, load_document_manifest


def test_loads_configured_document_manifest() -> None:
    entries = load_document_manifest()

    assert len(entries) == 9
    assert entries[0].id == "return-policy-v1"
    assert entries[0].source_name == "01_return_and_exchange_procedure_2024.pdf"
    assert entries[0].family_id == "return-policy"
    assert entries[0].document_type == "procedure"


def test_multiple_documents_can_belong_to_same_family() -> None:
    entries = load_document_manifest()

    assert [entry.family_id for entry in entries[:2]] == ["return-policy", "return-policy"]


def test_malformed_yaml_raises_clear_error(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.yaml"
    manifest_path.write_text("documents:\n  - id: broken\n    source_name: [", encoding="utf-8")

    with pytest.raises(DocumentManifestError, match="YAML biçimi geçersiz"):
        load_document_manifest(manifest_path)


def test_manifest_entry_rejects_fields_outside_the_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "unexpected-field.yaml"
    manifest_path.write_text(
        """documents:
  - id: return-policy-v1
    source_name: return-policy.pdf
    family_id: return-policy
    document_type: procedure
    version: v1
""",
        encoding="utf-8",
    )

    with pytest.raises(DocumentManifestError, match="izin verilmeyen alanlar: version"):
        load_document_manifest(manifest_path)
