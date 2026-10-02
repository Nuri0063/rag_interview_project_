from app.ingestion.chunking import chunk_document
from app.ingestion.document_ingestion import build_document
from app.ingestion.manifest import load_document_manifest
from app.resolution.version_resolver import resolve_versions
from app.retrieval.retriever import RetrievalResult


def _candidate(
    *,
    chunk_id: str,
    document_id: str,
    family_id: str,
    version: str,
    published_at: str,
    score: float,
    content: str = "Candidate text.",
    source_name: str = "source.pdf",
    position: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        document_id=document_id,
        family_id=family_id,
        version=version,
        published_at=published_at,
        document_type="guide",
        source_name=source_name,
        page_start=1,
        page_end=1,
        position=position,
        score=score,
    )


def test_higher_version_is_preferred_within_the_same_family() -> None:
    old = _candidate(
        chunk_id="return-policy-v1:0",
        document_id="return-policy-v1",
        family_id="return-policy",
        version="1.1",
        published_at="2024-06-18",
        score=0.99,
    )
    current = _candidate(
        chunk_id="return-policy-v2:0",
        document_id="return-policy-v2",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        score=0.10,
    )

    assert resolve_versions([old, current]) == [current]


def test_newer_published_date_breaks_equal_version_ties() -> None:
    earlier = _candidate(
        chunk_id="guide-a:0",
        document_id="guide-a",
        family_id="guide",
        version="2.0",
        published_at="2026-01-01",
        score=0.9,
    )
    later = _candidate(
        chunk_id="guide-b:0",
        document_id="guide-b",
        family_id="guide",
        version="2.0",
        published_at="2026-02-01",
        score=0.1,
    )

    assert resolve_versions([earlier, later]) == [later]


def test_versions_are_never_compared_across_families() -> None:
    faq = _candidate(
        chunk_id="faq:0",
        document_id="faq",
        family_id="customer-faq",
        version="5.2",
        published_at="2026-04-06",
        score=0.8,
    )
    payments = _candidate(
        chunk_id="payments:0",
        document_id="payments",
        family_id="payments-refunds",
        version="3.0",
        published_at="2026-06-22",
        score=0.7,
    )

    assert resolve_versions([faq, payments]) == [faq, payments]


def test_unrelated_candidates_remain_and_original_order_is_preserved() -> None:
    old = _candidate(
        chunk_id="return-policy-v1:0",
        document_id="return-policy-v1",
        family_id="return-policy",
        version="1.1",
        published_at="2024-06-18",
        score=0.9,
    )
    unrelated = _candidate(
        chunk_id="faq:0",
        document_id="faq",
        family_id="customer-faq",
        version="5.2",
        published_at="2026-04-06",
        score=0.8,
    )
    current_first = _candidate(
        chunk_id="return-policy-v2:0",
        document_id="return-policy-v2",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        score=0.7,
        position=0,
    )
    current_second = _candidate(
        chunk_id="return-policy-v2:1",
        document_id="return-policy-v2",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        score=0.6,
        position=1,
    )

    resolved = resolve_versions([old, unrelated, current_first, current_second])

    assert resolved == [unrelated, current_first, current_second]


def test_retrieval_score_does_not_override_a_newer_version() -> None:
    old_high_score = _candidate(
        chunk_id="old:0",
        document_id="old",
        family_id="return-policy",
        version="1.1",
        published_at="2024-06-18",
        score=0.99,
    )
    current_low_score = _candidate(
        chunk_id="current:0",
        document_id="current",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        score=0.01,
    )

    assert resolve_versions([old_high_score, current_low_score]) == [current_low_score]


def test_current_return_policy_survives_real_return_window_and_code_examples() -> None:
    documents = {entry.id: build_document(entry) for entry in load_document_manifest()}
    old_document = documents["return-policy-v1"]
    current_document = documents["return-policy-v2"]

    old_window = _chunk_candidate(old_document, "14 calendar days", score=0.99)
    current_window = _chunk_candidate(current_document, "30 calendar days", score=0.01)
    old_code = _chunk_candidate(old_document, "7 days", score=0.99)
    current_code = _chunk_candidate(current_document, "10 days", score=0.01)

    resolved_window = resolve_versions([old_window, current_window])
    resolved_code = resolve_versions([old_code, current_code])

    assert [candidate.document_id for candidate in resolved_window] == ["return-policy-v2"]
    assert [candidate.document_id for candidate in resolved_code] == ["return-policy-v2"]


def _chunk_candidate(document, phrase: str, score: float) -> RetrievalResult:
    chunk = next(chunk for chunk in chunk_document(document) if phrase in chunk.content)
    return _candidate(
        chunk_id=chunk.id,
        document_id=document.id,
        family_id=document.family_id,
        version=document.version or "",
        published_at=document.published_at.isoformat() if document.published_at else "",
        score=score,
        content=chunk.content,
        source_name=document.source_name,
        position=chunk.position,
    )
