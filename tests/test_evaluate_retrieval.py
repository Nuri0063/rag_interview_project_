from eval.evaluate_retrieval import calculate_metrics, first_relevant_rank
from app.retrieval.retriever import RetrievalResult


def _candidate(document_id: str, chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content="Content",
        document_id=document_id,
        family_id="family",
        version="1.0",
        published_at="2026-01-01",
        document_type="guide",
        source_name="source.pdf",
        page_start=1,
        page_end=1,
        position=0,
        score=0.5,
    )


def test_calculates_hit_at_k_and_mrr() -> None:
    questions = [
        {"id": "one", "expected_status": "answered", "relevant_document_ids": ["a"]},
        {"id": "two", "expected_status": "answered", "relevant_document_ids": ["b"]},
        {"id": "three", "expected_status": "answered", "relevant_document_ids": ["c"]},
    ]
    candidates = {
        "one": [_candidate("a", "a:0")],
        "two": [_candidate("x", "x:0"), _candidate("b", "b:0")],
        "three": [_candidate("x", "x:1")],
    }

    metrics = calculate_metrics(questions, candidates)

    assert metrics.hit_at_1 == 1 / 3
    assert metrics.hit_at_3 == 2 / 3
    assert metrics.hit_at_5 == 2 / 3
    assert metrics.mrr == 0.5


def test_multiple_relevant_documents_are_supported() -> None:
    candidates = [_candidate("other", "other:0"), _candidate("second-relevant", "second:0")]

    assert first_relevant_rank(candidates, ["first-relevant", "second-relevant"]) == 2


def test_duplicate_chunks_do_not_distort_document_rank() -> None:
    candidates = [
        _candidate("other", "other:0"),
        _candidate("other", "other:1"),
        _candidate("relevant", "relevant:0"),
    ]

    assert first_relevant_rank(candidates, ["relevant"]) == 2


def test_insufficient_evidence_questions_are_excluded_from_metrics() -> None:
    questions = [
        {"id": "answered", "expected_status": "answered", "relevant_document_ids": ["a"]},
        {"id": "unknown", "expected_status": "insufficient_evidence"},
    ]
    candidates = {
        "answered": [_candidate("a", "a:0")],
        "unknown": [_candidate("unrelated", "unrelated:0")],
    }

    metrics = calculate_metrics(questions, candidates)

    assert metrics.question_count == 1
    assert metrics.hit_at_1 == 1.0
    assert metrics.mrr == 1.0
