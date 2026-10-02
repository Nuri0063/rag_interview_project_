import pytest

from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalResult


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker()


def _candidate(
    chunk_id: str,
    content: str,
    score: float,
    page: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        document_id="document-1",
        family_id="family-1",
        version="2.0",
        published_at="2026-02-01",
        document_type="guide",
        source_name="source.pdf",
        page_start=page,
        page_end=page,
        position=page - 1,
        score=score,
    )


def _candidates() -> list[RetrievalResult]:
    return [
        _candidate(
            "chunk-return",
            "Customers can create a return request after delivery.",
            score=0.91,
            page=1,
        ),
        _candidate(
            "chunk-password",
            "To reset a password, use the password recovery option on the account sign-in page.",
            score=0.42,
            page=2,
        ),
        _candidate(
            "chunk-refund",
            "Refunds are sent to the original payment method after approval.",
            score=0.63,
            page=3,
        ),
    ]


def test_reranker_returns_same_number_of_candidates(reranker: Reranker) -> None:
    candidates = _candidates()

    assert len(reranker.rerank("Şifremi sıfırlamak istiyorum.", candidates)) == len(candidates)


def test_reranker_orders_by_relevance_score(reranker: Reranker) -> None:
    results = reranker.rerank("Şifremi sıfırlamak istiyorum.", _candidates())

    assert all(result.rerank_score is not None for result in results)
    assert [result.rerank_score for result in results] == sorted(
        (result.rerank_score for result in results), reverse=True
    )


def test_reranking_preserves_original_retrieval_scores(reranker: Reranker) -> None:
    candidates = _candidates()
    retrieval_scores = {candidate.chunk_id: candidate.score for candidate in candidates}

    results = reranker.rerank("Şifremi sıfırlamak istiyorum.", candidates)

    assert {result.chunk_id: result.score for result in results} == retrieval_scores


def test_reranking_preserves_source_metadata(reranker: Reranker) -> None:
    candidate = _candidates()[1]

    result = next(
        result
        for result in reranker.rerank("Şifremi sıfırlamak istiyorum.", _candidates())
        if result.chunk_id == candidate.chunk_id
    )

    assert result.document_id == candidate.document_id
    assert result.family_id == candidate.family_id
    assert result.version == candidate.version
    assert result.published_at == candidate.published_at
    assert result.source_name == candidate.source_name
    assert result.page_start == candidate.page_start
    assert result.page_end == candidate.page_end


def test_empty_candidate_list_returns_cleanly(reranker: Reranker) -> None:
    assert reranker.rerank("Şifremi sıfırlamak istiyorum.", []) == []
