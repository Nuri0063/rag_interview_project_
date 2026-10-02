from types import SimpleNamespace

import pytest

from app.analysis.evidence_analyzer import _SYSTEM_INSTRUCTIONS, EvidenceAnalyzer, EvidenceAnalyzerError
from app.retrieval.retriever import RetrievalResult


class FakeResponses:
    def __init__(self, output: object | None = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_parsed=self.output)


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def _candidate(chunk_id: str = "chunk-1") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        content="Returns can be requested within 30 calendar days.",
        document_id="return-policy-v2",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        document_type="procedure",
        source_name="return-policy.pdf",
        page_start=1,
        page_end=1,
        position=0,
        score=0.8,
        rerank_score=0.9,
    )


def _analyzer(output: object | None = None, error: Exception | None = None) -> tuple[EvidenceAnalyzer, FakeResponses]:
    responses = FakeResponses(output=output, error=error)
    return EvidenceAnalyzer(client=FakeClient(responses)), responses


def test_answered_result_is_parsed() -> None:
    analyzer, responses = _analyzer(
        {
            "status": "answered",
            "supporting_chunk_ids": ["chunk-1"],
            "conflicting_chunk_ids": [],
            "reason": "The chunk states a return window directly.",
        }
    )

    result = analyzer.analyze("Iade süresi nedir?", [_candidate()])

    assert result.status == "answered"
    assert result.supporting_chunk_ids == ["chunk-1"]
    sent_input = str(responses.calls[0]["input"])
    assert "rerank_score" not in sent_input
    assert "published_at" not in sent_input


def test_insufficient_evidence_result_is_parsed() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "insufficient_evidence",
            "supporting_chunk_ids": [],
            "conflicting_chunk_ids": [],
            "reason": "The candidates do not address international shipping.",
        }
    )

    result = analyzer.analyze("Uluslararası teslimat var mı?", [_candidate()])

    assert result.status == "insufficient_evidence"


def test_conflict_result_is_parsed() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "conflict",
            "supporting_chunk_ids": ["chunk-1"],
            "conflicting_chunk_ids": ["chunk-2"],
            "reason": "Different families give contradictory return windows.",
        }
    )

    result = analyzer.analyze("Iade süresi nedir?", [_candidate(), _candidate("chunk-2")])

    assert result.status == "conflict"
    assert result.conflicting_chunk_ids == ["chunk-2"]


def test_cross_family_return_window_conflict_contract_is_explicit() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "conflict",
            "supporting_chunk_ids": [],
            "conflicting_chunk_ids": ["chunk-14-days", "chunk-30-days"],
            "reason": "Different families give incompatible return windows.",
        }
    )
    fourteen_days = _candidate("chunk-14-days").model_copy(
        update={
            "document_id": "customer-faq",
            "family_id": "customer-faq",
            "content": "Returns are allowed within 14 days.",
        }
    )
    thirty_days = _candidate("chunk-30-days").model_copy(
        update={
            "document_id": "return-policy-v2",
            "family_id": "return-policy",
            "content": "Returns are allowed within 30 days.",
        }
    )

    result = analyzer.analyze("How many days can I return a product?", [fourteen_days, thirty_days])

    assert result.status == "conflict"
    assert result.conflicting_chunk_ids == ["chunk-14-days", "chunk-30-days"]
    assert "Before deciding answered or insufficient_evidence" in _SYSTEM_INSTRUCTIONS
    assert "family A: \"Returns are allowed within 14 days.\"" in _SYSTEM_INSTRUCTIONS
    assert "family B: \"Returns are allowed within 30 days.\"" in _SYSTEM_INSTRUCTIONS


def test_unknown_chunk_ids_are_rejected() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "answered",
            "supporting_chunk_ids": ["unknown-chunk"],
            "conflicting_chunk_ids": [],
            "reason": "Unsupported identifier.",
        }
    )

    with pytest.raises(EvidenceAnalyzerError, match="bulunmayan chunk kimlikleri"):
        analyzer.analyze("Iade süresi nedir?", [_candidate()])


def test_conflict_requires_conflicting_chunk_ids() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "conflict",
            "supporting_chunk_ids": [],
            "conflicting_chunk_ids": [],
            "reason": "No identifiers supplied.",
        }
    )

    with pytest.raises(EvidenceAnalyzerError, match="conflicting_chunk_id"):
        analyzer.analyze("Iade süresi nedir?", [_candidate()])


def test_insufficient_evidence_requires_empty_evidence_ids() -> None:
    analyzer, _ = _analyzer(
        {
            "status": "insufficient_evidence",
            "supporting_chunk_ids": ["chunk-1"],
            "conflicting_chunk_ids": [],
            "reason": "No evidence exists.",
        }
    )

    with pytest.raises(EvidenceAnalyzerError, match="insufficient_evidence sonucu"):
        analyzer.analyze("Iade süresi nedir?", [_candidate()])


def test_empty_candidates_return_insufficient_evidence_without_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    analyzer = EvidenceAnalyzer()

    result = analyzer.analyze("Iade süresi nedir?", [])

    assert result.status == "insufficient_evidence"
    assert analyzer.client is None


def test_api_errors_are_reported_clearly() -> None:
    analyzer, _ = _analyzer(error=RuntimeError("network failure"))

    with pytest.raises(EvidenceAnalyzerError, match="Responses API"):
        analyzer.analyze("Iade süresi nedir?", [_candidate()])
