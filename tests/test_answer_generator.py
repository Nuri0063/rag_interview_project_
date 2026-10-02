from types import SimpleNamespace

from app.analysis.evidence_analyzer import EvidenceAnalysisResult
from app.generation.answer_generator import AnswerGenerator, INSUFFICIENT_EVIDENCE_ANSWER
from app.retrieval.retriever import RetrievalResult


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="İade kodu on gün geçerlidir.")


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def _chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="return:1",
        content="The return shipping code is valid for 10 days.",
        document_id="return-policy-v2",
        family_id="return-policy",
        version="2.0",
        published_at="2026-02-01",
        document_type="procedure",
        source_name="return.pdf",
        page_start=3,
        page_end=3,
        position=0,
        score=0.8,
    )


def test_answered_generation_uses_selected_evidence() -> None:
    responses = FakeResponses()
    generator = AnswerGenerator(client=FakeClient(responses))
    result = EvidenceAnalysisResult(
        status="answered",
        supporting_chunk_ids=["return:1"],
        conflicting_chunk_ids=[],
        reason="Direct evidence.",
    )

    answer = generator.generate("İade kodu ne kadar geçerli?", result, [_chunk()])

    assert answer == "İade kodu on gün geçerlidir."
    sent_input = str(responses.calls[0]["input"])
    assert "return-policy" not in sent_input
    assert "page_start" not in sent_input
    assert "valid for 10 days" in sent_input


def test_insufficient_evidence_does_not_call_answer_llm() -> None:
    responses = FakeResponses()
    generator = AnswerGenerator(client=FakeClient(responses))
    result = EvidenceAnalysisResult(
        status="insufficient_evidence",
        supporting_chunk_ids=[],
        conflicting_chunk_ids=[],
        reason="No direct evidence.",
    )

    answer = generator.generate("Üyelik ücreti ne kadar?", result, [])

    assert answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert responses.calls == []
