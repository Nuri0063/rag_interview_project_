import pytest

from app.analysis.evidence_analyzer import EvidenceAnalysisResult, EvidenceAnalyzerError
from app.api.question_pipeline import QuestionPipeline, build_sources
from app.generation.answer_generator import INSUFFICIENT_EVIDENCE_ANSWER
from app.main import app
from app.models.api import AskResponse
from app.retrieval.retriever import RetrievalResult


class FakeStore:
    def __init__(self, count: int = 16) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakeRetriever:
    def __init__(self, candidates: list[RetrievalResult]) -> None:
        self.candidates = candidates
        self.top_k: int | None = None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        self.top_k = top_k
        return self.candidates


class FakeReranker:
    def __init__(self) -> None:
        self.candidate_count: int | None = None

    def rerank(self, question: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        self.candidate_count = len(candidates)
        return list(reversed(candidates))


class FakeAnalyzer:
    def __init__(self, result: EvidenceAnalysisResult) -> None:
        self.result = result
        self.candidate_count: int | None = None

    def analyze(self, question: str, candidates: list[RetrievalResult]) -> EvidenceAnalysisResult:
        self.candidate_count = len(candidates)
        return self.result


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.selected_chunks: list[RetrievalResult] | None = None

    def generate(self, question: str, result: EvidenceAnalysisResult, selected_chunks: list[RetrievalResult]) -> str:
        self.selected_chunks = selected_chunks
        return self.answer


def _candidate(index: int, page: int | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk:{index}",
        content=f"Evidence {index}",
        document_id=f"document:{index}",
        family_id=f"family:{index}",
        version="1.0",
        published_at="2026-01-01",
        document_type="guide",
        source_name="source.pdf",
        page_start=page or index + 1,
        page_end=page or index + 1,
        position=index,
        score=0.5,
    )


def test_pipeline_uses_top_16_then_limits_reranked_candidates_to_8() -> None:
    candidates = [_candidate(index) for index in range(16)]
    retriever = FakeRetriever(candidates)
    reranker = FakeReranker()
    analyzer = FakeAnalyzer(
        EvidenceAnalysisResult(
            status="answered",
            supporting_chunk_ids=["chunk:15"],
            conflicting_chunk_ids=[],
            reason="Evidence.",
        )
    )
    generator = FakeGenerator("Yanıt")
    pipeline = QuestionPipeline(FakeStore(), retriever, reranker, analyzer, generator)

    response = pipeline.ask("Soru")

    assert retriever.top_k == 16
    assert reranker.candidate_count == 16
    assert analyzer.candidate_count == 8
    assert response.sources[0].chunk_id == "chunk:15"


def test_conflict_response_includes_supporting_and_conflicting_sources() -> None:
    candidates = [_candidate(0), _candidate(1)]
    analyzer = FakeAnalyzer(
        EvidenceAnalysisResult(
            status="conflict",
            supporting_chunk_ids=["chunk:0"],
            conflicting_chunk_ids=["chunk:1"],
            reason="Conflict.",
        )
    )
    generator = FakeGenerator("Belgeler çelişiyor.")
    pipeline = QuestionPipeline(FakeStore(), FakeRetriever(candidates), FakeReranker(), analyzer, generator)

    response = pipeline.ask("Soru")

    assert response.status == "conflict"
    assert {source.chunk_id for source in response.sources} == {"chunk:0", "chunk:1"}


def test_build_sources_is_deterministic_and_removes_duplicate_pages() -> None:
    sources = build_sources([_candidate(0, page=3), _candidate(1, page=3), _candidate(2, page=4)])

    assert [(source.chunk_id, source.page_start) for source in sources] == [("chunk:0", 3), ("chunk:2", 4)]


def test_api_maps_application_errors_without_exposing_internal_details() -> None:
    class FailingPipeline:
        def ask(self, question: str) -> AskResponse:
            raise EvidenceAnalyzerError("private OpenAI failure")

    app.state.question_pipeline = FailingPipeline()
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Soru"})

    del app.state.question_pipeline
    assert response.status_code == 503
    assert "private OpenAI failure" not in response.json()["detail"]
