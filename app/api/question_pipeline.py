"""Small end-to-end orchestration for one Knowledge QA question."""

from app.analysis.evidence_analyzer import EvidenceAnalysisResult, EvidenceAnalyzer
from app.generation.answer_generator import AnswerGenerator
from app.models.api import AskResponse, Source
from app.resolution.version_resolver import resolve_versions
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalResult, Retriever
from app.retrieval.vector_store import VectorStore


RETRIEVAL_CANDIDATE_COUNT = 16
FINAL_CANDIDATE_COUNT = 8


class QuestionPipelineError(RuntimeError):
    """Raised for expected application-level question processing failures."""


class QuestionPipeline:
    """Apply the fixed production retrieval and evidence-decision pipeline."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        evidence_analyzer: EvidenceAnalyzer | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or Retriever(self.vector_store)
        self.reranker = reranker or Reranker()
        self.evidence_analyzer = evidence_analyzer or EvidenceAnalyzer()
        self.answer_generator = answer_generator or AnswerGenerator()

    def ask(self, question: str) -> AskResponse:
        """Answer one question with the evaluated top-16 then top-8 flow."""
        if self.vector_store.count() == 0:
            raise QuestionPipelineError("Arama indeksi boş. Sorular yanıtlanmadan önce korpus indekslenmelidir.")

        raw_candidates = self.retriever.retrieve(question, top_k=RETRIEVAL_CANDIDATE_COUNT)
        resolved_candidates = resolve_versions(raw_candidates)
        reranked_candidates = self.reranker.rerank(question, resolved_candidates)
        reranked_candidates = reranked_candidates[:FINAL_CANDIDATE_COUNT]
        evidence_result = self.evidence_analyzer.analyze(question, reranked_candidates)
        selected_chunks = _selected_chunks(evidence_result, reranked_candidates)
        answer = self.answer_generator.generate(question, evidence_result, selected_chunks)
        sources = build_sources(selected_chunks)
        return AskResponse(
            status=evidence_result.status,
            answer=answer,
            sources=sources,
        )


def _selected_chunks(
    evidence_result: EvidenceAnalysisResult,
    candidates: list[RetrievalResult],
) -> list[RetrievalResult]:
    selected_ids = set(evidence_result.supporting_chunk_ids)
    if evidence_result.status == "conflict":
        selected_ids.update(evidence_result.conflicting_chunk_ids)
    return [candidate for candidate in candidates if candidate.chunk_id in selected_ids]


def build_sources(selected_chunks: list[RetrievalResult]) -> list[Source]:
    """Map selected evidence chunks to stable, deduplicated source citations."""
    sources: list[Source] = []
    seen: set[tuple[str, int, int]] = set()
    for chunk in selected_chunks:
        key = (chunk.source_name, chunk.page_start, chunk.page_end)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                source_name=chunk.source_name,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
            )
        )
    return sources
