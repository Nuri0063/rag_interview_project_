"""Evaluate OpenAI evidence-decision statuses on the fixed question set."""

from dataclasses import dataclass
import os
from typing import Any, Literal

from dotenv import load_dotenv

from app.analysis.evidence_analyzer import EvidenceAnalysisResult, EvidenceAnalyzer
from app.api.question_pipeline import FINAL_CANDIDATE_COUNT, RETRIEVAL_CANDIDATE_COUNT
from app.resolution.version_resolver import resolve_versions
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalResult, Retriever
from app.retrieval.vector_store import VectorStore
from eval.evaluate_retrieval import load_questions


load_dotenv()

Status = Literal["answered", "insufficient_evidence", "conflict"]
STATUSES: tuple[Status, ...] = ("answered", "insufficient_evidence", "conflict")


@dataclass(frozen=True)
class StatusMetrics:
    question_count: int
    overall_accuracy: float
    per_class_accuracy: dict[Status, float | None]
    confusion: dict[Status, dict[Status, int]]


@dataclass(frozen=True)
class EvidenceEvaluationReport:
    metrics: StatusMetrics
    incorrect_question_ids: list[str]
    supporting_without_expected_document_ids: list[str]
    conflicts_without_chunk_ids: list[str]


def calculate_status_metrics(
    questions: list[dict[str, Any]],
    predicted_statuses: dict[str, Status],
) -> StatusMetrics:
    """Calculate status accuracy and expected-versus-predicted counts."""
    confusion: dict[Status, dict[Status, int]] = {
        expected: {predicted: 0 for predicted in STATUSES} for expected in STATUSES
    }
    correct = 0
    for question in questions:
        question_id = question["id"]
        expected = _as_status(question["expected_status"])
        try:
            predicted = predicted_statuses[question_id]
        except KeyError as error:
            raise ValueError(f"{question_id} için tahmin edilen status yok.") from error
        confusion[expected][predicted] += 1
        correct += expected == predicted

    question_count = len(questions)
    per_class_accuracy: dict[Status, float | None] = {}
    for status in STATUSES:
        expected_count = sum(confusion[status].values())
        per_class_accuracy[status] = (
            confusion[status][status] / expected_count if expected_count else None
        )
    return StatusMetrics(
        question_count=question_count,
        overall_accuracy=correct / question_count if question_count else 0.0,
        per_class_accuracy=per_class_accuracy,
        confusion=confusion,
    )


def run_evaluation() -> EvidenceEvaluationReport | None:
    """Run the live production candidate flow, or skip without an API key."""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY ayarlı değil; canlı kanıt değerlendirmesi atlandı.")
        return None

    questions = load_questions()
    retriever = Retriever(VectorStore())
    reranker = Reranker()
    analyzer = EvidenceAnalyzer()
    predicted_statuses: dict[str, Status] = {}
    results_by_question: dict[str, EvidenceAnalysisResult] = {}
    candidates_by_question: dict[str, list[RetrievalResult]] = {}

    for question in questions:
        raw_candidates = retriever.retrieve(
            question["question"], top_k=RETRIEVAL_CANDIDATE_COUNT
        )
        resolved_candidates = resolve_versions(raw_candidates)
        candidates = reranker.rerank(question["question"], resolved_candidates)[
            :FINAL_CANDIDATE_COUNT
        ]
        result = analyzer.analyze(question["question"], candidates)
        predicted_statuses[question["id"]] = result.status
        results_by_question[question["id"]] = result
        candidates_by_question[question["id"]] = candidates

    metrics = calculate_status_metrics(questions, predicted_statuses)
    incorrect_question_ids = [
        question["id"]
        for question in questions
        if question["expected_status"] != predicted_statuses[question["id"]]
    ]
    supporting_without_expected_document_ids = _supporting_without_expected_documents(
        questions, results_by_question, candidates_by_question
    )
    conflicts_without_chunk_ids = [
        question_id
        for question_id, result in results_by_question.items()
        if result.status == "conflict" and not result.conflicting_chunk_ids
    ]
    report = EvidenceEvaluationReport(
        metrics=metrics,
        incorrect_question_ids=incorrect_question_ids,
        supporting_without_expected_document_ids=supporting_without_expected_document_ids,
        conflicts_without_chunk_ids=conflicts_without_chunk_ids,
    )
    _print_report(questions, results_by_question, report)
    return report


def _supporting_without_expected_documents(
    questions: list[dict[str, Any]],
    results_by_question: dict[str, EvidenceAnalysisResult],
    candidates_by_question: dict[str, list[RetrievalResult]],
) -> list[str]:
    question_ids: list[str] = []
    for question in questions:
        if question["expected_status"] != "answered":
            continue
        result = results_by_question[question["id"]]
        chunk_document_ids = {
            candidate.chunk_id: candidate.document_id
            for candidate in candidates_by_question[question["id"]]
        }
        supporting_documents = {
            chunk_document_ids[chunk_id] for chunk_id in result.supporting_chunk_ids
        }
        if not supporting_documents.intersection(question["relevant_document_ids"]):
            question_ids.append(question["id"])
    return question_ids


def _as_status(value: str) -> Status:
    if value not in STATUSES:
        raise ValueError(f"Geçersiz beklenen status: {value}")
    return value


def _format_accuracy(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_report(
    questions: list[dict[str, Any]],
    results_by_question: dict[str, EvidenceAnalysisResult],
    report: EvidenceEvaluationReport,
) -> None:
    metrics = report.metrics
    print(f"questions={metrics.question_count}")
    print(f"overall_accuracy={metrics.overall_accuracy:.3f}")
    for status in STATUSES:
        print(f"{status}_accuracy={_format_accuracy(metrics.per_class_accuracy[status])}")

    print("\nConfusion summary")
    for expected in STATUSES:
        counts = metrics.confusion[expected]
        print(
            f"expected {expected} -> answered={counts['answered']} "
            f"insufficient={counts['insufficient_evidence']} conflict={counts['conflict']}"
        )

    print("\nAnswered evidence selections")
    for question in questions:
        if question["expected_status"] == "answered":
            result = results_by_question[question["id"]]
            print(f"{question['id']}: {result.supporting_chunk_ids}")

    print(f"\nIncorrect predictions: {report.incorrect_question_ids}")
    print(
        "Answered with no supporting chunk from an expected document: "
        f"{report.supporting_without_expected_document_ids}"
    )
    print(f"Conflicts without conflicting chunk IDs: {report.conflicts_without_chunk_ids}")


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_evaluation()
