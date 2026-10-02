"""Evaluate baseline and reranked retrieval against the local question set."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.resolution.version_resolver import resolve_versions
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import RetrievalResult, Retriever
from app.retrieval.vector_store import VectorStore


QUESTIONS_PATH = Path(__file__).with_name("questions.json")


@dataclass(frozen=True)
class Metrics:
    question_count: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load the small, explicit retrieval evaluation dataset."""
    with path.open(encoding="utf-8") as question_file:
        return json.load(question_file)


def first_relevant_rank(
    candidates: list[RetrievalResult], relevant_document_ids: list[str]
) -> int | None:
    """Return the rank of the first relevant unique document."""
    seen_document_ids: set[str] = set()
    rank = 0
    for candidate in candidates:
        if candidate.document_id in seen_document_ids:
            continue
        seen_document_ids.add(candidate.document_id)
        rank += 1
        if candidate.document_id in relevant_document_ids:
            return rank
    return None


def calculate_metrics(
    questions: list[dict[str, Any]],
    candidates_by_question: dict[str, list[RetrievalResult]],
) -> Metrics:
    """Calculate document-level Hit@K and MRR for answered questions only."""
    ranks = [
        first_relevant_rank(
            candidates_by_question.get(question["id"], []),
            question["relevant_document_ids"],
        )
        for question in questions
        if question["expected_status"] == "answered"
    ]
    question_count = len(ranks)
    if question_count == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)

    return Metrics(
        question_count=question_count,
        hit_at_1=sum(rank is not None and rank <= 1 for rank in ranks) / question_count,
        hit_at_3=sum(rank is not None and rank <= 3 for rank in ranks) / question_count,
        hit_at_5=sum(rank is not None and rank <= 5 for rank in ranks) / question_count,
        mrr=sum(1 / rank if rank is not None else 0.0 for rank in ranks) / question_count,
    )


def run_evaluation() -> None:
    """Run both fixed retrieval pipelines and print a compact comparison."""
    questions = load_questions()
    retriever = Retriever(VectorStore())
    reranker = Reranker()
    baseline_results: dict[str, list[RetrievalResult]] = {}
    reranked_results: dict[str, list[RetrievalResult]] = {}
    inspections: dict[str, list[RetrievalResult]] = {}
    version_checks: list[tuple[str, bool, bool, bool]] = []

    for question in questions:
        raw_candidates = retriever.retrieve(question["question"])
        resolved_candidates = resolve_versions(raw_candidates)
        baseline_results[question["id"]] = resolved_candidates

        if question["expected_status"] == "answered":
            reranked_results[question["id"]] = reranker.rerank(
                question["question"], resolved_candidates
            )
            if question.get("expected_family_id") == "return-policy":
                raw_versions = {candidate.version for candidate in raw_candidates if candidate.family_id == "return-policy"}
                resolved_versions = {
                    candidate.version
                    for candidate in resolved_candidates
                    if candidate.family_id == "return-policy"
                }
                version_checks.append(
                    (
                        question["id"],
                        "1.1" in raw_versions and "2.0" in raw_versions,
                        "1.1" in resolved_versions,
                        question["expected_version"] in resolved_versions,
                    )
                )
        else:
            inspections[question["id"]] = resolved_candidates[:5]

    baseline_metrics = calculate_metrics(questions, baseline_results)
    reranked_metrics = calculate_metrics(questions, reranked_results)
    print(f"questions={len(questions)} metric_questions={baseline_metrics.question_count}")
    print("Pipeline   Hit@1  Hit@3  Hit@5  MRR")
    print(_metrics_line("Baseline", baseline_metrics))
    print(_metrics_line("Reranked", reranked_metrics))
    print("\nPer-question comparison")
    for question in questions:
        if question["expected_status"] != "answered":
            continue
        baseline_rank = first_relevant_rank(
            baseline_results[question["id"]], question["relevant_document_ids"]
        )
        reranked_rank = first_relevant_rank(
            reranked_results[question["id"]], question["relevant_document_ids"]
        )
        print(
            f"{question['id']}: expected={','.join(question['relevant_document_ids'])} "
            f"baseline={_rank_label(baseline_rank)} reranked={_rank_label(reranked_rank)} "
            f"{_comparison_label(baseline_rank, reranked_rank)}"
        )

    print("\nInsufficient-evidence inspection")
    for question in questions:
        if question["expected_status"] == "insufficient_evidence":
            candidates = inspections[question["id"]]
            summary = ", ".join(
                f"{candidate.source_name}:p{candidate.page_start}" for candidate in candidates
            )
            print(f"{question['id']}: {summary}")

    print("\nReturn-policy version checks")
    for question_id, both_retrieved, old_survives, expected_survives in version_checks:
        print(
            f"{question_id}: both_raw={both_retrieved} "
            f"v1_survives={old_survives} v2_survives={expected_survives}"
        )


def _metrics_line(name: str, metrics: Metrics) -> str:
    return (
        f"{name:<10} {metrics.hit_at_1:.3f}  {metrics.hit_at_3:.3f}  "
        f"{metrics.hit_at_5:.3f}  {metrics.mrr:.3f}"
    )


def _rank_label(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def _comparison_label(baseline_rank: int | None, reranked_rank: int | None) -> str:
    if baseline_rank == reranked_rank:
        return "unchanged"
    if baseline_rank is None:
        return "improved" if reranked_rank is not None else "unchanged"
    if reranked_rank is None:
        return "worsened"
    return "improved" if reranked_rank < baseline_rank else "worsened"


if __name__ == "__main__":
    run_evaluation()
