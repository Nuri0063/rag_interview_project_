"""Cross-Encoder relevance reranking for semantic retrieval candidates."""

from sentence_transformers import CrossEncoder

from app.retrieval.retriever import RetrievalResult


MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Score query and chunk-content pairs with a multilingual Cross-Encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RetrievalResult]:
        """Return every candidate ordered by Cross-Encoder relevance."""
        if not candidates:
            return []

        scores = self.model.predict([(query, candidate.content) for candidate in candidates])
        scored_candidates = [
            candidate.model_copy(update={"rerank_score": float(score)})
            for candidate, score in zip(candidates, scores)
        ]
        return sorted(scored_candidates, key=lambda candidate: candidate.rerank_score, reverse=True)
