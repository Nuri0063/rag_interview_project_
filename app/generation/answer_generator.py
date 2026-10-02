"""OpenAI-backed Turkish answer generation from selected evidence chunks."""

import json
import os

from openai import OpenAI

from app.analysis.evidence_analyzer import EvidenceAnalysisResult
from app.retrieval.retriever import RetrievalResult


DEFAULT_MODEL = "gpt-4.1-mini"
INSUFFICIENT_EVIDENCE_ANSWER = "Mevcut dokümanlarda bu soruyu yanıtlamak için yeterli bilgi bulunamadı."


class AnswerGeneratorError(RuntimeError):
    """Raised when a grounded answer cannot be generated."""


class AnswerGenerator:
    """Generate concise Turkish answers from evidence selected upstream."""

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = client

    def generate(
        self,
        question: str,
        evidence_result: EvidenceAnalysisResult,
        selected_chunks: list[RetrievalResult],
    ) -> str:
        """Return a Turkish answer without using the LLM for insufficient evidence."""
        if evidence_result.status == "insufficient_evidence":
            return INSUFFICIENT_EVIDENCE_ANSWER

        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": _build_user_input(question, evidence_result.status, selected_chunks),
                    },
                ],
            )
            answer = response.output_text.strip()
        except Exception as error:
            raise AnswerGeneratorError("OpenAI Responses API yanıt üretimi sırasında hata verdi.") from error

        if not answer:
            raise AnswerGeneratorError("OpenAI boş bir yanıt üretti.")
        return answer

    def _get_client(self) -> OpenAI:
        if self.client is not None:
            return self.client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AnswerGeneratorError("OPENAI_API_KEY ortam değişkeni ayarlı değil.")
        self.client = OpenAI(api_key=api_key)
        return self.client


_SYSTEM_INSTRUCTIONS = """Answer in Turkish using only the supplied evidence chunks.
Do not use outside knowledge, infer missing facts, or invent details, source names,
or page numbers. Keep the answer concise and useful.

For answered, give a direct answer grounded in the evidence. For conflict, clearly
describe the conflicting claims and state that the available document families
contain conflicting information. Do not select one source as correct."""


def _build_user_input(
    question: str,
    status: str,
    selected_chunks: list[RetrievalResult],
) -> str:
    evidence = [
        {"chunk_id": chunk.chunk_id, "content": chunk.content}
        for chunk in selected_chunks
    ]
    return json.dumps(
        {"question": question, "evidence_status": status, "evidence_chunks": evidence},
        ensure_ascii=False,
    )
