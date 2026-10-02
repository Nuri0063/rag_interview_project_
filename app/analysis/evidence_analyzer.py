"""OpenAI-backed validation of retrieved evidence candidates."""

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from app.retrieval.retriever import RetrievalResult


DEFAULT_MODEL = "gpt-4.1-mini"


class EvidenceAnalysisResult(BaseModel):
    """The structured decision made from the supplied evidence only."""

    status: Literal["answered", "insufficient_evidence", "conflict"]
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    conflicting_chunk_ids: list[str] = Field(default_factory=list)
    reason: str


class EvidenceAnalyzerError(RuntimeError):
    """Raised when an evidence analysis cannot be completed safely."""


class EvidenceAnalyzer:
    """Ask OpenAI to assess whether retrieved chunks directly support a question."""

    def __init__(self, model: str | None = None, client: OpenAI | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = client

    def analyze(
        self,
        question: str,
        candidates: list[RetrievalResult],
    ) -> EvidenceAnalysisResult:
        """Return a validated evidence decision for the supplied candidates."""
        if not candidates:
            return EvidenceAnalysisResult(
                status="insufficient_evidence",
                reason="Analiz edilecek aday kanıt bulunamadı.",
            )

        client = self._get_client()
        try:
            response = client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
                    {
                        "role": "user",
                        "content": _build_user_input(question, candidates),
                    },
                ],
                text_format=EvidenceAnalysisResult,
            )
        except Exception as error:
            raise EvidenceAnalyzerError("OpenAI Responses API kanıt analizi sırasında hata verdi.") from error

        try:
            result = EvidenceAnalysisResult.model_validate(response.output_parsed)
        except (AttributeError, ValidationError) as error:
            raise EvidenceAnalyzerError("OpenAI yanıtı geçerli yapılandırılmış kanıt sonucu içermiyor.") from error

        _validate_chunk_ids(result, candidates)
        _validate_result_semantics(result)
        return result

    def _get_client(self) -> OpenAI:
        if self.client is not None:
            return self.client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EvidenceAnalyzerError("OPENAI_API_KEY ortam değişkeni ayarlı değil.")
        self.client = OpenAI(api_key=api_key)
        return self.client


_SYSTEM_INSTRUCTIONS = """You analyze only the provided evidence chunks.

Follow this decision order exactly:
1. Identify the chunks that directly address the original user question.
2. Before deciding answered or insufficient_evidence, compare their directly
   relevant factual claims across different family_id values.
3. If directly relevant chunks from two or more different document families make
   mutually incompatible factual claims about the same requested fact, return
   conflict. Do not choose one claim as correct.
4. Return answered only when directly relevant evidence is mutually consistent,
   or when there is only one supported claim.
5. Return insufficient_evidence only when the supplied chunks do not directly
   support an answer.

For answered, supporting_chunk_ids must identify the chunks used as direct evidence.
For insufficient_evidence, both chunk-ID lists must be empty. For conflict,
conflicting_chunk_ids must identify every chunk participating in the conflicting
claims. Identify all IDs from the supplied candidates only.

Example:
Question: "How many days can I return a product?"
family A: "Returns are allowed within 14 days."
family B: "Returns are allowed within 30 days."
Expected: status=conflict and conflicting_chunk_ids=[both chunks].

Do not use outside knowledge. Same-family version resolution has already happened.
Do not choose document versions, decide authority, or prefer any document type.
Do not use version, publication date, retrieval score, reranker score, document
type, source name, or perceived authority to resolve cross-family conflict."""


def _build_user_input(question: str, candidates: list[RetrievalResult]) -> str:
    evidence = [
        {
            "chunk_id": candidate.chunk_id,
            "content": candidate.content,
            "family_id": candidate.family_id,
            "document_id": candidate.document_id,
        }
        for candidate in candidates
    ]
    return json.dumps(
        {"question": question, "candidates": evidence},
        ensure_ascii=False,
    )


def _validate_chunk_ids(
    result: EvidenceAnalysisResult,
    candidates: list[RetrievalResult],
) -> None:
    known_ids = {candidate.chunk_id for candidate in candidates}
    returned_ids = set(result.supporting_chunk_ids) | set(result.conflicting_chunk_ids)
    unknown_ids = sorted(returned_ids - known_ids)
    if unknown_ids:
        raise EvidenceAnalyzerError(
            "OpenAI yanıtı adaylarda bulunmayan chunk kimlikleri döndürdü: "
            + ", ".join(unknown_ids)
        )


def _validate_result_semantics(result: EvidenceAnalysisResult) -> None:
    if result.status == "answered" and not result.supporting_chunk_ids:
        raise EvidenceAnalyzerError("answered sonucu en az bir supporting_chunk_id içermelidir.")
    if result.status == "insufficient_evidence" and (
        result.supporting_chunk_ids or result.conflicting_chunk_ids
    ):
        raise EvidenceAnalyzerError(
            "insufficient_evidence sonucu supporting veya conflicting chunk kimliği içeremez."
        )
    if result.status == "conflict" and not result.conflicting_chunk_ids:
        raise EvidenceAnalyzerError("conflict sonucu en az bir conflicting_chunk_id içermelidir.")
