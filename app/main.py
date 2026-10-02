from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analysis.evidence_analyzer import EvidenceAnalyzerError
from app.api.question_pipeline import QuestionPipeline, QuestionPipelineError
from app.generation.answer_generator import AnswerGeneratorError
from app.models.api import AskRequest, AskResponse

app = FastAPI(title="Turkish Knowledge QA")
STATIC_DIRECTORY = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the small end-user question page."""
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run the fixed source-grounded question-answering pipeline."""
    try:
        return _get_question_pipeline().ask(request.question)
    except QuestionPipelineError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (EvidenceAnalyzerError, AnswerGeneratorError) as error:
        raise HTTPException(status_code=503, detail="Soru şu anda işlenemiyor. Lütfen daha sonra tekrar deneyin.") from error


def _get_question_pipeline() -> QuestionPipeline:
    pipeline = getattr(app.state, "question_pipeline", None)
    if pipeline is None:
        pipeline = QuestionPipeline()
        app.state.question_pipeline = pipeline
    return pipeline
