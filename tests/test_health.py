from fastapi.testclient import TestClient

from app.main import app
from app.models.api import AskResponse


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakePipeline:
    def ask(self, question: str) -> AskResponse:
        assert question == "Kurulum nasıl yapılır?"
        return AskResponse(
            status="answered",
            answer="Kurulum bilgisi bulundu.",
            sources=[],
        )


def test_ask_endpoint_returns_pipeline_response() -> None:
    app.state.question_pipeline = FakePipeline()
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Kurulum nasıl yapılır?"})

    del app.state.question_pipeline
    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "answer": "Kurulum bilgisi bulundu.",
        "sources": [],
    }
