import html
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_turkish_question_page() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    page = html.unescape(response.text)
    assert "Nexa Bilgi Asistanı" in page
    assert "Şirket dokümanları hakkında sorunuzu yazın." in page
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_frontend_uses_relative_ask_request_and_hides_internal_fields() -> None:
    script = (Path(__file__).parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/ask"' in script
    assert "localhost" not in script
    assert "document_id" not in script
    assert "chunk_id" not in script
    assert "rerank_score" not in script
    assert "retrieval_score" not in script


def test_static_assets_are_served() -> None:
    with TestClient(app) as client:
        css_response = client.get("/static/styles.css")
        script_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert script_response.status_code == 200
    assert "const insufficientMessage" in script_response.text
