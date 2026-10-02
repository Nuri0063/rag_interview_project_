import pytest
from pydantic import ValidationError

from app.models.api import AskRequest, AskResponse


def test_ask_request_preserves_turkish_question() -> None:
    request = AskRequest(question="Kurulum nasıl yapılır?")

    assert request.question == "Kurulum nasıl yapılır?"


def test_ask_request_requires_question() -> None:
    with pytest.raises(ValidationError) as error:
        AskRequest()

    assert error.value.errors()[0]["loc"] == ("question",)


def test_ask_response_contains_answer_and_sources() -> None:
    response = AskResponse(status="answered", answer="Kurulum kılavuzuna bakın.", sources=[])

    assert response.status == "answered"
    assert response.answer == "Kurulum kılavuzuna bakın."
    assert response.sources == []
