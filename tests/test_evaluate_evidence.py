from eval.evaluate_evidence import calculate_status_metrics, run_evaluation


def test_calculates_status_accuracy_per_class_and_confusion_counts() -> None:
    questions = [
        {"id": "answered", "expected_status": "answered"},
        {"id": "insufficient", "expected_status": "insufficient_evidence"},
        {"id": "conflict", "expected_status": "conflict"},
    ]
    predictions = {
        "answered": "insufficient_evidence",
        "insufficient": "insufficient_evidence",
        "conflict": "conflict",
    }

    metrics = calculate_status_metrics(questions, predictions)

    assert metrics.question_count == 3
    assert metrics.overall_accuracy == 2 / 3
    assert metrics.per_class_accuracy == {
        "answered": 0.0,
        "insufficient_evidence": 1.0,
        "conflict": 1.0,
    }
    assert metrics.confusion["answered"] == {
        "answered": 0,
        "insufficient_evidence": 1,
        "conflict": 0,
    }


def test_per_class_accuracy_is_not_available_without_expected_examples() -> None:
    questions = [{"id": "answered", "expected_status": "answered"}]

    metrics = calculate_status_metrics(questions, {"answered": "answered"})

    assert metrics.per_class_accuracy["answered"] == 1.0
    assert metrics.per_class_accuracy["insufficient_evidence"] is None
    assert metrics.per_class_accuracy["conflict"] is None


def test_live_evaluation_skips_without_api_key(monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    report = run_evaluation()

    assert report is None
    assert "canlı kanıt değerlendirmesi atlandı" in capsys.readouterr().out
