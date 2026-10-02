import json
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://127.0.0.1:8010/ask"

TEST_CASES = [
    {
        "id": 1,
        "category": "normal",
        "question": "Param ne zaman hesabıma geri yatar?",
        "expected_status": "answered",
    },
    {
        "id": 2,
        "category": "normal",
        "question": "Şifremi unuttum, nasıl sıfırlayabilirim?",
        "expected_status": "answered",
    },
    {
        "id": 3,
        "category": "normal",
        "question": "Hasarlı gelen ürün için ne yapmalıyım?",
        "expected_status": "answered",
    },
    {
        "id": 4,
        "category": "normal",
        "question": "Faturamı nereden bulabilirim?",
        "expected_status": "answered",
    },
    {
        "id": 5,
        "category": "normal",
        "question": "İade talebini nasıl oluşturabilirim?",
        "expected_status": "answered",
    },
    {
        "id": 6,
        "category": "normal",
        "question": "Standart teslimat ne kadar sürer?",
        "expected_status": "answered",
    },
    {
        "id": 7,
        "category": "version",
        "question": "İade kodu ne kadar süre geçerli?",
        "expected_status": "answered",
        "expected_contains": ["10"],
    },
    {
        "id": 8,
        "category": "conflict",
        "question": "Ürünü kaç gün içinde iade edebilirim?",
        "expected_status": "conflict",
    },
    {
        "id": 9,
        "category": "unanswerable",
        "question": "Uluslararası gönderim yapıyor musunuz?",
        "expected_status": "insufficient_evidence",
    },
    {
        "id": 10,
        "category": "unanswerable",
        "question": "Üyelik ücreti ne kadar?",
        "expected_status": "insufficient_evidence",
    },
]


def ask(question: str) -> dict:
    body = json.dumps(
        {"question": question},
        ensure_ascii=False
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def evaluate(case: dict, response: dict) -> tuple[bool, list[str]]:
    errors = []

    actual_status = response.get("status")

    if actual_status != case["expected_status"]:
        errors.append(
            f"Beklenen status={case['expected_status']}, "
            f"gerçek={actual_status}"
        )

    answer = response.get("answer", "")

    for expected_text in case.get("expected_contains", []):
        if expected_text.lower() not in answer.lower():
            errors.append(
                f"Cevapta beklenen ifade bulunamadı: {expected_text}"
            )

    return len(errors) == 0, errors


def main():
    results = []

    passed = 0

    for case in TEST_CASES:
        print("=" * 80)
        print(f"[{case['id']}/10] {case['question']}")

        try:
            response = ask(case["question"])

            success, errors = evaluate(case, response)

            if success:
                passed += 1
                result_text = "PASS"
            else:
                result_text = "FAIL"

            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_status": case["expected_status"],
                "actual_status": response.get("status"),
                "answer": response.get("answer"),
                "sources": response.get("sources", []),
                "result": result_text,
                "errors": errors,
            }

            results.append(result)

            print(f"Sonuç : {result_text}")
            print(f"Status : {response.get('status')}")
            print(f"Cevap  : {response.get('answer')}")

            sources = response.get("sources", [])

            if sources:
                print("Kaynaklar:")

                for source in sources:
                    print(
                        f"  - {source.get('source_name')} "
                        f"(Sayfa {source.get('page_start')}"
                        + (
                            f"-{source.get('page_end')}"
                            if source.get("page_end") != source.get("page_start")
                            else ""
                        )
                        + ")"
                    )

            if errors:
                print("Kontrol hataları:")
                for error in errors:
                    print(f"  - {error}")

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")

            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "result": "ERROR",
                    "error": f"HTTP {exc.code}: {body}",
                }
            )

            print(f"ERROR: HTTP {exc.code}")
            print(body)

        except Exception as exc:
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "result": "ERROR",
                    "error": str(exc),
                }
            )

            print(f"ERROR: {exc}")

    output_path = Path("eval/final_test_results.json")

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FINAL TEST")
    print("=" * 80)
    print(f"PASS: {passed}/{len(TEST_CASES)}")
    print(f"Sonuç dosyası: {output_path}")


if __name__ == "__main__":
    main()