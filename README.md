# Setup

- Python 3.12 kullanın.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

`.env` dosyasındaki `OPENAI_API_KEY` değerini elle doldurun. `OPENAI_MODEL` için varsayılan değer `gpt-4.1-mini`dir.

# Indexing

```powershell
python -c "from app.retrieval.indexing import index_corpus; from app.retrieval.vector_store import VectorStore; print(index_corpus(VectorStore(), reset=True))"
```

# Run

```powershell
python -m uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8010
```

UI: <http://127.0.0.1:8010/>

Swagger: <http://127.0.0.1:8010/docs>

# Tests

```powershell
python -m pytest -q
```
