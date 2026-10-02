## Technical Decisions

- `paraphrase-multilingual-MiniLM-L12-v2` embedding modeli kullanılır.
- ChromaDB kalıcı vektör deposudur.
- İlk aşama retrieval, top 16 aday getirir.
- Deterministik Version Resolver yalnızca aynı ailede sürüm karşılaştırır.
- `BAAI/bge-reranker-v2-m3` adayları yeniden sıralar.
- Evidence Analyzer'a ilk 8 aday gönderilir.
- Evidence Analyzer, yanıt üretiminden önce çalışır.
- FastAPI API ve statik arayüzü sunar.
- Kaynak ve sayfa atıfları metadata'dan üretilir.

## Known Limitations

- İngilizce belgeler ve Türkçe sorgular bazı paraphrase'lerde retrieval kalitesini azaltabilir.
- Bazı sınır durumları korumacı biçimde `insufficient_evidence` döndürebilir.
- Yalnızca metin tabanlı PDF'ler desteklenir; OCR uygulanmaz.
- Evidence analizi ve yanıt üretimi OpenAI API gerektirir.
