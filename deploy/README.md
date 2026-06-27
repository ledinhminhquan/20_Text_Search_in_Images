# Deployment — Text Search within Images (`imgtextsearch`)

Three ways to serve the OCR-search system. The collection is OCR-indexed once at startup; queries
return ranked image ids + scores + the matching OCR snippet + an exact-match flag, and **"not found"**
when no image contains the query text.

## 1. Local (FastAPI + Gradio)

```bash
pip install -e ".[ocr,api,report]"
imgtextsearch serve --ui          # API at :8000, demo UI at :8000/ui
```

```bash
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "invoice", "k": 5}'
```

Response (abridged):

```json
{
  "query": "invoice", "query_kind": "keyword",
  "results": [
    {"id": "doc_0007", "score": 3.21, "snippet": "... TOTAL INVOICE 2023 ...",
     "exact_match": true, "rank": 1}
  ],
  "n_exact": 4, "abstained": false, "low_confidence": false,
  "model_version": "...", "status": "completed"
}
```

## 2. Docker

```bash
docker compose up --build        # bundles tesseract-ocr + fonts + libGL
```

`IMGTEXT_USE_HF=1` (+ `IMGTEXT_*` config) indexes a real dataset instead of the synthetic collection.

## 3. Hugging Face Space (Gradio)

Point a Gradio Space at [`app/app.py`](../app/app.py). Set `IMGTEXT_LOAD_MODEL=0` to force the
offline BM25 + SeedEngine path on CPU-only Spaces.

## Environment variables

| Var | Purpose |
|-----|---------|
| `IMGTEXT_ARTIFACTS_DIR` | root for models / runs / logs |
| `IMGTEXT_MODEL_DIR` | trained retriever location |
| `IMGTEXT_USE_HF` | `1` to index a real HF dataset (default synthetic) |
| `IMGTEXT_INFER_CONFIG` | path to a YAML config the server loads |
| `HF_HOME` | Hugging Face cache |
| `ANTHROPIC_API_KEY` | only if the optional LLM brain is enabled (off by default) |

## Notes

- **CPU vs GPU:** OCR (Tesseract) + BM25 run on CPU; the dense retriever uses a GPU when `torch`
  with CUDA is present, else falls back to BM25-only.
- **Privacy:** OCR-indexing scanned documents is sensitive — front the API with auth, do not log
  query text in production (metadata-only logging is the default), and keep the LLM brain off.
