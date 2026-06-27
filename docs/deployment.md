# P20 Text Search within Images — Deployment

> Package `imgtextsearch` · folder `20_Text_Search_in_Images` · author Le Dinh Minh Quan (23127460).
> Task: search a **collection of images by the text they CONTAIN** (via OCR) — *"find images containing the word INVOICE"*, *"receipts mentioning a 2023 date"*. This is OCR-based **document** search, not the visual/semantic search of the sibling P19 (`clipsearch`).

This document covers how to serve P20: the FastAPI service, the Gradio UI, Docker (with the OCR system dependencies), the Hugging Face Space, CPU-vs-GPU placement, the OCR-index build/refresh lifecycle, latency, environment configuration, and a concrete request/response pair.

---

## 1. What gets served

P20 is served as **one OCR-indexed collection plus a query endpoint**. The deployable unit is:

1. an **image collection** (scanned docs, receipts, forms, scene-text photos, born-digital PDF pages),
2. **OCR'd once at startup** into a per-image text "document" (`image_id`, OCR text, per-word boxes, mean confidence),
3. wrapped in the **hybrid index** (BM25 lead arm + dense bi-encoder arm → RRF), and
4. exposed through the **deterministic D1–D5 agent** that parses the query, searches, checks coverage, **verifies the literal term**, extracts a highlighted **snippet**, and **abstains** when no image contains the query.

The value-add over a blind semantic ranker — **literal-term verification + snippet highlighting + abstention** — lives entirely behind the `/search` endpoint and the Gradio UI, so every deployment surface returns honest "which images literally contain this text" answers, not "which images are vaguely about this topic."

The collection is **OCR-indexed once at startup** and held in memory. A query never triggers OCR; only the small query string is encoded/tokenized at request time. Re-indexing is an explicit, separate operation (see §6).

---

## 2. FastAPI service

The API mirrors the sibling templates (P18/P19) and exposes a small, stable surface. Default bind `0.0.0.0:8000`.

### 2.1 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness + readiness. Returns build/config + whether the OCR index is loaded. |
| `GET` | `/stats` | Collection stats: image count, OCR engine in use, dense model id, mean OCR confidence, fraction of images flagged `low_ocr_confidence`. |
| `POST` | `/search` | The core call: **query → ranked image ids + scores + matching OCR snippet + exact-match flag**, or an explicit abstention. |
| `POST` | `/reindex` | **Admin/optional, OFF by default.** Rebuild or refresh the OCR index from the configured collection (see §6). Guarded by `IMGTS_ENABLE_REINDEX` + an admin token. |

#### `GET /healthz`

Returns `200` once the collection is OCR-indexed and both index arms are ready. Until then it reports `"ready": false` so an orchestrator does not route traffic to a server that is still OCR-ing the collection (cold start can be long — see §7).

```json
{
  "status": "ok",
  "ready": true,
  "version": "0.1.0",
  "ocr_engine": "tesseract",
  "dense_model": "BAAI/bge-small-en-v1.5",
  "dense_enabled": true,
  "n_images_indexed": 1024,
  "device": "cpu"
}
```

When the dense arm is unavailable (no `sentence-transformers` / no GPU and `IMGTS_DENSE_ENABLED=0`), `dense_enabled` is `false` and the service runs **BM25-only** — still fully functional for literal-term search, since BM25 is the lead arm in P20.

#### `POST /search`

Request body:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `query` | string | — (required) | The word / date / number / ID / short phrase to find inside images. |
| `top_k` | int | `10` | Max images to return. |
| `exact_only` | bool | `false` | If `true`, drop any candidate that does not **literally** contain the term (no fuzzy survivors). |
| `fuzzy` | bool | `true` | Allow bounded edit-distance ≤ 1 matches at D4 to tolerate OCR noise (`INV0ICE`, `rn↔m`, `0↔O`). |
| `rerank` | bool | `false` | Apply the optional cross-encoder rerank over the fused top-k. |
| `return_boxes` | bool | `true` | Include per-word bounding boxes for the highlighted snippet. |

Response body (top-level):

| Field | Type | Meaning |
|---|---|---|
| `query` | string | Echo of the normalized query. |
| `intent` | string | D2 classification: `exact_code` / `keyword` / `phrase` (drives BM25-vs-dense weighting + whether literal verify is *required*). |
| `abstained` | bool | `true` when **no** image literally contains the query → D5 abstention. |
| `needs_review` | bool | `true` when results are only fuzzy / low-OCR-confidence survivors, or when abstained. |
| `results` | array | Ranked images (empty when `abstained`). |
| `trace` | array | The D1–D5 `ToolTrace` (decision id, branch, signal value, threshold) — a deterministic, replayable audit. |
| `latency_ms` | number | Server-side wall time. |

Each element of `results`:

| Field | Type | Meaning |
|---|---|---|
| `image_id` | string | The image identifier (filename / collection id). |
| `score` | number | The fused **RRF** score used for ranking. |
| `exact_match` | bool | **`true` iff the query LITERALLY appears** in this image's OCR text (case/diacritic-insensitive). `false` ⇒ fuzzy survivor. |
| `fuzzy_match` | bool | `true` when the term matched only via bounded edit distance (flagged, uncertain). |
| `snippet` | string | The matching OCR text span, with the matched term marked (`**…**`) for highlighting. |
| `boxes` | array | Per-word `[x0,y0,x1,y1]` of the matched span (when `return_boxes`), so a UI can draw a box on the image. |
| `ocr_confidence` | number | Mean OCR confidence in `[0,1]` for that image (drives the high / medium / weak label). |
| `confidence_label` | string | `high` / `medium` / `weak`. |

**Exact-match-first ordering.** Within the returned list, images whose OCR text literally contains the term (`exact_match: true`) are placed ahead of fuzzy survivors — D4 puts exact matches first. The `exact_match` flag is the single most load-bearing field: it is what separates an honest "this image contains INVOICE" hit from a semantically-similar-but-wrong neighbor.

#### `POST /reindex` (note)

`/reindex` is **disabled by default**. P20 OCR-indexes a *collection*, and re-OCR-ing is an expensive batch job (see §7), not a per-request action. When enabled (`IMGTS_ENABLE_REINDEX=1` + a valid `X-Admin-Token`), it rebuilds the BM25 + dense index from the configured collection path and atomically swaps it in. Until the swap completes, `/search` continues serving the previous index, and `/healthz` reports `"reindexing": true`. For most deployments, prefer rebuilding the index offline and shipping it as an artifact (see §6) over enabling a live `/reindex`.

### 2.2 Errors and abstention

- **Abstention is a `200`, not an error.** When no image literally contains the query, the response is `200` with `"abstained": true`, `"results": []`, and `"needs_review": true`. A nonexistent term (e.g. `zzqwx`) returns this verified-empty answer — the offline seed confirms it correctly abstains.
- `422` — empty / pure-punctuation / sub-`min_query_tokens` query (D2 `ABSTAIN_EARLY`; no encode wasted).
- `503` — index not yet loaded (still in cold-start OCR). Clients should retry after `/healthz` reports `"ready": true`.

---

## 3. Gradio UI

A thin demo UI (`app.py`) on top of the same agent — **type a query → see the matching images + highlighted snippets**.

- **Query box** + `Search` button; toggles for `exact_only`, `fuzzy`, `rerank`, and a `top_k` slider.
- **Results gallery:** each hit shows the **image thumbnail**, the **highlighted snippet** (matched term emphasized), the **exact-match badge** (literal vs fuzzy), the OCR-confidence label, and the RRF score. When `return_boxes` is on, the matched span's bbox is drawn on the thumbnail.
- **Abstention state:** when the agent abstains, the UI shows a clear *"No image in the collection contains this text"* panel with `needs_review` rather than a misleading gallery of "least irrelevant" images.
- **Trace panel (collapsible):** renders the D1–D5 `ToolTrace` so a reviewer can see why each decision fired (e.g. *D3 raw-cosine below `tau_soft` → widened once*; *D4 dropped 2 semantic false-positives*; *D5 abstain*).
- Mirrors P19's Gradio layout; the difference is that hits are accompanied by **OCR snippets + exact-match flags**, not opaque neighbor ids.

The UI calls the in-process agent directly (no HTTP hop) on a Space, or the `/search` endpoint when pointed at a remote API via `IMGTS_API_URL`.

---

## 4. Docker

OCR introduces **system-level dependencies** beyond `pip` — the image must carry the Tesseract binary, fonts (for the synthetic renderer and for non-default scripts), and the OpenGL/GLib shared libs that OpenCV-based OCR backends (PaddleOCR/EasyOCR/docTR) link against.

### 4.1 System dependencies

| Dependency | Why it is needed |
|---|---|
| `tesseract-ocr` | The **default OCR engine** (`pytesseract.image_to_data`). Without it, the service falls back to SeedEngine (synthetic, spec-embedded images) or the BM25-only stub — no real-image OCR. |
| `tesseract-ocr-eng` | English language pack (P20's primary supported language). Add `tesseract-ocr-<lang>` only for research-only non-EN runs. |
| `libgl1` | `libGL.so.1` — required by OpenCV (`cv2`), pulled in by the PaddleOCR/EasyOCR/docTR upgrade paths. The classic `ImportError: libGL.so.1` on slim images. |
| `libglib2.0-0` | GLib runtime, also an OpenCV transitive dep. |
| `fonts-dejavu-core` (+ optional `fonts-liberation`, `fonts-noto-core`) | TrueType fonts for the **synthetic rendered-text generator** (`render_page` discovers a TTF, falls back to `load_default`). Fonts also matter for any non-Latin script rendering. |
| `poppler-utils` *(optional)* | PDF rasterization fallback alongside PyMuPDF for the born-digital-vs-scanned router. |

### 4.2 Dockerfile sketch

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng \
        libgl1 libglib2.0-0 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV TESSERACT_CMD=/usr/bin/tesseract \
    IMGTS_OCR_ENGINE=tesseract \
    IMGTS_DENSE_MODEL=BAAI/bge-small-en-v1.5 \
    IMGTS_DEVICE=cpu \
    HF_HOME=/models

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Pre-cache the dense model into the image so cold start does not hit the network.
RUN python -m imgtextsearch.tools.prefetch_models || true

EXPOSE 8000
CMD ["uvicorn", "imgtextsearch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:
- `--no-install-recommends` keeps the image small; the four runtime libs above are the minimum for real-image OCR.
- Pre-caching the dense model into `HF_HOME` avoids a network fetch on first request and keeps the container runnable **offline**.
- The Tesseract binary is found via `TESSERACT_CMD` (`pytesseract.pytesseract.tesseract_cmd`); leave it default on Linux.
- If you only ever serve the **synthetic / spec-embedded** offline collection or BM25-only, you can drop `tesseract-ocr` and `libgl1` — SeedEngine reads the embedded gold text with no OCR binary, the documented P15/P18/P19 zero-heavy-dep contract.

### 4.3 GPU image

For the dense-on-GPU tier, base on `nvidia/cuda:12.x-runtime` (or an official PyTorch CUDA image), install the same OCR/system libs, install the CUDA build of `torch`, and set `IMGTS_DEVICE=cuda`. Run with `--gpus all`. OCR (Tesseract) and BM25 stay on CPU regardless (see §5).

---

## 5. GPU vs CPU serving

P20 is deliberately CPU-viable; the GPU only accelerates the **one trained component** and the optional reranker.

| Component | Placement | Rationale |
|---|---|---|
| **OCR (Tesseract)** | **CPU always** | CPU-bound binary; no GPU path. The dominant cold-start cost (see §7). |
| **BM25 sparse arm** | **CPU always** | Pure-python `BM25Index` (reused from P19 `lexical.py`); the **lead** arm for literal-term search; no GPU. |
| **RRF fusion + D1–D5 agent + snippet verifier** | **CPU always** | Rank arithmetic + string containment / bounded edit distance; trivially fast. |
| **Dense bi-encoder** (`bge-small-en-v1.5`) | **GPU if available, else CPU** | The trained retriever. GPU helps the **one-time index embedding pass** most; per-query encode of a short string is cheap on CPU too. |
| **Cross-encoder reranker** (optional) | **GPU preferred** | Heaviest per-query model; only enabled when latency budget allows; disabled on CPU-only / tight-latency tiers. |
| Optional `byt5-small` post-OCR corrector | **GPU preferred** | Applied at index build, before indexing; skipped on T4/CPU tiers. |

**CPU-only deployment is first-class.** With `IMGTS_DEVICE=cpu`, the dense arm still runs (slower index build); or set `IMGTS_DENSE_ENABLED=0` to serve **BM25-only**, which in P20 is a genuinely strong configuration because query and OCR text share the same lexical space. Either way the service is fully functional — only OCR-noise/paraphrase recall degrades when the dense arm is off.

**GPU deployment** keeps OCR + BM25 on CPU and offloads the dense embedding (and rerank/byt5 if enabled) to the GPU. The biggest GPU win is **batch-embedding the whole collection at index build**, not per-query latency.

---

## 6. OCR-index build and refresh

The OCR index is the deployable artifact. Two lifecycle models:

### 6.1 Build at startup (default for small/synthetic collections)

On boot the service:
1. **Ingests + routes** each item (D1): PyMuPDF checks PDF pages for a real text layer ≥ char threshold → read text directly, **skip OCR**; raster images and text-layer-less pages take the OCR path.
2. **OCRs** each image (Tesseract `image_to_data` → per-word text + bbox + confidence; or SeedEngine for spec-embedded synthetic images). Empty/unreadable images are **excluded and flagged**; `low_ocr_confidence` (< floor, e.g. 0.5) images are kept but tagged for cautious handling at D4/D5.
3. **Builds both arms:** `BM25Index` over the per-image OCR text, and (when the dense model is available) the dense `ImageIndex` over OCR-text embeddings.
4. Flips `/healthz` to `"ready": true`.

`/healthz` stays `"ready": false` during this pass, which can be long for large collections — keep readiness probes patient.

### 6.2 Pre-built index artifact (recommended for large collections)

For large or production collections, **build the index offline** (a CLI / batch job — ideal on the GPU tier to batch-OCR and batch-embed the whole collection in one pass) and ship it as an artifact:
- BM25 postings + doc-length stats,
- the dense vectors (FAISS `IndexFlatIP` on L2-normalized embeddings, or numpy fallback),
- the per-image OCR text, word boxes, confidences, and `image_id` map.

The service then **loads** the artifact at startup (fast, no OCR) and serves immediately. This is the preferred path because OCR-ing 400K-scale collections (e.g. RVL-CDIP) is a heavy batch job, not a startup task.

### 6.3 Refresh

- **Recommended:** rebuild the artifact offline and redeploy / hot-swap. Deterministic, reviewable, no live OCR load on the serving box.
- **Live `/reindex`:** OFF by default; when enabled it re-runs the build over the configured collection and atomically swaps the index while continuing to serve the old one. Use only for small, trusted, internal collections.
- **Index scaling:** `IndexFlatIP` is exact but O(N) per query; for large N, swap to an ANN index (HNSW/IVF) behind the same `retrieve()` interface. BM25-only remains the no-GPU floor.

---

## 7. Latency and cold start

**Cold start is dominated by OCR**, not by model load:
- Loading the dense model (`bge-small`, 33.4M params) is seconds.
- OCR-ing the collection scales linearly with image count and resolution — hundreds of ms to a few seconds **per image** on CPU Tesseract. A few thousand images can mean minutes of startup. This is why §6.2 (pre-built artifact) and a **patient `/healthz` readiness gate** matter.

**Per-query latency** (warm, index in memory) is small and CPU-friendly:
- BM25 lookup + RRF + snippet verify: low-ms.
- Dense query: one short-string encode (sub-10 ms GPU, tens of ms CPU) + an exact `IndexFlatIP` dot-product over N.
- Optional cross-encoder rerank over the top-k is the heaviest add — keep it off under tight latency, on when accuracy matters.

A query never triggers OCR; the OCR cost is paid once at index build.

---

## 8. Configuration (environment variables)

All deploy-time knobs are env vars (12-factor; same pattern as the sibling templates).

| Variable | Default | Purpose |
|---|---|---|
| `IMGTS_HOST` | `0.0.0.0` | API bind host. |
| `IMGTS_PORT` | `8000` | API bind port. |
| `IMGTS_COLLECTION_PATH` | `data/collection` | Directory of images / PDFs to OCR-index. |
| `IMGTS_INDEX_PATH` | *(unset)* | Path to a **pre-built index artifact** (§6.2). If set, load it instead of OCR-ing at startup. |
| `IMGTS_OCR_ENGINE` | `tesseract` | `tesseract` / `trocr` / `paddle` / `easyocr` / `seed` (offline) / `stub`. Surya is **not** a default (CC-BY-NC-SA — flagged, research-only). |
| `TESSERACT_CMD` | `/usr/bin/tesseract` | Path to the Tesseract binary. |
| `IMGTS_OCR_LANG` | `eng` | Tesseract language pack(s). EN is the supported target; others are research-only. |
| `IMGTS_DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | Dense retriever id (MIT). Upgrade `bge-base-en-v1.5`; T4 fallback `all-MiniLM-L6-v2`. |
| `IMGTS_DENSE_ENABLED` | `1` | `0` → serve **BM25-only**. |
| `IMGTS_DEVICE` | `cpu` | `cpu` / `cuda` for the dense + rerank + byt5 models (OCR/BM25 stay on CPU). |
| `IMGTS_RERANK` | `0` | Enable the optional `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank. |
| `IMGTS_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker id (Apache-2.0). |
| `IMGTS_POSTOCR_CORRECT` | `0` | Enable `google/byt5-small` post-OCR correction before indexing. |
| `IMGTS_RRF_C` | `60` | RRF constant `c`. |
| `IMGTS_TOP_K` | `10` | Default `top_k`. |
| `IMGTS_FUZZY` | `1` | Default bounded-fuzzy (edit distance ≤ 1) verification at D4. |
| `IMGTS_TAU_SOFT` / `IMGTS_TAU_FLOOR` | tuned | D3 raw-cosine coverage thresholds (gate on **raw** cosine, never the fused RRF score — the P08/P18 gotcha). |
| `IMGTS_MIN_QUERY_TOKENS` | `1` | D2 early-abstain floor. |
| `IMGTS_OCR_CONF_FLOOR` | `0.5` | Below this, an image is tagged `low_ocr_confidence`. |
| `IMGTS_ENABLE_REINDEX` | `0` | Enable the `/reindex` admin endpoint. |
| `IMGTS_ADMIN_TOKEN` | *(unset)* | Required `X-Admin-Token` for `/reindex`. |
| `IMGTS_LLM_BRAIN` | `0` | Optional advisory Anthropic LLM (query-expansion note only; **OFF by default, never changes ranking**). |
| `ANTHROPIC_API_KEY` | *(unset)* | Only read when `IMGTS_LLM_BRAIN=1`. |
| `IMGTS_RESEARCH_ONLY` | `0` | Gate that must be `1` to load non-commercial / unlicensed datasets (FUNSD, RVL-CDIP, DocVQA-1200, COCO-Text) or Surya weights. |
| `HF_HOME` | `~/.cache/huggingface` | Model cache; pre-populate in the image for offline serving. |

**Privacy defaults.** Query retention is **off** by default; the LLM brain is **off** by default. OCR-indexing user/scanned documents is sensitive (IDs, contracts, medical/financial PII inside the text) — run behind access control, enable PII redaction where applicable, and keep `IMGTS_LLM_BRAIN=0` unless a reviewer has approved it. OCR errors can cause false negatives (a document that *does* contain the term is missed) or false positives; the fuzzy verify + abstention mitigate, and OCR quality varies by script/font/scan quality (a bias source → unequal recall).

---

## 9. Request / response example

**Request**

```bash
curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "INVOICE", "top_k": 5, "exact_only": false, "fuzzy": true}'
```

**Response (hit)**

```json
{
  "query": "invoice",
  "intent": "keyword",
  "abstained": false,
  "needs_review": false,
  "results": [
    {
      "image_id": "page_0042.png",
      "score": 0.0328,
      "exact_match": true,
      "fuzzy_match": false,
      "snippet": "CUSTOMER 1187   **INVOICE**   DATE 2023-04-12",
      "boxes": [[212, 96, 388, 140]],
      "ocr_confidence": 0.94,
      "confidence_label": "high"
    },
    {
      "image_id": "page_0117.png",
      "score": 0.0161,
      "exact_match": false,
      "fuzzy_match": true,
      "snippet": "TOTAL 1,250.00   **INV0ICE** NO PO-4471",
      "boxes": [[64, 220, 240, 262]],
      "ocr_confidence": 0.71,
      "confidence_label": "medium"
    }
  ],
  "trace": [
    {"decision": "D1", "branch": "scanned->tesseract", "n_indexed": 1024},
    {"decision": "D2", "branch": "exact_intent", "intent": "keyword", "tokens": 1},
    {"decision": "D3", "branch": "strong_hit", "bm25_literal_hits": 7, "raw_cosine_top": 0.81},
    {"decision": "D4", "branch": "verified", "exact": 1, "fuzzy": 1, "dropped_semantic_fp": 3},
    {"decision": "D5", "branch": "finalize", "n_verified": 2}
  ],
  "latency_ms": 23.7
}
```

The first result is an **exact literal match** (`exact_match: true`, ordered first); the second is a **fuzzy survivor** recovered from an OCR confusion (`INV0ICE`), flagged `fuzzy_match` with `medium` confidence. D4 dropped 3 semantically-similar images where the word `INVOICE` never actually appears — the core filter that beats blind semantic ranking.

**Response (abstention)**

```bash
curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "zzqwx"}'
```

```json
{
  "query": "zzqwx",
  "intent": "exact_code",
  "abstained": true,
  "needs_review": true,
  "results": [],
  "trace": [
    {"decision": "D2", "branch": "exact_intent", "intent": "exact_code", "tokens": 1},
    {"decision": "D3", "branch": "no_literal_hit", "bm25_literal_hits": 0, "raw_cosine_top": 0.22},
    {"decision": "D5", "branch": "abstain", "n_verified": 0}
  ],
  "latency_ms": 9.1
}
```

No image's OCR text contains `zzqwx`, so the agent **abstains** (`200`, `abstained: true`, `needs_review: true`) rather than returning the least-irrelevant image — the honest "not found in any image" answer that distinguishes OCR document search from a semantic neighbor lookup.

---

## 10. Deployment checklist

- [ ] System deps installed: `tesseract-ocr`, `tesseract-ocr-eng`, `libgl1`, `libglib2.0-0`, a TTF font package.
- [ ] `TESSERACT_CMD` resolves to the binary; `IMGTS_OCR_LANG=eng`.
- [ ] Dense model cached into `HF_HOME` for offline cold start (or `IMGTS_DENSE_ENABLED=0` for BM25-only).
- [ ] `IMGTS_DEVICE=cuda` + `--gpus all` only if a GPU is present; OCR + BM25 stay on CPU regardless.
- [ ] Large collection → ship a **pre-built index artifact** (`IMGTS_INDEX_PATH`); small/synthetic → build at startup.
- [ ] Readiness probe points at `/healthz` and tolerates a long cold start (OCR-bound).
- [ ] Privacy: query retention off, `IMGTS_LLM_BRAIN=0`, access control + PII redaction for sensitive document collections.
- [ ] `IMGTS_RESEARCH_ONLY=0` in any commercial deployment (excludes FUNSD / RVL-CDIP / DocVQA-1200 / COCO-Text / Surya).
- [ ] `/reindex` left disabled unless an admin token is set and the collection is small and trusted.

---

## 11. Hugging Face Space

A Gradio Space gives a public, zero-install demo:
- **SDK:** Gradio; entrypoint `app.py` runs the in-process agent (no separate API hop).
- **System packages:** add `tesseract-ocr`, `tesseract-ocr-eng`, `libgl1`, `libglib2.0-0`, `fonts-dejavu-core` via the Space's `packages` / `apt.txt`, mirroring the Dockerfile in §4.
- **Default collection:** ship the **small synthetic rendered-text collection** (the offline generator) so the Space runs with **no Tesseract dependency on the gold text** — SeedEngine reads the embedded spec, and the demo works even where OCR binaries are unavailable. This keeps the Space within free-tier limits and gives a deterministic, license-clean demo (MIT/CC0/CC-BY-4.0/synthetic only).
- **Hardware:** CPU basic is sufficient (BM25 + small dense model on CPU). Upgrade to a GPU Space only to demo `bge-base` / rerank / a larger real collection.
- **Secrets:** set `ANTHROPIC_API_KEY` only if demonstrating the advisory LLM brain (off by default). Never bake sensitive real-document collections into a public Space — demo on synthetic + the commercially-safe corpora only.
