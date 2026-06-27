# P20 — System Architecture

**Project:** Text Search within Images (`imgtextsearch`, folder `20_Text_Search_in_Images`)
**Author:** Le Dinh Minh Quan — student 23127460

---

## 1. What this system is (and is not)

P20 searches a **collection of images by the text they CONTAIN**. Given a query that is a word, date, number, ID, or short phrase — *"find images containing the word INVOICE"*, *"receipts that mention a 2023 date"*, *"the page with PO-4471"* — it returns the ranked subset of images whose **OCR'd text literally contains that query**, each with a highlighted snippet, or it **abstains** when no image contains it.

This is **OCR-based document search**, not visual/semantic content search. The query and the indexed text live in the **same lexical space** (both are strings), so the system is built around *literal containment*, with a semantic arm only to recover OCR noise and paraphrase. This is the explicit contrast with the sibling project **P19 `clipsearch`** (CLIP text→image), where the query describes what an image *depicts* and shares no vocabulary with the pixels. Do not conflate the two: P19 indexes image embeddings; P20 indexes **per-image OCR text**.

**The "small trained surface, large deterministic core" split.** Of the whole system, exactly **one** component is trained: a **dense text retriever** (a sentence-transformers bi-encoder) fine-tuned over the OCR text. Everything else — the OCR front-end, BM25, RRF fusion, the snippet/exact-match verifier, and the agent state machine — is **pretrained or purely algorithmic**. This mirrors the lineage P07 `dococr`, P15 `imgtrans`, P18 `tkgqa`, P19 `clipsearch`.

---

## 2. The OCR-index-then-search pipeline

The system runs in two phases. **Indexing** happens once when the collection is loaded (e.g. at API startup). **Search** runs per query against the built index.

```
                      ╔════════════════════════ INDEXING (once, at startup) ════════════════════════╗
                      ║                                                                              ║
  images / PDFs ─────▶║  (1) INGEST + ROUTE ──▶ (2) OCR ──────▶ (3) BUILD TEXT INDEX               ║
  (scans, receipts,   ║   born-digital vs        per-image       per image = one "document":        ║
   forms, scene text, ║   scanned (PyMuPDF       OCR text +       • BM25  (sparse, ALWAYS)          ║
   synthetic PNGs)    ║   text-layer check)      mean conf        • dense (TRAINED bi-encoder,      ║
                      ║   text layer ≥ thr →     SeedEngine /       FAISS / numpy; optional)        ║
                      ║     read directly         Tesseract /      → ImageTextIndex (texts+metas)   ║
                      ║   else → OCR path         Stub                                               ║
                      ╚══════════════════════════════════════════════════════════════════════════════╝
                                                                            │  index in memory
                                                                            ▼
  query ─────────────▶ ┌──────────────────────── SEARCH (per query, the D1–D5 agent) ──────────────┐
  ("INVOICE",          │                                                                            │
   "a 2023 date",      │  D1 PARSE ──▶ D2 SEARCH ──▶ D3 COVERAGE ──▶ D4 VERIFY ──▶ D5 FINALIZE      │
   "PO-4471")          │  gate +       BM25 + dense   raw-score      literal-term    abstain if 0   │
                       │  classify     → RRF fuse     gate (raw,     containment +    verified;     │
                       │  (exact/      → top-k        NOT fused)     snippet extract  else ranked   │
                       │   keyword/    image cands                   + fuzzy (OCR     list +        │
                       │   phrase)     + raw score                   noise) + drop    snippets +    │
                       │                                             false-positives  flags         │
                       └────────────────────────────────────────────────────────────────────────────┘
                                                                            │
                                                                            ▼
                                                  ranked image ids + RRF scores + highlighted
                                                  OCR snippet + exact-match flag   |   ABSTAIN
                                                                                   ("not found",
                                                                                    needs_review)
```

### Phase A — Indexing (build once)

1. **Ingest + route.** For real PDFs, a PyMuPDF (`fitz`) text-layer router (reused from P07/P15) checks each page for an embedded text layer above a character threshold. Born-digital pages with real text are read directly and **skip OCR**; raster images and text-layer-less pages take the OCR path. Synthetic PNGs bypass the router (they carry an embedded gold spec instead).
2. **OCR each image.** Every image is reduced to one string of text plus a mean confidence. The engine is selected by `OcrConfig.engine` (`"auto"` by default) — see §4.
3. **Build the text index.** Each image becomes one *document* = its OCR text, paired with metadata (`image_id`, and where available word boxes / mean confidence). `ImageTextIndex.build(texts, metas)` constructs the BM25 index immediately and, if a dense encoder is present, embeds the same texts for the dense arm.

### Phase B — Search (per query, driven by the agent)

The five stages **D1–D5** are exactly the agent's decision points (§6). Parse and classify the query, run hybrid retrieval, gate on raw confidence, verify literal containment and extract the snippet, then finalize or abstain. The pipeline never returns an image just because it is *semantically near* the query — the literal verifier at D4 is the gate.

---

## 3. Module map — `src/imgtextsearch/`

```
src/imgtextsearch/
├── config.py                  AppConfig + dataclass sections (Data/Ocr/Model/Index/Agent/Serving),
│                              load_config / save_config / ensure_dirs; all defaults live here
├── logging_utils.py           get_logger — structured logging shared by every module
│
├── ocr/                       ── OCR FRONT-END (pretrained / algorithmic, NOT trained) ──
│   └── engine.py              SeedEngine · StubEngine · TesseractEngine · discover_font ·
│                              has_spec / read_spec · _noisify (CER/WER knob) ·
│                              load_ocr_engine(cfg, engine, image) → auto-select · ocr_text(...)
│
├── data/                      ── DATA: synthetic generator + samples + (optional) real corpora ──
│   ├── synth_text_images.py   make_snippet · render_text_image (PIL) · save_png_with_text (gold
│   │                          text in a PNG tEXt chunk) · generate_collection → PRIMARY offline data
│   ├── samples.py             seed_collection() / seed_queries() — tiny in-memory fixture, zero deps
│   ├── dataset.py             (HF loaders: TextOCR_OCR / cord-v2 / docvqa; off by default)
│   └── download.py            (cached HF download helpers)
│
├── models/                    ── THE TRAINABLE CORE + baselines ──
│   ├── encoder.py             DenseEncoder (the trained bi-encoder) · from_pretrained ·
│   │                          encode(texts) · load_encoder(cfg) → None if torch absent
│   ├── baseline.py            BM25Only · DenseOnly · RandomRetriever (the three baselines)
│   └── model_registry.py      make_version · write/read_metadata · update_latest_pointer ·
│                              resolve_latest — version + "latest" pointer
│
├── index/                     ── HYBRID TEXT INDEX over per-image OCR text ──
│   ├── lexical.py             tokenize · BM25Index (k1=1.5, b=0.75, pure-python) ·
│   │                          rrf_fuse(*rankings, k=60)
│   └── text_index.py          ImageTextIndex (BM25 + dense → RRF; raw-score for the agent) ·
│                              build_index(texts, metas, cfg, encoder)
│
├── training/                  ── retriever fine-tune + evaluation metrics ──
│   └── metrics.py             first_gold_rank · recall_at_k · mrr · median_rank ·
│                              retrieval_metrics · cer / wer · exact_match_precision
│
├── agent/                     ── THE OCR-SEARCH AGENT (mandatory agentic component) ──
│   ├── state.py               JobStatus · ToolTrace · Decision · JobState (replayable audit log)
│   ├── policy.py              classify_query · query_gate (D1) · coverage_gate (D3) ·
│   │                          verify_exact + _fuzzy_contains + _edit_le1 (D4) · make_snippet ·
│   │                          abstain_gate (D5)
│   ├── tools.py               tool_parse (D1) · tool_search (D2) · tool_coverage (D3) ·
│   │                          tool_verify (D4) · tool_finalize (D5) — wrap policy + index
│   ├── search_agent.py        SearchAgent (the D1→D5 driver) · build_index_from_images · get_agent
│   └── llm_orchestrator.py    LLMBrain — optional advisory note (anthropic), OFF by default
│
├── api/                       FastAPI app: POST /search → ranked ids + scores + snippet + flag
├── analysis/                  error analysis, per-query breakdowns, exact-match audits
├── autoreport/                run-report generation (metrics + config + traces → markdown)
├── monitoring/                latency / abstention-rate / OCR-confidence telemetry
├── automation/                end-to-end job runner (generate → OCR → index → eval)
└── grading/                   rubric / self-grade harness shared across the P-series
```

### Module responsibilities in detail

**`config.py` — single source of truth.** `AppConfig` aggregates six dataclasses, each owning one concern. Defaults seen in the code:

- `DataConfig` — `use_hf=False` (the synthetic generator is the spine), `collection_size=200`, `image_width=700`, `vocab_phrases=60`, `seed=42`; optional real ids `MiXaiLL76/TextOCR_OCR` (MIT), `naver-clova-ix/cord-v2` (CC-BY-4.0), OCR-text source `PleIAs/Post-OCR-Correction` (CC0), and the flagged `nielsr/docvqa_1200_examples` (**license UNSPECIFIED — research-only**).
- `OcrConfig` — `engine="auto"`, `lang="eng"`, `psm=6`, `min_word_conf=0.0`.
- `ModelConfig` — `base_model="BAAI/bge-small-en-v1.5"` (MIT, 384-d, the DEFAULT trained retriever), `retriever_fallback="sentence-transformers/all-MiniLM-L6-v2"` (Apache), MNRL fine-tune hyperparameters.
- `IndexConfig` — `top_k=20`, `use_bm25=True`, `use_dense=True`, `use_faiss=True`, `normalize=True`, `rrf_k=60`.
- `AgentConfig` — D1 `min_query_chars=2`; D3 `coverage_min_score=0.05`; D4 `require_exact=True`, `exact_match_fuzzy=True`, `snippet_window=6`; D5 `abstain_enabled=True`, `min_match_score=0.05`, `n_results=10`; LLM brain `llm_fallback_enabled=False`.
- `ServingConfig` — API title/version, model version.

**`ocr/engine.py` — the pretrained front-end.** Three engines behind one `.text(image)` contract plus an auto-selector:
- `SeedEngine` — reads the gold text **embedded in synthetic images** (`read_spec` / `has_spec`), reconstructing the per-image OCR text with **no OCR binary**. `_noisify(text, rate, seed)` injects realistic OCR confusions (`m↔rn`, `0↔O`, `1↔l/I`, …) at a controllable rate — this rate *is* the CER/WER knob and is what makes the dense arm earn its keep.
- `TesseractEngine` — the real default on Colab/H100; calls `pytesseract` (Apache-2.0). Same `.text()` contract so the index is identical offline and online.
- `StubEngine` — returns empty text when there is no spec and no binary; guarantees the pipeline never crashes.
- `load_ocr_engine(cfg, engine, image)` auto-selects: SeedEngine when the image carries a spec, else Tesseract, else Stub. `discover_font()` finds a TrueType font for rendering (falls back to PIL's default).

Neural upgrades named in the design (not the default): `microsoft/trocr-base-printed` (MIT), PaddleOCR PP-OCRv5 (Apache), docTR/EasyOCR (Apache). **`vikp/surya_rec2` is CC-BY-NC-SA-4.0 — non-commercial, excluded from the default stack (FLAG).** Optional post-OCR corrector `google/byt5-small` (Apache) cleans noisy text before indexing.

**`data/synth_text_images.py` — the PRIMARY offline data.** No public "find the image containing X" benchmark with a gold query→image map exists, so we synthesize one. `make_snippet(seed)` composes 1–3 short lines from a controlled vocabulary: rare **target needles** (`INVOICE`, `RECEIPT`, dates like `2023`, ids) and common **fillers** (`total`, `amount`, `date`, …) shared across images so target terms recur in a controlled number of images — giving a non-degenerate Recall@K/MRR and real multi-gold queries. `render_text_image(text, ...)` lays the snippet onto a PNG with a discovered font; `save_png_with_text(img, text, path)` writes the gold text into a PNG `tEXt` chunk so the SeedEngine reads it back. `generate_collection(out_dir, n_images=200, ...)` writes the collection plus the **gold map** (per target term → the image ids that contain it), which is exact and independent of OCR noise.

**`index/` — the hybrid OCR-text index (the heart of P20).**
- `lexical.py:BM25Index` — pure-python BM25 (`k1=1.5`, `b=0.75`, regex tokenizer, standard IDF/TF-saturation), reused from P19 but fed **per-image OCR text** instead of captions. In P20 BM25 is the **lead arm**: the query and the OCR text share the same lexical space, so BM25 fires on rare high-IDF tokens (`INVOICE`, `PO-4471`) and is exact-match-faithful by construction.
- `lexical.py:rrf_fuse(*rankings, k=60)` — rank-based Reciprocal Rank Fusion: `score(d) = Σ_arms 1/(k + rank_a(d))`. Rank-based fusion avoids calibrating BM25's unbounded scores against cosine's `[-1,1]`.
- `text_index.py:ImageTextIndex` — orchestrates both arms. `build(texts, metas)` always builds BM25; if an encoder is present it L2-normalizes the embeddings for the dense arm (and silently falls back to BM25-only if embedding fails). `search(query, top_k)` returns `(fused_ranking, raw_top)`: when both arms exist it RRF-fuses them, otherwise it returns whichever arm is present. Crucially, `raw_top` is the **raw dense cosine** (or a normalized raw BM25 ratio) — *not* the fused score — so the agent's coverage gate (D3) gets an honest confidence signal (the P08/P18 raw-vs-fused gotcha).

**`models/` — the trained core and the baselines.** `encoder.py:DenseEncoder` wraps the fine-tuned sentence-transformers bi-encoder; `load_encoder(cfg)` returns `None` when torch / sentence-transformers are absent (this is the lazy-import boundary that lets everything else run offline). `baseline.py` provides the three documented baselines: `BM25Only` (the strong lexical floor — genuinely competitive in P20), `DenseOnly` (isolates the trained retriever and exposes its semantic-false-positive weakness), and `RandomRetriever` (sanity floor). `model_registry.py` versions checkpoints and maintains a "latest" pointer.

**`training/metrics.py` — the evaluation surface.** `recall_at_k`, `mrr`, `median_rank`, `first_gold_rank`, `retrieval_metrics` (reused from P19) score query→gold-image retrieval at K ∈ {1,5,10}. `cer` / `wer` (reused from P07) measure OCR quality against the rendered gold text. `exact_match_precision(retrieved_texts, query)` is **new for P20**: the fraction of returned images whose OCR text *literally contains* the normalized query — it audits the returned list directly, catching the dense arm's semantically-similar-but-wrong results and quantifying the value BM25 + verification add.

**`agent/` — the deterministic OCR-search agent.** See §6.

**Serving and ops layers** (`api`, `analysis`, `autoreport`, `monitoring`, `automation`, `grading`) are the shared P-series template, specialized to P20's search/snippet/abstain output.

---

## 4. OCR engine selection (auto-routing)

`OcrConfig.engine` controls the front-end; `"auto"` (the default) routes per image:

| Condition | Engine chosen | Why |
|---|---|---|
| Image carries an embedded gold spec (`has_spec` true) | **SeedEngine** | Offline / synthetic / CI — reads gold text back, optionally noised to a target CER/WER. |
| Real image, Tesseract available | **TesseractEngine** | The production default — `pytesseract` over scans / receipts / scene text. |
| No spec and no OCR binary | **StubEngine** | Degraded but non-crashing; returns empty text so the rest of the pipeline still runs. |

The PyMuPDF born-digital-vs-scanned router sits *in front* of OCR for real PDFs (read the text layer, skip OCR), and is bypassed for synthetic PNGs. Because all three engines share the same `.text(image)` contract, the **exact same indexing/search/eval code path** runs offline (Seed) and on Colab/H100 (real Tesseract) — making CER/WER an honest measurement rather than two divergent paths.

---

## 5. Offline / degradation design

The offline backbone runs with **no Tesseract, no torch, no FAISS, no network** — the same contract as P15/P18/P19. Degradation is layered so that removing any heavy dependency cleanly drops to a lighter mode rather than failing:

1. **SeedEngine instead of a real OCR binary.** Synthetic images embed their gold text in the PNG; `SeedEngine` reconstructs per-image OCR text with zero OCR dependencies. `_noisify` adds controllable character noise so the offline index *looks like* real OCR output (and exercises the dense/fuzzy recovery path). `StubEngine` is the final fallback when there is neither a spec nor a binary.

2. **BM25-only when there is no encoder.** `load_encoder` returns `None` when torch / sentence-transformers are missing. `ImageTextIndex.build` then skips the dense arm entirely and `search` returns the pure BM25 ranking. Because P20 is literal-term search, **BM25-only is not a crippled mode** — it is a strong, sometimes near-perfect, retriever (verified offline seed: Recall@1 = MRR = 1.0 for unique codes; the nonexistent term `zzqwx` correctly abstains).

3. **numpy fallback for the dense arm.** When an encoder *is* present, `ImageTextIndex` does the dense search in plain numpy (`emb @ q` over L2-normalized vectors) — FAISS `IndexFlatIP` is an optimization, not a requirement. `use_faiss` can be off and the math is identical (exact cosine).

4. **Lazy imports at every heavy boundary.** numpy, torch, sentence-transformers, FAISS, Tesseract, and the optional `anthropic` LLM client are all imported *inside* the functions that use them (e.g. `import numpy as np` inside `_dense_search`; the LLM client inside `LLMBrain._get_client`). Importing any top-level module of `imgtextsearch` never pulls a heavy dependency, so the package imports, the agent runs, and the test suite passes on a bare Python install.

5. **Dense-embed failures degrade silently.** If the encoder is present but embedding raises (OOM, bad checkpoint), `build` logs and falls back to BM25-only rather than aborting the index build.

6. **A zero-dependency fixture.** `data/samples.py:seed_collection()` / `seed_queries()` give a tiny in-memory collection so the agent, the API, and the tests can run end-to-end with literally no data files.

Net effect: the full **generate → OCR → index → query → verify → eval** loop runs in CI with no GPU and no network, and each added dependency (real OCR, a trained encoder, FAISS, the LLM brain) is a strict *upgrade* over a working floor.

---

## 6. The OCR-search agent (D1–D5)

The mandatory agentic component is a **deterministic finite-state machine** with five decision points — not an LLM agent. `SearchAgent.run(query)` (in `agent/search_agent.py`) drives `tool_parse → tool_search → tool_coverage → tool_verify → tool_finalize` (in `agent/tools.py`), each backed by a pure policy function in `agent/policy.py`. Every gate appends a `Decision` and a `ToolTrace` record to the `JobState` for a fully replayable, deterministic audit.

| ID | Stage | Gates on | Branches |
|---|---|---|---|
| **D1** | parse (`tool_parse`) | non-empty query; `classify_query` → `exact_code` / `keyword` / `phrase`; `query_gate` vs `min_query_chars` | too short / empty → abstain early; else carry the query kind forward. |
| **D2** | search (`tool_search`) | `ImageTextIndex.search` → BM25 (exact arm) + dense (semantic arm) → RRF → top-k image candidates + the **raw** top score | candidates found → carry shortlist + `raw_top`; empty → mark for review. |
| **D3** | coverage (`tool_coverage`) | the **raw** score (raw cosine / raw BM25 ratio, **never** the fused score) vs `coverage_min_score` | below floor → `low_confidence` flag for cautious downstream handling. |
| **D4** | verify (`tool_verify`) | per candidate, does the query term **literally appear** in *that* image's OCR text? `verify_exact` + `_fuzzy_contains` (edit-distance ≤ 1 via `_edit_le1`, case/diacritic-insensitive) | literal hit → extract + highlight the snippet (`make_snippet`, `snippet_window=6`), keep, put exact matches first; `exact_intent` but no occurrence → **drop** as a semantic false-positive; near-match → keep but flag `fuzzy_match`. |
| **D5** | finalize (`tool_finalize`) | how many verified candidates survived D4 | zero verified → **ABSTAIN** ("not found in any image" + `needs_review`); ≥1 → ranked list with snippets, scores, OCR-confidence labels; only low-confidence survivors → return explicitly flagged. |

**Value-add over blind semantic ranking = literal-term VERIFICATION + snippet HIGHLIGHTING + ABSTENTION.** A naive dense retriever always returns k images by embedding similarity, so for `INVOICE` it surfaces images *about* billing where the word never appears — unacceptable for OCR document search. D4 re-reads the candidate's OCR text and confirms the term is actually there (tolerating OCR noise via bounded fuzzy matching), D5 abstains honestly when nothing contains the text, and BM25 + RRF keep the rare literal token exact where embeddings would smear it. The optional `LLMBrain` (`anthropic`, OFF by default) is advisory only — it may add a query-expansion note and **never changes the ranking**.

---

## 7. Deployment surface

- **FastAPI** (`api/`) — `POST /search`: a query returns ranked image ids + RRF scores + the matching OCR snippet + the exact-match flag. The collection is OCR-indexed **once at startup** into an in-memory `ImageTextIndex`.
- **Gradio UI** — type a query, see the matching images with highlighted snippets.
- **Docker** — ships `tesseract-ocr` + fonts + `libGL` for the real OCR path; the synthetic/offline path needs none of these.
- **HF Space** — the demo deployment.

**Ethics / privacy.** OCR-indexing user or scanned documents is sensitive (IDs, contracts, medical/financial records, PII in the text). Mitigations baked into the design: consent + access control, PII redaction, no query retention by default, and the LLM brain off by default. OCR errors can cause false negatives (a document that *does* contain the term is missed) or false positives — the fuzzy D4 verify + D5 abstention mitigate both. OCR quality varies by script/font/scan quality, so search recall is unequal across non-Latin scripts and low-quality scans; English is the only supported production target and non-EN is treated as research-only.

---

## 8. Reuse vs new

**Reused (verbatim / near-verbatim):** BM25 + RRF and the dense `ImageIndex`/`ImageTextIndex` pattern + retrieval metrics from P19 `clipsearch`; CER/WER + `normalize_ws` from P07 `dococr`; the synthetic renderer + `SeedEngine` + `_noisify` + font discovery from P15 `imgtrans`; the PyMuPDF born-digital router from P07/P15; BM25+dense+RRF fusion and the MNRL retriever-train from P18 `tkgqa`; and the whole config/logging/registry/autoreport/monitoring/automation/grading/cli/api template layer.

**New for P20:** (1) the **OCR-text index** — wiring BM25 + dense + RRF where the indexed document is *per-image OCR text* (+ image_id, boxes, confidence); (2) the **snippet + exact-match verifier** — D4/D5 literal-containment (fuzzy-tolerant), snippet span extraction, and `exact_match_precision`; (3) the **rendered-text generator** with the query-needle vocabulary, target-term recurrence control, and the gold query→image map; (4) the **OCR-search agent** — the deterministic D1–D5 machine with the raw-cosine coverage gate, literal verification, abstention, and `ToolTrace`.
