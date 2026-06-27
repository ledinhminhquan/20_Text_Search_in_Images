# P20 Text Search within Images — Project Plan

> Project: **P20 "Text Search within Images"** · package `imgtextsearch` · folder `20_Text_Search_in_Images`
> Author: **Le Dinh Minh Quan** (student 23127460)
> Companion docs: `docs/DESIGN_BRIEF.md` (design-locked architecture), this plan (milestones, timeline, risks, resources).

---

## 1. What we are building (one paragraph)

P20 searches a **collection of images by the TEXT they CONTAIN**. The user query is a word, date, number, ID, or short phrase (e.g. *"find images containing the word INVOICE"*, *"receipts mentioning a 2023 date"*, *"the page with PO-4471"*); the system OCRs every image, indexes the per-image OCR text, ranks the images by a **hybrid BM25 + dense + RRF** retriever, **verifies** the query term literally appears, returns each hit with a **highlighted snippet**, and **abstains** when no image contains the text. This is OCR-based *document* search, **not** the visual/semantic content search of the sibling project P19 (`clipsearch`, CLIP text→image). Exactly **one** component is trained — a dense text retriever (bi-encoder) fine-tuned over OCR text; the OCR front-end, BM25, RRF, the snippet verifier, and the agent FSM are pretrained or algorithmic.

The plan below assumes heavy reuse from the sibling projects (P07 `dococr`, P15 `imgtrans`, P18 `tkgqa`, P19 `clipsearch`) and follows the same "small trained surface, large deterministic core, offline-first" contract.

---

## 2. Pipeline diagram

The plan delivers the five-stage pipeline from the design brief. Only stage (3)'s dense arm is trained; everything else is pretrained or algorithmic.

```
                          ┌──────────────────────── P20 imgtextsearch pipeline ────────────────────────┐
                          │                                                                              │
  image / PDF page  ─────▶│  (1) INGEST + ROUTE        (2) OCR EACH IMAGE         (3) BUILD TEXT INDEX    │
                          │  PyMuPDF (fitz) text-layer  Tesseract image_to_data    per-image OCR text =   │
                          │  char count ≥ thr?          → word text + bbox +       one "document":        │
                          │   ├─ yes → read text layer   confidence (0–100)         ┌─ BM25  (sparse, LEAD)│
                          │   │        SKIP OCR          → per-image full_text       └─ dense (TRAINED)    │
                          │   └─ no  → OCR path          + mean conf in [0,1]         bi-encoder embeddings│
                          │  (reused P07/P15 D2 router) (offline: SeedEngine reads  (FAISS IndexFlatIP /  │
                          │                              embedded gold spec)         numpy fallback)       │
                          │                                                                              │
                          │  (4) QUERY → RANK                              (5) SNIPPET + VERIFY           │
                          │  parse query (keyword vs phrase;               extract matching snippet,     │
                          │  exact vs semantic intent)                     highlight matched span,       │
                          │  run BM25 + dense arms                         literal-containment check     │
                          │  fuse ranked lists with RRF (c=60)             (case/diacritic-insensitive,  │
                          │  optional cross-encoder rerank top-k           bounded fuzzy for OCR noise)   │
                          │                                                → ABSTAIN if NO image contains │
                          └──────────────────────────────────────────────────────────────────┬──────────┘
                                                                                              ▼
                                                     ranked image ids + scores + snippet + exact-match flag
                                                                       │  OR  "not found" + needs_review
                                                                       ▼
   AGENT (deterministic FSM, the mandatory agentic component) wraps stages 1–5 as 5 decision points:
   D1 ingest/OCR route + conf gate ─▶ D2 query parse + intent ─▶ D3 coverage (RAW BM25 hits + RAW cosine)
   ─▶ D4 literal verify + snippet extract (drop semantic false-positives) ─▶ D5 finalize | ABSTAIN
   every gate writes a ToolTrace (decision id, branch, signal, threshold) → fully replayable audit.

   DEPLOY: FastAPI POST /search (collection OCR-indexed once at startup) + Gradio UI + Docker (tesseract
   + fonts + libGL) + HF Space.
```

---

## 3. Milestones and timeline

Plan spans **8 weeks (≈40 working days)** for a single primary author with optional reviewer support, structured as 9 milestones (M0–M8). Each milestone has an explicit exit gate so work cannot silently slip. Because so much is reused, the heavy lifting is *integration and verification*, not greenfield code.

### 3.1 Milestone table

| ID | Milestone | Duration | Key deliverables | Exit gate (Definition of Done) | Reuse source |
|---|---|---|---|---|---|
| **M0** | Research, scope-lock, repo scaffold | Days 1–3 (3d) | Verified dataset/model ids; license posture table (MIT/CC0/CC-BY-4.0 ship; NC/unlicensed gated); `imgtextsearch` package skeleton; config/logging/registry/CLI/API templates copied; `DESIGN_BRIEF.md` finalized. | All HF ids resolve; every non-commercial/undeclared set flagged `research_only`; `pip install -e .` + `pytest` green on empty stubs. | Template layer (all siblings) |
| **M1** | Synthetic generator + SeedEngine (PRIMARY offline data) | Days 4–8 (5d) | `data/synth_text_images.py`: needle/filler vocab, controlled term-recurrence, **gold query→image map**; PIL renderer (`imgsearch_spec` namespace) writing `page_XXXX.png` + `manifest.jsonl`; SeedEngine reads gold back; `_noisify` CER/WER knob. | `generate_dataset` produces a collection with multi-gold queries; SeedEngine reconstructs per-image OCR text; **offline run needs NO tesseract/torch/network**; noise rate maps to a measurable CER. | P15 `synth_render.py`, `ocr_engine.py:SeedEngine` |
| **M2** | OCR integration (real front-end) | Days 9–13 (5d) | Tesseract via `pytesseract.image_to_data` → word text + bbox + confidence; PyMuPDF born-digital-vs-scanned router; `load_ocr_engine(..., engine='auto')` auto-selects SeedEngine (offline) vs Tesseract (Colab); optional TrOCR/PaddleOCR upgrade paths stubbed. | Same code path runs offline (seed) and on Colab (real Tesseract); CER/WER computed vs gold; router skips OCR on text-layer PDFs; empty/unreadable images excluded + flagged. | P07/P15 OCR stack + D2 router |
| **M3** | Hybrid index: BM25 + dense + RRF | Days 14–18 (5d) | `BM25Index` over per-image OCR text (LEAD arm); `ImageIndex` (FAISS `IndexFlatIP` / numpy) over OCR-text embeddings; RRF fusion (`c=60`); TF-IDF/BM25-only offline stand-in; optional cross-encoder rerank. | Hybrid retrieve() returns ranked image ids + scores; BM25-only, dense-only, hybrid all runnable; offline path runs with FAISS/torch absent; RRF beats either single arm on a smoke set. | P19 `lexical.py`, `image_index.py`; P18 RRF |
| **M4** | Dense retriever fine-tune (the trainable core) | Days 19–24 (6d) | Fine-tune `BAAI/bge-small-en-v1.5` (default) with MultipleNegativesRankingLoss on (query, OCR-text) pairs from TextOCR_OCR + DocVQA + synthetic; T4 fallback `all-MiniLM-L6-v2`; H100 upgrade `bge-base-en-v1.5`; checkpoint to registry. | Fine-tuned retriever beats the off-the-shelf checkpoint on dev Recall@K/MRR; training reproducible on Colab T4; checkpoint loads in the index path. | P18/P19 retriever-train (MNRL/InfoNCE) |
| **M5** | Agent FSM (mandatory agentic component) | Days 25–29 (5d) | `src/imgtextsearch/agent/`: D1 ingest+OCR gate, D2 query parse/intent, D3 coverage on **RAW** BM25 hits + **RAW** cosine, D4 literal verify + snippet extract + bbox highlight (fuzzy-tolerant, drop semantic false-positives), D5 finalize/abstain; `ToolTrace` per gate; optional advisory LLM brain OFF by default. | All **5 decision points** fire on the seed set; nonexistent term (`zzqwx`) correctly **abstains**; ToolTrace replayable/deterministic; literal verification drops semantic false-positives. | P07/P19 FSM + ToolTrace pattern |
| **M6** | Evaluation harness + baselines | Days 30–34 (5d) | Recall@{1,5,10} + MRR + median/mean rank; CER/WER vs gold; **Exact-Match precision@K** (P20-specific); baselines BM25-only / dense-only / random; autoreport. | Verified offline seed reproduces **Recall@1 = MRR = 1.0** on unique codes; Exact-Match precision quantifies BM25's value over dense-only; report auto-generated. | P19 `metrics.py`, P07 `corpus_cer/wer`; NEW Exact-Match |
| **M7** | Deployment: FastAPI + Gradio + Docker + HF Space | Days 35–38 (4d) | `POST /search` (collection OCR-indexed once at startup → ranked ids + scores + snippet + exact-match flag); Gradio UI (query → matching images + highlighted snippets); Docker (tesseract-ocr + fonts + libGL); HF Space. | API + UI run on the synthetic collection end-to-end; Docker image builds and serves; HF Space live; abstention surfaced in the UI. | API/UI/Docker templates (all siblings) |
| **M8** | Hardening, docs, ethics/privacy, release | Days 39–40 (2d) | README + usage docs; ethics/privacy/PII section; bias + robustness notes; license attribution (CORD-v2 CC-BY-4.0); final grading pass; tag release. | CI green; docs complete; PII/consent/no-query-retention defaults documented; `research_only` gating verified; grading rubric satisfied. | autoreport/grading/automation templates |

### 3.2 Timeline (Gantt-style, by week)

```
Week →            1        2        3        4        5        6        7        8
Day  →         1  2  3  4  5  6  7  8  9 ...                                    40
M0 Research    ███
M1 Synth+Seed     ████
M2 OCR integ          ████
M3 BM25+dense              ████
M4 Retriever FT                ██████
M5 Agent FSM                          ████
M6 Eval+base                              ████
M7 Deploy                                      ███
M8 Hardening                                       ▓
                                                    (M8 = days 39–40)
```

**Critical path:** M1 (synthetic gold map) → M2 (OCR) → M3 (index) → M4 (retriever) → M6 (eval). M5 (agent) depends on M3 and partially overlaps M4. M7/M8 depend on M5+M6. The synthetic generator (M1) is the single most important early deliverable because it is the **PRIMARY offline data** and unblocks every downstream milestone with zero heavy dependencies.

---

## 4. Risk register

Severity = Low / Medium / High; Likelihood = Low / Med / High. Risks below are P20-specific (not generic project risks).

| # | Risk | Likelihood | Impact | Mitigation | Owner / Milestone |
|---|---|---|---|---|---|
| R1 | **OCR errors break exact match.** A single char confusion (`INV0ICE`, `lnvoice`, `rn→m`) makes BM25 miss a literally-present term → false negative. | High | High | Dense arm + RRF recover near-misses; **D4 verification is fuzzy-tolerant** (bounded edit distance ≤1, case/diacritic-insensitive); optional `google/byt5-small` post-OCR correction before indexing; **CER/WER reported** so OCR error propagation is visible; synthetic `_noisify` knob tunes the regime under test. | M2/M4/M5 |
| R2 | **Image licensing.** Several useful corpora are NC / unlicensed (`nielsr/docvqa_1200_examples` no tag, `nielsr/funsd-layoutlmv3` research-use, `aharley/rvl_cdip` license:other tobacco-corpus, `howard-hou/COCO-Text` no tag) and one OCR model (Surya, `vikp/surya_*`) is CC-BY-NC-SA. | High | High | **Ship demo + tests only on MIT** (TextOCR_OCR, IIIT5K_OCR) **+ CC0** (PleIAs/Post-OCR-Correction) **+ CC-BY-4.0** (CORD-v2, with attribution) **+ synthetic**; gate every flagged set behind a `research_only` flag; **exclude Surya from the default stack**; document attribution. | M0 (posture), all |
| R3 | **Index scaling.** `IndexFlatIP` is exact but O(N) per query; OCR-ing 400K RVL-CDIP images is a heavy batch job. | Med | Med | Batch-OCR on H100 in one pass; born-digital router skips OCR on text-layer PDFs; for large N swap `IndexFlatIP` → ANN (HNSW/IVF) **behind the same interface**; BM25-only remains a no-GPU floor; synthetic CI collection stays small/fast. | M3/M7 |
| R4 | **No public "find image containing X" benchmark** with a gold query→image map exists on HF. | High | High | The **synthetic rendered-text generator (M1) is the PRIMARY offline data** — it owns the gold map (independent of OCR, exact even under char-noise); `nielsr/docvqa_1200_examples` (gated) supplies the closest real query→snippet signal. | M1/M6 |
| R5 | **Multi-word / phrase queries.** Phrase order, hyphenation, line breaks in OCR text complicate literal containment. | Med | Med | D2 distinguishes quoted-phrase vs keyword intent; D4 verification is token-boundary-aware and word-subsequence-tolerant; D3 can relax phrase→keywords **once** on a weak shortlist. | M5 |
| R6 | **Raw-cosine vs fused-score gotcha (P08/P18).** Gating coverage on the post-RRF fused score always looks confident and defeats abstention. | Med | High | **D3 gates on RAW BM25 hit count + RAW dense cosine, never the fused score** — explicitly encoded and unit-tested. | M5 |
| R7 | **Language / script bias.** OCR quality varies by script/font/scan quality → unequal recall; default target is EN only. | Med | Med | Treat non-EN (fr/it/de via CC0 Post-OCR + renderer) as **research-only**; do not claim multilingual production support; document the bias in the ethics section. | M2/M8 |
| R8 | **Semantic false-positives** (dense returns images *about* billing where `INVOICE` never appears) — unacceptable for literal search. | Med | High | **D4 literal verification DROPS** them; BM25 LEAD arm anchors exactness; **Exact-Match precision@K** quantifies and guards against regressions. | M5/M6 |
| R9 | **Privacy / PII in OCR-indexed documents** (IDs, contracts, medical/financial records). | Med | High | Consent + access control; PII redaction option; **no query retention by default**; LLM brain **OFF** by default; document in ethics section. | M8 |
| R10 | **Reuse drift** — sibling code (P15 spec key, P19 index interfaces) changes shape during port. | Low | Med | Pin reused modules; rename `imgtrans_spec → imgsearch_spec` to namespace; integration tests at each milestone gate; verbatim-reuse map tracked in `DESIGN_BRIEF.md` §8. | All |

---

## 5. Resource needs

### 5.1 Compute (Colab tiers, matched to the design brief)

| Tier | Hardware | Used for | Notes |
|---|---|---|---|
| **CPU / offline** | Local CPU, no GPU | Synthetic generation (M1), SeedEngine OCR, BM25-only index, agent FSM, full eval, CI | The **offline backbone runs with NO tesseract, NO torch, NO network** — the P15/P18/P19 contract. This is the default dev + CI loop. |
| **T4 fallback** | Colab T4 (16 GB) | Real-Tesseract OCR pass, lightweight retriever fine-tune (`all-MiniLM-L6-v2` 22.7M or `bge-small` 33.4M), `trocr-small-printed` (61.4M) | Reranker off, `byt5` skipped. Sufficient for the default deliverable. |
| **L4 / A100** | Colab L4 or A100 (Pro) | Default retriever fine-tune (`bge-small-en-v1.5`), cross-encoder rerank experiments, mid-scale OCR batches | The comfortable middle tier for M4 training runs. |
| **H100 upgrade** | Colab H100 / A100-80G (Pro+) | Batch-OCR the full collection in one pass; `bge-base-en-v1.5` (768-d) retriever; PP-OCRv5 server det+rec or TrOCR-base-printed; enable `byt5-small` post-OCR correction; large RVL-CDIP scale stress test | Top text-quality + large-top-k config. Only needed for the stretch/scale evaluation. |

All tiers stay **fully permissive (MIT/Apache)**; none require Surya's NC weights.

### 5.2 Accounts / services

- **HF Pro** — Colab Pro/Pro+ for L4/A100/H100 access; HF Hub for dataset/model pulls and the deployment **HF Space**; private repo for gated `research_only` artifacts.
- **HF token** — authenticated dataset/model access (already authenticated as `ledinhminhquan`).
- **Anthropic API key** — *optional only*; the advisory LLM brain is OFF by default and never changes ranking. Not required for any core deliverable.

### 5.3 Software / dependencies

- Core (always): Python, numpy, Pillow (PIL), pure-python BM25, PyMuPDF (`fitz`).
- OCR (Colab arm): `pytesseract` + system `tesseract-ocr`, fonts, `libGL` (Docker). Optional: TrOCR/PaddleOCR/docTR/EasyOCR.
- Retriever: `sentence-transformers`, FAISS (numpy fallback when absent), optional `cross-encoder` reranker, optional `google/byt5-small`.
- Deploy: FastAPI, Gradio, Docker, HF Spaces.

### 5.4 Data

- Ship/demo: `MiXaiLL76/TextOCR_OCR` (MIT, 112K), `MiXaiLL76/IIIT5K_OCR` (MIT eval), `naver-clova-ix/cord-v2` (CC-BY-4.0 receipts), `PleIAs/Post-OCR-Correction` (CC0 text-only) + **synthetic (PRIMARY offline)**.
- Research-only (gated, **flagged**): `nielsr/docvqa_1200_examples` (closest real query→snippet signal, license unspecified), `nielsr/funsd-layoutlmv3` (research-use), `aharley/rvl_cdip` (license:other), `howard-hou/COCO-Text` (no tag).

---

## 6. Division of work

Single primary author (Le Dinh Minh Quan, 23127460) owns all milestones end-to-end; the split below is by **work-stream** so effort and dependencies are explicit, and so an optional reviewer/collaborator can pick up a stream cleanly.

| Work-stream | Milestones | Description | Depends on |
|---|---|---|---|
| **A — Data & synthesis** | M1, part M0 | Synthetic generator, gold query→image map, vocab/recurrence control, SeedEngine noise knob, dataset license posture. | — (unblocks everything) |
| **B — OCR & ingest** | M2 | Tesseract `image_to_data`, PyMuPDF router, `load_ocr_engine` auto-select, CER/WER plumbing. | A (needs synthetic PNGs with spec) |
| **C — Index & retrieval** | M3, M4 | BM25 (lead) + dense (FAISS/numpy) + RRF; retriever fine-tune (MNRL); reranker. | A, B (needs per-image OCR text) |
| **D — Agent** | M5 | D1–D5 FSM, raw-cosine coverage gate, literal verify + snippet/bbox highlight, abstention, ToolTrace, optional advisory LLM. | B, C (needs OCR text + index) |
| **E — Eval & metrics** | M6 | Recall/MRR/rank, CER/WER, Exact-Match precision@K, baselines, autoreport. | A, C, D |
| **F — Deploy & docs** | M7, M8 | FastAPI + Gradio + Docker + HF Space; README, ethics/privacy, bias/robustness, release. | D, E |

**Reuse leverage (reduces effort across all streams):** BM25/dense/RRF + retriever-train from P18/P19; OCR + SeedEngine + renderer + D2 router from P07/P15; metrics (`recall_at_k`, `mrr`, `median_rank`, `corpus_cer/wer`, `normalize_ws`) from P07/P19; the full template layer (config/logging/registry/autoreport/monitoring/automation/grading/cli/api). **NEW for P20** (the genuine build effort): the OCR-text index, the snippet + exact-match verifier, the rendered-text generator's gold-map layer, and the OCR-search agent.

---

## 7. Definition of done (project-level)

The project is complete when:

1. The **offline backbone** (synthetic data → SeedEngine OCR → BM25 index → agent FSM → eval) runs end-to-end with **no tesseract, no torch, no network**, reproducing the verified seed result (**Recall@1 = MRR = 1.0** on unique codes; `zzqwx` correctly **abstains**; all 5 decisions fire).
2. The **real arm** (Colab Tesseract + fine-tuned `bge-small-en-v1.5` + RRF) beats BM25-only on OCR-noise/paraphrase Recall while **matching** its Exact-Match precision.
3. The **agent** drops semantic false-positives via D4 literal verification, highlights snippets, and abstains honestly.
4. **FastAPI + Gradio + Docker + HF Space** serve the collection (OCR-indexed once at startup) and surface abstention.
5. **License posture** holds: demo/tests on MIT + CC0 + CC-BY-4.0 + synthetic; every NC/undeclared set gated behind `research_only`; Surya excluded from the default stack.
6. **Ethics/privacy** documented: consent, access control, PII redaction, no query retention by default, LLM brain OFF, bias (script/font/scan-quality), robustness (OCR errors, phrase queries, scaling).

---

## 8. Out of scope (explicit)

Per the design brief, the following are **NOT** in P20 and must not creep in: visual/semantic image search (that is P19 `clipsearch`); VQA / answer generation; full layout parsing / table extraction; multilingual production support (English is primary; fr/it/de are research-only via the CC0 Post-OCR corpus and the renderer). Pulling P19's visual-semantic datasets or its image-embedding index into P20 is explicitly disallowed.
