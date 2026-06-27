# P20 Text Search within Images — Model Selection

> Author: Le Dinh Minh Quan (student 23127460)
> Package: `imgtextsearch` · Folder: `20_Text_Search_in_Images`
> Scope: this document justifies **every model choice** in P20 — the OCR front-end, the trainable dense retriever, the BM25 exact-term arm, RRF fusion, the reranker, and the optional post-OCR corrector — and maps each to a GPU tier. It is prescriptive: it records the decisions, not options to revisit.

---

## 0. What P20 is selecting models *for*

P20 searches a **collection of images by the text they CONTAIN** — "find the images containing the word INVOICE", "receipts mentioning a 2023 date", "the page with PO-4471". This is **OCR-based document search**, not visual/semantic content search (that is the sibling project P19 `clipsearch` / CLIP). The query and the indexed text live in the **same lexical space** (both are literal strings), which fundamentally shapes the model selection:

```
image ─▶ OCR (pretrained) ─▶ per-image OCR text ─▶ TEXT INDEX ─▶ query → rank ─▶ verify snippet ─▶ result | ABSTAIN
                                                    BM25 (algorithmic)              literal-term check
                                                    + dense (TRAINED)               + highlight + abstain
                                                    fused via RRF
```

The model-selection principle is the same "**small trained surface, large deterministic core**" split P20 inherits from P07/P15/P18/P19:

| Component | Role | Trained in P20? | Why |
|---|---|---|---|
| OCR front-end | image → text | **No (pretrained)** | OCR is a solved, off-the-shelf capability; training it is out of scope and would need char-level transcription labels we do not have. |
| **Dense bi-encoder** | query/OCR-text → vector | **YES — the one trainable core** | The single component we fine-tune (MNRL over `(query, OCR-text)` pairs). |
| BM25 | exact-term sparse arm | **No (algorithmic)** | Pure-python, no checkpoint, no GPU; faithful by construction on literal tokens. |
| RRF | rank fusion | **No (algorithmic)** | Parameter-free rank arithmetic. |
| Cross-encoder reranker | top-k reorder | **No (pretrained)** | Off-the-shelf MS-MARCO checkpoint; optional. |
| `byt5-small` corrector | OCR text cleanup | **No (pretrained)** | Off-the-shelf byte-level seq2seq; optional. |

Exactly **one** checkpoint is trained. Everything else is selected off-the-shelf or is parameter-free. **Every model in the DEFAULT stack is MIT or Apache-2.0** — fully commercially usable. The only non-commercial OCR option (Surya, CC-BY-NC-SA) is flagged and excluded from the default stack.

---

## 1. OCR front-end — why Tesseract (Apache-2.0) is the default

### 1.1 The decision

| Component | id | License | Status in P20 |
|---|---|---|---|
| **DEFAULT OCR** | **Tesseract** (system binary via `pytesseract`) | **Apache-2.0** | **Default.** `image_to_data(output_type=DICT)` → per-word text + bbox + confidence (0–100). |
| Neural upgrade (clean printed scans) | `microsoft/trocr-base-printed` | **MIT (in practice)** | Upgrade tier; recognition-only, needs an external line detector. |
| Lighter neural (T4) | `microsoft/trocr-small-printed` | **MIT (in practice)** | 5× smaller (61.4M params) for latency-constrained T4. |
| Alt full pipeline | `PaddlePaddle/PP-OCRv5_server_det` + `PP-OCRv5_server_rec` | **Apache-2.0** | Strong det+rec; heavier PaddleOCR runtime dep. |
| Library alternatives | docTR (`mindee/doctr`), EasyOCR (JaidedAI) | **Apache-2.0** | PyPI/GitHub libraries; `pip install`, not canonical HF weights. |
| **FLAG — non-commercial** | `vikp/surya_rec2` (+ `surya_det3`, `surya_layout*`, `surya_order`) | **cc-by-nc-sa-4.0** | **Excluded from the default stack.** Research/eval only. |

### 1.2 Why Tesseract wins the default slot

**License — Apache-2.0, the cleanest possible posture.** Tesseract is permissive, commercially usable, and adds no model-weight licensing burden. P20 indexes *user/scanned documents* — a privacy-sensitive setting (Section 7) where shipping a non-commercial OCR weight into the default path would be a legal liability. Apache-2.0 keeps the default stack uniformly commercially safe.

**Cost / footprint — CPU-friendly, offline, no GPU.** Tesseract runs on CPU with no accelerator. The DEFAULT and T4 tiers can OCR an entire collection on commodity hardware. Neural OCR (TrOCR, PaddleOCR, Surya) wants a GPU for throughput; P20's default deliberately does not require one, so the system is deployable on a plain Docker container (`tesseract-ocr` + fonts + `libGL`).

**Output shape fits the verifier exactly.** `pytesseract.image_to_data` returns **per-word text + bounding box + confidence (0–100)**. P20's value-add is **snippet highlighting** (D4 returns the matching span *with its word bbox*) and **OCR-confidence-aware** decisions (D1 tags `low_ocr_confidence`, D4/D5 treat those candidates cautiously). Word-level boxes + confidence are precisely the signals the agent's decision points consume. A recognition-only model that emits a single string (raw TrOCR) does **not** give per-word boxes — it cannot drive bbox highlighting without an external detector.

**Reuse — already wired in P07/P15.** Tesseract via `pytesseract` is the OCR stack of P07 `dococr` and P15 `imgtrans`, including font discovery and the `load_ocr_engine(...)` auto-selection path. P20 ports it verbatim. No new OCR plumbing is invented; the only NEW OCR-side artifact for P20 is the *index* over OCR text, not the OCR engine.

**Born-digital routing makes most OCR unnecessary.** Stage (1)'s PyMuPDF (`fitz`) router reads the real text layer of born-digital PDF pages directly and **skips OCR**, so Tesseract only runs on raster images and text-layer-less pages. The OCR engine's cost matters less because the router already removes the easy cases — and a CPU engine is more than adequate for the remainder.

### 1.3 Why each alternative is *not* the default (but is available)

- **`microsoft/trocr-base-printed` (MIT in practice).** VERIFIED on the Hub (8.6M downloads, 333M params). Best transcription quality on clean printed scans, and the license is MIT in practice (the card omits a tag, but the sibling `trocr-base-handwritten` is MIT and the source `microsoft/unilm` is MIT). But it is **recognition-only** — it transcribes a cropped text line and has **no built-in text detector**, so it needs an external line detector (e.g. the PaddleOCR detector) to locate words before it can read them, and it does not natively emit the per-word bbox/confidence stream the verifier wants. It is heavier and wants a GPU. → **Upgrade tier only** (H100), where text quality is worth the extra dependency and compute.
- **`microsoft/trocr-small-printed` (MIT in practice).** VERIFIED (2.1M downloads, 61.4M params), 5× smaller than base. A reasonable neural option on a small GPU, but it carries the same recognition-only / no-detector limitation. → **T4 tier alternative** when a neural reader is wanted under tight memory.
- **PaddleOCR (`PP-OCRv5_server_det` + `PP-OCRv5_server_rec`, Apache-2.0).** VERIFIED official org; a strong **det+rec** pipeline (en+zh) and the natural detector to pair with TrOCR. License is clean. But it pulls a heavier PaddleOCR runtime, and its advantage over Tesseract does not justify that dependency for the *default* English-first path. → **H100 upgrade** for top text quality.
- **docTR (`mindee/doctr`) / EasyOCR (JaidedAI), Apache-2.0.** Clean licenses, but these are **PyPI/GitHub libraries, not canonical HF weights**. We `pip install` them rather than pulling low-quality community HF mirrors. They are valid drop-ins behind the same `load_ocr_engine` interface but add a dependency without beating Tesseract on the default English path.
- **Surya (`vikp/surya_rec2` + det/layout/order), CC-BY-NC-SA-4.0 — FLAGGED.** High accuracy, but **non-commercial AND ShareAlike**. ShareAlike is doubly problematic: it would attempt to impose its license on derived output. P20 indexes commercial documents, so Surya is **excluded from the default stack** and confined to research/eval comparisons. It never appears in any shipped tier.

### 1.4 Offline OCR — SeedEngine (no model, the test backbone)

The PRIMARY offline data is a **synthetic rendered-text-image generator** (`data/synth_text_images.py`, built on the P15 renderer). Each PNG embeds its **gold text** in a PNG `tEXt` chunk (`imgsearch_spec`). The offline **SeedEngine** (`has_spec(image) → SeedEngine`) reconstructs per-word `Word(text, conf, bbox, …)` by reading that embedded gold spec — so with **no Tesseract binary, no torch, no network** we still get a faithful per-image "OCR output" to index. `_noisify(...)` injects realistic OCR confusions (`m↔rn`, `0↔O`, `1↔l/I`, `5↔S`, `8↔B`) at a **controllable rate** = the CER/WER knob — exactly what makes the dense arm earn its keep (BM25 misses `INV0ICE`; dense recovers it). `load_ocr_engine(cfg, engine='auto', image)` runs the **same code path** offline (seed) and on Colab/H100 (real Tesseract `image_to_data`), so CER/WER are honest. SeedEngine is not a *model* choice — it is the deterministic offline stand-in that lets the whole pipeline run with zero heavy deps, mirroring the P15/P18/P19 contract.

---

## 2. The trainable core — `BAAI/bge-small-en-v1.5` dense retriever (MIT)

### 2.1 The decision

| Component | id | License | Params / dim | Status |
|---|---|---|---|---|
| **DEFAULT dense retriever (TRAINED core)** | **`BAAI/bge-small-en-v1.5`** | **mit** | 33.4M / 384-d | **Default.** Fine-tuned with MultipleNegativesRankingLoss (InfoNCE) on `(query, OCR-text)` pairs. |
| **H100 upgrade** | `BAAI/bge-base-en-v1.5` | **mit** | 109.5M / 768-d | Higher accuracy on larger top-k. |
| **T4 fallback** | `sentence-transformers/all-MiniLM-L6-v2` | **apache-2.0** | 22.7M | Lightweight drop-in. |
| Alt | `intfloat/e5-base-v2` | **mit** | 109M / 768-d | Requires `query:` / `passage:` prefixes. |

### 2.2 Why bge-small is the trainable core

**It is the one component worth training.** OCR + BM25 + RRF + the verifier are all pretrained or algorithmic. The dense arm is where the system can *learn* the mapping from a user's query phrasing to noisy OCR text — the part no off-the-shelf component covers. Fine-tuning it with **MultipleNegativesRankingLoss (InfoNCE)** on `(query, OCR-text)` positive pairs teaches the embedding space to pull a query (e.g. "images mentioning a 2023 date") toward the OCR text of images that contain it, against in-batch negatives. Positive pairs come from `MiXaiLL76/TextOCR_OCR` `(image, gold text)` and the `(query → matched_text)` spans in `nielsr/docvqa_1200_examples` (research-only, gated).

**License — MIT.** Clean, commercially usable, no attribution burden, keeps the default retriever in the same permissive posture as the rest of the stack.

**Right size for the default tier.** 33.4M params / 384-d embeddings are small enough to fine-tune on a single modest GPU and to embed an entire OCR-text collection quickly, while still being a strong MTEB-class retriever. The 384-d vectors keep the FAISS `IndexFlatIP` footprint and per-query cost low.

**Reuse — already wired in P18/P19.** The BGE family is the dense retriever already integrated in P18 `tkgqa` and P19 `clipsearch`, with the MNRL/InfoNCE training loop and the `ImageIndex` (FAISS `IndexFlatIP` on L2-normalized vectors, numpy fallback). P20 reuses that index **but embeds OCR text instead of image/caption embeddings**. Reusing the family means the training script, the index, and the eval harness all transfer with minimal change.

**What the dense arm is *for* in P20 — recovering BM25's blind spots.** P20 is literal-term search, so BM25 is the *lead* arm (Section 3). The dense retriever exists specifically to catch what exact-term matching misses:
1. **OCR noise** — `INV0ICE`, `lnvoice` will not lexically match `INVOICE`, but their embeddings sit near it.
2. **Morphology / synonymy** — `invoices` vs `invoice`, `date: 2023` vs `dated 2023`.
3. **Paraphrase queries** — "images mentioning a 2023 date" is a descriptive phrase, not a literal token.

So the dense arm is a **recall booster** layered over a faithful lexical floor — not a standalone ranker. (Its semantic-false-positive weakness — returning images *about* billing where the word INVOICE never appears — is exactly what D4 literal verification + BM25 exist to filter; see Section 3.)

### 2.3 Why the alternatives sit in other tiers

- **`BAAI/bge-base-en-v1.5` (MIT, 109.5M / 768-d).** Same family, more capacity, higher accuracy on a larger top-k — but 3× the params and 2× the embedding dim. → **H100 upgrade**, where the extra compute and index footprint are affordable.
- **`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22.7M).** The lightweight workhorse; a drop-in when even bge-small is too heavy. → **T4 fallback** (or when the dense arm must coexist with a neural OCR on the same small GPU).
- **`intfloat/e5-base-v2` (MIT).** Strong, but requires `query:` / `passage:` prefixes — an extra wiring detail and a behavioral foot-gun if the prefixes are forgotten. Listed as an alternative, not the default.

---

## 3. The BM25 exact-term arm + RRF — why lexical is the *lead*, not a baseline

### 3.1 BM25 (pure-python, algorithmic — no checkpoint, no GPU)

**This is the single most important model-selection point in P20.** In the sibling P19 (CLIP text→image), the query describes *what an image depicts* and shares no vocabulary with the pixels, so a lexical arm over captions was only a **weak floor**. In P20 the query and the indexed OCR text live in the **same lexical space** — both are literal strings — so **BM25 is the strongest single signal, not a baseline.** The whole task is "which images literally CONTAIN this token", and BM25 is exact-match-faithful **by construction**.

Why BM25 is the lead arm for literal search:
- **(a) Fires on rare query tokens via IDF.** `INVOICE`, `PO-4471`, `2023-04-12` are high-IDF; BM25's `idf = log(1 + (N - n + 0.5)/(n + 0.5))` rewards exactly the rare literal needles users search for.
- **(b) Exact-match-faithful by construction.** A BM25 hit means the token genuinely overlaps the document — it cannot hallucinate a semantically-similar-but-absent term the way a dense embedding can. This directly underwrites P20's **Exact-Match precision@K** metric.
- **(c) No GPU, no checkpoint.** Pure-python, self-contained, runs in the offline/CI path with zero heavy deps and stands in for the dense arm when torch/FAISS are absent.

**Reuse:** `19_Text_to_Image_Retrieval/src/clipsearch/index/lexical.py:BM25Index` **verbatim** (`k1=1.5`, `b=0.75`, regex tokenizer, standard tf-saturation denominator) — fed per-image OCR text instead of captions. It is *selected*, not trained.

### 3.2 RRF fusion (algorithmic, parameter-free)

`score_RRF(d) = Σ_arms 1 / (c + rank_a(d))`, with `c = 60`, omitting an arm's term when `d` falls outside its top-M.

**Why RRF and not a weighted score sum.** BM25 scores are unbounded and IDF-scaled; cosine similarity is bounded in `[-1, 1]`. The two are **not commensurable** — a weighted sum would need per-arm score calibration that drifts with corpus and query. RRF is **rank-based**, so it needs no normalization, is robust when one arm is noisy, and is the same documented BM25+dense+RRF fusion already used in P18/P19. The net effect: **BM25 guarantees literal-term precision (the Exact-Match metric), and dense + RRF recover OCR-noise / morphology / paraphrase cases** — improving Recall@K / MRR without sacrificing top-1 exactness.

**Coverage-gate caveat (P08/P18 gotcha).** The agent's D3 coverage gate reads the **raw BM25 hit count** and the **raw dense cosine** of the top candidate — *never* the post-RRF fused score, which always looks confident and would defeat abstention. This is a model-*usage* decision that flows directly from the fusion choice.

### 3.3 The reranker — `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0, optional)

| Component | id | License | Status |
|---|---|---|---|
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **apache-2.0** | Optional cross-encoder rerank of the fused top-k. Disabled under tight latency. |

After RRF produces a fused top-k, an **optional** cross-encoder rerank reorders that shortlist by jointly encoding `(query, OCR-text)` — more accurate than the bi-encoder's independent embeddings, at the cost of one forward pass per candidate. It is **off by default** and on tight-latency tiers (T4) because:
- it adds GPU latency proportional to top-k,
- the bi-encoder + BM25 + RRF already produce a strong fused ranking, and
- P20's decisive quality gain comes from **D4 literal verification**, not from reranking semantic order.

It is a pretrained, off-the-shelf checkpoint (the `L-6` spelling redirects to this canonical model) — selected, never trained. → Enabled on the **H100 upgrade** over a large top-k.

---

## 4. Optional post-OCR corrector — `google/byt5-small` (Apache-2.0)

| Component | id | License | Params | Status |
|---|---|---|---|---|
| Post-OCR corrector | `google/byt5-small` | **apache-2.0** | ~300M | **Optional.** Cleans noisy OCR text **before** indexing. H100 tier only. |

**Why byt5-small, and why byte-level.** OCR errors are predominantly **character-level confusions** — `rn↔m`, `0↔O`, `1↔l/I`. A **byte-level** seq2seq (ByT5 operates directly on UTF-8 bytes, no subword vocabulary) is the natural fit: it can rewrite `INV0ICE → INVOICE` at the character granularity where the error actually lives, where a subword-tokenized model would first have to mis-segment the corrupted token. Running it as a pre-index cleanup step **restores literal-term matches that OCR noise would otherwise break**, lifting BM25 (which is the part most hurt by a single char flip) before fusion.

**Why optional / H100 only.** It is a ~300M-param seq2seq — a real per-document inference cost. The cheaper, already-present mitigations (the dense arm + RRF recovering near-misses, and D4's **fuzzy-tolerant** verification with bounded edit distance) cover most OCR-noise cases without it. So byt5 correction is an **accuracy upgrade for the H100 tier**, not a default dependency. License is clean Apache-2.0. It is pretrained and applied off-the-shelf (optionally fine-tuned on the CC0 `PleIAs/Post-OCR-Correction` `text → corrected_text` pairs) — never a required trained component.

---

## 5. GPU-tier table and the processor each needs

All three tiers stay **fully permissive (MIT / Apache only)**; **none require Surya's NC weights.** "Processor" is the OCR/image preprocessor each engine needs to turn raw image bytes into model input.

| Tier | OCR front-end | Processor the OCR needs | Dense retriever (TRAINED) | Reranker | Post-OCR (byt5) | Fusion |
|---|---|---|---|---|---|---|
| **T4 fallback** (16 GB) | **Tesseract (CPU)** or `microsoft/trocr-small-printed` (61.4M) | Tesseract: `pytesseract` (no GPU). TrOCR-small: `TrOCRProcessor` (= `ViTImageProcessor` + `RobertaTokenizer`) | `sentence-transformers/all-MiniLM-L6-v2` (22.7M) or `bge-small` (33.4M) | **off** | **skipped** | BM25 + dense + RRF |
| **DEFAULT** | **Tesseract** (`pytesseract.image_to_data`) | `pytesseract` → system Tesseract binary (word text + bbox + conf; no GPU) | **`BAAI/bge-small-en-v1.5`** (33.4M / 384-d) | optional | off | BM25 + dense + RRF |
| **H100 upgrade** | `PaddlePaddle/PP-OCRv5_server` **det + rec**, or `microsoft/trocr-base-printed` (333M) | PaddleOCR: built-in det+rec preprocessing pipeline. TrOCR-base: `TrOCRProcessor` (`ViTImageProcessor` + `RobertaTokenizer`); **needs an external line detector** (e.g. the PP-OCRv5 det model) since TrOCR is recognition-only | `BAAI/bge-base-en-v1.5` (109.5M / 768-d) | `cross-encoder/ms-marco-MiniLM-L-6-v2` on a large top-k | **`google/byt5-small`** enabled (`ByT5`/`AutoTokenizer`, byte-level) | BM25 + dense + RRF + cross-encoder rerank |

Notes on the L4 / A100 middle ground:

- **L4 (24 GB):** treat as **DEFAULT-plus** — Tesseract or `trocr-base-printed` for higher text quality, `bge-small` (or `bge-base` if headroom allows), reranker optionally on, byt5 off. Comfortably runs the default stack with room for one neural component.
- **A100 (40/80 GB):** treat as **near-H100** — `bge-base-en-v1.5`, cross-encoder rerank enabled, optional byt5 correction, and batch-OCR of the whole collection in one pass; choose PP-OCRv5 det+rec or TrOCR-base for OCR. The only practical gap vs H100 is throughput on very large collections (e.g. OCR-ing 400K RVL-CDIP images), where H100's batch headroom matters most.

**Processor summary (what each engine consumes):**

- **Tesseract via `pytesseract`** — no ML processor; the system binary handles binarization/layout internally and returns word text + bbox + confidence directly. This is precisely why it drives snippet highlighting and confidence gating with no extra components.
- **TrOCR (`*-printed`)** — `TrOCRProcessor` = a `ViTImageProcessor` (resize/normalize image patches) + a `RobertaTokenizer` (decode to text). **Recognition-only**: it consumes an already-cropped text line, so it needs an upstream **line detector** (PP-OCRv5 det) and does not emit per-word boxes on its own.
- **PP-OCRv5 (det + rec)** — PaddleOCR's own det→rec preprocessing pipeline (detection produces boxes, recognition reads each box); a complete OCR pipeline, heavier runtime dependency.
- **byt5-small** — a byte-level tokenizer (`AutoTokenizer` / `ByT5Tokenizer`); consumes the raw OCR string, emits a corrected string. No image processor — it sits *after* OCR.
- **Dense retriever / reranker** — sentence-transformers / cross-encoder text tokenizers only; they consume OCR text strings, never images.

---

## 6. License posture — summary

| Model | License | Commercial? | In default stack? |
|---|---|---|---|
| Tesseract (`pytesseract`) | Apache-2.0 | Yes | **Yes (default OCR)** |
| `microsoft/trocr-base-printed` / `trocr-small-printed` | MIT (in practice) | Yes | Upgrade / T4 alt |
| `PaddlePaddle/PP-OCRv5_server_det` / `_rec` | Apache-2.0 | Yes | H100 upgrade |
| docTR / EasyOCR | Apache-2.0 | Yes | Library alternative |
| **`vikp/surya_*`** | **CC-BY-NC-SA-4.0** | **No (NC + ShareAlike)** | **NO — flagged, research/eval only** |
| `BAAI/bge-small-en-v1.5` (trained core) | MIT | Yes | **Yes (default retriever)** |
| `BAAI/bge-base-en-v1.5` | MIT | Yes | H100 upgrade |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | Yes | T4 fallback |
| `intfloat/e5-base-v2` | MIT | Yes | Alternative |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | Yes | Optional reranker |
| `google/byt5-small` | Apache-2.0 | Yes | Optional corrector (H100) |

**Every shipped tier is MIT/Apache and fully commercially usable. Surya is the only non-commercial option and is excluded from the default stack** — confined to research/eval comparisons, never in a deployed path. This matters acutely for P20 because the system OCR-indexes sensitive user/scanned documents (IDs, contracts, financial/medical records); the model stack must not add licensing risk on top of the privacy obligations.

---

## 7. Selection rationale in one paragraph

P20 trains **exactly one** model — the `BAAI/bge-small-en-v1.5` dense retriever (MIT), fine-tuned with MNRL/InfoNCE on `(query, OCR-text)` pairs and reused from P18/P19 — because it is the only component that learns the query→noisy-OCR-text mapping. Everything else is selected off-the-shelf for permissiveness, fit, and reuse: **Tesseract (Apache-2.0)** is the default OCR because it is CPU-friendly, offline, commercially clean, and emits the per-word boxes + confidence the snippet verifier needs — with neural upgrades (TrOCR-MIT, PP-OCRv5-Apache) reserved for the H100 tier and **Surya (CC-BY-NC-SA) excluded entirely**; **BM25 (algorithmic)** is the *lead* arm, not a baseline, because P20 is literal-term search where query and OCR text share a lexical space, making exact-match the strongest single signal; **RRF (parameter-free)** fuses the lexical and dense arms without score calibration; an **optional cross-encoder reranker** and an **optional byt5-small byte-level post-OCR corrector** (both Apache-2.0) are accuracy upgrades gated to higher GPU tiers. The result is a fully commercially-usable default stack that answers "which images literally contain this text" honestly — exact where it must be, semantic where it helps.
