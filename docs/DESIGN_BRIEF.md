# P20 Text Search within Images — Design Brief

> Status: design-locked. This brief is prescriptive — it states the decisions, not options to revisit.
> Sibling lineage: P07 `dococr` (OCR + router + CER/WER), P15 `imgtrans` (synthetic render + SeedEngine OCR), P18 `tkgqa` (BM25 + dense + RRF + retriever-train), P19 `clipsearch` (BM25/dense index + Recall@k/MRR). P20 is the OCR-text-search sibling of P19's visual search.

---

## 1. Problem and scope

**Task.** Given a collection of images (scanned documents, receipts, forms, scene/street photos, born-digital PDF pages) and a user query that is a **word, date, number, ID, or short phrase**, return the ranked subset of images whose **OCR'd text literally contains that query**, each with a highlighted snippet — or **abstain** when no image contains it.

This is *document* search over the text **inside** images via OCR. The user is asking **"which images literally CONTAIN this text"** — e.g. *find images containing the word INVOICE*, *receipts that mention a 2023 date*, *the page with PO-4471*.

**Explicit contrast with P19 (sibling, do not conflate).** P19 (`clipsearch`, CLIP text→image) is **visual/semantic** search: the query describes *what the image depicts* and there is no shared vocabulary between the query text and the pixels, so a lexical arm over captions was only a weak floor. P20 is **lexical/literal**: the query and the indexed OCR text live in the **same lexical space**, so **BM25 is the strongest single signal, not a baseline**. Do **not** pull P19's visual-semantic datasets or its image-embedding index into P20.

**The trainable core.** Of the entire system, exactly **one** component is trained: a **dense text retriever (bi-encoder)** fine-tuned over the per-image OCR text. Everything else — the OCR front-end, BM25, FAISS/RRF, the snippet verifier, the agent state machine — is **pretrained or algorithmic**. This is the same "small trained surface, large deterministic core" split as P07/P15/P18/P19.

**Scope boundaries.**
- IN: English-first literal-term search over OCR text; scanned + born-digital + scene-text images; snippet highlighting; abstention; hybrid BM25+dense+RRF ranking; OCR-noise-tolerant verification.
- OUT: visual/semantic image search (that is P19); VQA / answer generation; full layout parsing / table extraction; multilingual production support (en is primary; fr/it/de available only via CC0 Post-OCR corpus and the renderer, not a supported target).

---

## 2. Pipeline architecture

Five algorithmic stages; **only stage (3)'s dense arm is trained**.

```
image/PDF ─▶ (1) INGEST+ROUTE ─▶ (2) OCR ─▶ (3) TEXT INDEX ─▶ (4) QUERY→RANK ─▶ (5) SNIPPET+VERIFY ─▶ result | ABSTAIN
            born-digital vs       per-image    BM25 (sparse)     parse query        literal-match check
            scanned (PyMuPDF)     text+boxes    + dense (TRAINED) BM25+dense         + highlight span
                                  +confidence   fused w/ RRF      fused via RRF      + abstain if none
```

| Stage | What it does | Trained? |
|---|---|---|
| (1) Ingest + route | PyMuPDF (`fitz`) checks each PDF page for a real text layer ≥ char threshold → read text directly, **skip OCR**; raster images & text-layer-less pages take the OCR path. | **Algorithmic** (reused P07/P15 D2 router) |
| (2) OCR each image | Default Tesseract via `pytesseract.image_to_data(output_type=DICT)` → per-word text + bbox + confidence (0–100). Words grouped to lines/blocks; per-image text concatenated; mean confidence kept in [0,1]. | **Pretrained** (no training) |
| (3) Build text index | Each image = one "document" = its OCR text (+ image_id, word boxes, confidence). Two arms: **BM25** (self-contained, no checkpoint) and a **dense bi-encoder** over the OCR text. | **Dense arm = TRAINED** (MNRL/InfoNCE); BM25 algorithmic |
| (4) Query → rank | Parse query (keyword vs quoted phrase; exact vs semantic intent). Run both arms, fuse ranked lists with **RRF** (rank-based, no score calibration). Optional cross-encoder rerank of top-k. | **Algorithmic** fusion; reranker pretrained |
| (5) Snippet + verify | Extract matching snippet, highlight matched span, **verify the query term literally appears** in that image's OCR text (case/diacritic-insensitive, fuzzy-tolerant). If **no** image contains it → **abstain**. | **Algorithmic** |

**Metric coupling.** Recall@k / MRR over (query → gold image-that-contains-the-text), reused from P18/P19. The OCR front-end and FAISS/RRF are pretrained/algorithmic; only the dense retriever is trained.

---

## 3. Datasets

Verified HF ids (resolved via `hub_repo_details`). The **primary offline data is a SYNTHETIC rendered-text-image generator** (Section 7); real corpora are demo/eval/fine-tune supplements. **License posture: ship demo + tests on MIT + CC0 + CC-BY-4.0 + synthetic; gate every non-commercial set behind a `research_only` flag.**

### 3a. Commercially safe (ship in default/demo)

| Dataset id | Role | Schema | Size | License | NC? |
|---|---|---|---|---|---|
| **`MiXaiLL76/TextOCR_OCR`** | **PRIMARY scene-text (image,text) corpus.** Build the searchable collection from `text`; positive (image_text, query) pairs to fine-tune the retriever. Largest clean MIT (image,text) set found. | `image: Image; text: string` (gold transcription) | 112.7K (train 91.4K / test 14.7K / +numbers) | **mit** | no |
| **`MiXaiLL76/IIIT5K_OCR`** | Secondary scene-text set; small clean **EVAL/dev** collection, drop-in same schema. | `image: Image; text: string` | 5.5K | **mit** | no |
| **`naver-clova-ix/cord-v2`** | **PRIMARY document-image-with-text** collection. Receipt photos + word-level transcription + layout. Realistic noisy real-world docs ("find receipts containing TOTAL / a price"). | `image: Image; ground_truth: string` (JSON: `valid_line[].words[].text` + quad bboxes) | 1.0K (800/100/100) | **cc-by-4.0** | no (attribution if redistributed) |
| **`PleIAs/Post-OCR-Correction`** | **Realistic OCR-TEXT source (NO images).** Real OCR'd `text` + `corrected_text`. Populate a noisy OCR-text index; supplies realistic noise strings for the synthetic renderer; pair with renderer for the image side. | `text` (raw OCR); `corrected_text`; `date/page/file_name/word_count` | 50.4K (en 31.3K / fr 16.5K / it 1.9K / de 672) | **cc0-1.0** | no |

### 3b. Research-only — FLAG, gate behind `research_only`, do NOT redistribute / no commercial release

| Dataset id | Role | License | Why flagged |
|---|---|---|---|
| **`nielsr/docvqa_1200_examples`** | **CLOSEST REAL retrieval-over-OCR signal** + the single best fine-tune/snippet-eval set: each row = image + full `words` OCR list + NL `query` + `answer.matched_text`/`start` span. Use as (query → document/snippet) supervision AND snippet-localization eval. | **Not specified (no tag)** | License unspecified — research-only / verify before commercial. Sampled from DocVQA. |
| **`nielsr/funsd-layoutlmv3`** | Supplementary form-image collection (per-token text + bboxes); tiny, good for fast offline integration tests. | **Not specified** | FUNSD is research-use only (derived from IIT-CDIP/RVL-CDIP legal-tobacco corpus). |
| **`aharley/rvl_cdip`** | Optional 400K scanned-doc **images-to-OCR scale stress test**. No gold text → must run OCR. | **other** | RVL-CDIP/IIT-CDIP derives from Legacy Tobacco Documents (non-commercial); viewer disabled. |
| **`howard-hou/COCO-Text`** | Optional scene-text-in-natural-images collection (~16K). | **Not specified** | No license tag; COCO-Text annotations research-oriented. Lower priority than TextOCR_OCR. |

**Avoid (do not cite):** `priyank-m/text_recognition`, `mteb/TextOCR`, `facebook/textocr`, `Cyrius/textocr`, `priyank-m/iiit_5k`, `Teklia/IIIT-5K` (all 404); `HuggingFaceM4/IIIT-5K`, `HugoLaurencon/IIIT-5K`, `Berzerker/iiit5k_ocr_dataset` (no license / arbitrary-code loaders); `jimmycarter/textocr-gpt4v` (cc-by-**nc**-4.0); `CaptionEmporium/TextOCR-GPT4o` (cc-by-sa, GPT4o captions not gold OCR, Meta TOS); mjsynth/SynthText (not cleanly on HF — generate our own synthetic instead).

**Key gap.** No public "find the image/document containing X" retrieval benchmark with a gold query→image map exists on HF. `nielsr/docvqa_1200_examples` is the closest real signal; the **synthetic generator (Section 7) is the PRIMARY offline data** that supplies the gold query→image map.

---

## 4. Models

Verified HF ids. **The DEFAULT stack is fully commercially usable (all MIT/Apache).** One non-commercial OCR option is FLAGGED and excluded from the default stack.

### 4a. OCR front-end

| Component | id | License | Role |
|---|---|---|---|
| **DEFAULT OCR** | **Tesseract** (system / `pytesseract`) | **Apache-2.0** | Default. `image_to_data` → word text + bbox + confidence (drives snippet highlighting). Reuses P07/P15 stack. Offline, CPU-friendly, permissive. |
| Neural OCR upgrade | `microsoft/trocr-base-printed` | **MIT (in practice)** | VERIFIED (8.6M dl, 333M params). Card omits tag; sibling `trocr-base-handwritten` is MIT and source `microsoft/unilm` is MIT. Recognition-only → needs an external line detector. Best on clean printed scans. |
| Lighter neural OCR (**T4**) | `microsoft/trocr-small-printed` | **MIT (in practice)** | VERIFIED (2.1M dl, 61.4M params). 5× smaller; for T4 / latency-constrained runs. |
| Alt full pipeline (rec) | `PaddlePaddle/PP-OCRv5_server_rec` | **apache-2.0** | VERIFIED official org. Pair with the detector below for a strong det+rec pipeline (en+zh). Heavier runtime dep (PaddleOCR). |
| Alt full pipeline (det) | `PaddlePaddle/PP-OCRv5_server_det` | **apache-2.0** | VERIFIED (616K dl). Text-line detector; pairs with PP-OCRv5 rec or feeds TrOCR (which has no detector). |
| Optional doc-type router | `microsoft/dit-base-finetuned-rvlcdip` | no tag (unilm code MIT) | NOT an OCR engine — optional born-digital-vs-scanned / doc-type router upstream of OCR. The PyMuPDF router already covers routing without a model dep; optional only. |
| Lib alternatives | docTR (`mindee/doctr`), EasyOCR (JaidedAI) | **Apache-2.0** | PyPI/GitHub libraries, **not** canonical HF weights — `pip install`, do not pull low-quality community HF mirrors. |
| **FLAG — non-commercial** | `vikp/surya_rec2` (+ `surya_det3`, `surya_layout*`, `surya_order`) | **cc-by-nc-sa-4.0** | High accuracy but **NON-COMMERCIAL + ShareAlike**. **Excluded from the default stack**; research/eval only. |

### 4b. Retriever / rerank / correction

| Component | id | License | Role |
|---|---|---|---|
| **DEFAULT dense retriever (TRAINED core)** | **`BAAI/bge-small-en-v1.5`** | **mit** | Default bi-encoder over OCR text. 33.4M params, 384-d. Fine-tuned with MultipleNegativesRankingLoss (InfoNCE). Same family already wired in P18/P19. |
| **H100 upgrade** retriever | `BAAI/bge-base-en-v1.5` | **mit** | 109.5M params, 768-d; higher accuracy on a larger top-k. |
| **T4 fallback** retriever | `sentence-transformers/all-MiniLM-L6-v2` | **apache-2.0** | 22.7M params; lightweight drop-in. |
| Alt dense retriever | `intfloat/e5-base-v2` | **mit** | Requires `query:` / `passage:` prefixes. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | **apache-2.0** | Cross-encoder rerank over top-k (the `L-6` spelling redirects here). Optional; disabled under tight latency. |
| Optional post-OCR corrector | `google/byt5-small` | **apache-2.0** | Byte-level seq2seq; clean noisy OCR text **before** indexing (handles `rn↔m`, `0↔O`). |

### 4c. GPU tiers

- **H100 upgrade:** OCR = PP-OCRv5 server det+rec **or** TrOCR-base-printed (top text quality) · retriever = `bge-base-en-v1.5` · cross-encoder rerank on a large top-k · enable `byt5-small` post-OCR correction · batch-OCR the whole collection in one pass.
- **DEFAULT:** Tesseract + `bge-small-en-v1.5` + BM25/RRF · reranker optional.
- **T4 fallback:** Tesseract (CPU) or `trocr-small-printed` (61.4M) · retriever = `all-MiniLM-L6-v2` (22.7M) or `bge-small` (33.4M) · reranker off, byt5 skipped.

All tiers stay fully permissive (MIT/Apache); **none require Surya's NC weights**.

---

## 5. Metrics, index, and baselines

### 5a. Metrics (with formulas)

Let `Q` = query set, `N` = images, `rank_i(q)` = 1-based position of gold image `i` for query `q`; not-retrieved → sentinel rank (1000).

1. **Recall@K (text-in-image retrieval)** — *primary*. Gold set `G(q)` = image(s) whose OCR text literally contains the query term/phrase.
   - single-gold: `hit@K = 1 if min_i rank_i(q) ≤ K else 0`
   - multi-gold: `Recall@K(q) = |{ i∈G(q) : rank_i(q) ≤ K }| / |G(q)|`
   - corpus: `Recall@K = (1/|Q|) · Σ_q Recall@K(q)`, reported for **K ∈ {1,5,10}**, range [0,1].
   - Monotone non-decreasing in K (`R@1 ≤ R@5 ≤ R@10`); a violation flags a ranking/gold-map bug.
   - **Reuse** `19_Text_to_Image_Retrieval/src/clipsearch/training/metrics.py:recall_at_k`. `rank_of(retrieved_ids, gold_id)` → 1-based rank or None (None → sentinel 1000).

2. **MRR (Mean Reciprocal Rank).** `MRR = (1/|Q|) · Σ_q (1 / rank*(q))`, where `rank*(q) = min_i rank_i(q)` is the first gold image; `1/rank = 0` if no gold retrieved. Range (0,1]. Rewards a correct image at the very top — the relevant behavior for "find THE image with this word." **Reuse** `metrics.py:mrr`.

3. **Median / mean rank of gold image.** `median_rank = median_q rank*(q)`; `mean_rank = (1/|Q|) Σ_q rank*(q)` (not-retrieved at sentinel 1000). Median robust to the sentinel tail; mean exposes catastrophic misses. **Reuse** `metrics.py:median_rank / mean_rank`. Diagnostic complement.

4. **OCR CER (Character Error Rate)** — upstream quality cap. `CER = ( Σ_images Levenshtein(ocr_norm, gold_norm) ) / ( Σ_images len(gold_norm) )`, char-level edit distance over whitespace-normalized strings, **micro-averaged corpus-wide** (total edits / total reference chars). Range [0,>1]; 0 = perfect. **Reuse** `07_Document_Level_OCR/src/dococr/training/metrics.py:corpus_cer`. On synthetic data the SeedEngine char-noise rate maps to a controllable CER.

5. **OCR WER (Word Error Rate).** `WER = ( Σ_images Levenshtein(ocr_norm.split(), gold_norm.split()) ) / ( Σ_images len(gold_norm.split()) )`, word-token edit distance, micro-averaged. Complements CER: one garbled word can break an exact-term match at low CER. **Reuse** `metrics.py:corpus_wer`. Report CER and WER together.

6. **Exact-Match precision@K (literal-term presence)** — *verification metric, P20-specific*. Of the top-K images **returned** for `q`, the fraction whose per-image OCR text literally contains the normalized query: `ExactP@K = (1/K) · |{ d∈topK(q) : query_norm is substring / word-subsequence of ocr_norm(d) }|`, averaged over queries; case/whitespace-normalized, token-boundary-aware for word queries. Range [0,1]. Unlike Recall (which uses the gold map), this audits the **returned** list directly against the literal query — catching the dense arm returning semantically-similar-but-wrong images and quantifying the value BM25 adds. Built on the same `normalize_ws` as CER/WER + a containment check; **NEW for P20** (not in siblings verbatim).

### 5b. Hybrid index = BM25 + dense + RRF

The unit of retrieval is the **per-image OCR TEXT** (the "document"); the query is a word/date/phrase expected to literally appear.

- **BM25 is the LEAD arm** (matters more here than in P19). P20 is literal-term document search; success means the chosen image's OCR text actually contains the token. BM25 scores exact term overlap with IDF + length normalization, so it (a) fires on the rare query token (high IDF for `INVOICE`), (b) is exact-match-faithful by construction, (c) needs no GPU. Query and indexed OCR text share the same lexical space → BM25 is the strongest single signal. **Reuse `19_.../src/clipsearch/index/lexical.py:BM25Index` verbatim** (pure-python, `k1=1.5`, `b=0.75`, regex tokenizer; `idf = log(1 + (N-n+0.5)/(n+0.5))`, standard tf-saturation denom) — feed it per-image OCR text instead of captions.
- **Dense arm recovers BM25's blind spots:** (i) OCR noise (`INV0ICE`, `lnvoice` won't lexically match), (ii) morphology/synonymy (`invoices` vs `invoice`, `date: 2023` vs `dated 2023`), (iii) paraphrase queries (`images mentioning a 2023 date`). A sentence-transformers bi-encoder embeds query + OCR text into a shared space. **Reuse** `19_.../src/clipsearch/index/image_index.py:ImageIndex` (FAISS `IndexFlatIP` on L2-normalized vectors, numpy fallback) — but index **OCR-text** embeddings, not image embeddings. Default `BAAI/bge-small-en-v1.5`; backup `all-MiniLM-L6-v2`.
- **RRF fusion (rank-based, score-scale-free).** `score_RRF(d) = Σ_arms 1 / (c + rank_a(d))`, `c=60`, omit the term if `d` is outside arm `a`'s top-M; sort by `score_RRF` desc. RRF beats score-weighted sum because BM25 scores (unbounded, IDF-scaled) and cosine ([-1,1]) are not commensurable; RRF needs no per-arm normalization and is robust when one arm is noisy. Mirrors P18/P19's documented BM25+dense+RRF. Net: BM25 guarantees literal-term precision (the Exact-Match metric), dense+RRF recover OCR-noise/paraphrase cases, improving Recall@K/MRR without sacrificing top-1 exactness.

### 5c. Baselines

| Baseline | Purpose |
|---|---|
| **BM25-only** | The strong lexical floor; in P20 it is genuinely competitive (literal-term search). Hybrid must beat it on OCR-noise/paraphrase Recall while matching its Exact-Match precision. |
| **Dense-only** | Isolates the trained retriever; exposes its semantic-false-positive weakness (high Recall, lower Exact-Match precision) that D4 verification + BM25 fix. |
| **Random** | Sanity floor for Recall@K / MRR. |

---

## 6. Agentic component

A **deterministic state machine** (5 decision points), not an LLM agent. Every gate writes a `ToolTrace` record (decision id, branch, signal value, threshold) for a fully replayable, deterministic audit (P07/P19 pattern).

| ID | Gates on (SIGNAL) | Branches |
|---|---|---|
| **D1** Ingest+OCR | Born-digital-vs-scanned route (PyMuPDF text-layer char count vs threshold) + per-image OCR confidence (`image_to_data` mean conf in [0,1] from per-word 0–100). Builds the index. | born-digital (text layer ≥ threshold) → read text layer, **skip OCR**; scanned/image → run Tesseract; conf < floor (e.g. <0.5) → tag `low_ocr_confidence` (D4/D5 treat cautiously → needs_review); empty/unreadable → **exclude from index + flag**. Fallback: sub-threshold text layer treated as scanned (never an empty doc); STUB SeedEngine runs with no OCR binary. |
| **D2** Query parse | Normalized token/char count; quotes (phrase); exact-literal-vs-semantic intent (quoted/ALL-CAPS/YEAR-DATE-number → exact; descriptive phrase → semantic). | empty / pure-punctuation / < `min_query_tokens` → **ABSTAIN_EARLY** (no encode wasted); quoted phrase or literal token → `exact_intent=True` (weight BM25, **require** literal verify at D4); descriptive phrase → `semantic_intent` (lean dense, still attempt verify); else → default hybrid. → D3. |
| **D3** Coverage | Top-k `(image_id, score)` shortlist — specifically the **RAW BM25 hit count** for the literal term AND the **RAW DENSE COSINE** of the top candidate (P08/P18 gotcha: gate on **raw** cosine, NOT the post-RRF fused score, which always looks confident). | index empty / < k hits / search error → **NEEDS_REVIEW** (terminal); ≥1 strong hit (raw cosine ≥ `tau_soft` and/or BM25 literal hit) → carry shortlist to D4; weak (top raw cosine ∈ [`tau_floor`, `tau_soft`) and zero BM25 literal hits) → **WIDEN once** (relax phrase→keywords / add synonyms / widen k), re-search, re-enter D3 (capped at 1 retry); whole shortlist raw cosine < `tau_floor` AND no BM25 literal hit → jump to **D5 to abstain**. |
| **D4** Literal verify | Per-candidate: does the query term actually appear in **that** image's OCR text? Extract matching snippet (the value-add). Uses per-image OCR text + word boxes; tolerant to OCR noise (case/diacritic-insensitive; bounded fuzzy/edit distance for `rn→m`, `0↔O`). | literal found → **VERIFIED**: extract + highlight snippet (with word bbox), keep, attach OCR confidence; found only via fuzzy/near-match → keep but flag `fuzzy_match` (uncertain); `exact_intent` but NO literal occurrence → **DROP** as semantic false-positive (the core filter beating blind semantic ranking); `low_ocr_confidence` candidate → keep but mark needs_review. Survivors → D5. |
| **D5** Finalize/abstain | How many VERIFIED candidates survived D4, and their best score/confidence. | zero verified → **ABSTAIN**: return "not found in any image" + `needs_review=True` (instead of a misleading semantic top-k); ≥1 verified → **FINALIZE**: ranked image list, each with highlighted snippet, fusion score, OCR confidence, high/medium/weak label; only fuzzy/low-OCR survivors → return **explicitly flagged** low-confidence + needs_review, never silently. |

**Value-add over blind semantic ranking = LITERAL-TERM VERIFICATION + SNIPPET HIGHLIGHTING + ABSTENTION.** A naive dense retriever always returns k images by embedding similarity, so for `INVOICE` it surfaces images *about* billing where the word never appears — unacceptable for OCR document search. The agent adds: **(1) D4 literal verification** — re-reads the candidate's OCR text, confirms the term appears (fuzzy-tolerant), DROPS semantic false-positives; **(2) snippet extraction + highlighting** — returns the exact matching span + word bbox + OCR confidence (explainable, not an opaque neighbor id); **(3) abstention** — when no image contains the text, D5 returns "not found" + needs_review rather than the least-irrelevant image. Reinforced by the BM25+RRF hybrid (lexical arm exact on rare literal tokens the embeddings smear), the D3 raw-cosine gate (P08/P18 gotcha), and the deterministic ToolTrace. Net: the system answers **"which images literally contain this text"** honestly, not "which images are vaguely about this topic."

---

## 7. Offline / test design

**Offline backbone runs with NO tesseract, NO torch, NO network** — exactly the P15/P18/P19 contract. Since no public "search a collection by contained text" benchmark with a gold query→image map exists, we **synthesize one and embed the gold text inside each PNG** so an offline SeedEngine reads it back as the "OCR output."

1. **Vocabulary + snippet builder (makes ranking non-trivial).** Controlled vocab: (a) rare **target needles** — `INVOICE`, `RECEIPT`, `CONFIDENTIAL`, `PURCHASE ORDER`, dates (`2023-04-12`, `March 2023`), amounts (`$1,250.00`), ids (`PO-4471`); (b) common **fillers** shared across images (`the, total, date, amount, page, customer, qty`). Each image's snippet = 1–3 short lines = a few fillers + 0–2 targets, so targets recur in a **controlled** number of images (some unique, some shared by k) → Recall@K/MRR non-degenerate, multi-gold queries exist. Record per target term the set of image ids containing it = the **GOLD map** (independent of OCR → exact even under char-noise). Pull realistic noisy strings from `PleIAs/Post-OCR-Correction` `text` (CC0) so the index looks like real OCR output.

2. **Render to image (PIL) + embed gold spec.** Reuse `15_Document_Image_Translation/src/imgtrans/data/synth_render.py` almost verbatim: `render_page(spec)` lays snippet lines onto a white page with a discovered TrueType font (fallback `load_default`), recomputes per-line bboxes from actual rendered geometry, stores the spec via `img.info['imgsearch_spec'] = json(spec)`; `save_png_with_spec` writes a PNG `tEXt` chunk so gold text+boxes survive reload. Vary font size (24–40), light `_degrade` (small rotation + GaussianBlur, scaled 0–1) so the real-Tesseract Colab arm sees realistic scans. `generate_dataset` writes `page_XXXX.png` + `manifest.jsonl` (one row/image: filename, lines, gold target-terms present). **Rename the spec key `imgtrans_spec` → `imgsearch_spec`** to namespace P20; structure identical.

3. **Offline OCR = SeedEngine (the key trick).** Reuse `15_.../src/imgtrans/models/ocr_engine.py:SeedEngine`: `_read_spec(image)` pulls the embedded gold spec (from `image.info` or PNG `.text`); `recognize()` reconstructs `Word(text,conf,bbox,block,line)` by splitting each gold line into tokens and distributing the line bbox — so with **no OCR binary** we get a faithful per-image OCR text (`OcrResult.full_text`) to index. `_noisify(token, rate, rng)` optionally injects realistic OCR confusions (`m↔rn`, `0↔O`, `1↔l/I`, `5↔S`, `8↔B`, deletion/duplication) at a **controllable rate** = the CER/WER knob — exactly what makes dense+RRF earn its keep (BM25 misses `INV0ICE`, dense recovers it). `load_ocr_engine(cfg, engine='auto', image)` auto-selects SeedEngine when `has_spec(image)`, else falls back tesseract→easyocr→stub — the **same code path** runs offline (seed) and on Colab/H100 (real Tesseract `image_to_data`) for an honest CER/WER. The PyMuPDF born-digital router sits in front for real PDFs, bypassed for synthetic PNGs.

4. **Query/gold construction + offline eval.** Queries = target terms + short paraphrases (`a 2023 date`, `mentions invoice`); `gold(query)` = the recorded image-id set (step 1). Offline pipeline: each image → `SeedEngine.recognize` → per-image OCR text → build `BM25Index` (P19 `lexical.py`) AND, when sentence-transformers present, the dense `ImageIndex` (P19 `image_index.py`) over OCR-text embeddings; score both arms, fuse via RRF (`c=60`), rank. Compute Recall@{1,5,10}/MRR/median-rank vs gold (P19 `metrics.py`), CER/WER vs gold text (P07 `corpus_cer`/`corpus_wer`), and Exact-Match precision@K by substring-checking each returned image's OCR text against the normalized query. A **StubEngine** path (no spec → empty result) and a **BM25-only** retriever guarantee eval, tests, and the agent run end-to-end with **zero heavy deps**.

---

## 8. Reuse map

### Reused (verbatim or near-verbatim — all paths verified to exist)

| Source (sibling) | What | Used for in P20 |
|---|---|---|
| `19_Text_to_Image_Retrieval/src/clipsearch/index/lexical.py:BM25Index` | Pure-python BM25 (`k1=1.5`, `b=0.75`, regex tokenizer, standard idf/tf) | The **lead** sparse arm over per-image OCR text |
| `19_.../src/clipsearch/index/image_index.py:ImageIndex` | FAISS `IndexFlatIP` on L2-normalized vectors + numpy fallback | Dense arm over **OCR-text** embeddings (not image embeddings) |
| `19_.../src/clipsearch/training/metrics.py` | `recall_at_k`, `mrr`, `median_rank`, `mean_rank`, `rank_of` | Recall@K / MRR / rank diagnostics |
| `07_Document_Level_OCR/src/dococr/training/metrics.py` | `corpus_cer`, `corpus_wer`, `normalize_ws` | OCR quality CER/WER; `normalize_ws` also backs Exact-Match |
| `15_Document_Image_Translation/src/imgtrans/data/synth_render.py` | `render_page`, `save_png_with_spec`, `generate_dataset` | Synthetic rendered-text image generator |
| `15_.../src/imgtrans/models/ocr_engine.py` | `SeedEngine`, `_noisify`, `load_ocr_engine`, `has_spec` | Offline OCR + controllable-noise CER/WER knob + auto front-end selection |
| P07/P15 D2 router | PyMuPDF born-digital-vs-scanned text-layer router | Stage (1) ingest routing |
| P18 `tkgqa` / P19 `clipsearch` | BM25+dense+RRF fusion + dense retriever-train (MNRL/InfoNCE) | Hybrid fusion + the trainable retriever |
| Template layer (all siblings) | config / logging / registry / autoreport / monitoring / automation / grading / cli / api | Project scaffolding |

### NEW for P20 (build these)

1. **The OCR-text index** — wiring BM25 + dense + RRF where the indexed "document" is **per-image OCR text** (+ image_id, word boxes, confidence), rather than captions (P19) or KG triples (P18).
2. **The snippet + exact-match verifier** — D4/D5 logic: literal-containment check (case/diacritic-insensitive, bounded-fuzzy for OCR noise), snippet span extraction + bbox highlight, and **Exact-Match precision@K** (built on `normalize_ws` + containment).
3. **The rendered-text generator** — the P20 query-needle vocabulary/gold-map layer on top of the P15 renderer (`imgsearch_spec` namespace; target-term recurrence control; gold query→image map).
4. **The OCR-search agent** — the deterministic D1–D5 state machine with the raw-cosine coverage gate, literal verification, abstention, and ToolTrace.

---

## 9. Indexing

- **Unit:** one "document" per image = its concatenated OCR text + metadata (`image_id`, per-word bboxes, mean OCR confidence).
- **Sparse arm — BM25** (reuse `lexical.py:BM25Index`, pure-python, self-contained, no checkpoint). Lead arm; exact on rare literal tokens, dates, part numbers.
- **Dense arm — FAISS / numpy** (reuse `image_index.py:ImageIndex`). L2-normalized OCR-text embeddings → `IndexFlatIP` (= exact cosine), pure-numpy fallback when FAISS/torch absent. Offline stand-in: TF-IDF behind the same `retrieve()` interface (P18 pattern).
- **Fusion — RRF over per-image OCR text.** Each arm returns a ranked list; `score_RRF(d) = Σ_arms 1/(c + rank_a(d))`, `c=60`. Rank-based → no cross-arm score calibration. Optional cross-encoder rerank of the fused top-k.
- **Offline guarantee:** with no torch/FAISS, the dense arm degrades to TF-IDF/numpy or is dropped to BM25-only, and the whole index/query/eval path still runs.

---

## 10. Risks and gotchas

1. **OCR errors break exact match.** A single char confusion (`INV0ICE`, `lnvoice`, `rn→m`) makes BM25 miss a literally-present term. *Mitigation:* the dense arm + RRF recover near-misses; D4 verification is **fuzzy-tolerant** (bounded edit distance, case/diacritic-insensitive); optional `byt5-small` post-OCR correction before indexing; CER/WER reported so the OCR error propagating into the term index is visible.
2. **Multi-word / phrase queries.** Phrase order, hyphenation, and line breaks in OCR text complicate "literal" containment. *Mitigation:* D2 distinguishes quoted-phrase vs keyword intent; verification is token-boundary-aware and word-subsequence-tolerant; D3 can relax phrase→keywords once on a weak shortlist.
3. **Language / script.** Default supported target is **English** (TextOCR/IIIT5K/DocVQA are EN). Post-OCR (fr/it/de) and the renderer can produce multilingual text, but Tesseract language packs, tokenization, and the en-only retriever are not validated cross-lingually. *Mitigation:* treat non-EN as research-only; do not claim multilingual production support.
4. **Image licensing.** Several useful corpora are **non-commercial / unlicensed** (`docvqa_1200_examples` no tag, `funsd` research-only, `rvl_cdip` tobacco-corpus, `COCO-Text` no tag) and one OCR model (Surya) is CC-BY-NC-SA. *Mitigation:* ship demo/tests only on MIT (TextOCR/IIIT5K) + CC0 (Post-OCR) + CC-BY-4.0 (CORD-v2, attribution) + synthetic; gate every flagged set behind a `research_only` flag; exclude Surya from the default stack.
5. **Index scaling.** `IndexFlatIP` is exact but O(N) per query and OCR-ing 400K images (RVL-CDIP) is a heavy batch job. *Mitigation:* batch-OCR on H100 in one pass; born-digital router skips OCR on text-layer PDFs; for large N swap `IndexFlatIP` → an ANN index (HNSW/IVF) behind the same interface; BM25-only remains a no-GPU floor; the synthetic offline collection stays small for fast CI.
6. **Raw-cosine vs fused-score gotcha (P08/P18).** Gating coverage on the **post-RRF fused score** always looks confident and defeats abstention. *Mitigation:* D3 gates on **raw** BM25 hit count + **raw** dense cosine, never the fused score.
