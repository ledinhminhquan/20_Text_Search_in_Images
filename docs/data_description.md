# P20 Text Search within Images — Data Description

> Project: **P20 "Text Search within Images"** · package `imgtextsearch` · folder `20_Text_Search_in_Images`
> Author: Le Dinh Minh Quan (student 23127460)
> Task: search a **collection of images** by the **text they contain** (via OCR) — e.g. *"find images containing the word INVOICE"*, *"receipts mentioning a 2023 date"*, *"the page with PO-4471"*. This is OCR-based **document** search, **not** visual/semantic content search (that is the sibling project P19 `clipsearch` / CLIP).

---

## 1. What "data" means for P20

P20 retrieves over the **text inside images**, so every data source has to be understood through one lens: *what searchable text does it give us, and is there a gold map from a query to the image(s) that literally contain it?*

The pipeline is:

```
image  ─▶  OCR  ─▶  per-image OCR text (the "document")  ─▶  TEXT index (BM25 + dense → RRF)  ─▶  query → ranked images + matching snippet
```

The unit of retrieval is **one image = one document = its concatenated OCR text** (plus `image_id`, per-word bounding boxes, and mean OCR confidence). Consequently a dataset is useful to P20 only insofar as it supplies one or more of:

1. **(image, gold text)** pairs — to build a searchable collection *and* to measure OCR quality (CER/WER) against a known reference;
2. **realistic OCR text** (with or without images) — to make the index look like real, noisy OCR output;
3. **(query → image/snippet) supervision** — the genuinely scarce signal: a query plus the gold span/image that contains the answer, which is what retrieval-over-OCR actually needs.

No public dataset gives us a clean *"find the image containing word X across a collection"* benchmark with a gold query→image map. That gap is the reason P20's **primary offline data is a synthetic rendered-text-image generator** (Section 6); the real corpora below are demo, fine-tune, and evaluation **supplements**.

**License posture (enforced throughout).** Ship the default/demo stack and all tests on **MIT + CC0 + CC-BY-4.0 + synthetic data only**. Every non-commercial or license-unspecified set is **flagged** and gated behind a `research_only` configuration flag; it is never bundled into the default demo and never redistributed.

---

## 2. Dataset inventory at a glance

| Dataset id | Role in P20 | Schema (key fields) | Size | License | Ship default? |
|---|---|---|---|---|---|
| `MiXaiLL76/TextOCR_OCR` | **Primary scene-text (image, text) corpus** — searchable collection + retriever fine-tune pairs | `image: Image`, `text: string` (gold transcription) | ~112.7K (train 91.4K / test 14.7K + numbers) | **MIT** | ✅ yes |
| `MiXaiLL76/IIIT5K_OCR` | Secondary scene-text set — small clean **eval/dev** collection, same schema | `image: Image`, `text: string` | ~5.5K | **MIT** | ✅ yes |
| `naver-clova-ix/cord-v2` | **Primary document-image-with-text** collection (receipts, word-level text + layout) | `image: Image`, `ground_truth: string` (JSON: `valid_line[].words[].text` + quad bboxes) | ~1.0K (800 / 100 / 100) | **CC-BY-4.0** | ✅ yes (attribution) |
| `PleIAs/Post-OCR-Correction` | **Realistic OCR-text source (no images)** — noisy OCR strings for the index and the synthetic renderer | `text` (raw OCR), `corrected_text`, `date`/`page`/`file_name`/`word_count` | ~50.4K (en 31.3K / fr 16.5K / it 1.9K / de 672) | **CC0-1.0** | ✅ yes (text only) |
| `nielsr/docvqa_1200_examples` | **Closest real retrieval-over-OCR signal** — query + gold answer span over OCR words | `image`, `words` (OCR list), `query`, `answer.matched_text` + `answer.start` (span) | ~1.2K | **Unspecified — FLAG** | ⚠️ `research_only` |
| `nielsr/funsd-layoutlmv3` | Supplementary form-image collection (per-token text + bboxes); fast integration tests | per-token `text` + `bbox` | small (FUNSD) | **Unspecified — FLAG** | ⚠️ `research_only` |
| `aharley/rvl_cdip` | Optional **OCR-at-scale stress test** (no gold text → must OCR) | scanned-doc `image` + doc-class label | ~400K images | **other (NC) — FLAG** | ⚠️ `research_only` |
| **Synthetic generator** (`data/synth_text_images.py`) | **PRIMARY offline data** — gold query→image map, embedded gold text, runs with no deps | rendered PNG + `imgsearch_spec` (lines, target terms) + `manifest.jsonl` | configurable (small for CI) | self-generated (MIT-compatible) | ✅ yes (primary) |

> `howard-hou/COCO-Text` (~16K scene-text-in-natural-images, no license tag) is a possible lower-priority research-only supplement and is **not** part of the default stack.

---

## 3. Commercially-safe sources (default / demo stack)

These four sources carry permissive licenses (MIT, CC-BY-4.0, CC0) and are the only real corpora allowed in the shipped demo, tests, and default fine-tune.

### 3.1 `MiXaiLL76/TextOCR_OCR` — primary scene-text (image, text) corpus · MIT

The **largest clean MIT (image, text) set** we found and P20's anchor real corpus.

- **Schema.** Each row is `image: Image` + `text: string`, where `text` is the gold transcription of the text visible in the image (cropped scene-text words / lines from natural images).
- **Size / splits.** ~112.7K rows: **train ~91.4K**, **test ~14.7K**, plus a numbers subset.
- **Role in P20.**
  - **Collection.** The `text` field is concatenated per image to build a real searchable collection — a stress test for literal-term search over genuine scene text (street signs, product labels, posters).
  - **Retriever fine-tune.** Positive `(query, OCR-text)` pairs are mined from `text` to fine-tune the dense bi-encoder with MultipleNegativesRankingLoss (InfoNCE): a token (or short phrase) drawn from an image's `text` becomes the query, the full `text` is the positive document, and other images' texts are in-batch negatives.
  - **OCR eval.** Because `text` is gold, it is a CER/WER reference when the real Tesseract / TrOCR path runs on these images.
- **License.** **MIT** — commercially usable, shippable in the default stack.

### 3.2 `MiXaiLL76/IIIT5K_OCR` — secondary scene-text set · MIT

A small, clean, drop-in eval companion to TextOCR_OCR.

- **Schema.** Identical: `image: Image` + `text: string`. No code changes to swap it in.
- **Size.** ~5.5K rows — small enough for a fast **eval/dev** collection and quick smoke tests.
- **Role in P20.** A held-out, same-schema collection for sanity-checking Recall@K / MRR and OCR CER/WER without touching the larger TextOCR_OCR collection. Useful as an independent dev set so the retriever is not tuned and evaluated on the same corpus.
- **License.** **MIT** — shippable.

### 3.3 `naver-clova-ix/cord-v2` — primary document-image collection · CC-BY-4.0

Real, noisy **document** images — the closest commercially-safe match to P20's actual use case (searching scanned business documents), as opposed to scene text.

- **Schema.** `image: Image` + `ground_truth: string`, where `ground_truth` is a JSON structure containing `valid_line[].words[].text` (word-level transcriptions) plus quad bounding boxes and layout. P20 flattens `valid_line[].words[].text` into the per-image OCR/gold text; the quad bboxes can seed snippet-highlight boxes.
- **Size / splits.** ~1.0K receipt images: **train 800 / validation 100 / test 100**.
- **Role in P20.** A realistic "find receipts containing **TOTAL** / a price / a date" collection — genuine photographed receipts with real-world noise (skew, lighting, thermal-print artifacts). It exercises the document side of the pipeline (the born-digital-vs-scanned router, snippet extraction over real layout) that scene text does not.
- **License.** **CC-BY-4.0** — commercially usable **with attribution if redistributed**. P20 retains the attribution notice; safe for the default stack.

### 3.4 `PleIAs/Post-OCR-Correction` — realistic OCR-text source · CC0-1.0

The only **text-only** source, and the one that makes the index look like *real* OCR rather than clean strings.

- **Schema.** `text` (raw OCR output, with real OCR errors), `corrected_text` (the clean target), and metadata `date` / `page` / `file_name` / `word_count`. **There are no images.**
- **Size / language mix.** ~50.4K rows: **en 31.3K**, fr 16.5K, it 1.9K, de 672. P20 is English-first, so the en subset is primary; fr/it/de exist for the renderer/corrector only and are **not** a supported production target.
- **Role in P20.**
  - **Noisy OCR-text index.** Populate a searchable index directly from real OCR `text` so retrieval is tested against authentic noise (`rn↔m`, `0↔O`, broken words), not synthetic-clean text.
  - **Realistic noise strings for the renderer.** The synthetic generator (Section 6) draws filler/background text from this corpus so rendered pages read like real OCR output.
  - **Post-OCR-correction supervision (optional).** The `(text → corrected_text)` pairs are the natural training/eval signal for the optional `google/byt5-small` corrector that cleans OCR text before indexing.
- **License.** **CC0-1.0** (public domain dedication) — maximally permissive, shippable.

---

## 4. Research-only sources (flagged — gated behind `research_only`)

These are valuable but carry **non-commercial or unspecified** licenses. Each is gated behind a `research_only` flag, excluded from the default demo, and **not redistributed**.

### 4.1 `nielsr/docvqa_1200_examples` — the retrieval-over-OCR signal · license UNSPECIFIED → FLAG

The **single most valuable** real source for what P20 actually does, and the closest thing to a real query→document/snippet benchmark on the Hub.

- **Schema.** Each row = `image` + a full `words` OCR list (the page's words) + a natural-language `query` + `answer.matched_text` and `answer.start` (the **gold span** inside the OCR words that answers the query). Sampled from DocVQA.
- **Why it matters.** It is the only real source that pairs a **query** with the **gold span/image** that contains the answer — exactly the `(query → document/snippet)` supervision retrieval needs. P20 uses it two ways:
  1. **Fine-tune / eval signal.** `(query, page-OCR-text)` as positive retrieval pairs.
  2. **Snippet-localization eval.** `answer.matched_text` / `answer.start` is a gold reference for the D4 snippet extractor — does the highlighted span actually cover the answer text?
- **Size.** ~1.2K examples.
- **License — FLAG.** **Not specified** (no license tag on the card). Treated as **research-only**: gated behind `research_only`, must be license-verified before any commercial use, never bundled into the shipped demo.

### 4.2 `nielsr/funsd-layoutlmv3` — supplementary form images · research-use → FLAG

- **Schema.** Form images with per-token `text` + `bbox` annotations — a small set of scanned forms.
- **Role.** A tiny, fast collection for offline integration tests of the document path (per-token text + layout), complementary to CORD-v2.
- **License — FLAG.** **Not specified**, and FUNSD is **research-use only** (derived from the IIT-CDIP / RVL-CDIP legal-tobacco corpus). Gated behind `research_only`.

### 4.3 `aharley/rvl_cdip` — OCR-at-scale stress test · license: other (NC) → FLAG

- **Schema.** ~400K scanned document images with a 16-class document-type label. **No gold text** — every image must be OCR'd to become searchable.
- **Role.** An optional **scale stress test**: batch-OCR a large collection (H100, one pass) to exercise index scaling (`IndexFlatIP` is exact but O(N); large N motivates an ANN swap to HNSW/IVF behind the same interface).
- **License — FLAG.** **`other`** — RVL-CDIP / IIT-CDIP derives from the Legacy Tobacco Documents corpus and is **non-commercial**; the dataset viewer is disabled. Gated behind `research_only`, not redistributed, never in the default stack.

### 4.4 Avoid entirely (do not cite or load)

Several superficially-relevant ids are explicitly **avoided**: broken/404 mirrors (`priyank-m/text_recognition`, `mteb/TextOCR`, `facebook/textocr`, `Cyrius/textocr`, `priyank-m/iiit_5k`, `Teklia/IIIT-5K`); no-license or arbitrary-code-loader mirrors (`HuggingFaceM4/IIIT-5K`, `HugoLaurencon/IIIT-5K`, `Berzerker/iiit5k_ocr_dataset`); and non-commercial / TOS-restricted captions (`jimmycarter/textocr-gpt4v` is CC-BY-**NC**-4.0; `CaptionEmporium/TextOCR-GPT4o` is CC-BY-SA with GPT-4o captions, not gold OCR, under Meta TOS). MJSynth / SynthText are not cleanly on HF — we generate our own synthetic data instead.

---

## 5. The key gap, restated

There is **no public "find the image/document containing X across a collection" retrieval benchmark with a gold query→image map** on Hugging Face. The available real data is:

- scene-text **(image, text)** pairs (TextOCR_OCR, IIIT5K_OCR) — good for collection + OCR eval, but each image is essentially one short string with no notion of "which images share term X";
- **document** images with word text (CORD-v2, FUNSD) — realistic but small and per-document, no cross-collection query→image map;
- **OCR text** without images (Post-OCR-Correction) — realistic noise, but no images to retrieve;
- **DocVQA** (`docvqa_1200_examples`) — the closest real query→span signal, but license-unspecified and per-page (a query→answer-span, not a query→*set-of-images-that-contain-the-term* map).

None of these gives a controllable gold map of *"these N images contain INVOICE, these M contain PO-4471, none contain zzqwx."* That map is exactly what Recall@K / MRR / median-rank over a **collection** require, and it must be **independent of OCR** so the metric is exact even when OCR is noisy. Hence the synthetic generator is P20's **primary offline data**, not an afterthought.

---

## 6. The synthetic rendered-text-image generator (PRIMARY offline data)

`data/synth_text_images.py` is P20's primary offline dataset. It builds a collection where **we control which images contain which terms**, embeds the gold text **inside each PNG**, and lets an offline `SeedEngine` read that gold text back as the "OCR output" — so the entire pipeline (OCR → index → query → rank → snippet → metrics) runs with **no Tesseract, no torch, no network**. This mirrors the P15 / P18 / P19 offline contract.

It is built on top of the P15 renderer (`15_Document_Image_Translation/src/imgtrans/data/synth_render.py`), reused almost verbatim, with the spec key renamed `imgtrans_spec → imgsearch_spec` to namespace P20.

### 6.1 Why synthetic is necessary (not just convenient)

1. **It supplies the missing gold query→image map.** Because we *decide* which target terms go into which images, the gold set `G(q)` = `{images whose snippet contains q}` is known exactly — recorded at generation time, **independent of OCR**. The metric is therefore exact even when OCR injects char noise.
2. **It makes ranking non-trivial.** By controlling term recurrence (some targets unique to one image, some shared by *k* images), Recall@K and MRR are **non-degenerate** and genuine **multi-gold** queries exist — unlike a corpus where every needle is unique or every image is identical.
3. **It gives a controllable OCR-error knob.** The SeedEngine's char-noise rate maps directly to a target CER/WER, so we can dial in exactly the regime where BM25 misses a literally-present term (`INV0ICE`) and the dense arm + RRF must recover it — i.e. the regime that justifies the hybrid index.
4. **It is dependency-free and CI-friendly.** No OCR binary, no GPU, no download; the collection stays small for fast CI.

### 6.2 How it works (four stages)

**Stage 1 — Vocabulary + snippet builder.** A controlled business-document vocabulary:

- **Rare target "needles"** — `INVOICE`, `RECEIPT`, `CONFIDENTIAL`, `PURCHASE ORDER`, dates (`2023-04-12`, `March 2023`), amounts (`$1,250.00`), ids (`PO-4471`, `REF####` unique codes). These are the searchable answers.
- **Common fillers** shared across images — `the`, `total`, `date`, `amount`, `page`, `customer`, `qty` — drawn in part from real noisy strings in `PleIAs/Post-OCR-Correction` (`text`, CC0) so the index reads like real OCR.

Each image's snippet is 1–3 short lines = a few fillers + 0–2 targets. Targets recur in a **controlled number of images** (some unique, some shared by *k*). For each target term the generator records the **set of image ids** containing it — this is the **GOLD map**, computed at generation time and independent of OCR.

**Stage 2 — Render to image (PIL) + embed gold spec.** Reusing the P15 renderer's `render_page(spec)`: each snippet's lines are laid onto a white page with a discovered TrueType font (fallback `load_default`), per-line bounding boxes are recomputed from the actual rendered geometry, and the spec is stored on the image via `img.info['imgsearch_spec'] = json(spec)`. `save_png_with_spec` writes the spec into a PNG `tEXt` chunk so the gold text + boxes survive a reload. Font size varies (24–40) and a light `_degrade` (small rotation + Gaussian blur, scaled 0–1) is applied so the real-Tesseract Colab arm sees realistic scans. `generate_dataset` writes `page_XXXX.png` files + a `manifest.jsonl` (one row per image: filename, lines, gold target-terms present).

**Stage 3 — Offline OCR = SeedEngine (the key trick).** Reusing `15_.../src/imgtrans/models/ocr_engine.py:SeedEngine`: `_read_spec(image)` pulls the embedded gold spec (from `image.info` or the PNG `.text` chunk); `recognize()` reconstructs `Word(text, conf, bbox, block, line)` by splitting each gold line into tokens and distributing the line bbox — so **with no OCR binary** we still get a faithful per-image OCR text (`OcrResult.full_text`) to index. `_noisify(token, rate, rng)` optionally injects realistic OCR confusions (`m↔rn`, `0↔O`, `1↔l/I`, `5↔S`, `8↔B`, deletion/duplication) at a **controllable rate** — this is the CER/WER knob. `load_ocr_engine(cfg, engine='auto', image)` auto-selects SeedEngine when `has_spec(image)`, otherwise falls back tesseract → easyocr → stub, so the **same code path** runs offline (seed) and on Colab/H100 (real Tesseract `image_to_data`) for an honest CER/WER. The PyMuPDF born-digital router sits in front for real PDFs and is bypassed for synthetic PNGs.

**Stage 4 — Query / gold construction + offline eval.** Queries = the target terms plus short paraphrases (`a 2023 date`, `mentions invoice`); `gold(query)` = the recorded image-id set from Stage 1. The offline pipeline: each image → `SeedEngine.recognize` → per-image OCR text → build a `BM25Index` (reused from P19 `lexical.py`) and, when sentence-transformers is present, a dense `ImageIndex` (P19 `image_index.py`) over OCR-text embeddings; score both arms and fuse via RRF (`c=60`). It then computes Recall@{1,5,10} / MRR / median-rank vs gold (P19 `metrics.py`), CER/WER vs the rendered gold text (P07 `corpus_cer` / `corpus_wer`), and **Exact-Match precision@K** by substring-checking each returned image's OCR text against the normalized query. A `StubEngine` path (no spec → empty result) and a **BM25-only** retriever guarantee that eval, tests, and the agent all run end-to-end with **zero heavy dependencies**.

### 6.3 Verified offline behavior (seed)

The offline seed run (BM25 over per-image OCR text) confirms the design: **Recall@1 = MRR = 1.0** for unique codes (exact term match is perfect when the needle is unique), the matching **snippet is highlighted**, a nonexistent term (`zzqwx`) **correctly ABSTAINS** rather than returning a least-irrelevant image, and **all five agent decisions (D1–D5) fire**.

---

## 7. Splits, sizes, and how each source is used

| Source | Split / size used | Used as | Trains the retriever? |
|---|---|---|---|
| **Synthetic generator** | configurable; small for CI | **Primary offline collection + gold query→image map + CER/WER knob** | Yes (synthetic `(query, OCR-text)` pairs available) |
| `MiXaiLL76/TextOCR_OCR` | train ~91.4K → fine-tune; test ~14.7K → eval | Real scene-text collection + retriever fine-tune pairs + OCR eval | **Yes (primary fine-tune)** |
| `MiXaiLL76/IIIT5K_OCR` | ~5.5K → dev/eval | Small held-out scene-text eval collection | No (eval only) |
| `naver-clova-ix/cord-v2` | 800 / 100 / 100 | Real document (receipt) collection + snippet/layout eval | Optionally (document-domain pairs) |
| `PleIAs/Post-OCR-Correction` | en 31.3K primary | Noisy OCR-text index + renderer noise strings + (optional) byt5 corrector data | No (corrector only) |
| `nielsr/docvqa_1200_examples` ⚠️ | ~1.2K | Retrieval (query→span) fine-tune/eval + snippet-localization eval | Yes, **research_only** |
| `nielsr/funsd-layoutlmv3` ⚠️ | small | Fast offline form-collection integration test | No, **research_only** |
| `aharley/rvl_cdip` ⚠️ | ~400K images | OCR-at-scale / index-scaling stress test | No, **research_only** |

The **English** subset is the supported target throughout. fr/it/de text exists only via the CC0 Post-OCR corpus and the renderer; multilingual production support is **not** claimed (Tesseract language packs, tokenization, and the en-only retriever are not validated cross-lingually).

---

## 8. Licensing summary and flags

| License class | Sources | Action |
|---|---|---|
| **MIT** | `TextOCR_OCR`, `IIIT5K_OCR` | Ship in default stack. |
| **CC0-1.0** | `Post-OCR-Correction` | Ship (public domain). |
| **CC-BY-4.0** | `cord-v2` | Ship **with attribution** if redistributed. |
| **Self-generated** | synthetic generator | Ship (MIT-compatible; primary offline data). |
| **Unspecified — FLAG** | `docvqa_1200_examples`, `funsd-layoutlmv3` | `research_only`; verify license before commercial use; do not redistribute. |
| **other / non-commercial — FLAG** | `rvl_cdip` (tobacco-corpus NC) | `research_only`; never ship; do not redistribute. |
| **No license tag (lower priority)** | `COCO-Text` | Optional research-only; not in default stack. |

**Net posture.** The shipped demo, automated tests, and default fine-tune use **only** MIT + CC0 + CC-BY-4.0 + synthetic data. Every non-commercial or undeclared-license set is behind the `research_only` flag, excluded from the default path, and not redistributed. This keeps the default P20 stack fully commercially usable while still allowing the richer real signals (DocVQA query→span supervision, RVL-CDIP scale) for research and evaluation.

### Privacy note on real document data

OCR-indexing real user/scanned documents is **sensitive** — the OCR text can contain PII (ids, contracts, medical/financial records). The flagged real document corpora (`cord-v2` receipts, `docvqa_1200_examples`, `funsd`, `rvl_cdip`) are research/eval material only; any production deployment over user documents requires consent, access control, PII redaction, no query retention by default, and the LLM brain kept off. The synthetic generator carries none of this risk, which is a further reason it is the primary offline data.

---

## 9. Reuse note

The data layer reuses sibling code: the synthetic renderer (`render_page`, `save_png_with_spec`, `generate_dataset`) and offline OCR (`SeedEngine`, `_noisify`, `load_ocr_engine`, `has_spec`) come from **P15 `imgtrans`**; the OCR-quality metrics (`corpus_cer`, `corpus_wer`, `normalize_ws`) from **P07 `dococr`**; the retrieval metrics (`recall_at_k`, `mrr`, `median_rank`, `rank_of`) and the BM25 / dense index (`lexical.py:BM25Index`, `image_index.py:ImageIndex`) from **P19 `clipsearch`**. **New for P20** at the data layer is the rendered-text **needle/gold-map generator** (`imgsearch_spec` namespace, target-term recurrence control, gold query→image map) — the layer that turns the P15 renderer into a retrieval benchmark with a known answer key.
