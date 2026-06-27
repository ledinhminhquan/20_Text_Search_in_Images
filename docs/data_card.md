# P20 Text Search within Images — Data Card

> Package: `imgtextsearch` · Folder: `20_Text_Search_in_Images` · Author: Le Dinh Minh Quan (student 23127460)
>
> **Task framing.** P20 searches a *collection of images* by the **text they CONTAIN** (via OCR): "which images literally contain the word INVOICE", "receipts that mention a 2023 date", "the page with PO-4471". This is **OCR-based document search**, not visual/semantic content search (that was the sibling project P19 `clipsearch` / CLIP). The unit of retrieval is the **per-image OCR text** ("the document"); the gold signal is the **set of images whose OCR text literally contains the query**.

This card documents every dataset and data-generating component used to build, train, and evaluate P20, plus the intended and out-of-scope uses, license posture, and known biases. It is written to be honest about what is real, what is synthetic, and what is gated.

---

## 1. Data at a glance

| Source | Role | Modality | Size | License | Ship tier |
|---|---|---|---|---|---|
| **Synthetic rendered-text generator** (`data/synth_text_images.py`) | **PRIMARY OFFLINE DATA** — the gold query→image map | image (PNG) + embedded gold text | configurable (CI: ~100s) | code = project license; output = synthetic | **default / CI / demo** |
| `MiXaiLL76/TextOCR_OCR` | Primary scene-text (image, text) corpus; retriever fine-tune pairs | image + gold transcription | 112.7K | MIT | default / demo |
| `MiXaiLL76/IIIT5K_OCR` | Secondary scene-text set; small clean eval/dev collection | image + gold text | 5.5K | MIT | default / demo |
| `naver-clova-ix/cord-v2` | Primary document-image-with-text collection (receipts) | image + word-level JSON | 1.0K | CC-BY-4.0 | demo (attribution) |
| `PleIAs/Post-OCR-Correction` | Realistic OCR-text source (text-only); noise strings | text (raw OCR + corrected) | 50.4K | CC0-1.0 | default / demo |
| `nielsr/docvqa_1200_examples` | Closest real retrieval-over-OCR signal; snippet-localization eval | image + OCR words + query + gold span | 1.2K | **UNSPECIFIED — FLAG** | research-only (gated) |
| `nielsr/funsd-layoutlmv3` | Supplementary form-image collection (tokens + boxes) | image + tokens + bboxes | small | **UNSPECIFIED — FLAG** | research-only (gated) |
| `aharley/rvl_cdip` | Optional 400K scanned-doc OCR-scale stress test | image only (no gold text) | 400K | **other (NC) — FLAG** | research-only (gated) |
| `howard-hou/COCO-Text` | Optional scene-text-in-natural-images collection | image + text | ~16K | **UNSPECIFIED — FLAG** | research-only (gated) |

**License posture (one line):** ship the default stack, demo, and tests on **MIT + CC0 + CC-BY-4.0 + synthetic** only; gate every dataset whose license is **non-commercial or unspecified** behind a `research_only` flag, do not redistribute it, and never include it in a commercial release.

---

## 2. The synthetic rendered-text generator (PRIMARY offline data)

**Why it is primary.** No public "find the image/document containing X" retrieval benchmark with a gold query→image map exists on Hugging Face. The closest real signal (`nielsr/docvqa_1200_examples`) has an unspecified license and only ~1.2K examples. P20's primary offline data is therefore a **synthetic rendered-text-image generator** that *constructs* the gold query→image map by construction, then **embeds the gold text inside each PNG** so an offline OCR stand-in (SeedEngine) reads it back deterministically. This is the same offline contract used by the siblings P15 (`imgtrans`) / P18 (`tkgqa`) / P19 (`clipsearch`): the backbone runs with **no Tesseract, no torch, no network**.

| Property | Specification |
|---|---|
| **Module** | `data/synth_text_images.py` (P20-new vocabulary/gold-map layer over the P15 renderer `15_Document_Image_Translation/src/imgtrans/data/synth_render.py`) |
| **Determinism** | Seeded RNG; identical seed → identical pages, identical gold map, identical OCR read-back. The gold map is recorded at generation time and is **independent of OCR**, so it stays exact even when char-noise is injected. |
| **Vocabulary** | Two pools. (a) Rare **target needles**: `INVOICE`, `RECEIPT`, `CONFIDENTIAL`, `PURCHASE ORDER`, dates (`2023-04-12`, `March 2023`), amounts (`$1,250.00`), ids (`PO-4471`, `REF####` unique codes). (b) Common **fillers** shared across images: `the, total, date, amount, page, customer, qty`. |
| **Snippet builder** | Each image's text = 1–3 short lines = a few fillers + 0–2 targets. Targets recur in a *controlled* number of images (some unique, some shared by `k`), so Recall@K / MRR are non-degenerate and multi-gold queries exist. Realistic noisy strings can be pulled from `PleIAs/Post-OCR-Correction` `text` (CC0) so the index looks like real OCR output. |
| **Rendering** | `render_page(spec)` lays snippet lines onto a white page with a discovered TrueType font (fallback `PIL.ImageFont.load_default`), recomputes per-line bboxes from actual rendered geometry, and stores the spec via `img.info['imgsearch_spec'] = json(spec)`. `save_png_with_spec` writes a PNG `tEXt` chunk so gold text + boxes survive reload. Font size varied 24–40; light `_degrade` (small rotation + GaussianBlur, scaled 0–1) so the real-Tesseract Colab arm sees realistic scans. |
| **Spec namespace** | Spec key renamed `imgtrans_spec` → **`imgsearch_spec`** to namespace P20; structure otherwise identical to P15. |
| **Outputs** | `generate_dataset` writes `page_XXXX.png` + `manifest.jsonl` (one row/image: `filename`, `lines`, gold target-terms present). |
| **Offline OCR read-back** | `SeedEngine` (reused from `15_.../src/imgtrans/models/ocr_engine.py`) pulls the embedded spec and reconstructs `Word(text, conf, bbox, block, line)` per token — a faithful per-image OCR text with **no OCR binary**. `_noisify(token, rate, rng)` injects realistic confusions (`m↔rn`, `0↔O`, `1↔l/I`, `5↔S`, `8↔B`, deletion/duplication) at a **controllable rate** = the CER/WER knob. |
| **Query / gold construction** | Queries = target terms + short paraphrases (`a 2023 date`, `mentions invoice`); `gold(query)` = the recorded image-id set. This is the only source in the project with an exact gold query→image map. |

**Verified offline seed result.** With BM25 over the per-image OCR text on a seed collection: **Recall@1 = MRR = 1.0** (exact term match is perfect for unique codes), the snippet highlights the match, a nonexistent term (`zzqwx`) correctly **ABSTAINS**, and all five agent decision points (D1–D5) fire.

### Limitations vs real scans (state these honestly)

- **Clean, regular layout.** Rendered pages are white-background, single-column, horizontal text in one font family at a time. Real scans have skew, multi-column layouts, tables, stamps, handwriting, bleed-through, dog-ears, JPEG artifacts, and variable lighting that the light `_degrade` (rotation + blur) does not reproduce.
- **OCR noise is synthetic.** `_noisify` injects a *parametric* confusion model. Real Tesseract errors are correlated with font, resolution, and layout, not i.i.d. per character; the synthetic CER is a controllable proxy, not a faithful replica of any one engine's error distribution.
- **Vocabulary is closed and business-doc-skewed.** Needles and fillers are a small business-document vocabulary (invoices/receipts/POs/dates/amounts). It does not cover natural-language prose, scientific notation, non-Latin scripts, or the long tail of real document language.
- **Gold map is term-presence only.** Gold is defined by literal containment of the rendered term; it does not model semantic relevance, partial matches, or human judgments about what "mentions a 2023 date" should return beyond the literal recorded set.
- **The same code path runs offline and on real OCR.** `load_ocr_engine(cfg, engine='auto', image)` auto-selects SeedEngine when `has_spec(image)`, else falls back tesseract → easyocr → stub. This keeps the offline (seed) and Colab/H100 (real Tesseract `image_to_data`) arms honest, but real-OCR CER/WER must be measured on the real corpora below, not inferred from the synthetic knob.

---

## 3. Real datasets — commercially safe (ship in default / demo)

### 3.1 `MiXaiLL76/TextOCR_OCR`
- **Role.** Primary scene-text `(image, text)` corpus. Build the searchable collection from `text`; form positive `(image_text, query)` pairs to fine-tune the dense retriever with MultipleNegativesRankingLoss (InfoNCE). Largest clean MIT `(image, text)` set found.
- **Schema.** `image: Image` · `text: string` (gold transcription).
- **Size.** 112.7K (train 91.4K / test 14.7K / + numbers split).
- **License.** **MIT** — commercially usable, no NC restriction.
- **Provenance.** Derived from the TextOCR scene-text dataset (text annotations over natural-scene / Open Images photos), repackaged on the Hub with one transcription per crop.
- **Known biases.** Scene-text skew: short isolated words/phrases on signage, products, and street scenes rather than dense documents. OCR difficulty driven by perspective, occlusion, and stylized fonts. English-dominant. As OCR-text "documents" these are very short, so per-image text is sparse compared with a real document page.

### 3.2 `MiXaiLL76/IIIT5K_OCR`
- **Role.** Secondary scene-text set; small, clean **eval/dev** collection, drop-in same schema as TextOCR.
- **Schema.** `image: Image` · `text: string`.
- **Size.** 5.5K.
- **License.** **MIT**.
- **Provenance.** Repackaging of the IIIT 5K-word scene-text recognition benchmark (cropped word images).
- **Known biases.** Cropped *single words* — even sparser "documents" than TextOCR; useful for OCR CER/WER sanity and tiny eval, but not representative of multi-line document search. English only.

### 3.3 `naver-clova-ix/cord-v2`
- **Role.** Primary **document-image-with-text** collection. Receipt photos with word-level transcription + layout. Realistic noisy real-world docs for queries like "find receipts containing TOTAL / a price".
- **Schema.** `image: Image` · `ground_truth: string` (JSON with `valid_line[].words[].text` + quad bboxes).
- **Size.** 1.0K (800 train / 100 val / 100 test).
- **License.** **CC-BY-4.0** — commercially usable **with attribution** if redistributed.
- **Provenance.** CORD (Consolidated Receipt Dataset) v2 from NAVER CLOVA; photographed Indonesian retail receipts with structured key-value annotations.
- **Known biases.** **Document-domain skew toward retail receipts** (menu/price/total structure). Photographed (not flatbed-scanned) → perspective, crumpling, thermal-print fade. Locale/currency skew. Small (1K), so it is a realism check and a demo collection, not a training-scale source.

### 3.4 `PleIAs/Post-OCR-Correction`
- **Role.** Realistic **OCR-text source (NO images)**. Real OCR `text` + `corrected_text`. Used to (a) populate a noisy OCR-text index, (b) supply realistic noise strings to the synthetic renderer, and (c) pair the text side with renderer-produced images.
- **Schema.** `text` (raw OCR) · `corrected_text` · `date` · `page` · `file_name` · `word_count`.
- **Size.** 50.4K (en 31.3K / fr 16.5K / it 1.9K / de 672).
- **License.** **CC0-1.0** (public domain dedication) — fully unrestricted.
- **Provenance.** Real post-OCR correction pairs over historical/printed material assembled by PleIAs.
- **Known biases.** **No images** (text-only — cannot be a standalone image collection). Skewed toward historical/printed material; error patterns reflect older print and scan quality. Strongly **English-skewed** with a long multilingual tail (fr/it/de) that is *not* a supported production target for P20 (en-first retriever). Useful precisely because its noise is *real*, but its domain is not modern business documents.

---

## 4. Real datasets — research-only (FLAGGED, gated behind `research_only`)

> **These are non-commercial or unlicensed. They are excluded from the default stack, demo, and tests; gated behind a `research_only` config flag; and must NOT be redistributed or included in any commercial release.**

### 4.1 `nielsr/docvqa_1200_examples` — license UNSPECIFIED (FLAG)
- **Role.** The **closest real retrieval-over-OCR signal** in the project, and the single best fine-tune / snippet-eval set: each row = image + full `words` OCR list + a natural-language `query` + `answer.matched_text` / `start` span. Used as `(query → document/snippet)` supervision AND as snippet-localization eval.
- **Schema.** `image: Image` · `words` (OCR word list) · `query: string` · `answer` with `matched_text` + `start` span.
- **Size.** 1.2K (sampled from DocVQA).
- **License.** **Not specified (no tag on the Hub).** → **research-only; verify licensing before any commercial use.**
- **Provenance.** Sampled from the DocVQA benchmark (scanned industry/government documents, UCSF Industry Documents Library lineage).
- **Known biases.** DocVQA document-domain skew (forms, reports, letters, tables from industry/government archives). Query distribution is VQA-style ("what is the date", "who signed"), not pure literal-term lookup — closest available but not a perfect match for P20's "contains word X" task. English. Small.

### 4.2 `nielsr/funsd-layoutlmv3` — license UNSPECIFIED, research-use (FLAG)
- **Role.** Supplementary **form-image** collection (per-token text + bboxes); tiny, good for fast offline integration tests.
- **Schema.** `image` · per-token `text` + `bboxes` + entity tags.
- **Size.** small (FUNSD is ~199 forms).
- **License.** **Not specified.** FUNSD is **research-use only** (derived from the IIT-CDIP / RVL-CDIP legal-tobacco corpus).
- **Provenance.** FUNSD (Form Understanding in Noisy Scanned Documents), a subset of RVL-CDIP scanned forms, reformatted for LayoutLMv3.
- **Known biases.** Heavy **document-domain skew to scanned business/legal forms** from the tobacco-litigation archive; noisy low-resolution grayscale scans; English; very small. Inherits the tobacco-corpus non-commercial lineage.

### 4.3 `aharley/rvl_cdip` — license `other` (non-commercial, FLAG)
- **Role.** Optional **400K scanned-document OCR-scale stress test**. No gold text → must run OCR; used to exercise batch OCR and index scaling.
- **Schema.** `image` only (no transcription) + a 16-class document-type label.
- **Size.** 400K images.
- **License.** **`other`** — RVL-CDIP / IIT-CDIP derives from the **Legacy Tobacco Documents Library (non-commercial)**; the Hub dataset viewer is disabled.
- **Provenance.** Ryerson Vision Lab Complex Document Information Processing dataset, a labeled subset of IIT-CDIP tobacco-litigation documents.
- **Known biases.** Strong **document-domain skew** (mid/late-20th-century US corporate/legal/tobacco paperwork); grayscale, low-resolution, heavily degraded scans → high and **uneven OCR error rates**. English. Non-commercial provenance and viewer-disabled status make it research-only.

### 4.4 `howard-hou/COCO-Text` — license UNSPECIFIED (FLAG)
- **Role.** Optional scene-text-in-natural-images collection (~16K). Lower priority than `TextOCR_OCR`.
- **Schema.** `image` · `text` annotations.
- **Size.** ~16K.
- **License.** **No license tag.** COCO-Text annotations are research-oriented. → **research-only.**
- **Provenance.** COCO-Text text annotations layered over MS-COCO natural images.
- **Known biases.** Scene-text skew (incidental text in everyday photos); many images contain little or hard-to-read text; English-dominant. Overlaps the TextOCR domain but with weaker licensing, so it is a fallback, not a default.

> **Explicitly avoided / not cited** (404, no license, arbitrary-code loaders, or NC/non-gold): `priyank-m/text_recognition`, `mteb/TextOCR`, `facebook/textocr`, `Cyrius/textocr`, `priyank-m/iiit_5k`, `Teklia/IIIT-5K`, `HuggingFaceM4/IIIT-5K`, `HugoLaurencon/IIIT-5K`, `Berzerker/iiit5k_ocr_dataset`, `jimmycarter/textocr-gpt4v` (CC-BY-**NC**-4.0), `CaptionEmporium/TextOCR-GPT4o` (CC-BY-SA + GPT4o captions, not gold OCR), and mjsynth / SynthText (not cleanly on HF — we generate our own synthetic instead).

---

## 5. Intended use

- **Build a searchable image collection** by OCR-ing each image to a per-image text "document," indexing it (BM25 lead arm + a fine-tuned dense bi-encoder, fused with RRF), and answering literal-term / date / id / short-phrase queries with a **ranked image list + highlighted matching snippet**, or an **abstention** when no image contains the term.
- **Train exactly one component** — the dense text retriever (bi-encoder over OCR text, default `BAAI/bge-small-en-v1.5`, MIT) — using `(query, OCR-text)` positives. Everything else (OCR, BM25, RRF, the snippet/exact-match verifier, the D1–D5 agent) is pretrained or algorithmic.
- **Evaluate** with text-in-image **Recall@{1,5,10}**, **MRR**, **median/mean rank** (gold = images whose OCR text contains the query), **OCR CER/WER** (OCR vs rendered gold text), and **Exact-Match precision@K** (fraction of returned images whose OCR text literally contains the query). Baselines: BM25-only (exact-term floor), dense-only (semantic, no exact guarantee), random.
- **Develop and CI offline** entirely on the synthetic generator + SeedEngine, with no Tesseract / torch / network, using BM25 as the dense stand-in.
- **Demo / commercial-eligible runs** use only MIT + CC0 + CC-BY-4.0 + synthetic sources and the fully-permissive (MIT/Apache) default model stack.

## 6. Out-of-scope use

- **Visual / semantic image search** ("find a photo of a dog on a beach"). That is the sibling P19 `clipsearch`; do **not** route such queries here, and do not pull P19's visual-semantic datasets or image-embedding index into P20.
- **Visual question answering / answer generation.** P20 returns *which images contain the text* + a snippet, not a synthesized answer. (DocVQA is used only as a retrieval/snippet signal, not to build a QA system.)
- **Full layout parsing / table extraction / key-value document understanding.** Word boxes drive snippet highlighting only.
- **Multilingual production support.** English is the supported target. The fr/it/de tail in `PleIAs/Post-OCR-Correction` and the renderer can produce non-EN text, but Tesseract language packs, tokenization, and the en-only retriever are not validated cross-lingually → treat non-EN as research-only; do **not** claim multilingual production support.
- **Commercial use of any FLAGGED dataset** (`docvqa_1200_examples`, `funsd`, `rvl_cdip`, `COCO-Text`) or of Surya OCR (CC-BY-NC-SA, excluded from the default stack). These are gated behind `research_only` and must not be redistributed.
- **Indexing sensitive personal documents without safeguards.** OCR-indexing user/scanned documents exposes PII in the text (IDs, contracts, medical/financial records). Out of scope without consent, access control, PII redaction, and the no-query-retention default; the optional LLM brain stays OFF.

## 7. Known biases and risks across all data

- **OCR-quality bias.** OCR accuracy varies by script, font, and scan quality. Non-Latin scripts and low-quality scans (RVL-CDIP, FUNSD, photographed receipts) yield higher CER/WER → **unequal search recall** across document types and languages. OCR errors cause **false negatives** (a document that *does* contain the term is missed because OCR mangled it) and occasional false positives; the fuzzy D4 verification (bounded edit distance, case/diacritic-insensitive), the dense+RRF arm, optional `byt5-small` post-OCR correction, and abstention mitigate but do not eliminate this.
- **Script-coverage skew.** Every gold-bearing real set (TextOCR, IIIT5K, DocVQA, CORD-v2, FUNSD, RVL-CDIP, COCO-Text) is **English-dominant**; the multilingual signal is confined to text-only CC0 Post-OCR data. Recall is validated for English only.
- **Document-domain skew.** Real corpora cluster into narrow domains — retail receipts (CORD-v2), tobacco-litigation forms (FUNSD, RVL-CDIP), VQA-style industry/government docs (DocVQA), and scene text (TextOCR, IIIT5K, COCO-Text). The synthetic generator adds a **business-document** vocabulary (invoices/receipts/POs/dates/amounts). None of these is a representative sample of arbitrary user documents; measured quality may not transfer to unseen domains.
- **Synthetic-vs-real gap.** The primary offline data is synthetic with clean layout and a parametric noise model (Section 2). It guarantees an exact gold map and zero-dependency CI, but real-scan layout, real-OCR error correlation, and open-vocabulary language are under-represented; real-OCR CER/WER must be measured on the real corpora, not inferred from the synthetic knob.
- **Licensing risk.** Several useful corpora are non-commercial or unlicensed (`docvqa_1200_examples`, `funsd`, `rvl_cdip`, `COCO-Text`) and one OCR model (Surya) is CC-BY-NC-SA. Mitigation: default/demo/tests on MIT + CC0 + CC-BY-4.0 + synthetic only; every flagged set gated behind `research_only`, not redistributed, excluded from commercial release; Surya excluded from the default stack.
