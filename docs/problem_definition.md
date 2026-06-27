# P20 — Problem Definition: Text Search within Images

**Project:** P20 "Text Search within Images" · package `imgtextsearch` · folder `20_Text_Search_in_Images`
**Author:** Le Dinh Minh Quan (student 23127460)

---

## 1. The problem in one sentence

Given a **collection of images** — scanned documents, receipts, forms, screenshots, born-digital PDF pages, scene/street photos — and a **text query** (a word, date, number, ID, or short phrase), return the **ranked subset of images whose visible text literally contains that query**, each with a **highlighted matching snippet**, or **abstain** when no image contains it.

Concretely, P20 answers questions like:

- *"Find images containing the word **INVOICE**."*
- *"Which receipts **mention a 2023 date**?"*
- *"Show me the page that contains **PO-4471**."*
- *"Find screenshots that say **CONFIDENTIAL**."*

The user is not asking *what an image depicts*; they are asking **which images literally CONTAIN this text**. That distinction is the whole project.

---

## 2. OCR-based text-in-image search vs. visual/semantic search (P19)

P20 has a sibling, **P19 `clipsearch`** (Text-to-Image Retrieval with CLIP). The two solve fundamentally different problems and must not be conflated.

| Dimension | **P19 — Visual/semantic search (CLIP)** | **P20 — OCR text-in-image search (this project)** |
|---|---|---|
| Question answered | "Which image **depicts** this?" | "Which image **literally contains** this text?" |
| Query refers to | The **visual content** (a dog on a beach, a red car) | The **characters/words printed inside** the image |
| Shared vocabulary | **None** — query words and pixels live in different spaces; matching is cross-modal | **Yes** — the query and the indexed OCR text live in the **same lexical space** |
| Core signal | An image embedding (CLIP) aligned to a text embedding | The **OCR'd text** of each image, indexed as a document |
| Strongest single arm | Dense cross-modal embedding; BM25 over captions was only a weak floor | **BM25 over OCR text** — the literal-term arm is the *lead* signal, not a baseline |
| Correct answer test | Visual/semantic relevance | **Literal containment** of the query term in the image's text |
| Example | "a photo of a beach at sunset" → beach photos | "the word INVOICE" → images whose text *prints* "INVOICE" |

Because P20's query and its index share a lexical space, **exact-term matching (BM25) is the strongest single signal**, and the success criterion is *literal presence of the term in the text*, not vague topical similarity. A naive semantic retriever would happily return an image that is *about* billing for the query `INVOICE` even when the word never appears on it — which is exactly the failure mode P20 is built to prevent.

**Boundary rule.** P20 does **not** reuse P19's image-embedding index or its visual-semantic datasets. P20 indexes **OCR text**, not pixels.

---

## 3. Why OCR → text index (and not pixels)

The query is text and the answer is "does this image's text contain it." The only honest way to answer is to **read the text out of each image** and search that text:

1. **OCR each image once** (Tesseract by default; neural upgrades available) → per-image OCR text + per-word bounding boxes + confidence.
2. Treat **each image's OCR text as one "document."** The collection becomes a text corpus where each document is tied back to an `image_id`.
3. Build a **text index** over those documents: a **BM25** (sparse, exact-term) arm + a **dense** (semantic, typo-robust) arm, fused with **Reciprocal Rank Fusion (RRF)**.
4. At query time, **rank the images** by their OCR-text relevance, then **verify** the query term literally appears in each candidate and **highlight the matching snippet**.

This is **document search over the text inside images**. The pixels matter only insofar as OCR converts them to text; once converted, the problem is a (well-understood) text-retrieval problem with one twist: the text is **noisy**, because OCR makes mistakes (`INVOICE` → `INV0ICE`, `m` → `rn`, `0` → `O`). That noise is what makes the dense arm, RRF, and a **fuzzy literal-verification** step earn their keep.

---

## 4. Real-world use cases

P20 targets situations where an organization or user has a **pile of images** and needs to find the ones containing specific text:

- **Document / receipt / invoice archives.** "Find every receipt that mentions `TOTAL` or an amount over a threshold"; "pull the invoice with reference `PO-4471`." (Corpus analog: CORD-v2 receipts.)
- **Scanned-contract search.** Locate the scanned page containing a clause, a counterparty name, an effective date, or a signatory — across thousands of scans with no text layer.
- **Screenshot search.** Find the screenshot that says `Error 0x80070057`, or the one mentioning a meeting date, in a folder of captures.
- **Compliance / e-discovery.** Search a scanned-document corpus for regulated terms, case numbers, or dates; return the exact pages with highlighted snippets as evidence (with abstention when a term is genuinely absent).
- **Photo libraries with signage / text.** Find street photos containing a shop name, a license plate fragment, a street sign, or a poster's wording. (Corpus analog: TextOCR / IIIT5K scene text.)
- **Forms and structured documents.** Find the form that contains a field value or token. (Corpus analog: FUNSD — research-only.)

In all of these, **visual/semantic search (P19) is the wrong tool** — the user knows the *text* they want, and a near-miss "looks similar" result is not acceptable.

---

## 5. Inputs and outputs

### Inputs

- **A collection of images** (indexed once, up-front): raster images (PNG/JPG), scanned-document pages, and born-digital PDF pages (which may already carry a text layer).
- **A query** at search time: a **word, date, number, ID code, or short phrase** (e.g. `INVOICE`, `2023-04-12`, `$1,250.00`, `PO-4471`, `"purchase order"`).

### Outputs

For each query, one of:

- **A ranked list of images** whose OCR text literally contains the query, each annotated with:
  - the **image id**,
  - a **fusion (RRF) score** and a confidence label (high / medium / weak),
  - the **matching OCR snippet** with the matched span **highlighted** (plus the word bounding box),
  - the **OCR confidence** of that image, and a `fuzzy_match` / `needs_review` flag where applicable; **or**
- an explicit **abstention**: *"not found in any image"* + `needs_review=True`, returned **instead of** a misleading semantically-nearest image when no image literally contains the query.

The snippet highlight, the exact-match flag, and the abstention are the project's user-facing value: the system explains *why* an image was returned and is honest when the answer is "none."

---

## 6. Scope and non-goals

### In scope

- **English-first, literal-term search** over OCR text.
- **Scanned + born-digital + scene-text** images (born-digital PDFs are routed to read their existing text layer and skip OCR).
- **Snippet highlighting** of the matched span.
- **Abstention** when no image contains the query.
- **Hybrid BM25 + dense + RRF** ranking, with **OCR-noise-tolerant** literal verification.

### Out of scope (non-goals)

- **Visual / semantic image search** — that is P19 (`clipsearch`). P20 never matches on what an image *depicts*.
- **Visual question answering / answer generation** — P20 retrieves and highlights; it does not synthesize answers.
- **Full layout parsing / table extraction / key-value structuring** — OCR text is concatenated per image; structured document understanding is out of scope.
- **Multilingual production support** — English is the primary, validated target. French/Italian/German text can be *rendered* (synthetic) or appear via the CC0 Post-OCR corpus, but cross-lingual OCR, tokenization, and retrieval are **not** validated and are treated as research-only.

---

## 7. Success criteria

P20 succeeds when it **finds the images that literally contain the query, ranks them well, and abstains honestly** — measured by a metric suite that combines retrieval quality, OCR quality, and literal-match faithfulness.

### Retrieval quality (primary)

- **Recall@K for K ∈ {1, 5, 10}.** A query *hits* if **any** image whose OCR text literally contains the query (the gold set) appears in the top-K; the rank used is the first gold image. Primary headline metric.
- **MRR (Mean Reciprocal Rank).** Rewards placing a correct image at the very top — the right behavior for "find **the** image with this word."
- **Median / mean rank** of the first gold image — diagnostic complements (median robust to the not-retrieved tail; mean exposes catastrophic misses).

### OCR quality (upstream cap)

- **CER (Character Error Rate)** and **WER (Word Error Rate)** of the OCR text vs. the rendered/known gold text — they bound how well any term index can perform. On synthetic data, an injected character-noise rate maps to a controllable CER/WER.

### Literal-match faithfulness (P20-specific verification)

- **Exact-Match precision@K** — of the top-K images **returned** for a query, the fraction whose OCR text **literally contains** the normalized query (case/whitespace-normalized, token-boundary-aware). Unlike Recall (which consults the gold map), this audits the **returned** list directly against the literal query — catching the dense arm's semantically-similar-but-wrong results and quantifying the value BM25 + verification add.

### Abstention

- On a **nonexistent term** (e.g. `zzqwx`), the system must **abstain** ("not found" + `needs_review`) rather than return the least-irrelevant image. Correct abstention is a first-class success condition, not an error.

### Baselines to beat / match

| Baseline | Role |
|---|---|
| **BM25-only** | The strong lexical floor (genuinely competitive for literal-term search). Hybrid must **beat it** on OCR-noise / paraphrase Recall while **matching** its Exact-Match precision. |
| **Dense-only** | Isolates the trained retriever; exposes its semantic-false-positive weakness (high Recall, lower Exact-Match precision) that verification + BM25 correct. |
| **Random** | Sanity floor for Recall@K / MRR. |

### Verified offline seed (sanity floor met)

A no-dependency offline seed (BM25 over per-image OCR text on the synthetic collection) already achieves **Recall@1 = MRR = 1.0** for unique target codes, **highlights the matching snippet**, **correctly abstains** on a nonexistent term (`zzqwx`), and exercises **all five agent decision points** — establishing that the pipeline and metrics are wired end-to-end with zero heavy dependencies.

---

## 8. Why this is hard (the core challenges)

1. **OCR errors break exact matching.** A single character confusion (`INV0ICE`, `lnvoice`, `rn→m`) makes a pure literal/BM25 match miss a term that is genuinely present. P20 mitigates with the dense arm + RRF (recover near-misses), a **fuzzy-tolerant** literal verifier (bounded edit distance, case/diacritic-insensitive), and optional post-OCR correction.
2. **Multi-word / phrase queries.** Phrase order, hyphenation, and OCR line breaks complicate "literal" containment; the system distinguishes quoted-phrase from keyword intent and is token-boundary-aware.
3. **Born-digital vs. scanned routing.** PDF pages with a real text layer should be read directly (no OCR); raster/text-less pages must be OCR'd. The correct route avoids both wasted OCR and missed text.
4. **Semantic false positives.** A dense retriever always returns *k* nearest images, even when the literal term is absent — unacceptable here. Literal verification + abstention are the fix.
5. **Language / script and image quality bias.** OCR quality varies by script, font, and scan quality, producing **unequal search recall** across content; English is the validated target.
6. **Index scaling.** Exact cosine is O(N) and OCR-ing very large collections is a heavy batch job; the design keeps a no-GPU BM25 floor and an ANN-swappable dense arm.

---

## 9. The system's value-add (what makes the answer trustworthy)

Beyond ranking, P20 adds three things a blind semantic retriever cannot:

1. **Literal-term verification** — re-read each candidate's OCR text and confirm the query term actually appears (fuzzy-tolerant to OCR noise); **drop** semantic false positives.
2. **Snippet highlighting** — return the exact matching span + word bounding box + OCR confidence, so a result is **explainable**, not an opaque neighbor id.
3. **Abstention** — when no image literally contains the query, return "not found" + `needs_review` rather than the least-irrelevant image.

Net: P20 answers **"which images literally contain this text"** honestly, instead of "which images are vaguely about this topic."

---

## 10. Assignment mapping

| Assignment requirement | How P20 satisfies it |
|---|---|
| **A real, useful NLP task** | OCR-based document/text-in-image retrieval — find images by the text they contain (invoices, receipts, contracts, screenshots, compliance/e-discovery, signage). |
| **A trainable model core** | Exactly **one** trained component: a **dense text retriever** (sentence-transformers bi-encoder, default `BAAI/bge-small-en-v1.5`, MIT) fine-tuned with MultipleNegativesRankingLoss on (query, OCR-text) pairs. Everything else (OCR, BM25, RRF, verifier) is pretrained/algorithmic. |
| **A pretrained component** | OCR front-end (Tesseract / TrOCR / docTR / PaddleOCR), the reranker, and optional post-OCR corrector — used, not trained. |
| **A retrieval / index component** | Hybrid **BM25 (lead exact-term arm) + dense + RRF** over per-image OCR text. |
| **An agentic component** | A deterministic 5-decision FSM (parse → search → coverage → verify → finalize) with a replayable trace; value-add = literal verification + snippet highlighting + abstention. |
| **Evaluation with metrics & baselines** | Recall@{1,5,10} + MRR + median/mean rank + OCR CER/WER + **Exact-Match precision@K**; baselines = BM25-only, dense-only, random. |
| **Reproducible offline path** | A **synthetic rendered-text-image generator** with an embedded gold text spec read back by an offline SeedEngine — runs with **no Tesseract, no torch, no network**, providing the otherwise-nonexistent gold *query → image-containing-it* map. |
| **Deployment** | FastAPI (`POST /search` → ranked image ids + scores + snippet + exact-match flag), Gradio UI, Docker (tesseract-ocr + fonts + libGL), HF Space. |
| **Ethics / privacy / robustness** | OCR-indexing of user/scanned documents is sensitive (PII, IDs, contracts, medical/financial records) → consent, access control, PII redaction, no query retention by default, LLM brain off; bias from script/font/scan-quality acknowledged; OCR-error false negatives/positives mitigated by fuzzy verification + abstention. |

### License posture (flagged, per ground-truth facts)

The default/demo stack is **fully permissive** (MIT / Apache / CC0 / CC-BY-4.0 + synthetic). The following are **flagged** and gated behind a `research_only` switch (not shipped commercially):

- **Surya** OCR (`vikp/surya_*`) — **CC-BY-NC-SA-4.0** (non-commercial + ShareAlike) → excluded from the default stack.
- **`nielsr/docvqa_1200_examples`** — license **unspecified** (closest real query→OCR-snippet signal; research-only).
- **`nielsr/funsd-layoutlmv3`** — research-use only (derived from the IIT-CDIP/RVL-CDIP legal-tobacco corpus).
- **`aharley/rvl_cdip`** — license `other` (non-commercial; tobacco-document derivation).
- **`howard-hou/COCO-Text`** — no license tag.

---

*This document defines **what** P20 solves and **why**. The pipeline architecture, datasets, models, metrics, agent, offline design, reuse map, and risks are specified in `DESIGN_BRIEF.md`.*
