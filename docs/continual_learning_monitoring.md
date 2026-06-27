# Continual Learning & Monitoring — P20 Text Search within Images

> Package `imgtextsearch` · Author: Le Dinh Minh Quan (23127460)
> Scope: how the P20 OCR-document-search system is kept healthy and improving *after* the first deploy — what we measure, how the index stays fresh as the image collection grows, how we turn user corrections into training signal, and when we re-fine-tune the dense retriever and the post-OCR corrector.

P20 searches a **collection of images by the text they literally contain** (via OCR), e.g. *"find images containing the word INVOICE"* or *"receipts mentioning a 2023 date"*. The only **trained** component is the dense text retriever (a `BAAI/bge-small-en-v1.5` bi-encoder over per-image OCR text); the OCR front-end (Tesseract / TrOCR), BM25, RRF fusion, the cross-encoder reranker, and the D1–D5 agent are pretrained or algorithmic. Continual learning in P20 therefore has a narrow trainable surface (the retriever, plus an optional `byt5-small` post-OCR corrector) but a wide **monitoring** surface, because the system can degrade through three independent channels that have nothing to do with model weights:

1. the **query distribution** drifts (users start asking for new ID formats, dates, phrasings);
2. the **document distribution** drifts (new images, new document types, new vendors/fonts/scripts enter the collection);
3. the **OCR quality** drifts (a new scanner, a worse camera, a degraded batch, or conversely an OCR-engine upgrade that changes the indexed text under our feet).

This document describes the monitoring signals that detect each channel, the index-freshness machinery that keeps the searchable corpus current, the feedback loop that captures corrections, and the periodic retraining cadence.

---

## 1. What we monitor

Every served query and every offline eval run emits a structured **job log** (one JSON line per query) consumed by `monitoring/drift_report.py`. The agent already writes a deterministic `ToolTrace` per request (decision id, branch taken, signal value, threshold) at each of the five decision points D1–D5; the monitoring layer aggregates those traces plus latency timers into time-bucketed metrics. Nothing here requires gold labels — these are **production health signals computed from the trace alone**, so they run continuously without human annotation.

### 1.1 Core health signals (from job logs → `drift_report.py`)

| Signal | Where it comes from | What it tells us | Alert condition |
|---|---|---|---|
| **Abstain rate** | D5 returns "not found in any image" + `needs_review` | Fraction of queries where *no* image's OCR text literally contained the term. Healthy baseline is non-zero (honest abstention is a feature). | Sustained spike vs the trailing baseline → either OCR is dropping terms (false negatives) or queries have drifted to vocabulary the collection genuinely lacks. |
| **Low-confidence / needs_review rate** | D1 `low_ocr_confidence`, D4 `fuzzy_match`, D5 weak-only finalize | Fraction of queries answered only via fuzzy/near-match or over low-OCR-confidence images. | Rising trend → OCR quality degradation in newly ingested images; triggers a re-OCR / source audit. |
| **Exact-match rate (ExactP@K in production)** | D4 literal-verification outcome over the returned top-K | Fraction of *returned* images whose OCR text literally contains the normalized query. This is the production analogue of the offline **Exact-Match precision@K** metric and the single most P20-specific quality number. | Drop → the dense arm is increasingly surfacing semantic-but-wrong images that D4 then has to drop, i.e. retriever drift. |
| **Top-score drift** | D3 raw signals — **raw BM25 hit count** and **raw dense cosine** of the top candidate (never the post-RRF fused score) | Distribution of the top-candidate raw cosine and raw BM25 hits over time. | A downward shift in raw cosine across the query stream is the earliest retriever-drift / corpus-drift signal — it moves before Recall does. |
| **Widen rate** | D3 "WIDEN once" branch fired | How often the shortlist was too weak on the first pass and had to be relaxed (phrase→keywords / widen k). | Rising → queries are landing in coverage gaps; correlates with both query drift and OCR drift. |
| **Fuzzy-match rate** | D4 `fuzzy_match` flag (literal term found only within bounded edit distance) | How often exact BM25 matching failed and the bounded-edit-distance fallback rescued the result. | Rising → OCR is introducing more character confusions (`INV0ICE`, `rn→m`); a direct proxy for indexed-text CER without needing gold. |
| **Latency (p50/p95/p99)** | Per-stage timers: OCR (ingest only), BM25, dense encode+search, RRF, rerank, D4 verify | End-to-end and per-stage query latency. | p95 regression localizes to a stage — dense-encode latency rising hints at a model/index-size change; verify latency rising hints at long OCR docs. |
| **Index size / coverage** | Index metadata (`#images`, `#excluded`, mean OCR confidence) | How many images are searchable, how many were excluded (empty/unreadable at D1), and collection-wide mean OCR confidence. | Excluded-image count climbing → an ingest or OCR-source problem upstream. |

`monitoring/drift_report.py` buckets these by time window (hourly/daily) and writes a report comparing the **current window** against a **trailing reference window** (e.g. the prior 7 days), surfacing per-metric deltas, a population-stability-style score on the raw-cosine and abstain-rate distributions, and a ranked list of the queries that most contributed to any regression. It runs both on the live job-log stream and on the synthetic offline eval (Section 4 of the design brief), so the same report format covers CI and production.

### 1.2 Labelled signals (when gold is available)

When an evaluation slice with a gold query→image map is available — the synthetic generator's recorded gold map, or `nielsr/docvqa_1200_examples` (research-only, gated) — `drift_report.py` additionally reports the full offline metric suite so we can correlate the unlabelled production signals above with ground-truth quality:

- **Recall@{1,5,10}, MRR, median rank** of the gold image (reused from `19_.../clipsearch/training/metrics.py`).
- **Exact-Match precision@K** (the literal-containment audit of the returned list).
- **OCR CER / WER** of the indexed text vs gold (reused from `07_.../dococr/training/metrics.py:corpus_cer/corpus_wer`).

The labelled and unlabelled signals are designed to move together: e.g. a real CER increase should show up *first* as a rising production fuzzy-match rate (unlabelled, immediate) and *later* be confirmed by a CER regression on the next labelled eval. That correlation is what lets us trust the unlabelled signals as early-warning triggers.

---

## 2. The index-freshness problem

The search collection is **not static**: new images arrive continuously (new scans, new uploads, new PDF pages), and the OCR engine itself improves over time. Both events stale the index. Because the searchable unit is the **per-image OCR text** (the "document" = concatenated OCR text + `image_id` + word boxes + mean confidence), keeping the index fresh means keeping that derived text current — and re-deriving it when the way we read images changes.

### 2.1 Incremental ingest of new images

New images must not trigger a full rebuild. The freshness pipeline is **incremental and append-only** per image:

1. **Route (D1).** PyMuPDF checks each new PDF page for a real text layer ≥ char threshold → read the text layer directly and **skip OCR**; raster images and text-layer-less pages take the OCR path.
2. **OCR the new image only** (Tesseract `image_to_data` by default) → per-word text + bbox + confidence → per-image OCR text. Empty/unreadable → excluded from the index and flagged (counted in the monitoring "excluded" signal).
3. **Append to both arms.** Add the new document to the **BM25** index and append its embedding to the **dense** index (`ImageIndex` / FAISS `IndexFlatIP`, numpy fallback). BM25 corpus statistics (`N`, document frequencies, average length) are updated incrementally; the dense index appends a single L2-normalized vector.
4. **No retrain needed for ingest.** Adding images does **not** require re-fine-tuning the retriever — the bi-encoder embeds the new OCR text with current weights. Retraining is a separate, slower cadence (Section 4).

A small caveat that the monitor watches: BM25's IDF depends on `N` and document frequency, so a large ingest batch shifts IDF for every term. This is cheap to recompute and is recomputed on each incremental commit; it is *not* a reason to retrain anything. For very large collections (the RVL-CDIP-scale 400K stress test), `IndexFlatIP` (exact, O(N) per query) is swapped behind the same `retrieve()` interface for an ANN index (HNSW/IVF) so incremental adds and query latency both stay bounded.

### 2.2 Re-OCR when the engine improves

The indexed OCR text is a *function of the OCR engine*. When we upgrade the engine — Tesseract → `microsoft/trocr-base-printed`, or enable `PP-OCRv5` det+rec on an H100 tier, or turn on the `byt5-small` post-OCR corrector — the same image now produces **different, better** text. The index must be re-derived, not just appended to.

- Each indexed document records the **OCR provenance**: engine id, engine version, corrector on/off, and a content hash of the produced text. `drift_report.py` and the ingest job compare the configured engine fingerprint against each document's stored fingerprint.
- On an engine upgrade, documents whose fingerprint no longer matches are queued for **re-OCR** in the background (batch-OCR the whole collection in one pass on the H100 tier; born-digital text-layer pages are skipped, only genuinely OCR'd images are re-read). Re-OCR'd documents replace their old text in both arms.
- Re-OCR is also triggered **selectively** by monitoring: images that repeatedly produce `low_ocr_confidence` (D1) or that show up only via `fuzzy_match` (D4) are prioritized for re-OCR with a stronger engine, rather than re-OCR-ing the whole collection.

Because the offline path embeds the gold text inside each synthetic PNG and reads it back with `SeedEngine`, re-OCR logic is fully testable with **no OCR binary**: the `_noisify` rate is the CER knob, so we can simulate an "engine improvement" by lowering the noise rate and assert that re-OCR'd documents now satisfy previously-failing exact-term queries (BM25 starts matching `INVOICE` where it previously only got `INV0ICE`).

### 2.3 Freshness SLAs

- New image searchable within the incremental-ingest cycle (minutes for a small batch; the batch OCR job for large drops runs on a schedule).
- Re-OCR backlog after an engine upgrade is drained as a low-priority background job; until a document is re-OCR'd, it is served with its old text and tagged in the trace so its (possibly stale) contribution to monitoring is identifiable.

---

## 3. Feedback capture → hard pairs

P20 is a search system, so the most valuable continual-learning signal is **what users tell us was wrong**. Two failure modes are worth capturing, and both map cleanly onto contrastive training pairs for the retriever.

### 3.1 What we capture

| Failure the user reports | Meaning | Training signal it produces |
|---|---|---|
| **Missed result (false negative)** | "Image X *does* contain my query but you didn't return it (or ranked it low / abstained)." | A **hard positive**: `(query, OCR-text-of-image-X)` that the retriever should rank highly. Especially valuable when BM25 missed it due to OCR noise — exactly the dense arm's job. |
| **Wrong result (false positive)** | "Image Y was returned but does *not* contain my query." | A **hard negative**: `(query, OCR-text-of-image-Y)` the retriever should rank *below* true positives. These are the semantic-but-wrong images D4 already drops; making them hard negatives teaches the bi-encoder not to surface them in the first place. |

Capture happens via the Gradio UI (a thumbs-up/down + "this one's wrong" / "you missed one" control on each returned image and on an abstention) and via the FastAPI `/search` response carrying stable `image_id`s so a client can post structured corrections back. Each correction is logged with the query, the implicated `image_id`, the image's stored OCR text, the OCR provenance fingerprint, and the agent's trace for that query (so we know whether the miss was a BM25 miss, a dense miss, a D3 abstain, or a D4 drop).

### 3.2 Turning corrections into a training set

Corrections are accumulated into a **hard-pair store** and periodically distilled into MNRL/InfoNCE training tuples for the retriever (same loss already used in P18/P19):

- A reported **missed image** → an additional positive for that query, mined alongside in-batch negatives.
- A reported **wrong image** → an explicit **hard negative** for that query (these are far more informative than random negatives because they are the retriever's actual mistakes).
- Corrections are de-duplicated by `(query_norm, image_id)` and aged out / re-validated when the underlying image is re-OCR'd (an old hard negative may no longer apply once the OCR text changes).

A crucial guard: a correction's positivity is checked against the **gold definition of the task** — "the OCR text literally contains the query." If a user marks an image as a missed result but its (current, correct) OCR text genuinely does not contain the term, that is an OCR/recall problem (route to re-OCR), **not** a retriever-training positive. We never train the retriever to "find" a term that isn't in the text; that would corrupt the literal-search contract. This routing — correction → (retriever hard pair) vs (re-OCR queue) vs (post-OCR-corrector example) — is decided from the trace and the verified containment check.

---

## 4. Periodic re-fine-tuning

### 4.1 Retriever re-fine-tuning

The dense retriever is re-fine-tuned on a cadence, not continuously, to keep the served model stable and auditable:

- **Trigger.** Either scheduled (e.g. monthly) or signal-driven: a sustained drop in production **exact-match rate** or **top raw-cosine**, a rising **widen rate**, or accumulation of enough new hard pairs from the feedback store (Section 3).
- **Data.** The base fine-tune set (positive `(query, OCR-text)` pairs from `MiXaiLL76/TextOCR_OCR`, `cord-v2` receipts, and the synthetic generator's gold map) **plus** the accumulated hard positives and hard negatives from feedback, **plus** fresh OCR text from newly ingested document types so the retriever sees the current document distribution.
- **Loss.** MultipleNegativesRankingLoss (InfoNCE) over the bi-encoder, identical to the P18/P19 recipe; hard negatives from feedback are injected into the batch.
- **Gate before promotion.** A new retriever checkpoint is **never** shipped on training loss alone. It must (a) not regress Recall@{1,5,10}/MRR on the frozen synthetic eval and the labelled `docvqa_1200_examples` slice, and (b) **not lower Exact-Match precision@K** — the literal-search guarantee is non-negotiable, and a retriever that improves semantic recall while smearing exact-term precision is rejected. Because BM25 + D4 verification sit downstream, the retriever change is also evaluated *in the full pipeline*, not in isolation.
- **Rollout.** Promote behind the registry with the ability to roll back; the BM25 arm and the D4 verifier are unchanged, so worst case the system degrades to BM25-only behaviour, which is a strong literal-search floor (in P20, BM25 is the lead arm, not a weak baseline). This makes retriever rollouts low-risk.

The OCR front-end, BM25, RRF, reranker, and the D1–D5 agent are **not** retrained — they are pretrained/algorithmic. Continual learning touches only the retriever weights and the optional corrector below.

### 4.2 Post-OCR corrector updates

The optional `google/byt5-small` post-OCR corrector (byte-level seq2seq, cleans `rn↔m`, `0↔O`, `1↔l/I` before indexing) is the second trainable surface. It is updated when monitoring shows the OCR-noise channel degrading (rising fuzzy-match rate, rising CER on labelled slices, recurring confusions in specific document types):

- **Training data.** Real `(text, corrected_text)` pairs from `PleIAs/Post-OCR-Correction` (CC0), augmented with confusions observed in production (mined from corrections that were routed to "re-OCR / correction" rather than to retriever training in Section 3.2), and with the synthetic generator's controllable `_noisify` confusions for which the clean gold is known exactly.
- **Effect on the index.** Enabling or updating the corrector changes the indexed OCR text, so it triggers the **re-OCR / re-index** path of Section 2.2 (provenance fingerprint changes → documents re-derived). The win is measured directly: more exact BM25 hits, lower fuzzy-match rate, higher Exact-Match precision@K.
- **Gate.** The corrector must lower CER/WER on the labelled slice **without** introducing over-correction that erases a genuinely-present rare token (a corrector that "fixes" a real part number `PO-4471` into a common word is worse than the original noise). Exact-Match precision@K on the literal needles is the guard here too.

---

## 5. Drift: the three channels, watched together

P20 can degrade through three independent distributions. The monitoring design's purpose is to attribute a quality drop to the right channel so the right remedy (re-OCR vs retrain vs index update) fires.

### 5.1 Query distribution drift

Users start asking for terms, ID formats, dates, or phrasings the system wasn't tuned for (a new vendor's `REF########` code scheme, a new fiscal year, more natural-language paraphrase queries like *"images mentioning a 2023 date"*).

- **Detection.** Shift in query-token IDF distribution and OOV-rate against the indexed vocabulary; rising **widen rate** (D3) and **abstain rate** (D5) concentrated on a new query cluster; `drift_report.py` surfaces the new high-frequency tokens.
- **Remedy.** If the terms exist in the collection but rank poorly (a retriever/paraphrase gap), feed the new query patterns into retriever re-fine-tuning (Section 4.1). If the terms genuinely aren't in any image (honest abstention), no model change is needed — the abstention is correct, and the signal is product feedback, not a defect.

### 5.2 Document distribution drift

New document types, vendors, layouts, fonts, or scripts enter the collection (receipts → contracts, a new printer, a non-Latin script).

- **Detection.** Shift in indexed-text length / token distribution; new clusters in the dense embedding space; rising **excluded-image** and **low_ocr_confidence** counts on the new images; falling top raw-cosine for queries against the new subset.
- **Remedy.** Re-fine-tune the retriever on OCR text sampled from the new document distribution (Section 4.1). If the new documents are a new **script/language**, that is explicitly out of the supported (English-first) scope — flag as research-only rather than silently degrading; do not claim multilingual production recall.

### 5.3 OCR quality drift

The most P20-specific channel: the *same* search logic silently gets worse because the **indexed text** got worse — a new scanner, lower-quality cameras, a degraded scan batch, or compression artifacts raise CER, so literally-present terms stop matching.

- **Detection.** Rising **fuzzy-match rate** (D4) and **low_ocr_confidence** rate (D1) are the immediate unlabelled proxies; confirmed by a CER/WER regression on the next labelled slice; per-source breakdown in `drift_report.py` localizes the bad batch.
- **Remedy.** Route affected images to **re-OCR with a stronger engine** (Section 2.2) and/or update the **post-OCR corrector** (Section 4.2). This channel is typically *not* fixed by retraining the retriever — the retriever can't recover a character the OCR never produced; the fix lives in the OCR/correction layer. The fuzzy-tolerant D4 verifier and D5 abstention keep the system *honest* while the re-OCR drains (it reports low-confidence rather than silently returning wrong images), but they are mitigation, not a cure.

### 5.4 Attribution summary

| Symptom (monitored) | Most likely channel | Remedy |
|---|---|---|
| Abstain rate up, terms genuinely absent | Query drift (correct abstention) | None — product signal |
| Exact-match rate down, abstain flat | Retriever drift / query-paraphrase gap | Re-fine-tune retriever (gate on ExactP@K) |
| Fuzzy-match + low-OCR-confidence up | OCR quality drift | Re-OCR stronger engine / update byt5 corrector |
| Excluded-image count up | Ingest / document drift | Audit source; re-OCR; scope-check script |
| Top raw-cosine down on a new cluster | Document distribution drift | Re-fine-tune on new doc distribution |
| Verify-stage latency up | Long OCR docs / index scaling | ANN index swap; snippet-window cap |

---

## 6. Privacy & governance constraints on the loop

OCR-indexing scanned/user documents is **sensitive** — the indexed text can contain PII, IDs, contracts, medical/financial records. This constrains continual learning:

- **No query retention by default.** Production monitoring runs on **aggregate** trace signals (rates, latencies, score distributions), not on stored raw queries. Per-query logs needed for drift attribution are short-lived and access-controlled; the LLM brain is **OFF by default** and never sees retained data.
- **Feedback / hard pairs are PII-bearing.** The OCR text in a captured hard pair can contain PII, so the hard-pair store inherits the collection's access controls and PII-redaction, and is consent-gated. Corrections are usable as training signal only under the same governance as the source documents.
- **License hygiene carries into retraining.** Default training/eval data stays on MIT (`TextOCR_OCR`, `IIIT5K_OCR`) + CC0 (`PleIAs/Post-OCR-Correction`) + CC-BY-4.0 (`cord-v2`, attribution) + synthetic. Every non-commercial / unlicensed set used for *evaluation only* (`docvqa_1200_examples` unspecified, `funsd` research-only, `rvl_cdip` license:other, `COCO-Text` no tag) stays behind the `research_only` flag and is **never** mixed into a shippable retriever's training data. The CC-BY-NC-SA Surya OCR weights are excluded from any production OCR path.
- **Bias monitoring.** OCR quality varies by script, font, and scan quality, which translates directly into **unequal search recall** across document populations. `drift_report.py` breaks the OCR-confidence, fuzzy-match, and exact-match signals down per source/document-type so a subpopulation with systematically worse recall is visible and can be prioritized for re-OCR or excluded from claims of coverage, rather than hidden in a corpus-wide average.

---

## 7. Operational cadence (summary)

| Activity | Trigger | Touches |
|---|---|---|
| Incremental ingest + index append | New images arrive | BM25 + dense index (no retrain) |
| BM25 IDF recompute | Each ingest commit | BM25 stats only |
| Selective re-OCR | `low_ocr_confidence` / `fuzzy_match` recurrence; engine upgrade | OCR text → both arms |
| Full re-OCR / re-index | OCR engine or corrector upgrade (fingerprint change) | OCR text → both arms |
| Drift report | Continuous (hourly/daily windows) + every CI eval | `monitoring/drift_report.py` |
| Retriever re-fine-tune | Scheduled or signal-driven (exact-match/raw-cosine drop, widen-rate rise, enough hard pairs) | Retriever weights only; gated on Recall + ExactP@K |
| Post-OCR corrector update | OCR-noise channel degrading | `byt5-small`; triggers re-index |
| Bias / governance review | Periodic + on new document population | Per-source signal breakdown |

The throughline: **the index stays fresh through incremental OCR + re-OCR, the retriever and corrector improve on a gated cadence from real corrections, and three distinct drift channels are watched separately so the right fix fires** — all while preserving P20's core promise that a returned image *literally contains* the queried text, and that the system abstains honestly when none does.
