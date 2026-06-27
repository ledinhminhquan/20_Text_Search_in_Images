# P20 Search Evaluation Methodology

> Project: **Text Search within Images** (`imgtextsearch`, folder `20_Text_Search_in_Images`)
> Author: Le Dinh Minh Quan — student 23127460
> Scope: how P20 is measured. This is the *evaluation contract*: what each metric means, how it is computed, which baselines isolate which capability, how OCR noise distorts the literal-match signal and how the fuzzy verifier recovers it, and what the verified offline numbers actually prove.

---

## 1. What we are evaluating

P20 answers exactly one question: **"which images in this collection literally CONTAIN this text?"** — e.g. *find images containing the word `INVOICE`*, *receipts that mention a `2023` date*, *the page with `PO-4471`*. The user query is a word, date, number, ID, or short phrase; the system returns the ranked subset of images whose **OCR'd text literally contains that query**, each with a highlighted snippet — or **abstains** when no image contains it.

This is **OCR-based document search**, not visual/semantic content search. The sibling project P19 (`clipsearch`, CLIP text→image) answers *what the image depicts*; there the query text and the pixels share no vocabulary. P20 is the opposite: **query and indexed OCR text live in the same lexical space**, so the literal token is supposed to physically appear in the document. That single fact drives the entire evaluation design:

- **BM25 is the lead arm and a genuinely strong baseline**, not a weak floor (the inverse of P19).
- The headline retrieval metrics (Recall@K, MRR, rank) are joined by **two P20-specific quality gates**: **OCR CER/WER** (the upstream ceiling on what is even searchable) and **Exact-Match precision@K** (does the *returned* image actually contain the term, not merely rank highly?).
- Because no public "find the image containing X" benchmark with a gold query→image map exists, the **PRIMARY offline evaluation is a synthetic rendered-text-image generator** with an OCR-independent gold map; real corpora (TextOCR, DocVQA) are eval/supervision supplements.

The trainable surface is small: **only the dense text retriever (bi-encoder over OCR text)** is fine-tuned. OCR, BM25, RRF, the cross-encoder reranker, and the snippet verifier are pretrained or algorithmic. The evaluation therefore measures a *mostly deterministic* system around one trained component.

---

## 2. Metric catalogue (with formulas)

Notation. `Q` = query set; `N` = number of images in the collection; for query `q` and gold image `i`, `rank_i(q)` = the 1-based position of image `i` in the ranked result list. An image that is not retrieved at all is assigned a **sentinel rank of 1000** so that "missing" is finite and median/mean stay well-defined. `G(q)` = the **gold set** of images whose OCR text literally contains the query term/phrase (multi-gold by construction — a common term recurs in several images). `topK(q)` = the top-K image ids the system actually returns for `q`. `ocr_norm(d)` / `query_norm` = whitespace- and case-normalized (and diacritic-folded) OCR text of image `d` / query string, via the shared `normalize_ws` helper reused from P07.

### 2.1 Recall@K (text-in-image retrieval) — PRIMARY

A query **hits** at K if *any* image whose OCR text contains the query (the gold set) appears in the top-K.

- single-gold form: `hit@K(q) = 1 if min_{i∈G(q)} rank_i(q) ≤ K else 0`
- multi-gold form: `Recall@K(q) = |{ i ∈ G(q) : rank_i(q) ≤ K }| / |G(q)|`
- corpus: `Recall@K = (1/|Q|) · Σ_q Recall@K(q)`

Reported for **K ∈ {1, 5, 10}**, range `[0, 1]`. Recall is **monotone non-decreasing in K** (`R@1 ≤ R@5 ≤ R@10`); a violation is an automatic red flag for a ranking or gold-map bug and fails the eval sanity check. Reuses `19_Text_to_Image_Retrieval/src/clipsearch/training/metrics.py:recall_at_k`; `rank_of(retrieved_ids, gold_id)` returns the 1-based rank or `None` → sentinel 1000.

### 2.2 MRR (Mean Reciprocal Rank)

`MRR = (1/|Q|) · Σ_q ( 1 / rank*(q) )`, where `rank*(q) = min_{i∈G(q)} rank_i(q)` is the rank of the **first** gold image; the reciprocal is `0` if no gold image is retrieved. Range `(0, 1]`. MRR rewards putting a correct image *at the very top* — the relevant behaviour for "find THE image with this word." Reuses `metrics.py:mrr`.

### 2.3 Median / mean rank of the gold image

`median_rank = median_q rank*(q)`; `mean_rank = (1/|Q|) · Σ_q rank*(q)` (unretrieved → sentinel 1000). The **median** is robust to the sentinel tail and reports typical behaviour; the **mean** deliberately exposes catastrophic misses (every abstention or total miss drags it toward 1000). Reported together as a diagnostic complement to Recall/MRR. Reuses `metrics.py:median_rank` / `mean_rank`.

### 2.4 OCR CER (Character Error Rate) — upstream quality cap

`CER = ( Σ_images Levenshtein(ocr_norm, gold_norm) ) / ( Σ_images len(gold_norm) )`

Character-level edit distance over whitespace-normalized strings, **micro-averaged corpus-wide** (total edits ÷ total reference characters). Range `[0, >1]`; `0` = perfect OCR. CER is the *ceiling* on searchability: characters the OCR mangles can never be matched literally. On the synthetic backbone, the SeedEngine character-noise rate maps directly to a **controllable, known CER**, which is what makes the OCR-noise ablations honest. Reuses `07_Document_Level_OCR/src/dococr/training/metrics.py:corpus_cer`.

### 2.5 OCR WER (Word Error Rate)

`WER = ( Σ_images Levenshtein(ocr_norm.split(), gold_norm.split()) ) / ( Σ_images len(gold_norm.split()) )`

Word-token edit distance, micro-averaged. WER complements CER because **one garbled word can break an exact-term match even at low corpus CER** — a single `INVOICE → INV0ICE` confusion is a tiny CER bump but a total miss for the literal token. CER and WER are always reported **together**. Reuses `metrics.py:corpus_wer`.

### 2.6 Exact-Match precision@K (literal-term presence) — P20-specific verification metric

Of the top-K images **actually returned** for `q`, the fraction whose per-image OCR text literally contains the normalized query:

`ExactP@K(q) = (1/K) · |{ d ∈ topK(q) : query_norm is a substring / word-subsequence of ocr_norm(d) }|`

`ExactP@K = (1/|Q|) · Σ_q ExactP@K(q)`, range `[0, 1]`; case/whitespace-normalized, token-boundary-aware for word queries.

This is the metric that makes P20 *P20*. Unlike Recall (which consults the gold map), Exact-Match precision audits the **returned** list directly against the literal query. It catches the failure mode that a pure dense retriever cannot avoid: returning an image that is *semantically about* the topic but where **the word never appears**. ExactP@K therefore quantifies, in one number, the value BM25 + the D4 verifier add over blind semantic ranking. It is built on the same `normalize_ws` as CER/WER plus a containment check, and is **NEW for P20** (not copied verbatim from a sibling).

> When `dense-only` posts high Recall@10 but low Exact-Match precision, that gap *is* the report's headline: the embeddings are surfacing plausible-but-wrong images, and the hybrid + verifier are what close it.

---

## 3. Metrics summary table

| Metric | Formula (corpus) | Range | What it measures | Reuse |
|---|---|---|---|---|
| **Recall@1** | `(1/|Q|) Σ_q [min_{i∈G(q)} rank_i(q) ≤ 1]` (multi-gold: `|gold≤1|/|G(q)|`) | `[0,1]` | A gold image at the very top | `clipsearch/.../metrics.py:recall_at_k` |
| **Recall@5** | same, K = 5 | `[0,1]` | A gold image in top-5 | same |
| **Recall@10** | same, K = 10 | `[0,1]` | A gold image in top-10 | same |
| **MRR** | `(1/|Q|) Σ_q 1/rank*(q)` | `(0,1]` | How high the *first* gold image ranks | `metrics.py:mrr` |
| **Median rank** | `median_q rank*(q)` (miss → 1000) | `[1,1000]` | Typical rank of first gold; robust to tail | `metrics.py:median_rank` |
| **Mean rank** | `(1/|Q|) Σ_q rank*(q)` (miss → 1000) | `[1,1000]` | Exposes catastrophic misses | `metrics.py:mean_rank` |
| **OCR CER** | `Σ lev(ocr,gold)_char / Σ len(gold)_char` | `[0,>1]` | Char-level OCR error; searchability ceiling | `dococr/.../metrics.py:corpus_cer` |
| **OCR WER** | `Σ lev(ocr,gold)_word / Σ len(gold)_word` | `[0,>1]` | Word-level OCR error; breaks exact match | `metrics.py:corpus_wer` |
| **Exact-Match P@K** | `(1/|Q|) Σ_q (1/K)·|{d∈topK : query_norm ⊑ ocr_norm(d)}|` | `[0,1]` | Do returned images literally contain the term | **NEW (P20)** |

`⊑` = "is a substring / word-subsequence of" after normalization.

---

## 4. Baselines — what each one isolates

P20 evaluates the hybrid retriever against three references. Because P20 is literal-term search, the baseline story is the **mirror image of P19's**: here BM25 is competitive, not a weak floor.

| Baseline | Configuration | What it isolates |
|---|---|---|
| **BM25-only** | Pure-python `BM25Index` over per-image OCR text (`k1=1.5`, `b=0.75`, regex tokenizer, `idf = log(1 + (N−n+0.5)/(n+0.5))`). No GPU, no checkpoint. | The **exact-term arm**. It is *exact-match-faithful by construction* and fires hardest on rare, high-IDF tokens (`INVOICE`, `PO-4471`). It is the **upper bound on Exact-Match precision** and a genuinely strong Recall/MRR competitor. Its weakness is OCR noise and paraphrase: `INV0ICE` and "a 2023 date" do not lexically match. |
| **Dense-only** | The trained bi-encoder (default `BAAI/bge-small-en-v1.5`, 384-d) over OCR-text embeddings; FAISS `IndexFlatIP` on L2-normalized vectors (exact cosine), numpy/TF-IDF fallback offline. | The **semantic / typo-robust arm** = the trained core in isolation. It recovers OCR-noise and paraphrase cases BM25 misses, so it can post **high Recall@K**, but it has **no exact-match guarantee** — it readily returns semantically-similar-but-wrong images, which shows up as **lower Exact-Match precision**. This gap is exactly what D4 verification and BM25 fix. |
| **Random** | Uniform random ranking over the collection. | A **sanity floor** for Recall@K / MRR. Confirms the metric harness and gold map are wired correctly (a real system must clear this by a wide margin). |

**The hybrid (BM25 + dense → RRF) is the system under test.** It must:
1. **match BM25-only's Exact-Match precision** (never sacrifice literal-term faithfulness for semantic recall), and
2. **beat BM25-only on OCR-noise / paraphrase Recall@K and MRR** (this is where the trained dense arm earns its keep).

RRF fusion is rank-based: `score_RRF(d) = Σ_arms 1/(c + rank_a(d))`, `c = 60`, omitting the term when `d` is outside an arm's top-M. RRF is used (over a score-weighted sum) precisely because BM25 scores (unbounded, IDF-scaled) and cosine (`[-1,1]`) are **not commensurable**; RRF needs no per-arm calibration and stays robust when one arm is noisy. Mirrors the documented P18/P19 fusion.

> **Net positioning.** BM25 guarantees the literal-term precision (the Exact-Match metric); dense + RRF recover the OCR-noise and paraphrase cases, lifting Recall@K / MRR *without* surrendering top-1 exactness.

---

## 5. Offline (synthetic) vs real (TextOCR / DocVQA) evaluation

### 5.1 Why synthetic is PRIMARY

There is **no public "search a collection by contained text" benchmark with a gold query→image map** on the Hub. `nielsr/docvqa_1200_examples` is the closest real signal (image + full OCR `words` + an NL `query` + a gold `answer.matched_text`/`start` span), but its **license is unspecified → research-only flag**, and it is a *snippet-localization* signal, not a clean multi-gold retrieval map. So the **PRIMARY offline evaluation is a synthetic rendered-text-image generator** (`data/synth_text_images.py`), and real corpora are supplements.

### 5.2 The synthetic backbone (PRIMARY) — runs with NO tesseract, NO torch, NO network

The offline contract matches P15/P18/P19 exactly. We synthesize a collection and **embed the gold text inside each PNG** so an offline `SeedEngine` reads it back as the "OCR output":

1. **Controlled vocabulary + gold map.** Rare **target needles** (`INVOICE`, `RECEIPT`, `CONFIDENTIAL`, `PURCHASE ORDER`, dates like `2023-04-12`, amounts like `$1,250.00`, ids like `PO-4471`) plus common **fillers** (`the, total, date, amount, page, customer, qty`). Each image's snippet = 1–3 short lines = a few fillers + 0–2 targets. Targets recur in a **controlled** number of images (some unique, some shared by *k*), so Recall@K/MRR are **non-degenerate** and **multi-gold queries genuinely exist**. The recorded per-term image-id set is the **GOLD map** — derived from the rendered spec, **independent of OCR**, so it stays exact even under heavy char-noise.
2. **Render + embed spec.** The P15 renderer (`render_page`, `save_png_with_spec`, `generate_dataset`) lays lines onto a page with a discovered TrueType font, recomputes per-line bboxes, and stores the spec in a PNG `tEXt` chunk (`imgsearch_spec` namespace) so gold text + boxes survive reload. Light degradation (rotation + blur) makes the *real-Tesseract* arm see realistic scans.
3. **Offline OCR = SeedEngine.** `SeedEngine.recognize()` reconstructs per-word text/conf/bbox from the embedded spec — a faithful per-image OCR text with **no OCR binary**. `_noisify(token, rate, rng)` injects realistic confusions (`m↔rn`, `0↔O`, `1↔l/I`, `5↔S`, `8↔B`, deletion/duplication) at a **controllable rate** = the CER/WER knob. The same `load_ocr_engine` code path runs offline (seed) and on Colab/H100 (real Tesseract `image_to_data`), so CER/WER is honest across both.

**Why this is the right primary harness.** The gold map is OCR-independent and exact, the needle-recurrence is tuned so ranking is non-trivial, and the noise rate is a dial — so we can plot Recall/Exact-Match *as a function of known CER* and prove the dense arm and fuzzy verifier earn their place. A **StubEngine** path (no spec → empty result) and a **BM25-only** retriever guarantee the eval, tests, and agent run end-to-end with **zero heavy deps**.

### 5.3 Real corpora (supplementary)

| Corpus | License | Role in eval |
|---|---|---|
| `MiXaiLL76/TextOCR_OCR` (112.7K, MIT) | mit | PRIMARY *real* scene-text `(image, text)` source: build a real searchable collection from `text`; positive `(query, OCR-text)` pairs to fine-tune the retriever and to sanity-check Recall on non-synthetic text. |
| `MiXaiLL76/IIIT5K_OCR` (5.5K, MIT) | mit | Small clean **real eval/dev** collection, drop-in same schema. |
| `naver-clova-ix/cord-v2` (1.0K, CC-BY-4.0) | cc-by-4.0 | Real noisy **document-image** collection (receipts): realistic "find receipts containing `TOTAL` / a price" queries; attribution required if redistributed. |
| `PleIAs/Post-OCR-Correction` (50.4K, CC0) | cc0-1.0 | Real raw-OCR `text` to populate a **noisy OCR-text index** and to seed the synthetic renderer with realistic noise strings (text-only, no images). |
| `nielsr/docvqa_1200_examples` | **unspecified → FLAG research-only** | Closest **real retrieval-over-OCR + snippet** signal: `(query → matched_text/start span)` supervision and **snippet-localization eval**. Gated behind `research_only`; never shipped in demo/tests. |
| `nielsr/funsd-layoutlmv3` | **unspecified → FLAG** | Tiny supplementary form collection for fast integration tests; research-use only. |
| `aharley/rvl_cdip` (400K) | **other → FLAG (non-commercial)** | Optional OCR-at-scale / index-scaling stress test; no gold text (must run OCR); viewer disabled. |

**License posture for evaluation reporting.** Default/demo/CI numbers are reported on **MIT + CC0 + CC-BY-4.0 + synthetic** only. Every flagged set (`docvqa_1200_examples`, `funsd`, `rvl_cdip`, `COCO-Text`) is gated behind a `research_only` flag and reported separately, never in the shippable default. The non-commercial OCR engine **Surya** (`vikp/surya_*`, CC-BY-NC-SA) is excluded from the default stack and used for research/eval comparison only.

### 5.4 The two-tier story in one line

> Synthetic gives an **exact, OCR-independent, tunable gold map** to prove correctness, the abstention contract, and the noise→recovery curve; real corpora (TextOCR / CORD-v2 / DocVQA) confirm the pipeline holds on genuine scene-text and document images with genuine OCR error.

---

## 6. OCR noise, exact match, and how the fuzzy verifier recovers it

This is the central failure mode of literal OCR search, and the reason P20 reports CER/WER alongside retrieval metrics.

### 6.1 How noise breaks exact match

BM25 and any substring containment check are **brittle to a single character**. `INVOICE → INV0ICE` (`O→0`), `m → rn`, `l ↔ I ↔ 1` — each is a tiny CER bump but a **complete miss** for the literal token: BM25's tokenizer never produces `INVOICE`, so the IDF term simply is not in that document. WER captures this better than CER (one broken word = one word error, regardless of how few characters changed), which is why both are reported. At the corpus level a low CER can still hide a meaningful drop in Recall whenever the *needle* tokens are the ones being corrupted.

### 6.2 The recovery stack (and where each piece is measured)

1. **Dense arm + RRF.** The bi-encoder embeds `INV0ICE` near `INVOICE` in vector space, so the dense arm still ranks the right image and RRF pulls it back into the top-K. **Measured by:** Recall@K / MRR lift of `hybrid` over `BM25-only` *as CER rises* on the synthetic noise sweep.
2. **D4 fuzzy literal verification.** The verifier re-reads the candidate's OCR text and checks the term appears, **tolerant to OCR noise**: case/diacritic-insensitive, with **bounded edit distance up to 1** to absorb `rn→m`, `0↔O`, etc. A near-match is kept but flagged `fuzzy_match` (uncertain) rather than silently accepted. **Measured by:** Exact-Match precision recovered, and the count of `fuzzy_match`-flagged survivors.
3. **Optional `byt5-small` post-OCR correction** before indexing (byte-level, handles `rn↔m`, `0↔O`). Off by default; when on, it lowers CER/WER upstream, which should lift both Recall and Exact-Match.

The synthetic harness makes this directly observable: because `_noisify` exposes the noise rate as a dial and the gold map is OCR-independent, we can sweep CER from 0 upward and plot the **BM25-only collapse vs. hybrid+verify resilience** — the empirical justification for the dense arm and the fuzzy verifier.

---

## 7. The abstention / coverage trade-off

P20 is allowed — and required — to **abstain** when no image contains the query. Abstention is not failure; it is the honest answer to "no document contains this." It is evaluated as a first-class behaviour.

### 7.1 The mechanism

Abstention is governed by the agent's coverage and finalize gates (deterministic FSM):

- **D3 (coverage)** gates on the **RAW** signals — the raw BM25 literal-hit count and the **raw dense cosine** of the top candidate — **never the post-RRF fused score** (the P08/P18 gotcha: the fused score always looks confident and would defeat abstention). If the whole shortlist's raw cosine is below `tau_floor` **and** there is no BM25 literal hit, the agent jumps straight to abstain. A borderline shortlist triggers a single capped **WIDEN** retry (relax phrase→keywords / widen *k*) before re-deciding.
- **D4 (verify)** drops any `exact_intent` candidate with **no literal occurrence** as a semantic false-positive.
- **D5 (finalize/abstain)** returns **ABSTAIN + `needs_review=True`** ("not found in any image") when zero candidates survive verification — instead of a misleading semantic top-k. When only fuzzy / low-OCR-confidence candidates survive, it returns them **explicitly flagged** low-confidence, never silently.

### 7.2 The trade-off and how it is scored

There is a tension between **coverage** (answer as many queries as possible) and **precision** (only answer when an image truly contains the term):

- **Too eager** (thresholds too low / no abstention) → the dense arm's semantic false-positives leak through → **Exact-Match precision falls**.
- **Too cautious** (thresholds too high) → genuine matches behind OCR noise get abstained away → **Recall falls** (false negatives: a document that *does* contain the term is reported "not found").

The evaluation scores both sides explicitly:
- **Correct abstention** is a *positive* outcome and is verified directly: a query for a term present in **no** image (e.g. `zzqwx`) must return ABSTAIN, not a least-irrelevant image.
- **Exact-Match precision@K** penalizes over-eager answering (returning images that do not contain the term).
- **Recall@K / MRR** penalize over-cautious abstention (missing images that do contain the term).

Reporting Exact-Match precision and Recall *together* makes the trade-off legible: the hybrid+verify configuration should hold Exact-Match precision at BM25's level while keeping Recall above dense-only's — i.e. it does not buy precision by abstaining on everything.

---

## 8. Verified offline numbers (what the seed run actually proves)

The offline seed run (BM25 over per-image SeedEngine OCR text, no torch / no tesseract / no network) establishes the **correctness floor and the upper bound** the trained hybrid is measured against:

| Quantity | Verified offline result | Interpretation |
|---|---|---|
| **Recall@1 (unique-code queries)** | **1.0** | For rare, unique needle tokens (e.g. `PO-4471`), BM25 over OCR text places the single gold image first every time — the **exact-term upper bound**. |
| **MRR (unique-code queries)** | **1.0** | First gold image is rank 1 → reciprocal rank 1 throughout. |
| **Snippet highlighting** | matching span returned | The verifier extracts and highlights the exact matching snippet (with word bbox) — explainable, not an opaque neighbour id. |
| **Absent term (`zzqwx`)** | **correctly ABSTAINS** | A term present in no image returns "not found" + `needs_review`, not a misleading top-k. The abstention contract fires. |
| **Agent decision points** | **all 5 fire** | D1 (ingest/OCR), D2 (query parse), D3 (coverage, raw-signal gate), D4 (literal verify + snippet), D5 (finalize/abstain) all execute on the seed path. |

**Reading these numbers correctly.** `Recall@1 = MRR = 1.0` is the *expected* result for unique high-IDF codes on clean OCR — it confirms BM25 is exact-match-faithful and the gold map / metric harness are wired correctly. It is **not** the headline accuracy of the full system on noisy real data; it is the **ceiling** and the **floor of correctness**. The trained dense arm's job is *not* to beat 1.0 on unique clean codes (impossible) but to **hold Recall up as CER rises and as queries become paraphrases**, while the verifier holds Exact-Match precision at the BM25 level. The interesting evaluation lives in the noise sweep and the paraphrase/multi-gold queries, not in the clean-unique-code corner where BM25 alone is already perfect.

---

## 9. Evaluation checklist (sanity invariants)

A run is considered valid only if:

1. `R@1 ≤ R@5 ≤ R@10` for every configuration (monotonicity); a violation indicates a ranking or gold-map bug.
2. **Random** is cleared by a wide margin on Recall@K / MRR.
3. **BM25-only** ExactP@K is the ceiling; the **hybrid** matches it (does not regress literal-term precision).
4. The **hybrid** beats **BM25-only** on Recall@K / MRR under non-zero synthetic CER and on paraphrase queries (otherwise the dense arm adds nothing and should be questioned).
5. The **absent-term query abstains** (no silent least-irrelevant answer).
6. CER and WER are reported **together**, and the noise rate used to produce them is recorded.
7. Flagged research-only corpora and the non-commercial Surya engine are excluded from the reported default/demo numbers and only appear in clearly separated `research_only` sections.

---

*All metric implementations are reused from siblings where noted (`clipsearch` retrieval metrics, `dococr` CER/WER + `normalize_ws`); Exact-Match precision@K is new for P20. The synthetic generator, OCR-text index, snippet/exact-match verifier, and the D1–D5 agent are the new P20 components under evaluation.*
