# P20 Agent Architecture — The OCR-Search Decision Engine

> Project: **P20 Text Search within Images** (`imgtextsearch`, folder `20_Text_Search_in_Images`)
> Author: Le Dinh Minh Quan — student 23127460
> Scope: the mandatory **agentic component** — a deterministic finite-state machine (FSM) with **5 decision points**, located in `src/imgtextsearch/agent/`.

---

## 1. What the agent is (and is not)

P20 answers one question: **"which images in this collection literally CONTAIN this text?"** — e.g. *find images containing the word INVOICE*, *receipts that mention a 2023 date*, *the page with PO-4471*. The text lives **inside** the images and is recovered by OCR; the searchable unit is therefore the **per-image OCR text** (the "document"), and the query is a word / date / number / ID / short phrase expected to appear **verbatim** in that text.

The agent is a **deterministic state machine**, not an LLM agent. There is no autonomous planning, no tool-choosing model in the loop, and no temperature. Given the same collection, query, and config, it produces the **same ranked output and the same audit trace, byte-for-byte**. This is the same "small trained surface, large deterministic core" split used across the sibling projects (P07 `dococr`, P15 `imgtrans`, P18 `tkgqa`, P19 `clipsearch`): exactly **one** component in the whole system is trained — the dense text retriever (a sentence-transformers bi-encoder fine-tuned with MultipleNegativesRankingLoss over `(query, OCR-text)` pairs). The OCR front-end, BM25, RRF fusion, the snippet verifier, and **the entire agent FSM are pretrained or purely algorithmic**.

**Why an FSM and not a free-form LLM agent.** For literal-term document search the failure mode that matters is a retriever confidently returning a semantically-similar image where the word never actually appears. That is a *verification* problem, not a *reasoning* problem. A deterministic FSM with an explicit literal-verification gate solves it exactly and reproducibly, whereas an LLM agent would add nondeterminism, latency, cost, and a privacy surface (OCR-indexed documents are sensitive) with no accuracy gain. The optional LLM "brain" (Section 8) is therefore **OFF by default and strictly advisory** — it can suggest query expansions but **never** touches ranking or the abstain decision.

### 1.1 The value-add over blind semantic ranking

A naive dense retriever always returns *k* images by embedding similarity, so for `INVOICE` it will happily surface images *about* billing where the word never appears — unacceptable for OCR document search. The agent adds three things on top of retrieval:

1. **Literal-term verification (D4)** — it re-reads each candidate's OCR text and confirms the query term actually appears (fuzzy-tolerant to OCR noise), **dropping semantic false-positives**.
2. **Snippet extraction + highlighting (D4)** — it returns the exact matching span plus the word bounding box and OCR confidence, so the answer is **explainable**, not an opaque neighbor id.
3. **Abstention (D5)** — when *no* image contains the text, it returns **"not found" + `needs_review`** instead of the least-irrelevant image.

These are reinforced by the BM25 + dense → RRF hybrid (the lexical arm is exact on rare literal tokens the embeddings smear), the D3 **raw-cosine** coverage gate (the P08/P18 gotcha, below), and a fully replayable `ToolTrace`. Net effect: the system answers *"which images literally contain this text"* honestly, not *"which images are vaguely about this topic."*

---

## 2. State machine overview

The agent runs five states in a fixed forward order, with two controlled back-edges (a D3 single-retry widen, and the D3→D5 abstain shortcut). Every state reads one or more **intermediate signals**, compares them against thresholds drawn from `AgentConfig`, picks a **branch**, and writes a `ToolTrace` record before advancing.

```
        query + OCR-indexed collection
                     │
        ┌────────────▼────────────┐
        │  D1  PARSE / QUERY GATE  │  signal: norm token/char count, quotes, intent class
        └────────────┬────────────┘
             empty/too-short → ABSTAIN_EARLY ─────────────────────────┐
                     │ valid (exact | semantic | hybrid intent)        │
        ┌────────────▼────────────┐                                    │
        │  D2  SEARCH (BM25+dense  │  signal: top-k (image_id, score)  │
        │       → RRF shortlist)   │          shortlist                 │
        └────────────┬────────────┘                                    │
           index empty / search error → NEEDS_REVIEW ─────────────────┤
                     │ shortlist                                        │
        ┌────────────▼────────────┐                                    │
        │  D3  COVERAGE / CONF.    │  signal: RAW BM25 literal-hit cnt │
        │       GATE               │          + RAW top dense cosine    │
        └───┬─────────────┬───────┘                                    │
   weak shortlist │       │ strong (raw cosine ≥ tau_soft               │
   (widen once) ◄─┘       │  or BM25 literal hit)                       │
   all < tau_floor &      │                                            │
   no BM25 hit → D5 ──────┼──────────────────────────────┐            │
                     │ candidates                          │            │
        ┌────────────▼────────────┐                       │            │
        │  D4  LITERAL VERIFY +    │  signal: literal     │            │
        │       SNIPPET (fuzzy)    │   containment / edit  │            │
        └────────────┬────────────┘   distance per cand.   │            │
                     │ verified survivors                   │            │
        ┌────────────▼────────────┐                         │            │
        │  D5  FINALIZE / ABSTAIN  │◄────────────────────────┴────────────┘
        └────────────┬────────────┘
            0 verified → ABSTAIN ("not found", needs_review)
            ≥1 verified → FINALIZE (ranked images + snippets + labels)
```

The collection is **OCR-indexed once at startup** (ingest → route born-digital-vs-scanned → OCR → build BM25 + dense index); that build is described in the design brief's Section 2 pipeline (and corresponds to the brief's ingest gate). The per-query agent below begins at the query and operates over that pre-built index.

---

## 3. The five decision points in detail

Each subsection states: the **signal** the gate reads, the **`AgentConfig` thresholds** it compares against, the **branches**, and the resulting trace.

### D1 — Parse / query gate + classify

**Purpose.** Reject degenerate queries before any encoding work is wasted, and classify the query's *intent* so downstream stages know how to weight the arms and how strictly to verify.

**Signal.** The normalized query string and its derived features: normalized token count and character count, presence of quote marks (a quoted phrase), and a lightweight pattern classification.

**Classification (`exact_code` / `keyword` / `phrase`).**
- **`exact_code`** — the query looks like a literal identifier/date/amount: ALL-CAPS tokens (`INVOICE`, `CONFIDENTIAL`), `YYYY-MM-DD` or year patterns (`2023`, `2023-04-12`), part/PO numbers (`PO-4471`, `REF####`), currency amounts (`$1,250.00`). Sets `exact_intent = True`.
- **`phrase`** — quoted input (`"purchase order"`) or a multi-token literal; literal containment is required at D4 with token-boundary / word-subsequence awareness.
- **`keyword`** — a single ordinary descriptive word; default hybrid handling.
- A descriptive natural-language phrase (`images mentioning a 2023 date`) sets `semantic_intent` (lean dense, but **still attempt** verify at D4).

**Thresholds (`AgentConfig`).**
- `min_query_tokens` (e.g. `1`) — minimum normalized tokens to proceed.
- `min_query_chars` (e.g. `2`) — guards single stray punctuation.

**Branches.**
| Condition | Branch | Effect |
|---|---|---|
| empty / pure-punctuation / `< min_query_tokens` | **ABSTAIN_EARLY** | terminal; no encode, no search; returns "empty/too-short query" + `needs_review` |
| quoted phrase or literal token (caps/date/number/id) | `exact_intent = True` | weight BM25; **require** literal verify at D4 → D2 |
| descriptive NL phrase | `semantic_intent = True` | lean dense; still attempt verify → D2 |
| otherwise | default hybrid | balanced BM25+dense → D2 |

**Trace.** `D1 {branch, query_norm, n_tokens, intent_class, exact_intent}`.

---

### D2 — Search: BM25 + dense → RRF shortlist

**Purpose.** Produce a top-*k* candidate shortlist over the **per-image OCR text** by fusing the exact-term arm and the semantic arm.

**Signal.** The two ranked lists and the fused shortlist of `(image_id, score)` pairs, plus the **raw** per-arm signals carried forward (raw BM25 hit count for the literal term; raw top-candidate dense cosine) — these are what D3 will gate on, *not* the fused score.

**How it runs.**
- **BM25 (lead arm).** P20 is literal-term document search, so BM25 — exact-match-faithful by construction, high IDF on rare needles like `INVOICE`, no GPU — is the **strongest single signal**, not a baseline. Reuses `BM25Index` (`k1=1.5`, `b=0.75`, regex tokenizer) over per-image OCR text.
- **Dense arm.** The trained bi-encoder (`BAAI/bge-small-en-v1.5`, 384-d, MIT, default) embeds the query and the OCR text into a shared space, recovering BM25's blind spots: OCR noise (`INV0ICE`, `lnvoice`), morphology/synonymy (`invoices` vs `invoice`), and paraphrase queries. FAISS `IndexFlatIP` on L2-normalized vectors, numpy/TF-IDF fallback offline.
- **RRF fusion.** `score_RRF(d) = Σ_arms 1 / (c + rank_a(d))`, `c = 60` (`AgentConfig.rrf_c`), rank-based so the unbounded IDF-scaled BM25 scores and the `[-1,1]` cosines never need calibration. Optional cross-encoder rerank (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the fused top-k.

**Thresholds (`AgentConfig`).**
- `top_k` (e.g. `10`) — shortlist size carried to D3/D4.
- `rrf_c` (`60`) — RRF constant.
- `rerank` (bool, default off under tight latency).

**Branches.**
| Condition | Branch |
|---|---|
| index empty / `< k` hits available / search error/exception | **NEEDS_REVIEW** (terminal) |
| otherwise | carry shortlist + raw per-arm signals → D3 |

**Trace.** `D2 {n_bm25_hits, top_dense_cosine, shortlist_ids, fused_scores, reranked}`.

---

### D3 — Coverage / confidence gate

**Purpose.** Decide whether the shortlist is strong enough to verify, whether it should be **widened once**, or whether it is hopeless and should jump straight to abstention. This gate is where P20 avoids the classic over-confidence trap.

**Signal — and the critical gotcha.** D3 gates on the **RAW BM25 literal-hit count** for the query term **and** the **RAW dense cosine of the top candidate** — **never the post-RRF fused score**. This is the P08/P18 lesson: the fused RRF score is rank-based and always looks "confident" (the top item always gets `1/(c+1)` regardless of how bad it actually is), so gating on it silently defeats abstention. Raw cosine and raw BM25 hit count are the only signals that honestly reflect whether anything good was found.

**Thresholds (`AgentConfig`).**
- `tau_soft` (e.g. `0.55`) — "strong enough" raw-cosine bar.
- `tau_floor` (e.g. `0.35`) — below this, with no BM25 literal hit, the candidate is hopeless.
- `max_widen` (`1`) — at most one widen retry.

**Branches.**
| Condition (signal vs threshold) | Branch |
|---|---|
| index empty / `< k` hits / search error | **NEEDS_REVIEW** (terminal) |
| `≥ 1` BM25 literal hit **OR** raw top cosine `≥ tau_soft` | **STRONG** → carry shortlist to D4 |
| raw top cosine `∈ [tau_floor, tau_soft)` **AND** zero BM25 literal hits | **WIDEN once** (capped at `max_widen`): relax phrase→keywords / add synonyms / widen `k`, re-search (D2), re-enter D3 |
| whole shortlist raw cosine `< tau_floor` **AND** no BM25 literal hit | **jump to D5 → ABSTAIN** |

The widen back-edge is taken **at most once**; on the second visit the retry budget is exhausted, so a still-weak shortlist falls through to the abstain shortcut rather than looping.

**Trace.** `D3 {branch, n_bm25_hits, top_raw_cosine, tau_soft, tau_floor, widened}`.

---

### D4 — Literal verify + snippet extraction (fuzzy-tolerant)

**Purpose.** This is the **value-add core**. For each surviving candidate, confirm the query term *actually appears* in **that** image's OCR text, extract and highlight the matching snippet, and **drop semantic false-positives**.

**Signal.** Per candidate: whether the normalized query is a literal containment match (substring for codes; word-subsequence / token-boundary-aware for keyword/phrase) in that image's normalized OCR text, and if not, the minimum bounded edit distance to any OCR token/span.

**Fuzzy exact-match logic (the OCR-noise tolerance).** OCR is imperfect: `INVOICE` can be read as `INV0ICE` (`O→0`), `lnvoice` (`I→l`), or `rn→m` confusions. A strictly exact substring check would produce false negatives — a document that genuinely contains the term would be missed. So matching proceeds in tiers:

1. **Normalize both sides** — lowercase, strip/normalize diacritics, collapse whitespace (the same `normalize_ws` that backs CER/WER and Exact-Match precision), so case and accent differences never block a match.
2. **Exact containment** on the normalized text → **VERIFIED (exact)**.
3. **Bounded fuzzy match** — if no exact hit, compare the query against OCR tokens/spans with **edit (Levenshtein) distance up to 1** (`AgentConfig.fuzzy_max_edits = 1`). A single-character OCR confusion (`0↔O`, `1↔l/I`, `5↔S`, `8↔B`, `rn↔m`) is therefore tolerated; two or more edits are **not**, which keeps the match precise rather than promiscuous. For multi-token phrases the budget applies per token, not across the whole phrase.
4. A token that matches **only** via the fuzzy tier is kept but **flagged `fuzzy_match`** (uncertain), so D5 can surface it as lower-confidence.

The edit-distance ceiling of **1** is deliberate: empirically most single-glyph OCR errors are one substitution/insertion/deletion, so distance-1 recovers the realistic noise the SeedEngine `_noisify` knob injects (and that real Tesseract produces) while still rejecting genuinely different tokens. This is exactly where the dense arm + fuzzy verify earn their keep — BM25 alone misses `INV0ICE`, dense retrieves it, and D4's fuzzy check confirms it.

**Snippet extraction (explainability).** Once a candidate verifies, D4 locates the matched span in the OCR text, extracts a short surrounding snippet (a few tokens of context), highlights the matched span, and attaches the **word bounding box** (from `image_to_data` / the SeedEngine word boxes) and the **per-image OCR confidence**. This is what turns an opaque neighbor id into an auditable answer.

**Thresholds (`AgentConfig`).**
- `fuzzy_max_edits` (`1`) — bounded edit-distance ceiling.
- `low_conf_floor` (e.g. `0.5`) — OCR confidence below this tags the candidate `low_ocr_confidence`.
- `snippet_window` (e.g. `6` tokens) — context width around the matched span.

**Branches (per candidate).**
| Condition | Branch |
|---|---|
| literal term found exactly | **VERIFIED**: extract + highlight snippet (+ word bbox), keep, attach OCR confidence |
| found only via bounded fuzzy / near-match | **VERIFIED (fuzzy)**: keep but flag `fuzzy_match` |
| `exact_intent` **but NO** literal/fuzzy occurrence | **DROP** as a semantic false-positive (the core filter beating blind semantic ranking) |
| candidate already tagged `low_ocr_confidence` | keep but mark `needs_review` |

Survivors (verified, possibly flagged) advance to D5.

**Trace.** `D4 {image_id, match_kind: exact|fuzzy|none, edit_distance, snippet, bbox, ocr_conf, dropped, flags}` per candidate.

---

### D5 — Finalize / abstain

**Purpose.** Convert the verified survivors into either an honest **"not found"** or a ranked, explainable result — never the least-irrelevant image.

**Signal.** The count of VERIFIED candidates that survived D4 and their best fused score / OCR confidence.

**Thresholds (`AgentConfig`).**
- `high_conf` / `medium_conf` score bands (e.g. `≥ 0.7` high, `≥ 0.4` medium, else weak) used only for the human-readable confidence label — they **do not** gate abstention.

**Branches.**
| Condition | Branch |
|---|---|
| zero verified survivors | **ABSTAIN**: return *"not found in any image"* + `needs_review = True` (instead of a misleading semantic top-k) |
| `≥ 1` verified survivor | **FINALIZE**: ranked image list, each with highlighted snippet, fusion score, OCR confidence, and a high/medium/weak label |
| only `fuzzy_match` / `low_ocr_confidence` survivors | **FINALIZE but explicitly flagged** low-confidence + `needs_review` — never returned silently |

**Trace.** `D5 {branch, n_verified, best_score, confidence_label, needs_review}`.

---

## 4. Decision table (compact)

| ID | Gates on (intermediate signal) | Key thresholds (`AgentConfig`) | Branches → next |
|---|---|---|---|
| **D1** Parse + classify | norm token/char count; quotes; intent pattern (caps/date/number/id vs NL phrase) | `min_query_tokens`, `min_query_chars` | empty/short → **ABSTAIN_EARLY**; quoted/literal → `exact_intent` → D2; NL phrase → `semantic_intent` → D2; else hybrid → D2 |
| **D2** Search (BM25+dense→RRF) | top-k `(image_id, score)` shortlist; **raw** BM25 hit count + **raw** top cosine carried fwd | `top_k`, `rrf_c=60`, `rerank` | index empty / error → **NEEDS_REVIEW**; else shortlist → D3 |
| **D3** Coverage gate | **RAW BM25 literal-hit count** + **RAW top dense cosine** (never the fused RRF score) | `tau_soft`, `tau_floor`, `max_widen=1` | `≥1` BM25 hit OR cosine `≥ tau_soft` → **STRONG**→D4; cosine `∈[tau_floor,tau_soft)` & 0 BM25 hits → **WIDEN once**→D2; all `< tau_floor` & no BM25 hit → **→D5 ABSTAIN**; empty/error → **NEEDS_REVIEW** |
| **D4** Literal verify + snippet | per-candidate literal containment / min bounded edit distance in *that* image's OCR text | `fuzzy_max_edits=1`, `low_conf_floor`, `snippet_window` | exact → **VERIFIED**(+snippet+bbox); fuzzy(≤1 edit) → **VERIFIED** flag `fuzzy_match`; `exact_intent` & no occurrence → **DROP**; low OCR conf → keep + `needs_review` |
| **D5** Finalize / abstain | count of VERIFIED survivors + best score/conf | `high_conf`, `medium_conf` (labels only) | 0 verified → **ABSTAIN** "not found" + `needs_review`; `≥1` → **FINALIZE** ranked+snippets+labels; only fuzzy/low-conf → **FINALIZE flagged** + `needs_review` |

---

## 5. Example traces

### 5.1 Happy path — a unique code that exists

**Collection:** 200 synthetic rendered pages; image `page_0042.png` OCR text contains `... REF PO-4471 TOTAL $1,250.00 ...`. **Query:** `PO-4471`.

```
D1  query_norm="po-4471"  n_tokens=1  intent_class=exact_code  exact_intent=True
      → branch: exact_intent → D2
D2  BM25: PO-4471 is high-IDF, fires on 1 doc (page_0042) score=8.91
      dense top cosine=0.62  RRF fuse → shortlist=[page_0042, page_0117, page_0008,...]
      n_bm25_hits=1  top_raw_cosine=0.62
      → branch: shortlist ok → D3
D3  raw n_bm25_hits=1 (≥1) AND top_raw_cosine 0.62 ≥ tau_soft 0.55
      → branch: STRONG → D4
D4  page_0042: normalize → "po-4471" found EXACTLY in OCR text
      → VERIFIED; snippet="...REF [PO-4471] TOTAL $1,250.00..."  bbox=(412,233,196,38)  ocr_conf=0.97
      page_0117: "po-4471" not present, exact_intent → DROP (semantic false-positive)
      page_0008: not present → DROP
      → survivors=[page_0042]
D5  n_verified=1  best_score=… confidence_label=high  needs_review=False
      → FINALIZE: [ page_0042 | snippet | bbox | high ]
```

### 5.2 OCR-noise path — fuzzy match earns its keep

**Setup:** `page_0091` truly contains `INVOICE`, but the SeedEngine char-noise read it as `INV0ICE` (`O→0`). **Query:** `INVOICE`.

```
D1  query_norm="invoice"  intent_class=exact_code (ALL-CAPS)  exact_intent=True → D2
D2  BM25: "invoice" does NOT lexically match "inv0ice" → n_bm25_hits=0
      dense: bge embeds INVOICE≈INV0ICE → page_0091 top cosine=0.71  RRF shortlist top=page_0091
      → D3
D3  n_bm25_hits=0  BUT top_raw_cosine 0.71 ≥ tau_soft 0.55
      → branch: STRONG (cosine arm) → D4
D4  page_0091: exact containment of "invoice" FAILS
      bounded fuzzy: edit_distance("invoice","inv0ice")=1 ≤ fuzzy_max_edits=1
      → VERIFIED(fuzzy); flag fuzzy_match; snippet="...[INV0ICE] no. 553..."  ocr_conf=0.88
      → survivors=[page_0091 (fuzzy)]
D5  n_verified=1, only fuzzy survivor
      → FINALIZE but FLAGGED: [ page_0091 | snippet | medium | fuzzy_match | needs_review=True ]
```

### 5.3 Abstain path — term not in the collection

**Query:** `zzqwx` (a token present in no image).

```
D1  query_norm="zzqwx"  n_tokens=1  intent_class=keyword → D2
D2  BM25: 0 hits  dense: top cosine=0.22 (nearest junk)  RRF shortlist exists but weak
      n_bm25_hits=0  top_raw_cosine=0.22
      → D3
D3  top_raw_cosine 0.22 < tau_floor 0.35  AND n_bm25_hits=0
      → branch: whole shortlist hopeless → jump to D5 ABSTAIN
D5  n_verified=0
      → ABSTAIN: "not found in any image"  needs_review=True
      (note: the high-looking RRF fused top score is correctly IGNORED — D3 gated on RAW cosine)
```

This is the decisive contrast with a blind semantic retriever, which would have returned `zzqwx`'s nearest neighbor as a confident-looking top-1. Verified offline, this is the `"zzqwx"` ABSTAIN case in the seed run.

---

## 6. The `ToolTrace` audit

Every gate appends one structured record to a per-query `ToolTrace` before transitioning. Each record carries: the **decision id** (`D1`…`D5`), the **branch taken**, the **signal value(s)** that drove it, and the **threshold(s)** compared against. Because every signal is a deterministic function of the (collection, query, config) and there is no model sampling anywhere in the FSM, the trace is **fully replayable**: re-running the same inputs reproduces the identical sequence of branches and the identical final ranking — the P07/P19 audit pattern.

The trace is the primary debugging and explainability surface. It answers, for any result, *why this image ranked here* (D2 scores), *why it survived* (D4 match kind + edit distance + snippet), and *why the query abstained* (D3 raw-signal values vs `tau_floor`, D5 zero survivors). It is also what makes the offline contract checkable: the seed run asserts that **all five decisions fire** on a representative query batch.

---

## 7. Fail-soft behavior

The FSM never crashes a query into an exception that reaches the user; every error path resolves to a defined terminal state plus a flag:

- **Degenerate query (D1):** `ABSTAIN_EARLY` + `needs_review` — no wasted encoding.
- **Empty index / `< k` hits / search exception (D2, D3):** `NEEDS_REVIEW` (terminal) rather than a fabricated ranking.
- **No literal occurrence anywhere (D3 floor / D5):** `ABSTAIN` "not found" + `needs_review` — the honest answer, not the least-irrelevant image.
- **Low OCR confidence or fuzzy-only survivors (D4/D5):** the result is **returned but explicitly flagged** low-confidence + `needs_review`, **never silently**.
- **No OCR binary / no torch / no FAISS (offline):** the SeedEngine reads the embedded gold spec, BM25 stands in for the dense arm, and the whole D1–D5 path still runs end-to-end with zero heavy dependencies — the same P15/P18/P19 offline contract.

The unifying principle: when the system is uncertain it **degrades to abstention + a review flag**, because for OCR document search a confident wrong answer (a document that does *not* contain the term, returned as if it does) is worse than an honest "not found."

---

## 8. The optional LLM brain (off by default)

An optional LLM "brain" (Anthropic) can be enabled but is **OFF by default and strictly advisory**. Its only role is to propose a **query-expansion note** (e.g. synonyms, alternate spellings, abbreviation expansions) that D3's widen step *could* consult. It **never** changes ranking, **never** overrides verification, and **never** decides abstention — those remain purely deterministic. It is kept off by default for three reasons: (1) determinism — the FSM's replayable trace is a core property; (2) privacy — OCR-indexed user/scanned documents are sensitive (IDs, contracts, medical/financial records, PII in the text), so sending query/OCR context to an external model is opt-in only; (3) it adds latency and cost with no accuracy gain on literal-term search, where BM25 + the fuzzy verifier already cover the hard cases. When enabled, its suggestions are logged in the `ToolTrace` as advisory metadata, clearly separated from the deterministic branch records.

---

## 9. Why this design wins (summary)

- **Literal verification (D4)** turns *"images about billing"* into *"images that literally contain INVOICE,"* dropping semantic false-positives that a blind dense retriever would return.
- **Bounded fuzzy match (edit distance ≤ 1)** recovers realistic OCR noise (`INV0ICE`, `lnvoice`, `rn↔m`) without becoming promiscuous — the dense arm finds the candidate, the fuzzy verifier confirms it.
- **Raw-signal coverage gate (D3)** sidesteps the P08/P18 fused-score trap, so abstention actually works.
- **Snippet + bbox + OCR confidence** make every answer explainable.
- **Abstention (D5)** gives an honest "not found" instead of the least-irrelevant image.
- **Deterministic FSM + `ToolTrace`** make every decision reproducible and auditable, with the LLM brain off by default to protect determinism and document privacy.
