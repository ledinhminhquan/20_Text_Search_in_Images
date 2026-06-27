# P20 Text Search within Images — Privacy & Robustness

> Package `imgtextsearch` · author Le Dinh Minh Quan (student 23127460).
> This document covers the two non-functional pillars that decide whether P20 is safe to deploy: **Privacy** (because OCR-indexing scanned documents turns the most sensitive content a user owns into a searchable text index) and **Robustness** (because the whole task hinges on imperfect OCR, where a single character confusion can hide a document that genuinely contains the query).

P20 answers a deceptively narrow question — *"which images in this collection literally contain this text?"* — by OCR'ing each image, building a hybrid BM25 + dense (RRF) index over the per-image OCR text, and ranking with a deterministic D1–D5 verification agent. Both pillars below are framed against that exact pipeline, not generic ML advice.

---

## Part A — Privacy

### A.1 Why P20 is unusually sensitive: the text *is* the content

In a visual/semantic search system (the sibling P19 `clipsearch`), the index stores image embeddings — opaque 384/768-d vectors with no human-readable content. P20 is the opposite. Its index is, by construction, a **plain-text transcript of every document in the collection**: the per-image OCR text is the searchable unit. When the collection is scanned documents — and the headline use cases (receipts, invoices, forms, contracts, ID pages) are exactly that — the index becomes a high-density store of personally identifiable and regulated information:

- **Identity / government data** — names, addresses, dates of birth, passport / national-ID / driver-licence numbers, signatures (as text-near-bbox).
- **Financial data** — bank-account and card numbers, invoice/PO amounts (`$1,250.00`), `PO-4471`-style references, tax IDs, salaries on payslips.
- **Health / legal data** — medical record text, diagnoses, contract clauses, case numbers, anything in a scanned legal-tobacco-style corpus (`rvl_cdip`).
- **Free-text PII** — email addresses, phone numbers, and arbitrary personal narrative that OCR lifts verbatim out of the pixels.

Two properties make this worse than an ordinary document store:

1. **OCR concentrates and exposes.** Content that was previously locked inside an image (not greppable, not indexed by most tooling) becomes fully text-searchable. The act of OCR-indexing *creates* a new, easier-to-exfiltrate copy of the sensitive content.
2. **Snippet highlighting surfaces PII by design.** P20's value-add is returning the *matching snippet* + word bbox (D4). A query for a common token can legitimately return a snippet that also contains adjacent PII (e.g. a name next to the matched `INVOICE`). The feature that makes results explainable is also a PII-leak surface.

The governing principle for P20: **treat the OCR-text index as a PII datastore with the same controls a primary document store would get** — not as a disposable cache.

### A.2 Threat model

| Asset | Threat | Where it lives in P20 |
|---|---|---|
| The OCR-text index (BM25 postings + dense vectors + raw OCR text + word bboxes) | Unauthorized read → bulk PII exfiltration | The index built at Stage (3) / FastAPI startup |
| Individual query terms (e.g. a person's name, a card number) | Query-log retention reveals *who searched for whom* and leaks the search term itself | The FastAPI `POST /search` request path; any logging |
| Returned snippets | Over-disclosure — a snippet exposes PII the searcher had no right to see | D4 snippet extraction → API/Gradio response |
| The raw image collection on disk | At-rest theft of the source documents | Ingest staging + any cached rasterized pages |
| The optional LLM brain (`anthropic`) | Sensitive OCR text / queries sent to a third-party API | The optional advisory LLM (see A.6) |

### A.3 Consent & lawful basis (before any image is indexed)

OCR-indexing a collection is a **processing** decision, not a neutral technical step. P20's posture:

- **Explicit, documented consent / lawful basis** to OCR-index a collection of personal documents before ingest runs. The owner of the collection must affirm they are entitled to index that content for search.
- **Purpose limitation** — the index exists to answer literal-term search, nothing else. No secondary use (no analytics over the OCR text, no model training on real user documents — the trainable retriever is fine-tuned on the MIT/CC0/CC-BY/synthetic corpora in §3, never on a deployed user's private collection).
- **Data minimization** — index only the collection the user submits; do not pull in research-only corpora (`docvqa_1200_examples`, `funsd`, `rvl_cdip`, `COCO-Text`) into any user-facing deployment. Those are gated behind the `research_only` flag and excluded from the default/demo path.
- **Right to deletion** — because the index unit is one document per `image_id`, a deletion request maps cleanly: drop the BM25 postings, the dense vector, and the stored OCR text + bboxes for that `image_id`, then re-fuse. Deletion is per-image and complete.

### A.4 Access control on the index

The index is the crown jewels; it is protected as such:

- **Authenticated, authorized access only.** `POST /search` and the Gradio UI require authentication; the index is never world-readable. No anonymous public Space serving a real user collection (the public HF Space ships **only** synthetic + permissively-licensed demo data — never a real document set).
- **Collection-scoped isolation.** A searcher only queries collections they are entitled to. The `image_id → collection` mapping enforces tenant/collection boundaries so one user's query can never rank or snippet another user's document.
- **Least privilege on snippets.** Snippet length is bounded to the matched span plus minimal context (not the whole page) to limit incidental PII disclosure around the hit. The D4 word-bbox is returned for highlighting, not the full OCR transcript.

### A.5 PII detection & redaction

Because OCR text is raw PII, P20 supports a **redaction layer between OCR and index** (and again on the response):

- **Pre-index redaction (opt-in, recommended for sensitive collections).** After Stage (2) OCR and before Stage (3) indexing, run a PII detector over the per-image OCR text and mask high-risk entities (national IDs, card/account numbers, emails, phone numbers) — e.g. token-pattern + entity rules over the same normalized text. The masked token is what gets indexed and what can appear in a snippet, so the regulated value is never stored in plain text in the index.
- **Response-time redaction.** Even when the index retains full text (internal trusted deployment), the snippet returned to a less-privileged searcher can be redaction-filtered so the highlighted match is shown but neighbouring PII is masked.
- **Honest trade-off, stated.** Redacting `PO-4471` or a card number *before* indexing means a literal query for that exact value will (correctly) no longer match it — privacy is bought at the cost of searchability for the redacted class. P20 makes this a deliberate, configured choice per collection rather than a silent default, and documents that redaction reduces recall for the redacted entity types.

### A.6 No query retention; LLM brain OFF by default

- **No query retention by default.** Query strings are treated as sensitive (a query *is* a name / number / clause). The default configuration logs **no raw queries** — only non-identifying operational counters (latency, abstention rate, arm-hit counts, the `ToolTrace` decision ids/branches with signal *values* but not the query text). Any query logging is opt-in, time-boxed, and access-controlled.
- **The `ToolTrace` is privacy-aware.** The deterministic D1–D5 trace records decision id, branch, signal value, and threshold for replayable audit — it is designed to capture *why* the agent abstained/finalized without persisting the searcher's raw query term or the document PII.
- **The optional LLM brain (`anthropic`) is OFF by default.** P20's intelligence is the deterministic FSM, not an LLM. The optional LLM is **advisory only** (a query-expansion note), **never changes ranking**, and is **disabled by default** precisely because enabling it would send the searcher's query — and potentially OCR text — to a third-party API. Operators must explicitly opt in, with full awareness that doing so transmits sensitive content off-box. For any deployment over real personal documents, it stays off.

### A.7 Encryption at rest & in transit

- **Encryption at rest** for both the source image collection and the derived OCR-text index (the index is PII and is encrypted with the same rigor as the source documents). Cached rasterized PDF pages are treated as sensitive and encrypted/cleaned up too.
- **TLS in transit** for the FastAPI `/search` endpoint and the Gradio UI; queries and snippets never cross the network in clear text.
- **No PII in container images / artifacts.** The Docker image ships `tesseract-ocr` + fonts + `libGL` and code only — never a baked-in real collection or index. The HF Space and any published artifact contain synthetic/demo data exclusively.

### A.8 Privacy posture summary

| Control | Default | Notes |
|---|---|---|
| Consent / lawful basis before indexing | Required | Purpose-limited to search |
| Access control on index + snippets | Required (auth + collection-scoped) | No anonymous real-collection serving |
| PII detection / redaction | Available; recommended for sensitive sets | Pre-index and/or response-time; trades recall for the redacted class |
| Query retention | **Off** | Only non-identifying counters + privacy-aware ToolTrace |
| LLM brain (`anthropic`) | **Off** | Advisory only, never changes ranking; opt-in sends data off-box |
| Encryption at rest + TLS | Required | Index encrypted as PII; no baked-in real data in images/Spaces |
| Research-only corpora in deployment | Excluded | `docvqa_1200_examples` / `funsd` / `rvl_cdip` / `COCO-Text` gated behind `research_only` |

---

## Part B — Robustness

P20's hardest robustness problem is intrinsic: **it searches the OCR text, but the user means the real text in the image.** Every failure mode below is some version of that gap, and P20's defence is a layered one — the hybrid index recovers near-misses, and the deterministic D1–D5 gates verify, flag, or abstain rather than confidently returning a wrong image.

### B.1 OCR errors break exact match — the central failure mode

A single character confusion silently defeats literal matching. Tesseract reads `INVOICE` as `INV0ICE` (`O→0`), `lnvoice` (`I→l`), `rn→m`, `1↔l/I`, `5↔S`, `8↔B`. The token is *present to a human* but *absent to BM25*, which matches on exact tokens. Left unhandled, the document that genuinely contains the query is missed → a **false negative**, the worst outcome for "find the document that contains X."

P20 mitigates this on three independent layers:

1. **The dense arm recovers what BM25 misses.** BM25 is the lead, exact-match arm; but `INV0ICE` and `lnvoice` are *embedding-near* `INVOICE`. The `bge-small-en-v1.5` bi-encoder (over OCR text) ranks the noisy variant near the clean query, and **RRF fusion** (`c=60`, rank-based, score-scale-free) floats it back into the candidate shortlist even though BM25 gave it zero. This is precisely why the system is hybrid rather than BM25-only — the dense arm *earns its keep* exactly on OCR noise.
2. **D4 verification is fuzzy-tolerant.** The literal-containment check at D4 is **case- and diacritic-insensitive** and uses **bounded edit distance (up to 1)** so `INV0ICE`/`lnvoice`/`rn→m` still verify as the query term. A fuzzy match is kept but **flagged `fuzzy_match`** so the result is explicitly marked uncertain, never silently promoted to a clean hit.
3. **Optional pre-index correction + visible CER/WER.** `google/byt5-small` post-OCR correction can normalize `0↔O` / `rn↔m` *before* indexing (default off for latency, on for high-quality tiers). And **CER/WER are reported as first-class metrics** (vs the rendered gold text), so the OCR error rate propagating into the index is measured, not hidden — a rising CER is a visible early warning that recall is about to degrade.

The synthetic offline harness makes this testable: the SeedEngine `_noisify` knob injects exactly these confusions at a **controllable rate**, mapping directly to a target CER/WER — so the recovery behaviour of dense+RRF and the D4 fuzzy verify is exercised under known noise rather than assumed.

### B.2 False negatives vs false positives — and how the gates balance them

| Error | What it means in P20 | Primary mitigation |
|---|---|---|
| **False negative** | A document that *does* contain the term is missed (OCR mangled the token, or it ranked below K). | Dense arm + RRF recover OCR-noise variants (B.1); D3 can **WIDEN once** on a weak shortlist; D4 fuzzy edit-distance verify catches near-misses. |
| **False positive** | A returned image does *not* literally contain the term — the dense arm surfaced a semantically-similar-but-wrong document. | **D4 literal verification DROPS** any `exact_intent` candidate with no literal (or bounded-fuzzy) occurrence — the core filter that beats blind semantic ranking. |

This is the asymmetry that justifies P20's whole agent design. A naive dense retriever *always* returns K images by similarity, so for `INVOICE` it confidently surfaces images *about* billing where the word never appears — unacceptable for literal document search. P20's **Exact-Match precision@K** metric audits the *returned* list directly for literal containment, quantifying the false-positive rate the dense arm would otherwise hide, and showing what BM25 + D4 verification recover.

### B.3 Multi-word and phrase queries

"Literal containment" gets complicated when the query spans tokens. OCR introduces line breaks, hyphenation (`pur-chase`), reordering, and stray whitespace, so a phrase that is visually contiguous may be non-contiguous in the OCR text stream.

- **D2 classifies intent.** A **quoted phrase** or ALL-CAPS/date/number/ID token is tagged `exact_intent` (weight BM25, **require** literal verify at D4); a descriptive phrase (`images mentioning a 2023 date`) is `semantic_intent` (lean dense, still attempt verify). The system knows whether the user demands literal contiguity or is describing content.
- **Token-boundary-aware, subsequence-tolerant verification.** D4 verification is word-token-boundary-aware and tolerates word-subsequence matching, so a phrase split across an OCR line break can still verify.
- **D3 widens phrase → keywords once.** On a weak shortlist (top raw cosine in the soft band, zero BM25 literal hits), D3 relaxes the phrase into keywords / adds the dense arm's reach **once** (capped at one retry) and re-searches — recovering phrase queries that strict contiguity would have dropped, without unbounded re-querying.

### B.4 Language & script coverage

OCR and retrieval quality vary sharply by script, and P20 is explicit about its supported envelope:

- **English is the validated target.** TextOCR / IIIT5K / DocVQA are EN; the retriever (`bge-small-en-v1.5`) and tokenizer are validated on EN. P20 claims English literal-term search as the supported surface.
- **Non-Latin / non-EN is research-only, not silently broken.** The renderer and the CC0 `Post-OCR-Correction` corpus can produce fr/it/de text, but Tesseract language packs, tokenization, and the EN retriever are **not validated cross-lingually**. P20 does **not** claim multilingual production support; non-EN is gated as research-only so a user is never given a false impression of recall on, say, an Arabic or CJK scan.
- **Bias is named, not hidden.** OCR accuracy is systematically worse on non-Latin scripts, handwriting, and low-quality scans → **unequal search recall across populations**. This is a fairness risk (a non-Latin-script user gets worse search), surfaced in the ethics posture rather than buried — the honest claim is "English-first," and CER/WER reporting per collection makes the disparity measurable.

### B.5 Low-quality scans & image-condition robustness

Real scans are skewed, blurred, low-DPI, and noisy. P20's defences:

- **Born-digital vs scanned routing (D1).** The PyMuPDF text-layer router reads the embedded text directly when a real text layer exists (≥ char threshold) and **skips OCR entirely** — eliminating OCR error for the entire born-digital class. Only raster / text-layer-less pages take the lossy OCR path.
- **Per-image OCR confidence as a gate.** `image_to_data` mean confidence (0–1) is carried through. Confidence `< floor` (e.g. < 0.5) tags the image `low_ocr_confidence`; D4/D5 keep such candidates but **mark `needs_review`** — a low-quality scan never silently produces a confident hit.
- **Unreadable images are excluded and flagged**, not indexed as empty documents that would pollute ranking (D1 fallback).
- **The degrade knob is tested.** The synthetic renderer applies light rotation + Gaussian blur (scaled 0–1), so the real-Tesseract path is exercised on realistically degraded pages and the CER/WER cost of degradation is measured offline.

### B.6 Abstention — the ultimate robustness backstop

The single most important robustness behaviour is knowing when **not** to answer:

- **D2 abstains early** on empty / pure-punctuation / sub-`min_query_tokens` queries (`ABSTAIN_EARLY`) — no wasted encode, no garbage shortlist.
- **D3 routes to abstain** when the whole shortlist is below `tau_floor` *and* there is no BM25 literal hit (jump to D5).
- **D5 abstains** when zero candidates survive D4 verification: it returns *"not found in any image"* + `needs_review=True` instead of a misleading semantic top-K. A nonexistent term (`zzqwx`) correctly **abstains** — verified in the offline seed run.
- **Gating on raw signals, not the fused score.** D3 gates coverage on the **raw BM25 hit count** and **raw dense cosine** — never the post-RRF fused score, which always looks confident (the P08/P18 gotcha) and would defeat abstention. This is what makes "I don't know" trustworthy: the confidence gate sees the un-smoothed evidence.

Abstention converts the dangerous failure mode (confidently returning the least-irrelevant image when the term is genuinely absent) into a safe, honest one (saying so).

### B.7 Index scaling robustness

- `IndexFlatIP` is exact but O(N) per query, and OCR-ing 400K images (`rvl_cdip` scale) is a heavy batch job. *Mitigation:* batch-OCR in one pass on the H100 tier; the D1 born-digital router skips OCR on text-layer PDFs; for large N swap `IndexFlatIP` → an ANN index (HNSW/IVF) **behind the same `retrieve()` interface**; **BM25-only remains a no-GPU floor** that always runs; the synthetic offline collection stays small for fast CI.
- **Graceful degradation is a contract.** With no torch / FAISS, the dense arm degrades to TF-IDF/numpy or drops to BM25-only, and the whole index → query → eval → agent path still runs end-to-end (the P15/P18/P19 offline guarantee). Robustness here means the system never *requires* the heavy dependency to produce a correct, if leaner, answer.

### B.8 Robustness posture summary

| Failure mode | Gate(s) that mitigate | Residual risk |
|---|---|---|
| OCR char confusion breaks exact match | Dense+RRF recovery; D4 fuzzy verify (edit-distance ≤ 1); optional `byt5` correction; CER/WER reported | High noise still drops recall — measured, not hidden |
| Semantic false positives | D4 literal verification **drops** non-containing `exact_intent` candidates; Exact-Match precision@K audit | Fuzzy band can admit a near-miss (flagged) |
| Multi-word / phrase queries | D2 intent classify; token-boundary subsequence verify; D3 phrase→keyword widen (once) | Severely reordered OCR phrases may still miss |
| Non-EN / non-Latin script | EN validated; non-EN gated research-only; bias named; CER/WER per collection | Unequal recall by script — disclosed, not solved |
| Low-quality scans | D1 born-digital route (skip OCR); OCR-confidence gate → `needs_review`; exclude unreadable | Borderline scans flagged, not perfected |
| Term genuinely absent | D2/D3/D5 **abstention** on raw signals (not fused score) | None — abstention is the correct answer |
| Index scale / missing heavy deps | ANN swap behind same interface; BM25-only floor; born-digital skip; full offline degrade | Exact O(N) cost at very large N until ANN swap |

---

## Closing

P20 makes a sharp privacy claim — **the OCR-text index is a PII datastore and is governed as one** (consent, collection-scoped access control, optional pre-index redaction, no query retention, LLM brain off, encryption at rest + TLS, research-only corpora excluded from deployment) — and a sharp robustness claim — **the system would rather abstain than confidently return a document that does not contain the term**, with the hybrid BM25+dense+RRF index and the fuzzy-tolerant D4 verifier closing the OCR-error gap and the D1–D5 gates flagging or refusing every low-confidence case. The two pillars reinforce each other: the same literal-verification + abstention machinery that makes results *honest* (robustness) also bounds the snippet disclosure that makes them *safe* (privacy).
