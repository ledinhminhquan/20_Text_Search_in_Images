# Ethics Statement — P20 Text Search within Images (`imgtextsearch`)

**Author:** Le Dinh Minh Quan (student 23127460)
**System:** OCR-based full-text search over a *collection of images* — find the images whose **OCR'd text literally contains a query** (e.g. *"images containing the word INVOICE"*, *"receipts mentioning a 2023 date"*, *"the page with PO-4471"*), each returned with a highlighted snippet, or an **abstention** when no image contains the term.

This document states the ethical risks specific to P20 and the concrete mechanisms in the system that mitigate them. It is deliberately narrow: P20 is *lexical, literal* document search over OCR text, **not** the visual/semantic content search of its sibling P19 (`clipsearch`). The risks below follow from one fact — **P20 makes the text inside a pile of images searchable** — and the most sensitive images a user is likely to throw at it are scanned documents.

---

## 1. Why this system carries ethics weight

In isolation, "rank some images by a query" sounds low-stakes. It is not, because of *what the indexed text usually is*. The realistic input collections for P20 are **scanned documents, receipts, forms, contracts, and PDF pages** (CORD-v2 receipts, FUNSD forms, RVL-CDIP scanned documents, DocVQA pages are all in scope as research/eval corpora, and real deployments index whatever the user uploads). Full-text-searching that kind of collection means the OCR text index can hold, verbatim:

- personal identifiers (names, addresses, national IDs, account/policy numbers);
- financial records (invoice amounts, prices, bank details, salary figures);
- medical and legal content (diagnoses, case references, contract clauses);
- any **PII that happens to be printed on a scanned page**.

Building an inverted/lexical (BM25) index plus a dense embedding index over that text is, functionally, **building a search engine over a private document trove**. That is the central ethical fact of this project and the reason every section below exists.

---

## 2. Surveillance and privacy risk: full-text search over scanned documents

### The risk

The same capability that lets a user *"find every receipt containing TOTAL"* lets an operator *"find every document containing this person's name / this account number / this medical term"* across a whole archive in one query. Concretely:

- **Mass document search.** Once a collection is OCR-indexed (the design indexes the collection *once at startup* and answers many queries against it), there is no per-document cost to a sweeping query. The privacy exposure of one query against 1 image and one query against 400K scanned documents (the RVL-CDIP-scale stress path) is wildly different, and the system makes both equally cheap.
- **E-discovery / investigative overreach.** A literal-term index over contracts, emails-as-images, or case files is exactly an e-discovery tool. Without scoping, it enables fishing expeditions far beyond a legitimate, authorized purpose.
- **Re-identification by aggregation.** Snippet highlighting — a core P20 feature — returns the *surrounding text* of a match, not just an image id. A query for one term can therefore surface adjacent PII (a name next to a queried diagnosis, an account number next to a queried amount).
- **Query logs as a second leak.** A retained log of queries ("searched for `<person>` in the medical-scans collection") is itself sensitive intelligence about who is being looked for, independent of the documents.

### Mitigations built in / required by P20

- **No query retention by default.** The deployment posture (FastAPI `POST /search` + Gradio UI) does **not** persist queries by default. Query logging is opt-in, must be justified, and where enabled should be access-controlled and time-limited. This is a default, not a suggestion.
- **The LLM "brain" is OFF by default and advisory-only.** P20 includes an optional Anthropic-backed advisory note (query-expansion hints) that is **disabled by default**, **never changes ranking**, and — critically — should never be sent raw document text or PII. Keeping it off by default means OCR'd document content is not shipped to a third-party API in the normal path.
- **Consent and lawful basis before indexing.** Indexing someone's scanned documents requires their consent or another lawful basis. Operators must confirm they are authorized to OCR-index the collection *before* it is ingested — not after a query surfaces something.
- **Access control on the collection, not just the app.** Because the index *is* the document trove, who can query must be restricted to who is authorized to read the underlying documents. Search permission must not silently exceed document-read permission.
- **PII redaction.** Where the collection is sensitive, PII should be redacted from the OCR text *before* it enters the BM25/dense index, so the index cannot be queried for what should not be searchable. The word-box output from `image_to_data` makes targeted redaction feasible.
- **Auditability.** The agent is a **deterministic state machine** in which every decision point (D1–D5) writes a `ToolTrace` record (decision id, branch, signal value, threshold). Every search is therefore **replayable and auditable** — who queried what, which images were returned, and why. This is the technical substrate for an access-and-use audit; operators of sensitive collections should retain and review these traces.
- **Scope discipline.** P20 is single-purpose: literal text search. It does not do face matching, visual/semantic search (that is P19, deliberately *not* pulled in), VQA, or answer generation. Keeping the tool narrow limits the surveillance surface.

**Responsible-use line:** this tool is for searching documents you are *authorized* to search. Using it to build a dragnet over people's documents without consent, lawful basis, and access control is out of scope and out of bounds.

---

## 3. OCR-quality bias: unequal search recall

### The risk

P20 has a structural fairness problem that pure-text search engines do not: **a document is findable only if OCR reads it correctly.** The query and the index share a lexical space, so if OCR mangles a word, the literal term is simply not in the index and the document silently drops out of results. OCR quality is **not uniform** — it varies systematically by:

- **script and language** — Tesseract and the en-only trained retriever are validated on English (TextOCR, IIIT5K, DocVQA are EN); non-Latin scripts and non-English languages have higher error rates and the retriever is not validated cross-lingually;
- **font and rendering** — unusual fonts, handwriting, stylized scene text;
- **scan quality** — low-resolution, skewed, blurred, or degraded scans (the synthetic generator deliberately models this with rotation + blur `_degrade` and a controllable character-noise rate that maps to CER).

The consequence is **unequal search recall across populations**: a collection of cleanly-printed English receipts is highly searchable; a collection of low-quality scans, non-Latin-script documents, or handwritten forms is *less* searchable — and the user is given no native signal that recall is degraded for that subset. A document that **does** contain the queried term but was OCR'd as `INV0ICE` / `lnvoice` is a **false negative** caused not by the query but by the input's script/quality. People whose documents are in harder-to-OCR scripts or come as poorer scans are therefore *systematically less served*.

### How P20 surfaces and mitigates it

P20 does not pretend this bias away — it **measures and reports** it so it is visible rather than hidden:

- **CER / WER reporting (the honest quality cap).** OCR Character- and Word-Error-Rate are first-class metrics (micro-averaged corpus-wide, reused from P07's `corpus_cer`/`corpus_wer`). CER/WER are the upstream quality cap on recall and are reported alongside retrieval metrics, so a high error rate on a subset is *exposed*, not buried.
- **Per-script / per-subset evaluation.** CER/WER and Recall@K should be reported **broken down by script, language, font class, and scan quality**, not only as a single corpus number. A single aggregate number hides exactly the disparity that matters; per-subset reporting is how unequal recall becomes a documented, reviewable fact.
- **Per-image OCR confidence.** Each indexed image carries a mean OCR confidence (from `image_to_data`, 0–1). Low-confidence images are tagged `low_ocr_confidence` at D1 and flagged downstream (D4/D5 `needs_review`), and truly unreadable images are excluded from the index *and flagged* rather than silently dropped — so the user can see that some documents may not be reliably searchable.
- **Fuzzy verification recovers some OCR errors.** D4 literal verification is OCR-noise-tolerant (case/diacritic-insensitive, bounded edit distance for `rn↔m`, `0↔O`, `1↔l/I`), and the dense + RRF arm recovers near-misses that BM25 cannot. Optional `byt5-small` post-OCR correction can clean text before indexing. These narrow — but do **not** eliminate — the gap; the residual disparity is what per-script CER reporting exists to keep honest.

**Honesty boundary:** P20 claims **English** as its supported target and explicitly treats other scripts/languages as research-only. It must **not** be marketed as offering equal multilingual recall. Where it is deployed on multi-script collections, the per-script CER/recall numbers must travel with it.

---

## 4. False negatives in high-stakes search (legal / medical)

### The risk

A **false negative** — a document that *does* contain the queried term but is not returned — is the most dangerous failure mode for P20, far more than a false positive. In a legal or medical context the harms are concrete:

- a contract clause or piece of discoverable evidence that *exists* is reported as "not found," and a decision is made on the false belief that it does not exist;
- a patient record containing a relevant term is missed in a records search.

These are exactly the situations where a confident-looking "no results" is worse than no tool at all. False negatives in P20 arise from OCR errors (Section 3), phrase/line-break artifacts, and the inherent recall ceiling of any retrieval system.

### Mitigations built in

P20 is explicitly designed to *fail loudly and hand the decision to a human* rather than fail silently:

- **Abstention instead of a misleading answer.** At D5, if **no** image is verified to literally contain the query, the system returns an explicit *"not found in any image"* with `needs_review=True`, rather than returning the least-irrelevant semantic neighbor as if it were a hit. Abstention beats blind semantic ranking precisely because a confident wrong answer in legal/medical search is dangerous. Verified offline, a nonexistent term (`zzqwx`) correctly **abstains**.
- **Fuzzy verification to catch OCR-induced near-misses.** D4's bounded-edit-distance, case/diacritic-insensitive verification is specifically there to recover documents that *do* contain the term but were OCR'd imperfectly — directly attacking the largest source of false negatives. Matches found only via fuzzy/near-match are kept but flagged `fuzzy_match` (uncertain), not presented as exact.
- **The snippet is shown for human confirmation — the tool does not adjudicate.** Every returned image comes with the **highlighted matching snippet** (the matched span + word bbox) and the image's OCR confidence. In high-stakes use the human reads the snippet and confirms the match; the system surfaces evidence, it does not make the legal/medical determination. A flagged `fuzzy_match` or `low_ocr_confidence` result tells the reviewer exactly where to look harder.
- **Low-confidence results are returned explicitly flagged, never silently.** When only fuzzy or low-OCR survivors exist, D5 returns them **explicitly labelled** low-confidence + `needs_review` — so a thin or shaky result set cannot masquerade as a clean one.

**Responsible-use line:** in legal, medical, or other high-stakes settings, an **"abstain" / "not found" from P20 must not be treated as proof of absence.** OCR error means absence-of-evidence is not evidence-of-absence. P20 is a recall aid for a human reviewer, not an authority on whether a term exists in a collection. Critical "does this document exist" determinations require human review of the source documents, not reliance on the search result alone.

---

## 5. Transparency

P20 is built to be **explainable, not an opaque neighbor-id ranker**. Every result a user sees carries the evidence for *why* it was returned and *how much* to trust it:

- **Snippet + highlight.** Each returned image shows the exact matching text span (with word bbox), so the user sees *what* matched and *where*, not just an image id.
- **Exact-match flag.** Each result indicates whether the query term **literally** appears in that image's OCR text (verified at D4), versus a fuzzy/near-match (`fuzzy_match`) versus a low-OCR-confidence case. The user is never left guessing whether "this image was returned" means "this image actually contains the word."
- **Scores and confidence shown.** The fusion (RRF) score, an OCR confidence value, and a high/medium/weak label accompany each result, so weak results are visibly weak.
- **Honest "not found."** Abstention is a first-class, visible outcome (Section 4), not a hidden empty list dressed up as a confident answer.
- **Deterministic, replayable trace.** The D1–D5 `ToolTrace` makes the entire decision path inspectable after the fact — which arm fired, which threshold was crossed, why a candidate was dropped or kept.
- **Exact-Match precision@K, reported.** The system measures the fraction of *returned* images that literally contain the query, auditing the result list directly against the literal query — catching (and quantifying) the dense arm's tendency to return semantically-similar-but-wrong images. This is published as a metric, so the system's literal-faithfulness is a stated, measured property.

The transparency goal: a user (or auditor) can always answer *"why was this image returned, does it actually contain my term, and how sure is the system?"* from what the system shows them.

---

## 6. Responsible-use guidance (summary)

**Intended use.** Literal text search over an image collection you are **authorized** to search — finding which images contain a given word, date, number, ID, or short phrase, with a snippet for human confirmation. English is the supported target.

**Do:**
- Confirm consent / lawful basis and apply access control **before** OCR-indexing a sensitive collection; restrict query access to those authorized to read the documents.
- Keep query retention off by default; keep the LLM brain off for sensitive content; never send raw document text/PII to a third-party API.
- Redact PII before indexing where appropriate; retain and review the D1–D5 audit trace for sensitive deployments.
- Report CER/WER and Recall **per script / language / scan quality**, not just as one number; ship those numbers with any multi-script deployment.
- Treat every result as a lead for **human review** — read the snippet, heed the `fuzzy_match` / `low_ocr_confidence` / `needs_review` flags.

**Do not:**
- Use P20 for surveillance, dragnet document search, or e-discovery beyond an authorized, scoped purpose.
- Treat **"not found" / abstain as proof a term is absent** — OCR error means a present document can be missed; this matters most in legal and medical search.
- Claim equal multilingual / multi-script recall; non-English is research-only.
- Repurpose it for face matching, visual/semantic search (use P19), VQA, or automated high-stakes decisions without a human in the loop.

**Dataset / model licensing note (ethical-use adjacent).** P20 ships its default demo and tests only on permissive data and models — MIT (`MiXaiLL76/TextOCR_OCR`, `MiXaiLL76/IIIT5K_OCR`, `BAAI/bge-small-en-v1.5`), Apache (Tesseract, `all-MiniLM-L6-v2`, reranker, `byt5-small`), CC0 (`PleIAs/Post-OCR-Correction`), CC-BY-4.0 (`naver-clova-ix/cord-v2`, attribution required if redistributed), plus the synthetic generator. Non-commercial / unlicensed assets are flagged and gated behind a `research_only` flag and **not** redistributed: `nielsr/docvqa_1200_examples` (license **unspecified**), `nielsr/funsd-layoutlmv3` (research-use only), `aharley/rvl_cdip` (license **other**, derived from the non-commercial tobacco-documents corpus), `howard-hou/COCO-Text` (no license tag), and the Surya OCR weights `vikp/surya_rec2` et al. (**CC-BY-NC-SA-4.0**, ShareAlike, excluded from the default stack). Respecting these terms — non-commercial, attribution, non-redistribution — is part of using P20 responsibly.
