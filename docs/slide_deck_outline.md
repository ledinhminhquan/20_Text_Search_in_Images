# P20 — Text Search within Images: Slide Deck Outline

> A ~12-slide presentation outline for **P20 "Text Search within Images"** (`imgtextsearch`, folder `20_Text_Search_in_Images`).
> Author: **Le Dinh Minh Quan**, student **23127460**.
> One slide per section below; each carries 3–5 sub-bullets and a one-line speaker note.
> Target audience: technical reviewers familiar with the sibling projects (P07 OCR, P15 image translation, P18 KGQA, P19 CLIP search).

---

## Slide 1 — Title

- **Text Search within Images** — find every image in a collection whose **OCR'd text literally contains** a query word, date, ID, or short phrase, with a highlighted snippet.
- Package `imgtextsearch` · folder `20_Text_Search_in_Images` · Author **Le Dinh Minh Quan** · student **23127460**.
- One-line pitch: *"Which images literally CONTAIN this text?"* — e.g. *find images containing the word INVOICE*, *receipts mentioning a 2023 date*, *the page with PO-4471*.
- Lineage badge: built on P07 `dococr` (OCR + CER/WER), P15 `imgtrans` (synthetic render + SeedEngine), P18 `tkgqa` (BM25 + dense + RRF), P19 `clipsearch` (Recall@k / MRR). P20 is the **OCR-text sibling of P19's visual search**.
- *Speaker note:* the trainable surface is tiny — exactly one dense TEXT retriever; everything else is pretrained or algorithmic.

---

## Slide 2 — Problem & Use Cases (vs Visual Search)

- The task is **OCR-based DOCUMENT search**: index the text *inside* each image, then return the ranked subset whose OCR text literally contains the query term — or **abstain** when none do.
- **Explicit contrast with P19 (do not conflate).** P19 `clipsearch` (CLIP text→image) is **visual/semantic** — the query describes *what the image depicts*, sharing no vocabulary with the pixels. P20 is **lexical/literal** — query and indexed OCR text live in the **same lexical space**, so BM25 is the strongest single signal, not a weak floor.
- Use cases: search scanned contracts/receipts/forms for a term, locate the page bearing a part number or PO code, retrieve every scene/street photo showing a sign word, audit a collection for a date or amount.
- IN scope: English-first literal-term search over scanned + born-digital + scene-text images; snippet highlighting; abstention; hybrid BM25 + dense + RRF; OCR-noise-tolerant verification.
- OUT of scope: visual/semantic search (that is P19), VQA / answer generation, full layout/table parsing, multilingual production support.
- *Speaker note:* "contains the word X" demands the token to actually appear — semantic similarity is not enough.

---

## Slide 3 — Pipeline: OCR → Text Index → Rank

- Five algorithmic stages; **only stage (3)'s dense arm is trained**:
  `image/PDF → (1) ingest+route → (2) OCR → (3) text index → (4) query→rank → (5) snippet+verify → result | ABSTAIN`.
- **(1) Ingest + route:** PyMuPDF (`fitz`) checks each PDF page for a real text layer ≥ char threshold → read text directly, **skip OCR**; raster pages take the OCR path (reused P07/P15 D2 router).
- **(2) OCR each image:** the per-image OCR text becomes the searchable **"document"** (+ `image_id`, word boxes, mean confidence in [0,1]).
- **(3) Build index:** each image = one document; two arms — **BM25** (sparse, no checkpoint) and a **dense bi-encoder** over the OCR text (the trained arm).
- **(4) Query → rank** parse query, run both arms, fuse with **RRF**; **(5) snippet + verify** extract + highlight the matching span, confirm the term literally appears, else abstain.
- *Speaker note:* the retrieval unit is OCR TEXT, not pixels — that is the whole architectural difference from P19.

---

## Slide 4 — Data: Scene-Text + Synthetic (the gold map)

- **No public "find the image containing X" benchmark with a gold query→image map exists** → the **PRIMARY offline data is a SYNTHETIC rendered-text-image generator** (`data/synth_text_images.py`); real corpora are demo/eval/fine-tune supplements.
- Commercially safe real sets: **`MiXaiLL76/TextOCR_OCR`** (MIT, 112K `(image, gold text)` — primary scene-text), **`MiXaiLL76/IIIT5K_OCR`** (MIT, small eval), **`naver-clova-ix/cord-v2`** (CC-BY-4.0 receipts with word text), **`PleIAs/Post-OCR-Correction`** (CC0, real OCR text, no images).
- **FLAG — research-only, gate behind `research_only`:** `nielsr/docvqa_1200_examples` (license **unspecified** — the closest real retrieval-over-OCR signal: image + OCR words + a query + a gold `matched_text`/`start` span), `nielsr/funsd-layoutlmv3` (research-use), `aharley/rvl_cdip` (license:other, NC tobacco corpus, 400K).
- **Synthetic generator** composes a business-doc vocabulary — rare needles (`INVOICE`, `RECEIPT`, `CONFIDENTIAL`, dates, amounts, `PO-4471`) + shared fillers — renders to PNG (PIL) and **embeds the gold spec**, so target-term recurrence is controlled and the **gold query→image map is exact, independent of OCR**.
- License posture: ship demo + tests on MIT + CC0 + CC-BY-4.0 + synthetic only; every non-commercial / undeclared set is flagged.
- *Speaker note:* the synthetic gold map is what makes Recall@K / MRR measurable at all — it is the benchmark we lack.

---

## Slide 5 — OCR Front-End (Tesseract, pretrained)

- The OCR front-end is **pretrained, NOT trained**: default **Tesseract** via `pytesseract` (**Apache-2.0**), `image_to_data` → per-word text + bbox + confidence (0–100) that drives snippet highlighting. Offline, CPU-friendly, permissive; reused from P07/P15.
- Neural upgrades: `microsoft/trocr-base-printed` (MIT, best on clean printed scans, recognition-only → needs a detector), `microsoft/trocr-small-printed` (MIT, T4 tier); alt pipelines PaddleOCR PP-OCRv5 det+rec / docTR / EasyOCR (Apache).
- Optional post-OCR corrector `google/byt5-small` (Apache) cleans `rn↔m`, `0↔O` **before** indexing.
- **FLAG:** Surya (`vikp/surya_rec2` etc.) is **CC-BY-NC-SA-4.0 — non-commercial + ShareAlike**, excluded from the default stack, research/eval only.
- **Offline trick — SeedEngine:** with no OCR binary it reads back the gold text embedded in each synthetic image, distributing the line bbox over tokens; `_noisify` injects controllable OCR confusions = the CER/WER knob. Same code path runs offline (seed) and on Colab/H100 (real Tesseract) for an honest CER/WER.
- *Speaker note:* OCR quality is the upstream cap on retrieval — a garbled token silently drops a literally-present term.

---

## Slide 6 — Hybrid Index: BM25 + Dense + RRF (exact + semantic)

- The retrieval unit is **per-image OCR text**; **BM25 is the LEAD arm** here (not a baseline as in P19): the query and indexed text share one lexical space, BM25 fires on rare high-IDF tokens (`INVOICE`, `PO-4471`) and is exact-match-faithful by construction, no GPU. Reuses P19 `lexical.py:BM25Index` verbatim (`k1=1.5`, `b=0.75`).
- The **dense arm recovers BM25's blind spots:** OCR noise (`INV0ICE`, `lnvoice` won't lexically match), morphology/synonymy (`invoices` vs `invoice`), and paraphrase queries (*"images mentioning a 2023 date"*). Reuses P19 `image_index.py:ImageIndex` (FAISS `IndexFlatIP` on L2-normalized vectors, numpy fallback) — but over **OCR-text** embeddings.
- **RRF fusion** (`score_RRF(d) = Σ_arms 1/(c + rank_a(d))`, `c=60`) is rank-based and score-scale-free — BM25 (unbounded IDF) and cosine ([-1,1]) are not commensurable, so RRF beats weighted sum. Mirrors P18/P19.
- Net: BM25 guarantees literal-term precision; dense + RRF recover OCR-noise/paraphrase cases — raising Recall@K / MRR **without** sacrificing top-1 exactness.
- Offline guarantee: with no torch/FAISS the dense arm degrades to TF-IDF/numpy or drops to BM25-only, and index/query/eval still run end-to-end.
- *Speaker note:* this is the inversion of P19 — there CLIP led; here BM25 leads and dense is the robustness arm.

---

## Slide 7 — The Trainable Retriever + MNRL

- Of the entire system, **exactly one component is trained**: a **dense TEXT retriever (bi-encoder)** fine-tuned over the per-image OCR text — the "small trained surface, large deterministic core" split shared with P07/P15/P18/P19.
- Default `BAAI/bge-small-en-v1.5` (**MIT**, 33.4M params, **384-d**); H100 upgrade `bge-base-en-v1.5` (768-d); T4 fallback `sentence-transformers/all-MiniLM-L6-v2` (Apache, 22.7M).
- Fine-tuned with **MultipleNegativesRankingLoss (InfoNCE)** on `(query, OCR-text)` positive pairs mined from TextOCR gold text and the DocVQA query→matched-text signal; in-batch negatives.
- Optional reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache) over the fused top-k; disabled under tight latency.
- BM25, FAISS/RRF, the OCR engine, the snippet verifier, and the agent FSM are all **pretrained or algorithmic** — none are trained.
- *Speaker note:* training targets only OCR-noise / paraphrase robustness; literal precision is owned by BM25 + the verifier, not the embedding.

---

## Slide 8 — The 5-Decision Agent (verify + snippet + abstain = value-add)

- A **deterministic FSM** (5 decision points), not an LLM agent; every gate writes a replayable `ToolTrace` (decision id, branch, signal, threshold). `src/imgtextsearch/agent/`.
- **D1 ingest+OCR:** born-digital-vs-scanned route + per-image OCR confidence; low conf → `low_ocr_confidence`, empty → exclude + flag. **D2 query parse:** classify `exact_code` / `keyword` / `phrase`; empty/too-short → ABSTAIN_EARLY; quoted/ALL-CAPS/year-date → `exact_intent` (require literal verify).
- **D3 coverage:** gates on the **RAW BM25 hit count + RAW dense cosine** of the top candidate (P08/P18 gotcha: *never* the post-RRF fused score, which always looks confident); weak shortlist → WIDEN once and re-search; all below floor with no BM25 hit → jump to abstain.
- **D4 literal verify (the core filter):** for each candidate, confirm the query term **literally appears** in *that* image's OCR text — case/diacritic-insensitive, **fuzzy-tolerant up to edit-distance 1** for `rn→m`, `0↔O`; extract + highlight the matching **snippet** with word bbox; DROP `exact_intent` candidates with no literal occurrence (kills semantic false-positives), put exact matches first.
- **D5 finalize/abstain:** zero verified → **ABSTAIN** ("not found in any image" + `needs_review`); else ranked list with snippets, fusion score, OCR confidence, and a high/medium/weak label. Optional LLM brain (`anthropic`) is **OFF by default**, advisory only (query-expansion note), and **never changes ranking**.
- **Value-add over blind semantic ranking = literal-term VERIFICATION + snippet HIGHLIGHTING + ABSTENTION** — answering *"which images literally contain this text"* honestly, not "which are vaguely about it."
- *Speaker note:* a naive dense retriever always returns k images; the agent is what makes "not found" a valid, honest answer.

---

## Slide 9 — Metrics & Results (Recall@k + Exact-Match + the BM25-vs-dense story)

- **Primary:** text-in-image **Recall@1/5/10** + **MRR** + **median rank** — a query hits if ANY gold image (whose OCR text literally contains the query) is in top-K; rank = first gold. Reuses P19 `metrics.py`.
- **OCR quality cap:** **CER / WER** (OCR vs the rendered gold text, micro-averaged), reused from P07 `corpus_cer`/`corpus_wer`; on synthetic data the SeedEngine char-noise rate maps to a controllable CER.
- **P20-specific:** **Exact-Match precision@K** — fraction of *returned* images whose OCR text **literally contains** the normalized query — audits the returned list directly (not the gold map), catching the dense arm's semantically-similar-but-wrong images and quantifying the value BM25 adds.
- **Baselines:** BM25-only (the strong exact-term floor, genuinely competitive here), dense-only (high Recall but lower Exact-Match — exposes the semantic false-positive weakness D4 + BM25 fix), random (sanity floor).
- **Verified offline seed (BM25 over per-image OCR text):** Recall@1 = MRR = **1.0** (exact term match perfect for unique codes); snippet highlights the match; a nonexistent term (`"zzqwx"`) correctly **ABSTAINS**; all 5 agent decisions fire.
- *Speaker note:* the story is that hybrid must beat BM25 on OCR-noise/paraphrase Recall while *matching* its Exact-Match precision.

---

## Slide 10 — Deployment

- **FastAPI** `POST /search`: query → ranked image ids + scores + the matching OCR snippet + an exact-match flag; the collection is **OCR-indexed once at startup**.
- **Gradio UI:** type a query → see the matching images with highlighted snippets and high/medium/weak confidence labels.
- **Docker:** image bundles `tesseract-ocr` + fonts + `libGL`; ships to a **Hugging Face Space**.
- **GPU tiers:** H100 (PP-OCRv5 / TrOCR-base + `bge-base` + reranker + `byt5` correction + batch-OCR the whole collection) · DEFAULT (Tesseract + `bge-small` + BM25/RRF) · T4 (Tesseract CPU or `trocr-small` + `all-MiniLM-L6-v2`, reranker off). All tiers stay fully permissive (MIT/Apache); none require Surya's NC weights.
- Index scaling: `IndexFlatIP` is exact but O(N) per query — for large N swap in an ANN index (HNSW/IVF) behind the same interface; BM25-only remains a no-GPU floor.
- *Speaker note:* the offline backbone runs with no Tesseract / no torch / no network — the same contract as P15/P18/P19.

---

## Slide 11 — Ethics, Privacy & OCR Bias

- **Indexing user/scanned documents is sensitive:** OCR text can carry PII — IDs, contracts, medical/financial records. Mitigations: **consent, access control, PII redaction, no query retention by default, the LLM brain OFF**.
- **OCR errors cut both ways:** a false negative misses a document that *does* contain the term; a false positive returns one that does not. The **fuzzy D4 verify + D5 abstention** mitigate both, and CER/WER make the error rate visible.
- **Bias:** OCR quality varies by **script, font, and image quality** — non-Latin scripts and low-quality scans get **unequal search recall**; English is the only validated target, so non-EN is treated as research-only (no multilingual production claim).
- **Licensing discipline:** non-commercial / unlicensed corpora (`docvqa_1200_examples`, `funsd`, `rvl_cdip`, COCO-Text) and the Surya OCR model are flagged and gated behind `research_only`; the shipped demo uses MIT + CC0 + CC-BY-4.0 + synthetic only.
- **Robustness watch-list:** OCR errors breaking exact match, multi-word/phrase queries, language/script coverage, image licensing, and index scaling (BM25 + ANN).
- *Speaker note:* abstention is itself an ethics feature — returning "not found" beats surfacing a misleading least-irrelevant image.

---

## Slide 12 — Conclusion & Future Work

- **Conclusion:** P20 turns a collection of images into a searchable corpus of their *contained* text — a hybrid **BM25-led + dense + RRF** index over per-image OCR text, wrapped in a deterministic 5-decision agent whose value-add is **literal verification + snippet highlighting + abstention**.
- **Honest design:** one trained component (the dense retriever); everything else pretrained/algorithmic; runs fully offline via SeedEngine + BM25; verified seed gives Recall@1 = MRR = 1.0 and correct abstention on a nonexistent term.
- **Future work — accuracy:** enable `byt5-small` post-OCR correction by default, add the cross-encoder reranker at scale, and a stronger detector for TrOCR (recognition-only) on scene text.
- **Future work — scale & coverage:** swap `IndexFlatIP` → ANN (HNSW/IVF) for 400K+ collections, validate multilingual OCR + retrieval beyond English, and build a real human-labeled "find the image containing X" benchmark to replace the synthetic gold map.
- **Future work — product:** richer phrase/proximity matching, layout-aware snippets (table cells), and an opt-in audited LLM advisory mode for query expansion.
- *Speaker note:* the open gap is the missing real benchmark — the synthetic generator is a stand-in, not a substitute.

---

*End of outline — 12 slides. Generated for P20 `imgtextsearch`; consistent with `docs/DESIGN_BRIEF.md`.*
