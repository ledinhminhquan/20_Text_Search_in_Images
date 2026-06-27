# Model Card — P20 Dense OCR-Text Retriever (`imgtextsearch`)

This card documents the **single trainable component** of P20 *Text Search within Images*: a fine-tuned dense **text** bi-encoder that retrieves images by the **OCR'd text they contain**. Every other component in the pipeline — OCR, BM25, RRF fusion, the snippet/exact-match verifier, and the D1–D5 agent — is pretrained or purely algorithmic and is **not** covered by this card except where it bounds this model's behavior.

> **One-line scope.** This model embeds a query and a per-image OCR-text "document" into a shared vector space so that images whose OCR text matches the query rank near the top. It does **not** look at pixels, does not perform OCR, and does not decide the final answer — literal-term verification and abstention are downstream (agent D4/D5).

---

## 1. Model details

| Field | Value |
|---|---|
| **Model name** | P20 dense OCR-text retriever (`imgtextsearch` retriever core) |
| **Package / project** | `imgtextsearch` — folder `20_Text_Search_in_Images` |
| **Author** | Le Dinh Minh Quan (student 23127460) |
| **Model type** | Sentence-transformers **bi-encoder** (dual-encoder), single shared text tower, mean-pooled, L2-normalized embeddings, cosine/inner-product scoring |
| **Base model (DEFAULT)** | `BAAI/bge-small-en-v1.5` — **MIT**, 33.4M params, **384-d** output |
| **Fine-tuning objective** | `MultipleNegativesRankingLoss` (MNRL / in-batch InfoNCE) over `(query, OCR-text)` pairs |
| **Input** | English text: a short query (word, date, number, ID, or short phrase) **or** a per-image OCR text document |
| **Output** | A single dense vector (384-d default); retrieval score = cosine similarity / inner product on L2-normalized vectors |
| **Language** | **English** (en) — primary and only validated target |
| **License (this fine-tuned model)** | Inherits the base model license — **MIT** for the `bge-small-en-v1.5` default. All default-tier variants below are MIT/Apache and commercially usable. |

### Configurable base variants (all permissive)

| Tier | Base id | License | Params / dim | When |
|---|---|---|---|---|
| **DEFAULT** | `BAAI/bge-small-en-v1.5` | MIT | 33.4M / 384-d | Default everywhere; same family already wired in P18/P19 |
| **H100 upgrade** | `BAAI/bge-base-en-v1.5` | MIT | 109.5M / 768-d | Higher accuracy on a larger top-k |
| **T4 fallback** | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | 22.7M / 384-d | Latency/memory-constrained drop-in |
| Alt | `intfloat/e5-base-v2` | MIT | — | Requires `query:` / `passage:` prefixes |

> No variant requires non-commercial weights. The high-accuracy Surya OCR family (`vikp/surya_*`, **CC-BY-NC-SA-4.0**) is **excluded from the default stack** and is research/eval only — but note Surya is an *OCR* model, upstream of this retriever, not a retriever option.

---

## 2. Intended use

### Primary intended use
Rank a **collection of images** by whether their **OCR'd text matches a text query**, so a user can find *which images literally contain* a word, date, number, ID, or short phrase (e.g. *"find images containing the word INVOICE"*, *"receipts that mention a 2023 date"*, *"the page with PO-4471"*).

Concretely, this model:
1. Embeds each image's concatenated OCR text once at index time (the image is treated as a **text document**).
2. Embeds the query at search time.
3. Provides the **dense arm** of a hybrid index; its ranked list is fused with BM25 via **Reciprocal Rank Fusion (RRF, `c=60`)**.

### Role in the pipeline (where it sits)
```
image → OCR (pretrained) → per-image OCR text → [BM25 arm] + [THIS dense arm] → RRF → agent D3/D4/D5 → result | ABSTAIN
```
The model contributes **recall under conditions BM25 cannot handle**:
- **OCR noise** — `INV0ICE`, `lnvoice`, `rn→m` confusions that break exact lexical match.
- **Morphology / synonymy** — `invoices` vs `invoice`, `date: 2023` vs `dated 2023`.
- **Paraphrase queries** — *"images mentioning a 2023 date"* rather than the bare token.

**BM25 is the lead arm** in P20 (literal-term document search, shared lexical space); this dense model is the **noise/paraphrase-robust complement**, not the primary signal.

### Intended users
Developers building OCR-document search over a private or curated image collection (scanned documents, receipts, forms, born-digital PDF pages, scene-text photos), via the P20 FastAPI `/search` endpoint or Gradio UI.

### Out-of-scope / not intended for
- **Visual or semantic image search** ("a photo of a beach") — that is the sibling project **P19 `clipsearch`** (CLIP). This model never sees pixels and shares no vocabulary with image content; it only matches **text inside images**.
- **VQA / answer generation**, layout parsing, table extraction.
- **Multilingual production search.** Only English is validated. The renderer and the CC0 Post-OCR corpus can produce fr/it/de text, but the retriever, tokenization, and OCR language coverage are **not** validated cross-lingually — treat non-EN as research-only.
- **A standalone "does this image contain X" decision.** This model only *ranks*; the **literal-term guarantee comes from the downstream BM25 arm + the D4 verifier**, never from embedding similarity alone. Using the raw dense ranking without D4 verification will surface images that are semantically *about* a term where the term never appears (see Limitations).

---

## 3. Training

### Data
The retriever is fine-tuned on `(query, OCR-text)` positive pairs. Sources, in license-posture order:

| Source | Role in training | License | Notes |
|---|---|---|---|
| **Synthetic rendered-text images** (`data/synth_text_images.py` + SeedEngine) | **PRIMARY offline signal.** Business-doc vocabulary (`INVOICE`, `RECEIPT`, `TOTAL`, dates, `PO-4471`, `REF####`) rendered to PNG with the gold text embedded; SeedEngine reads it back as OCR text. Supplies a gold query→image map and controllable OCR noise (CER/WER knob). | Generated (project-owned) | No public "find image containing X" benchmark exists; this is the only set with a clean gold query→image map. |
| `MiXaiLL76/TextOCR_OCR` | Real scene-text `(image, gold text)`; build `(image_text, query)` positives. Largest clean MIT (image,text) set. | **MIT**, 112.7K | Default/demo-safe. |
| `naver-clova-ix/cord-v2` | Real document images + word-level transcription (receipts). | **CC-BY-4.0**, ~1K | Attribution if redistributed. |
| `PleIAs/Post-OCR-Correction` | Realistic noisy OCR **text** (no images) to make the index look like real OCR; noise strings for the renderer. | **CC0-1.0**, 50.4K | Text-only. |
| `nielsr/docvqa_1200_examples` | **Closest real retrieval-over-OCR signal** (image + OCR `words` + NL `query` + `answer.matched_text`/`start` span). Best (query→doc/snippet) supervision + snippet eval. | **License UNSPECIFIED — FLAG** | **Research-only**, gated behind `research_only`; verify before any commercial use. |

> Datasets `nielsr/funsd-layoutlmv3` (research-use), `aharley/rvl_cdip` (`license:other`, NC tobacco corpus), and `howard-hou/COCO-Text` (no tag) are **flagged research-only** and are **not** used in the default/demo training set.

### Objective
**MultipleNegativesRankingLoss (MNRL / in-batch InfoNCE).** For each `(query, OCR-text)` positive in a batch, every other OCR-text in the batch is an implicit negative; the model is trained to maximize cosine similarity for the true pair against all in-batch negatives. This is the same retriever-training recipe used in **P18 `tkgqa`** and **P19 `clipsearch`** — reused, with the indexed "document" being **per-image OCR text** instead of KG triples or captions.

### What is and is not trained
- **Trained:** this dense bi-encoder (the *only* trained surface in P20).
- **Not trained / pretrained / algorithmic:** OCR front-end (Tesseract default), BM25, RRF fusion, the optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, Apache), the optional post-OCR corrector (`google/byt5-small`, Apache), the snippet/exact-match verifier, and the D1–D5 agent FSM.

### Offline / no-dependency training and eval contract
The offline backbone runs with **no Tesseract, no torch, no network** (the P15/P18/P19 contract). When `sentence-transformers` is absent, the dense arm degrades to a TF-IDF/numpy stand-in behind the same `retrieve()` interface, or the system runs **BM25-only** — so eval, tests, and the agent run end-to-end with zero heavy deps.

---

## 4. Evaluation

### Metrics
Retrieval is scored over (query → gold image-that-contains-the-text). The metrics that bound **this model** are:

1. **Recall@K** (K ∈ {1, 5, 10}) — *primary*. A query hits if any image whose OCR text literally contains the query (the gold set) appears in the top-K; multi-gold queries averaged. Reused from `19_.../training/metrics.py:recall_at_k`. Monotone non-decreasing in K.
2. **MRR (Mean Reciprocal Rank)** — reciprocal of the first gold image's rank; rewards a correct image at top-1. Reused from `metrics.py:mrr`.
3. **Median / mean rank** of the first gold image (not-retrieved → sentinel rank 1000); diagnostic.
4. **Exact-Match precision@K** — *P20-specific verification metric.* Of the top-K images **returned**, the fraction whose OCR text literally contains the normalized query (case/whitespace-normalized, token-boundary-aware). This audits the **returned** list directly against the literal query and **exposes the dense arm's semantic-false-positive weakness** — i.e. it quantifies exactly where this model needs BM25 + D4 to enforce literalness.
5. **OCR CER / WER** — *upstream quality cap, not a property of this model.* Reported because OCR error in the indexed text bounds achievable retrieval: a garbled term (`INV0ICE`) defeats BM25 and is precisely the case this dense model is meant to recover. Micro-averaged corpus-wide (`07_.../dococr/training/metrics.py:corpus_cer/corpus_wer`).

### Baselines (for contextualizing this model's contribution)
| Baseline | What it isolates |
|---|---|
| **BM25-only** | The strong lexical floor — *genuinely competitive in P20*. The hybrid (BM25 + this dense model + RRF) must **beat BM25 on OCR-noise/paraphrase Recall while matching its Exact-Match precision**. |
| **Dense-only (this model alone)** | Isolates the trained retriever; **exposes its semantic-false-positive weakness** (can show high Recall but lower Exact-Match precision) that D4 verification + BM25 fix. |
| **Random** | Sanity floor. |

### Verified offline behavior (seed run)
The verified offline seed uses **BM25 over per-image OCR text** (the dense arm stands in as BM25 when sentence-transformers is absent):
- **Recall@1 = MRR = 1.0** on the synthetic collection — exact-term match is perfect for unique codes (e.g. `REF####`, `PO-4471`).
- The matching **snippet is highlighted**.
- A **nonexistent term** (`"zzqwx"`) correctly triggers **ABSTAIN** (D5), not a misleading top-k.
- All **5 agent decision points (D1–D5) fire**.

> This seed validates the *pipeline contract and the gold map*, and confirms the floor (BM25 exactness + abstention). The trained dense model's distinct contribution — recovering OCR-noise/paraphrase cases the lexical arm misses — is measured against the **BM25-only** baseline on the noise-injected (CER/WER) synthetic and real (TextOCR / CORD-v2 / DocVQA) splits, where it should lift Recall@K/MRR without degrading Exact-Match precision below the BM25 floor.

---

## 5. Limitations

1. **OCR-error dependence (the dominant limitation).** This model only ever sees **OCR output**, never the image. If the OCR front-end drops or garbles a term, the retriever cannot recover information that was never in its input — and OCR CER/WER place a hard ceiling on achievable Recall. The model is *trained to be robust* to char-level confusions (`rn↔m`, `0↔O`, `1↔l/I`), and the dense arm + RRF + the fuzzy D4 verifier recover many near-misses, but no embedding can recover a term the OCR never produced.
2. **Semantic false positives — no literal-term guarantee.** Embedding similarity ranks images that are *about* a topic even when the query term is absent (for `INVOICE`, images about billing). The literal-containment guarantee is **not** provided by this model; it is enforced downstream by the BM25 arm and the **D4 verifier**, which DROPS dense candidates whose OCR text does not contain the term under `exact_intent`. **Do not deploy the raw dense ranking without D4.**
3. **Domain shift.** Trained primarily on synthetic business-document vocabulary plus scene-text (TextOCR), receipts (CORD-v2), and DocVQA documents. Collections far from these domains (handwriting, dense legal text, heavy tables/forms, unusual layouts) will see degraded embedding quality. The synthetic vocabulary, while controlled and useful for a clean gold map, is narrower than open-world document text.
4. **Script / language coverage.** **English only** is validated. Non-Latin scripts and non-EN languages are unsupported by the retriever, the en-only tokenization, and (separately) by OCR language packs. This produces **unequal search recall across scripts** — a fairness concern, not just a capability gap (see Ethics).
5. **Phrase / multi-word queries.** Bi-encoder mean-pooling blurs word order; phrase order, hyphenation, and OCR line breaks complicate literal phrase matching. D2 (quoted-phrase vs keyword intent) and the token-boundary-aware D4 verifier mitigate this, but the dense arm alone is not order-faithful.
6. **Short-query degeneracy.** Single tokens, bare IDs, and dates carry little distributional signal for a sentence embedder; here **BM25 is the stronger arm** and RRF leans on it. The dense model earns its place mainly on noisy/paraphrased queries, not clean unique codes (where BM25 already scores 1.0).
7. **Index scaling.** The dense arm uses exact `IndexFlatIP` (O(N) per query). At large N (e.g. RVL-CDIP's 400K) swap to an ANN index (HNSW/IVF) behind the same interface; this trades exactness for latency and can change recall.

---

## 6. Ethical considerations

1. **Sensitive-text indexing (primary concern).** OCR-indexing user/scanned documents surfaces whatever text they contain — IDs, contracts, medical and financial records, and other **PII embedded in the image text**. The retriever's embeddings are derived from that text. Mitigations are **mandatory** at deployment: explicit consent for indexing, access control on the collection and the `/search` endpoint, **PII redaction** before/at indexing, and **no query retention by default**.
2. **LLM brain OFF by default.** The optional `anthropic` "brain" is **advisory only** (a query-expansion note), **never** changes ranking, and is **disabled by default** — no document text is sent to an external LLM unless an operator explicitly opts in.
3. **OCR-error harms cut both ways.** OCR errors cause **false negatives** (a document that *does* contain the term is missed → a relevant record is not found, potentially consequential in legal/medical/financial search) and **false positives**. The fuzzy D4 verification and D5 abstention mitigate, but operators must understand that **a "not found" is not proof of absence**.
4. **Bias / unequal recall by script and scan quality.** OCR quality — and therefore retrieval recall — varies by **script, font, and image quality** (non-Latin scripts, low-quality scans). This yields **systematically unequal search recall** across populations and document types. Do not claim uniform performance; report per-condition metrics where feasible and treat non-EN/low-quality scans as degraded, not equivalent.
5. **Honest failure over confident error.** The system is designed to **abstain** (D5: "not found in any image" + `needs_review`) rather than return the least-irrelevant image, and to surface **explicit `low_ocr_confidence` / `fuzzy_match` flags** rather than silent results — reflecting a deliberate preference for honest "I don't know" over a misleading top-k.
6. **Licensing discipline.** The default/demo training and eval ship only on **MIT (TextOCR/IIIT5K) + CC0 (Post-OCR) + CC-BY-4.0 (CORD-v2, with attribution) + synthetic**. Every flagged set (`docvqa_1200_examples` license-unspecified, `funsd` research-only, `rvl_cdip` NC tobacco corpus, `COCO-Text` no tag) is gated behind a `research_only` flag and **not redistributed**; the non-commercial Surya OCR weights are excluded from the default stack.

---

## 7. How to reproduce / cite

- **Base model:** `BAAI/bge-small-en-v1.5` (MIT). Fine-tune with `MultipleNegativesRankingLoss` on `(query, OCR-text)` pairs.
- **Index it with:** the BM25 lead arm (`19_.../clipsearch/index/lexical.py:BM25Index`, `k1=1.5`, `b=0.75`) + this dense arm (`19_.../clipsearch/index/image_index.py:ImageIndex`, `IndexFlatIP` on L2-normalized OCR-text embeddings) fused via **RRF (`c=60`)**.
- **Evaluate with:** `recall_at_k` / `mrr` / `median_rank` (P19 `metrics.py`), `corpus_cer` / `corpus_wer` (P07 `dococr/training/metrics.py`), and the **P20 Exact-Match precision@K** containment check (built on `normalize_ws`).
- **Always run behind:** the D1–D5 agent — specifically **D4 literal verification** and **D5 abstention** — which provide the literal-term guarantee this model alone does not.

> **Summary.** This is a small, permissively licensed, English dense text retriever whose job is to make OCR-document search **robust to OCR noise and paraphrase**. It is one arm of a hybrid index, deliberately subordinate to BM25 on literal tokens, and its semantic-false-positive tendency is contained by downstream verification and abstention — so the overall system answers *"which images literally contain this text"* honestly, not *"which images are vaguely about this topic."*
