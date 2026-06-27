# Text Search within Images (`imgtextsearch`)

> **Search a collection of images by the TEXT they CONTAIN.** OCR each image → index the per-image
> text with a **trainable dense retriever + BM25 (RRF)** → for a query, rank the images, **verify
> the term literally appears** (fuzzy to OCR errors), show the matching **snippet**, and **abstain
> ("not found")** when no image contains the text. A deterministic agent (D1–D5) orchestrates it.

NLP in Industry — Final Assignment, **Project 20**. Author: **Le Dinh Minh Quan (23127460)**.

This is OCR-based **document/text search**, distinct from visual/semantic search (that was P19,
CLIP). Only the **dense text retriever** is trained; the OCR front-end, BM25 and the literal
verifier are pretrained/algorithmic. The whole pipeline runs **fully offline** (a synthetic
rendered-text generator + a `SeedEngine` that reads gold text embedded in each image + pure-python
BM25), and upgrades to **Tesseract + a fine-tuned bge-small retriever** on Colab/H100.

---

## What it does

```
INDEX:   images ──► OCR (Tesseract / SeedEngine) ──► per-image text ──► BM25 + dense embeddings
SEARCH:  query ──► parse(D1) ──► search BM25+dense→RRF(D2) ──► coverage(D3)
                  ──► snippet + literal-term verify(D4) ──► abstain / finalize(D5)
OUTPUT:  ranked image ids + scores + matching snippet + exact-match flag  (or "not found")
```

**Why an agent over a raw index?** BM25/dense can rank a *semantically similar* image, but for
"contains the word X" the term must actually **appear**. The agent's value-add is **literal-term
verification** (edit-distance ≤ 1, to survive OCR errors), **snippet highlighting**, and
**abstention** when nothing matches.

## Quickstart (offline, no GPU / no Tesseract)

```bash
pip install -e .                       # core deps only (numpy, pyyaml, pydantic, Pillow)

imgtextsearch demo-agent --fast        # run the agent on the seed queries
imgtextsearch search --query invoice --fast
imgtextsearch search --query REF3386 --fast    # exact code -> rank-1 exact match
imgtextsearch search --query zzqwx --fast      # absent -> ABSTAINS ("not found")
imgtextsearch evaluate --fast          # Recall@k/MRR vs baselines + OCR CER + exact-match
imgtextsearch autopilot --no-train     # full pipeline -> report.pdf + slides.pptx + bundle
imgtextsearch grade                    # rubric self-check (target score 1.0)
```

Everything above runs with **no torch, no Tesseract, no network**.

## Train on Colab / H100

Open [`notebooks/Text_Search_in_Images_Colab_H100.ipynb`](notebooks/Text_Search_in_Images_Colab_H100.ipynb),
set the GPU runtime, **Run all** → the **one-button autopilot** fine-tunes the dense retriever and
writes `report.pdf` + `slides.pptx` + a submission bundle to your Drive. Full walkthrough:
[`notebooks/COLAB_GUIDE.md`](notebooks/COLAB_GUIDE.md).

```bash
pip install -e ".[all]"                # torch + sentence-transformers + tesseract bindings + serving + report
imgtextsearch train-retriever          # fine-tune bge-small with MNRL on (query, OCR-text) pairs
imgtextsearch evaluate                 # full eval with the trained dense retriever
imgtextsearch serve --ui               # FastAPI at :8000 + Gradio demo at :8000/ui
```

## Model & data stack (verified on the HF Hub)

| Component | Default | License |
|-----------|---------|---------|
| OCR front-end | **Tesseract** (`pytesseract`); `SeedEngine` offline | Apache-2.0 |
| OCR (neural upgrade) | `microsoft/trocr-base-printed`; docTR/Paddle/EasyOCR | MIT / Apache |
| **Dense text retriever (trained)** | **`BAAI/bge-small-en-v1.5`** (MNRL) | **MIT** |
| Lexical arm | pure-python **BM25** (exact terms) + **RRF** fusion | — |
| Reranker (optional) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 |
| Real collections | `MiXaiLL76/TextOCR_OCR` (MIT), `naver-clova-ix/cord-v2` (CC-BY-4.0) | see card |
| Retrieval signal | `nielsr/docvqa_1200_examples` (query + gold span) | **unspecified — flagged** |
| OCR text source | `PleIAs/Post-OCR-Correction` | CC0 |

⚠️ Some datasets/models are **non-commercial or unspecified-license** (Surya OCR CC-BY-NC-SA;
`aharley/rvl_cdip`, `nielsr/funsd-layoutlmv3`, `nielsr/docvqa_1200_examples`,
`howard-hou/COCO-Text`). They are **flagged** in [`docs/data_card.md`](docs/data_card.md) and
[`LICENSE`](LICENSE). The default **synthetic generator** avoids all licensing concerns — there is
no public "find the image containing word X" benchmark, so it is the primary offline data.

## Metrics

- **Text-in-image Recall@1/5/10 + MRR + median rank** — a query *hits* if ANY image whose OCR text
  contains it is in the top-K (multi-gold; rank = first gold).
- **OCR CER/WER** — OCR output vs the gold rendered text.
- **Exact-match precision** — fraction of returned images whose OCR text *literally* contains the
  query term.
- **Baselines:** BM25-only (exact-term floor), dense-only (semantic), random.

Offline-verified: clean OCR → **Recall@1 = MRR = 1.0** for unique codes; with injected OCR noise
the exact match degrades (demonstrating the OCR-error false-negative problem the fuzzy verify +
dense retriever mitigate); absent terms **abstain**.

## The agent — five decision points

| # | State | Decision (acts on) | Branches |
|---|-------|--------------------|----------|
| **D1** | parse | query gate + classify (exact_code / keyword / phrase) | proceed / fail |
| **D2** | search | BM25 (exact) + dense (semantic) → **RRF** → top-k candidates | candidates / empty |
| **D3** | coverage | top score vs `coverage_min_score` | confident / low-confidence |
| **D4** | verify | snippet + **literal-term match** (fuzzy, edit-distance ≤ 1) | exact-first / semantic |
| **D5** | finalize | **abstain** if no image literally contains the query | results / not-found |

An optional LLM **brain** (`anthropic`, **OFF by default**) only adds an advisory note — it never
changes the ranking, and the default runs with **zero paid API calls**.

## Repository layout

```
src/imgtextsearch/
  config.py  cli.py  logging_utils.py
  ocr/           engine.py            # Tesseract / SeedEngine / Stub + font discovery
  data/          synth_text_images.py samples.py dataset.py download_dataset.py
  models/        encoder.py baseline.py model_registry.py
  index/         text_index.py lexical.py   # BM25 + dense + RRF over per-image OCR text
  training/      train_retriever.py train_baseline.py evaluate.py tune.py metrics.py
  agent/         search_agent.py tools.py policy.py state.py llm_orchestrator.py
  api/           main.py schemas.py dependencies.py ui.py app_combined.py
  analysis/      error_analysis.py latency.py search_quality.py
  autoreport/    artifact_loader.py charts.py report_pdf.py slides_pptx.py
  monitoring/    drift_report.py
  automation/    autopilot.py
  grading/       checklist.py
docs/   (14 Section-I docs + DESIGN_BRIEF)   notebooks/   tests/   configs/   app/   deploy/   sample_data/
```

## Documentation

All Section-I deliverables live in [`docs/`](docs/): problem definition, data description &
[data card](docs/data_card.md), [model selection](docs/model_selection.md),
[architecture](docs/architecture.md), [agent architecture](docs/agent_architecture.md),
[search evaluation](docs/search_evaluation.md), [deployment](docs/deployment.md),
[continual learning & monitoring](docs/continual_learning_monitoring.md),
[privacy & robustness](docs/privacy_robustness.md), [project plan](docs/project_plan.md),
[ethics statement](docs/ethics_statement.md), [model card](docs/model_card.md), and the slide outline.

## Tests

```bash
pip install -e . pytest && pytest -q     # offline; no torch / tesseract / network
```

## License

Code: [MIT](LICENSE). Third-party models/datasets retain their own licenses (some non-commercial /
unspecified — see the data card). This MIT license covers only the code in this repository.
