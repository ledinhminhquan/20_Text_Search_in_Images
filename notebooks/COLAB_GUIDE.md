# Colab / H100 Guide — Text Search within Images

End-to-end: upload to Drive → open the notebook → run cells → train → collect `report.pdf` +
`slides.pptx` + the submission bundle. The notebook auto-adapts to **H100 / A100 / L4 / T4**.

## Option A — clone from GitHub (recommended)

1. Push this folder to a GitHub repo (`20_Text_Search_in_Images`).
2. Open `notebooks/Text_Search_in_Images_Colab_H100.ipynb` in Colab
   (or upload it). **Runtime → Change runtime type → GPU** (pick H100 with Colab Pro+).
3. In the **Controls** cell, set `GIT_URL` to your repo. Leave `CLONE_FROM_GIT = True`.
4. **Runtime → Run all.** The **ONE-BUTTON AUTOPILOT** cell does everything.

## Option B — upload the repo to Drive

1. Zip this folder and upload it to `MyDrive/imgtextsearch/`, unzip so that
   `MyDrive/imgtextsearch/20_Text_Search_in_Images/src/imgtextsearch/` exists.
2. In **Controls** set `CLONE_FROM_GIT = False`. Run all.

## What the autopilot produces (in your Drive)

```
MyDrive/imgtextsearch/artifacts/
  models/<version>/                 fine-tuned dense OCR-text retriever
  runs/<run>/eval.json              Recall@k/MRR vs baselines + OCR CER + exact-match
  runs/<run>/{error_analysis,search_quality,benchmark,tune,monitoring}/
  submission/submission-*/report.pdf
  submission/submission-*/slides.pptx
  submission/submission-*/submission_bundle.zip   <-- hand this in
```

## Controls

| Control | Meaning |
|---------|---------|
| `TRAIN_RETRIEVER` | fine-tune the dense retriever (off → BM25-only, still produces a report) |
| `USE_REAL_DATA` | index a real HF collection (`MiXaiLL76/TextOCR_OCR`) instead of synthetic |
| `EPOCHS` / `PAIR_LIMIT` | training length / number of (query, OCR-text) pairs |
| `RUN_AUTOPILOT` | one-button everything vs the individual step cells |

## GPU auto-profile

| GPU | batch size | bf16 / tf32 |
|-----|-----------|-------------|
| H100 | 192 | yes |
| A100 | 128 | yes |
| L4 | 64 | yes |
| T4 | 32 | no |
| CPU | 16 | no (training skipped by autopilot) |

## Testing after training

Cell **10** loads the fine-tuned retriever and searches `invoice`, `REF3386`, `2023`, `zzqwx`
(the last must **abstain**). Cell **9** runs `grade` (target score **1.0**) and `demo-agent`.

## Notes

- **Colab-safe install:** the ML deps come from `requirements_colab.txt`, then the package installs
  with `pip install -e . --no-deps` so it never perturbs Colab's torch/CUDA.
- **Offline-friendly:** with no GPU the autopilot skips training and runs the BM25 + SeedEngine path
  end-to-end (still writes `report.pdf`, `slides.pptx`, and grades).
- **Licenses:** `USE_REAL_DATA` pulls third-party datasets — some are non-commercial / unspecified
  (see `docs/data_card.md`). The synthetic generator avoids all licensing concerns.
