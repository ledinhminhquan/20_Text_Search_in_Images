# Sample data — Text Search within Images

This system needs **no downloaded data to run**: the synthetic rendered-text generator
(`imgtextsearch.data.synth_text_images`) and the seed collection
(`imgtextsearch.data.samples`) produce an image-text collection + queries deterministically, and
the `SeedEngine` reads the gold text embedded in each synthetic image — so the whole pipeline
(index → search → verify → snippet → eval → agent) runs offline with no Tesseract/torch.

- [`sample_queries.json`](sample_queries.json) — a handful of example queries (an exact code, a
  shared keyword, and a deliberately-absent term that should make the agent **abstain**).

Render a synthetic collection to disk (PNG images with the gold text in their metadata):

```bash
imgtextsearch gen-synthetic        # -> $IMGTEXT_DATA_DIR/synthetic/eval/
```

To index a **real** image collection instead, set `data.use_hf: true` and point
`data.collection_dataset` at e.g. `MiXaiLL76/TextOCR_OCR` (MIT) or `naver-clova-ix/cord-v2`
(CC-BY-4.0). See [`docs/data_card.md`](../docs/data_card.md) for licenses (some are flagged
non-commercial / unspecified).
