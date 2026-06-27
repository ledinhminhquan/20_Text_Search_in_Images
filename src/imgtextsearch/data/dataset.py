"""Data loading: an image-text collection (real scene-text/document images or synthetic).

* ``load_collection`` - (items, gold_texts) where items are PIL images (real) or text specs
  (synthetic); for real data the gold text comes with the dataset, else the synthetic snippet.
``datasets`` is imported lazily; everything falls back to the synthetic seed collection offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, data_dir
from ..logging_utils import get_logger
from . import samples

logger = get_logger(__name__)


def load_collection(cfg: AppConfig, limit: Optional[int] = None) -> Tuple[List[Any], List[Dict]]:
    dc = cfg.data
    cap = limit or dc.collection_size
    if dc.use_hf:
        try:
            from datasets import load_dataset  # lazy
            ds = load_dataset(dc.collection_dataset, split="train", streaming=True)
            items, metas = [], []
            for i, r in enumerate(ds):
                if len(items) >= cap:
                    break
                img = r.get("image")
                text = r.get("text") or r.get("ground_truth") or ""
                if img is None:
                    continue
                iid = f"img_{i:05d}"
                items.append({"id": iid, "image": img})
                metas.append({"id": iid, "gold_text": str(text)})
            if len(items) > 4:
                logger.info("Loaded %d images from %s", len(items), dc.collection_dataset)
                return items, metas
        except Exception as exc:
            logger.warning("Could not load %s (%s); using synthetic collection.", dc.collection_dataset, exc)
    coll = samples.seed_collection()
    items = [{"id": c["id"], "text": c["text"]} for c in coll]
    metas = [{"id": c["id"], "gold_text": c["text"]} for c in coll]
    return items, metas


def synthetic_dir(split: str = "eval") -> Path:
    return data_dir() / "synthetic" / split


def build_synthetic(cfg: AppConfig, n_images: Optional[int] = None, split: str = "eval") -> Dict:
    from .synth_text_images import generate_collection
    return generate_collection(str(synthetic_dir(split)), n_images=n_images or cfg.data.collection_size,
                               width=cfg.data.image_width, seed=cfg.data.seed)


__all__ = ["load_collection", "synthetic_dir", "build_synthetic"]
