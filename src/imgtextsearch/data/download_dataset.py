"""Prefetch + sanity-check: probe the retriever + (optional) the scene-text collection, report
the synthetic seed stats, and optionally render the synthetic collection. Degrades gracefully.
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _probe(loader) -> Dict[str, Any]:
    try:
        return {"ok": True, **loader()}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def download_all(cfg: AppConfig, render_synthetic: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {"retriever": {}, "collection": {}, "seed": {}, "synthetic": {}}

    def retriever_probe():
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return {"model": cfg.model.base_model, "reachable": True}

    def collection_probe():
        from datasets import load_dataset
        ds = load_dataset(cfg.data.collection_dataset, split="train", streaming=True)
        first = next(iter(ds))
        return {"dataset": cfg.data.collection_dataset, "reachable": True, "columns": list(first.keys())}

    out["retriever"] = _probe(retriever_probe)
    out["collection"] = _probe(collection_probe) if cfg.data.use_hf else \
        {"ok": True, "note": "use_hf off; synthetic rendered-text collection (no benchmark for image-text-search exists)"}

    from . import samples
    out["seed"] = {"ok": True, "images": len(samples.seed_collection()), "queries": len(samples.seed_queries())}

    if render_synthetic:
        try:
            from .dataset import build_synthetic
            out["synthetic"] = {"ok": True, **build_synthetic(cfg)}
        except Exception as exc:
            out["synthetic"] = {"ok": False, "error": str(exc)}

    logger.info("download_all: retriever=%s collection=%s seed=%d images",
                out["retriever"].get("ok"), out["collection"].get("ok"), out["seed"]["images"])
    return out


__all__ = ["download_all"]
