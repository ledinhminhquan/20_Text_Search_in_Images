"""Persist the BM25 baseline artifact (the offline floor + reference).

The BM25 arm needs no training; this builder records it as a versioned baseline so the registry,
report and API can reference the exact-term floor the fine-tuned dense retriever must complement.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_baseline(cfg: AppConfig, limit: Optional[int] = None) -> Dict:
    out_path = cfg.model.output_dir / cfg.model.baseline_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": "baselines", "version": "baseline-1.0",
               "bm25_only": "self-contained BM25 over the per-image OCR text (exact-term floor)",
               "dense_only": cfg.model.base_model, "random": "random ranking floor"}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("baselines -> %s", out_path)
    return {"baseline_path": str(out_path)}


__all__ = ["build_baseline"]
