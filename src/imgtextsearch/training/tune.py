"""Lightweight sweep of the OCR-robustness / abstain behaviour.

Sweeps the injected OCR char-noise and reports the hybrid Recall@1 + the OCR CER + the abstain
rate on a held-out synthetic collection - a cheap proxy for how the fuzzy literal-match + the
abstain gate hold up as OCR quality degrades.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger
from .evaluate import evaluate

logger = get_logger(__name__)


def tune(cfg: AppConfig, noises: Optional[List[float]] = None, save: bool = True,
         load_model: bool = True) -> Dict:
    noises = noises or [0.0, 0.05, 0.1, 0.2]
    trials: List[Dict[str, Any]] = []
    for noise in noises:
        rep = evaluate(cfg, limit=80, load_model=load_model, ocr_noise=noise, save=False)
        trials.append({"ocr_noise": noise, "hybrid_recall@1": rep["systems"]["hybrid"]["recall"]["@1"],
                       "ocr_cer": rep["ocr"]["cer"], "exact_match_precision": rep["exact_match_precision"]})
    best = max(trials, key=lambda t: t["hybrid_recall@1"]) if trials else {}
    result = {"trials": trials, "best": best}
    if save:
        out = run_dir() / "tune"
        out.mkdir(parents=True, exist_ok=True)
        (out / "tune.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("tune: best noise=%s R@1=%s", best.get("ocr_noise"), best.get("hybrid_recall@1"))
    return result


__all__ = ["tune"]
