"""Error analysis (offline): hit@1 / abstained / miss buckets + worst examples.

Runs the agent on the seed queries, checks whether a gold image (one whose OCR text contains the
query) is ranked #1, and buckets the outcomes. Short keys ``correct``/``abstained``/``wrong`` feed
the charts.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)


def error_analysis(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.search_agent import SearchAgent
        from ..data import samples
        agent = SearchAgent(cfg, load_model=False)
        queries = samples.seed_queries()
    except Exception as exc:
        return _stub(str(exc), save)
    if limit:
        queries = queries[:limit]

    correct = abstained = wrong = 0
    worst: List[Dict] = []
    for q in queries:
        try:
            out = agent.search(q["query"])
        except Exception:
            continue
        if out["abstained"]:
            abstained += 1
            continue
        ids = [r["id"] for r in out["results"]]
        if ids and ids[0] in q["gold_ids"]:
            correct += 1
        else:
            wrong += 1
            if len(worst) < 8:
                worst.append({"query": q["query"], "top": ids[:3], "gold": q["gold_ids"][:3]})
    n = max(1, len(queries))
    result = {"n": len(queries), "correct": correct, "abstained": abstained, "wrong": wrong,
              "hit_at_1": round(correct / n, 4), "abstain_rate": round(abstained / n, 4),
              "worst_examples": worst}
    if save:
        _save(result)
    logger.info("error analysis: correct=%d abstained=%d wrong=%d", correct, abstained, wrong)
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n": 0, "correct": 0, "abstained": 0, "wrong": 0, "hit_at_1": 0.0,
              "abstain_rate": 0.0, "worst_examples": [], "error": error}
    if save:
        _save(result)
    return result


def _save(result: Dict) -> None:
    try:
        d = run_dir() / "error_analysis"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"errors-{utc_stamp()}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.info("error_analysis: could not save (%s)", exc)


__all__ = ["error_analysis"]
