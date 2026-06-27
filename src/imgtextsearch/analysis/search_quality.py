"""Search-quality report (the P20 special quality analysis).

Runs the agent on the seed queries and reports Recall@1/5/10 + MRR, the mean exact-match
precision (fraction of returned images that literally contain the query term), and the abstention
rate - quantifying both the ranking quality and the OCR-search value-add (literal verification).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp
from ..training import metrics as M

logger = get_logger(__name__)


def search_quality_report(cfg: AppConfig = None, limit: Optional[int] = None, save: bool = True) -> Dict:
    cfg = cfg or AppConfig()
    try:
        from ..agent.search_agent import SearchAgent
        from ..data import samples
        agent = SearchAgent(cfg, load_model=False)
        coll = {c["id"]: c["text"] for c in samples.seed_collection()}
        queries = samples.seed_queries()
    except Exception as exc:
        return _stub(str(exc), save)
    if limit:
        queries = queries[:limit]

    retrieved, golds = [], []
    exact: List[float] = []
    n_abstain = 0
    for q in queries:
        out = agent.search(q["query"])
        if out["abstained"]:
            n_abstain += 1
        ids = [r["id"] for r in out["results"]]
        retrieved.append(ids)
        golds.append(q["gold_ids"])
        exact.append(M.exact_match_precision([coll.get(i, "") for i in ids], q["query"]))
    rm = M.retrieval_metrics(retrieved, golds)
    n = max(1, len(queries))
    result = {"n": len(queries), **rm, "abstain_rate": round(n_abstain / n, 4),
              "exact_match_precision": round(sum(exact) / n, 4)}
    if save:
        _save(result)
    logger.info("search quality: R@1=%s MRR=%s exact=%s", rm["recall"]["@1"], rm["mrr"],
                result["exact_match_precision"])
    return result


def _stub(error: str, save: bool) -> Dict:
    result = {"n": 0, "recall": {}, "mrr": 0.0, "abstain_rate": 0.0,
              "exact_match_precision": 0.0, "error": error}
    if save:
        _save(result)
    return result


def _save(out: Dict) -> None:
    try:
        d = run_dir() / "search_quality"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"sq-{utc_stamp()}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        (d / "latest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.info("search_quality: could not save (%s)", exc)


__all__ = ["search_quality_report"]
