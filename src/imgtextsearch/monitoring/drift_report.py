"""Production monitoring report from the serving job log (imgtext_search JSONL).

Turns raw ``imgtext_search`` events into a health picture: query volume, status mix, the
**abstention / low-confidence rate** (queries no image satisfied), the mean exact-match count,
mean top score, latency, and a drift signal comparing a recent window to an earlier baseline (a
rising abstention rate or falling top-score flags out-of-domain queries or OCR/index degradation).
Stdlib only; never raises past its entrypoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_stamp

logger = get_logger(__name__)

_EVENT = "imgtext_search"


def _read_logs(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("could not read job log %s: %s", path, exc)
        return rows
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 1)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[rank]), 1)


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _window_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    statuses: Dict[str, int] = {}
    abstained = low_conf = 0
    exacts: List[float] = []
    tops: List[float] = []
    lats: List[float] = []
    for r in rows:
        statuses[str(r.get("status", "?"))] = statuses.get(str(r.get("status", "?")), 0) + 1
        if bool(r.get("abstained")):
            abstained += 1
        if bool(r.get("low_confidence")):
            low_conf += 1
        if _is_num(r.get("n_exact")):
            exacts.append(float(r["n_exact"]))
        if _is_num(r.get("top_score")):
            tops.append(float(r["top_score"]))
        metrics = r.get("metrics") or {}
        if isinstance(metrics, dict) and _is_num(metrics.get("latency_ms")):
            lats.append(float(metrics["latency_ms"]))
    return {"n": n, "status_distribution": statuses, "abstain_rate": round(abstained / n, 4),
            "low_confidence_rate": round(low_conf / n, 4), "mean_exact_matches": _mean(exacts),
            "mean_top_score": _mean(tops), "mean_latency_ms": _mean(lats), "p95_latency_ms": _percentile(lats, 95)}


def _delta(base: Dict[str, Any], recent: Dict[str, Any], key: str) -> Optional[float]:
    a, b = base.get(key), recent.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return round(float(b) - float(a), 4)
    return None


def _drift(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) < 6:
        return {"available": False, "reason": "need >=6 events to split baseline/recent windows"}
    half = len(rows) // 2
    base = _window_stats(rows[:half])
    recent = _window_stats(rows[half:])
    d_ab = _delta(base, recent, "abstain_rate")
    d_top = _delta(base, recent, "mean_top_score")
    d_lat = _delta(base, recent, "mean_latency_ms")
    flags: List[str] = []
    if (d_ab or 0) > 0.15:
        flags.append("rising_abstention_rate")
    if d_top is not None and d_top < -0.1:
        flags.append("falling_top_score")
    if d_lat is not None and base.get("mean_latency_ms"):
        if d_lat / (base["mean_latency_ms"] or 1.0) > 0.5:
            flags.append("latency_regression")
    return {"available": True, "baseline_window": base, "recent_window": recent,
            "delta_abstain_rate": d_ab, "delta_mean_top_score": d_top, "delta_mean_latency_ms": d_lat,
            "flags": flags, "alert": bool(flags)}


def _recommendations(overall: Dict[str, Any], drift: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    ab = overall.get("abstain_rate") or 0.0
    flags = drift.get("flags") or []
    if ab > 0.4:
        recs.append("High abstention rate ({:.0%}): queries are out-of-domain or OCR/index coverage is "
                    "missing content - re-OCR / re-index and re-fine-tune the retriever.".format(ab))
    if "rising_abstention_rate" in flags or "falling_top_score" in flags:
        recs.append("Search confidence is drifting vs the baseline window: the query or document "
                    "distribution (or OCR quality) may have shifted - re-evaluate and refresh the index.")
    if not recs:
        recs.append("No action needed: monitoring metrics within healthy operating ranges.")
    return recs


def monitoring_report(cfg: AppConfig, log_path: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    path = Path(log_path) if log_path else cfg.serving.job_log_path
    rows = _read_logs(path)
    events = [r for r in rows if r.get("event", _EVENT) == _EVENT]
    if not events:
        logger.info("monitoring: no imgtext_search events at %s", path)
        result = {"status": "no_data", "log_path": str(path), "n_events": 0, "request_volume": 0,
                  "overall": {"n": 0}, "drift": {"available": False, "reason": "no events"},
                  "recommendations": ["No job logs yet: exercise the agent / API to populate the log."],
                  "generated_at": utc_stamp()}
    else:
        overall = _window_stats(events)
        drift = _drift(events)
        result = {"status": "ok", "log_path": str(path), "n_events": len(events),
                  "request_volume": len(events), "overall": overall, "drift": drift,
                  "recommendations": _recommendations(overall, drift), "generated_at": utc_stamp()}
    if save:
        try:
            out = run_dir() / "monitoring"
            out.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(result, indent=2, ensure_ascii=False)
            (out / f"monitor-{utc_stamp()}.json").write_text(payload, encoding="utf-8")
            (out / "latest.json").write_text(payload, encoding="utf-8")
        except Exception as exc:
            logger.warning("monitoring: could not save report: %s", exc)
    logger.info("monitoring: %s events, abstain=%.0f%% mean_top=%s p95=%s ms, drift_alert=%s",
                result["n_events"], 100 * (result["overall"].get("abstain_rate") or 0.0),
                result["overall"].get("mean_top_score"), result["overall"].get("p95_latency_ms"),
                result.get("drift", {}).get("alert", False))
    return result


__all__ = ["monitoring_report"]
