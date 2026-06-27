"""Collect generated run artifacts into one dict for the report + slides generators.

Reads the JSON under ``run_dir()`` - the eval (hybrid vs baselines Recall@k/MRR + OCR CER +
exact-match), the error analysis, the search-quality report, a latency benchmark, and a monitoring
snapshot - plus the trained-retriever metadata. Every read is defensive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppConfig, run_dir
from ..models.model_registry import read_metadata, resolve_latest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def load_artifacts(cfg: AppConfig) -> Dict[str, Any]:
    rd = run_dir()
    arts: Dict[str, Any] = {
        "eval": _load_json(rd / "eval.json"),
        "error_analysis": _load_json(rd / "error_analysis" / "latest.json"),
        "search_quality": _load_json(rd / "search_quality" / "latest.json"),
        "benchmark": _load_json(rd / "benchmark" / "latest.json"),
        "tune": _load_json(rd / "tune" / "tune.json"),
        "monitoring": _load_json(rd / "monitoring" / "latest.json"),
    }
    try:
        latest = resolve_latest(cfg.model.output_dir)
        arts["model_meta"] = read_metadata(latest) if latest else {}
    except Exception:
        arts["model_meta"] = {}
    return arts


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def systems(arts: Dict[str, Any]) -> Dict[str, Any]:
    return (arts.get("eval") or {}).get("systems") or {}


def sys_recall(arts: Dict[str, Any], system: str, k: str) -> Optional[float]:
    return _num(((systems(arts).get(system) or {}).get("recall") or {}).get(k))


def sys_mrr(arts: Dict[str, Any], system: str) -> Optional[float]:
    return _num((systems(arts).get(system) or {}).get("mrr"))


def ocr_cer(arts: Dict[str, Any]) -> Optional[float]:
    return _num(((arts.get("eval") or {}).get("ocr") or {}).get("cer"))


def exact_match(arts: Dict[str, Any]) -> Optional[float]:
    return _num((arts.get("eval") or {}).get("exact_match_precision"))


def headline(arts: Dict[str, Any], key: str) -> Optional[float]:
    return _num(((arts.get("eval") or {}).get("headline") or {}).get(key))


def has_eval(arts: Dict[str, Any]) -> bool:
    return bool(systems(arts))


def encoder_name(arts: Dict[str, Any]) -> str:
    return str((arts.get("eval") or {}).get("encoder") or "bm25-only")


def model_version(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("version") or (arts.get("eval") or {}).get("model_version") or "untrained (BM25)")


def base_model(arts: Dict[str, Any]) -> str:
    mv = arts.get("model_meta") or {}
    return str(mv.get("base_model") or "BAAI/bge-small-en-v1.5")


def buckets(arts: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ea = arts.get("error_analysis") or {}
    return {"correct": _num(ea.get("correct")), "abstained": _num(ea.get("abstained")),
            "wrong": _num(ea.get("wrong"))}


def latency(arts: Dict[str, Any], pct: str = "p50") -> Optional[float]:
    b = (arts.get("benchmark") or {}).get("latency_ms") or {}
    return _num(b.get(pct))


def read_doc(name: str) -> str:
    p = repo_root() / "docs" / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


__all__ = ["load_artifacts", "read_doc", "repo_root", "systems", "sys_recall", "sys_mrr", "ocr_cer",
           "exact_match", "headline", "has_eval", "encoder_name", "model_version", "base_model",
           "buckets", "latency"]
