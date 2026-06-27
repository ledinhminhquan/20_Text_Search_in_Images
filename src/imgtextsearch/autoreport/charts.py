"""Matplotlib charts for the text-search-within-images report/slides.

  * a **Recall@k** grouped bar chart - hybrid vs BM25-only vs random;
  * a **quality** chart - Recall@1 + exact-match precision + (1-OCR CER) + abstain rate;
  * an **outcome bucket** chart (hit@1 / abstained / miss) from error analysis.

Returns saved PNG paths under ``run_dir()/report``; matplotlib lazy-imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_utils import get_logger
from . import artifact_loader as AL

logger = get_logger(__name__)

_HYBRID = "#2b6cb0"
_BM25 = "#9aa7b4"
_RAND = "#cbd5e0"
_GOOD = "#2f855a"
_MED = "#dd6b20"
_POOR = "#c53030"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def recall_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    if not AL.has_eval(arts):
        return None
    ks = ["@1", "@5", "@10"]
    series = [("hybrid", _HYBRID), ("bm25_only", _BM25), ("random", _RAND)]
    try:
        plt = _mpl()
        x = list(range(len(ks)))
        width = 0.8 / len(series)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for si, (name, color) in enumerate(series):
            vals = [(AL.sys_recall(arts, name, k) or 0.0) * 100 for k in ks]
            offs = [i + (si - (len(series) - 1) / 2) * width for i in x]
            bars = ax.bar(offs, vals, width=width, label=name, color=color)
            for rect, v in zip(bars, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"Recall{k}" for k in ks])
        ax.set_ylim(0, 105); ax.set_ylabel("%")
        ax.set_title("Text-in-image Recall@k: hybrid vs BM25 vs random")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("recall_chart skipped (%s)", exc)
        return None


def quality_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    r1 = AL.sys_recall(arts, "hybrid", "@1")
    ex = AL.exact_match(arts)
    cer = AL.ocr_cer(arts)
    if r1 is None and ex is None:
        return None
    try:
        plt = _mpl()
        labels = ["Recall@1", "exact-match", "OCR acc\n(1-CER)"]
        vals = [(r1 or 0.0) * 100, (ex or 0.0) * 100, (1.0 - (cer or 0.0)) * 100]
        fig, ax = plt.subplots(figsize=(5.8, 3.4))
        ax.bar(labels, vals, color=[_HYBRID, _GOOD, "#553c9a"])
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105); ax.set_ylabel("%")
        ax.set_title("Search quality + OCR accuracy")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("quality_chart skipped (%s)", exc)
        return None


def buckets_chart(arts: Dict[str, Any], out_path: Path) -> Optional[Path]:
    b = AL.buckets(arts)
    vals = [b.get("correct"), b.get("abstained"), b.get("wrong")]
    if not any(isinstance(v, (int, float)) for v in vals):
        return None
    try:
        plt = _mpl()
        labels = ["hit@1", "abstained", "miss"]
        nums = [float(v) if isinstance(v, (int, float)) else 0.0 for v in vals]
        fig, ax = plt.subplots(figsize=(5.6, 3.3))
        ax.bar(labels, nums, color=[_GOOD, _MED, _POOR])
        for i, v in enumerate(nums):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("# queries"); ax.set_title("Agent outcomes (hit@1 / abstained / miss)")
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        return out_path
    except Exception as exc:
        logger.info("buckets_chart skipped (%s)", exc)
        return None


def build_all(arts: Dict[str, Any], out_dir: Path) -> List[Tuple[str, Path]]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return []
    charts: List[Tuple[str, Path]] = []
    jobs = [("recall", lambda p: recall_chart(arts, p)),
            ("quality", lambda p: quality_chart(arts, p)),
            ("buckets", lambda p: buckets_chart(arts, p))]
    for name, fn in jobs:
        try:
            p = fn(out_dir / f"{name}.png")
        except Exception as exc:
            logger.info("chart %s skipped (%s)", name, exc)
            p = None
        if p:
            charts.append((name, p))
    return charts


__all__ = ["recall_chart", "quality_chart", "buckets_chart", "build_all"]
