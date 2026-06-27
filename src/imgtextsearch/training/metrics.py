"""Metrics for text-search-within-images.

* **Retrieval Recall@K / MRR / median rank** - a query "hits" if ANY image whose OCR text
  contains the query (the gold set) is in the top-K; the rank is that of the first gold image.
* **OCR CER / WER** - character / word error rate of the OCR text vs the rendered gold text.
* **Exact-match precision** - fraction of the returned top-k images whose OCR text LITERALLY
  contains the query term (the OCR-search guarantee, beyond semantic similarity).
Pure-python, no heavy deps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def first_gold_rank(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> Optional[int]:
    gold = set(gold_ids)
    for r, rid in enumerate(retrieved_ids, 1):
        if rid in gold:
            return r
    return None


def ranks(retrieved_lists: Sequence[Sequence[str]], gold_lists: Sequence[Sequence[str]],
          missing_rank: int = 1000) -> List[int]:
    out = []
    for ret, gold in zip(retrieved_lists, gold_lists):
        r = first_gold_rank(ret, gold)
        out.append(r if r is not None else missing_rank)
    return out


def recall_at_k(rank_list: Sequence[int], k: int) -> float:
    if not rank_list:
        return 0.0
    return round(sum(1 for r in rank_list if r <= k) / len(rank_list), 4)


def mrr(rank_list: Sequence[int]) -> float:
    if not rank_list:
        return 0.0
    return round(sum(1.0 / r for r in rank_list) / len(rank_list), 4)


def median_rank(rank_list: Sequence[int]) -> float:
    if not rank_list:
        return 0.0
    s = sorted(rank_list)
    n = len(s)
    return float(s[n // 2]) if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 2)


def retrieval_metrics(retrieved_lists: Sequence[Sequence[str]], gold_lists: Sequence[Sequence[str]],
                      ks: Sequence[int] = (1, 5, 10)) -> Dict[str, Any]:
    rk = ranks(retrieved_lists, gold_lists)
    return {"recall": {f"@{k}": recall_at_k(rk, k) for k in ks},
            "mrr": mrr(rk), "median_rank": median_rank(rk), "n": len(rk)}


def _levenshtein(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    tot_err = tot = 0
    for h, r in zip(hyps, refs):
        rr = list((r or "").strip())
        tot += len(rr)
        tot_err += _levenshtein(list((h or "").strip()), rr)
    return round(tot_err / max(1, tot), 4)


def exact_match_precision(retrieved_texts: Sequence[str], query: str) -> float:
    """Fraction of returned images whose OCR text literally contains a query token."""
    qtoks = _TOKEN.findall((query or "").lower())
    if not retrieved_texts or not qtoks:
        return 0.0
    n = sum(1 for t in retrieved_texts if any(qt in _TOKEN.findall((t or "").lower()) for qt in qtoks))
    return round(n / len(retrieved_texts), 4)


__all__ = ["first_gold_rank", "ranks", "recall_at_k", "mrr", "median_rank", "retrieval_metrics",
           "cer", "exact_match_precision"]
