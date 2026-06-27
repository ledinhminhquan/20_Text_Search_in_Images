"""Decision-point logic for the text-search-within-images agent (pure, testable).

Five explicit decision points:
* **D1** query parse/gate (non-empty; classify exact-code vs keyword vs phrase).
* **D2** search (BM25 + dense -> RRF; top-k image candidates + scores).
* **D3** coverage / confidence gate (top score below threshold -> low confidence).
* **D4** snippet + literal-term verification (does the OCR text actually CONTAIN the query term?).
* **D5** abstain (no image contains the query text -> "not found"; else rank).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..config import AgentConfig

_TOKEN = re.compile(r"[a-z0-9]+")


def classify_query(query: str) -> str:
    q = (query or "").strip()
    toks = _TOKEN.findall(q.lower())
    if len(toks) == 1 and re.fullmatch(r"[a-z]{2,4}\d{2,}", toks[0]):
        return "exact_code"
    if len(toks) >= 3:
        return "phrase"
    return "keyword"


def query_gate(query: str, cfg: AgentConfig) -> Dict[str, Any]:
    if len((query or "").strip()) < cfg.min_query_chars:
        return {"ok": False, "branch": "empty_query"}
    return {"ok": True, "branch": "ok"}


def coverage_gate(top_score: Optional[float], cfg: AgentConfig) -> Dict[str, Any]:
    if top_score is None:
        return {"ok": True, "branch": "no_score"}
    if top_score < cfg.coverage_min_score:
        return {"ok": False, "branch": "low_confidence", "top_score": top_score}
    return {"ok": True, "branch": "ok", "top_score": top_score}


def _fuzzy_contains(text_tokens: List[str], qtok: str, allow_fuzzy: bool) -> bool:
    if qtok in text_tokens:
        return True
    if not allow_fuzzy:
        return False
    for t in text_tokens:
        if abs(len(t) - len(qtok)) <= 1 and _edit_le1(t, qtok):
            return True
    return False


def _edit_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # at most one substitution/insertion/deletion
    if len(a) == len(b):
        return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1; j += 1
        elif skipped:
            return False
        else:
            skipped = True; j += 1
    return True


def verify_exact(text: str, query: str, cfg: AgentConfig) -> bool:
    """Does the OCR text literally contain (a fuzzy match of) every query token?"""
    text_tokens = _TOKEN.findall((text or "").lower())
    qtoks = _TOKEN.findall((query or "").lower())
    if not qtoks:
        return False
    return all(_fuzzy_contains(text_tokens, qt, cfg.exact_match_fuzzy) for qt in qtoks)


def make_snippet(text: str, query: str, window: int) -> str:
    tokens = (text or "").split()
    low = [t.lower() for t in tokens]
    qtoks = _TOKEN.findall((query or "").lower())
    for i, t in enumerate(low):
        if any(qt in t for qt in qtoks):
            lo, hi = max(0, i - window), min(len(tokens), i + window + 1)
            return ("... " if lo > 0 else "") + " ".join(tokens[lo:hi]) + (" ..." if hi < len(tokens) else "")
    return " ".join(tokens[: 2 * window]) + (" ..." if len(tokens) > 2 * window else "")


def abstain_gate(n_exact: int, top_score: Optional[float], n_results: int, cfg: AgentConfig) -> Dict[str, Any]:
    if not cfg.abstain_enabled:
        return {"ok": True, "branch": "disabled"}
    if n_results == 0:
        return {"ok": False, "branch": "no_results"}
    if cfg.require_exact and n_exact == 0:
        return {"ok": False, "branch": "no_literal_match"}
    if top_score is not None and top_score < cfg.min_match_score and n_exact == 0:
        return {"ok": False, "branch": "no_good_match", "top_score": top_score}
    return {"ok": True, "branch": "ok", "top_score": top_score}


__all__ = ["classify_query", "query_gate", "coverage_gate", "verify_exact", "make_snippet", "abstain_gate"]
