"""Self-contained BM25 over the per-image OCR text + RRF fusion.

Pure-python (no sklearn / no torch). BM25 is the EXACT-term arm of the hybrid index - critical
for "find images containing the word X", where literal term presence matters more than semantics.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(d) for d in docs]
        self.N = len(self.docs_tokens)
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(t) for t in self.docs_tokens]
        df: Counter = Counter()
        for toks in self.docs_tokens:
            for w in set(toks):
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.tf[i].get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-9, self.avgdl))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        scores = self.score(query)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in order if scores[i] > 0]


def rrf_fuse(*rankings: List[Tuple[int, float]], k: int = 60, top_k: int = 20) -> List[Tuple[int, float]]:
    fused: dict = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return order


__all__ = ["tokenize", "BM25Index", "rrf_fuse"]
