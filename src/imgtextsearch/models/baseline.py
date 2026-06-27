"""Baselines for OCR-text search - the floors the hybrid (dense+BM25) index must beat.

* ``BM25Only`` - the exact-term BM25 arm alone (the strong, no-torch floor).
* ``DenseOnly`` - the dense retriever alone (semantic, no exact-term guarantee).
* ``RandomRetriever`` - random ranking (the absolute floor).

All expose ``search(query, top_k) -> [(idx, score)]`` over the indexed texts, so they slot into
the same evaluation as the hybrid index.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from ..index.lexical import BM25Index


class BM25Only:
    name = "bm25_only"
    version = "bm25-1.0"

    def __init__(self, texts: List[str]):
        self.bm25 = BM25Index(texts)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        return self.bm25.search(query, top_k=top_k)


class DenseOnly:
    name = "dense_only"
    version = "dense-1.0"

    def __init__(self, texts: List[str], encoder):
        import numpy as np
        self.encoder = encoder
        emb = encoder.encode(texts)
        self._emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None) if emb.size else emb

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        import numpy as np
        q = self.encoder.encode([query])[0]
        q = q / max(1e-9, float(np.linalg.norm(q)))
        sims = self._emb @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]


class RandomRetriever:
    name = "random"
    version = "random-1.0"

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.rng = random.Random(seed)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        order = list(range(self.n))
        self.rng.shuffle(order)
        return [(i, 0.0) for i in order[:top_k]]


__all__ = ["BM25Only", "DenseOnly", "RandomRetriever"]
