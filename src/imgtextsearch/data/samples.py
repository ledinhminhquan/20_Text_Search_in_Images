"""Built-in seed dataset (offline fallback + tests).

A small fixed collection of synthetic text-image "documents" (each = the text rendered on one
image) + queries whose gold is the image(s) whose text contains the query term. Plain dicts (no
PIL needed for the text path), so the index/search/eval/agent run with no torch and no tesseract;
the text IS the (perfect-OCR) document offline. Queries include a UNIQUE code per image (single
gold) and a few COMMON words (multi-gold, so ranking matters). On Colab a real OCR engine reads
the rendered images.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from . import synth_text_images as G


@lru_cache(maxsize=1)
def _build():
    collection: List[Dict] = []
    for i in range(40):
        snip = G.make_snippet(100 + i)
        collection.append({"id": f"doc_{i:04d}", "text": snip["text"], "code": snip["code"],
                           "words": snip["words"]})
    queries: List[Dict] = []
    # unique-code queries (single gold)
    for c in collection:
        queries.append({"query": c["code"], "gold_ids": [c["id"]], "kind": "exact_code"})
    # a few common-word queries (multi-gold)
    for word in ("invoice", "receipt", "2023", "total"):
        gold = [c["id"] for c in collection if word in c["words"]]
        if gold:
            queries.append({"query": word, "gold_ids": gold, "kind": "keyword"})
    return collection, queries


def seed_collection() -> List[Dict]:
    import copy
    coll, _ = _build()
    return copy.deepcopy(coll)


def seed_queries() -> List[Dict]:
    import copy
    _, qs = _build()
    return copy.deepcopy(qs)


__all__ = ["seed_collection", "seed_queries"]
