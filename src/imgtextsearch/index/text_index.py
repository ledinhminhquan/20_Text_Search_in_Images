"""The image-text index: BM25 + dense over the per-image OCR text, fused with RRF.

Each "document" is the OCR'd text of one image. BM25 (always) handles exact term matching; when a
dense encoder is available its embeddings (FAISS / numpy) add semantic / typo-robust matching;
the two rankings are fused with RRF. ``search`` returns the top-k image indices + a raw top score
(for the agent's coverage gate). Offline (no encoder) it is BM25-only - still strong for literal
text search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import IndexConfig
from ..logging_utils import get_logger
from .lexical import BM25Index, rrf_fuse

logger = get_logger(__name__)


class ImageTextIndex:
    def __init__(self, cfg: Optional[IndexConfig] = None, encoder=None):
        self.cfg = cfg or IndexConfig()
        self.encoder = encoder
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Index] = None
        self._emb = None

    def build(self, texts: List[str], metas: List[Dict[str, Any]]) -> "ImageTextIndex":
        self.texts = list(texts)
        self.metas = list(metas)
        self.bm25 = BM25Index(self.texts)
        self._emb = None
        if self.encoder is not None and self.cfg.use_dense:
            try:
                emb = self.encoder.encode(self.texts)
                import numpy as np
                if self.cfg.normalize and emb.size:
                    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
                self._emb = emb
            except Exception as exc:
                logger.info("dense embedding failed (%s); BM25-only", exc)
                self._emb = None
        return self

    def __len__(self) -> int:
        return len(self.texts)

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self._emb is None or self.encoder is None:
            return []
        import numpy as np
        q = self.encoder.encode([query])[0]
        q = q / max(1e-9, float(np.linalg.norm(q)))
        sims = self._emb @ q
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]

    def search(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Tuple[int, float]], float]:
        top_k = top_k or self.cfg.top_k
        bm = self.bm25.search(query, top_k=top_k) if (self.cfg.use_bm25 and self.bm25) else []
        dense = self._dense_search(query, top_k) if self._emb is not None else []
        raw_top = 0.0
        if dense:
            raw_top = dense[0][1]
        elif bm:
            top1, top2 = bm[0][1], (bm[1][1] if len(bm) > 1 else 0.0)
            raw_top = round(top1 / (top1 + top2 + 1e-9), 4)
        if dense and bm:
            fused = rrf_fuse(dense, bm, k=self.cfg.rrf_k, top_k=top_k)
        else:
            fused = dense or bm
        return fused, round(float(raw_top), 4)

    def doc(self, idx: int) -> str:
        return self.texts[idx] if 0 <= idx < len(self.texts) else ""

    def meta(self, idx: int) -> Dict[str, Any]:
        return self.metas[idx] if 0 <= idx < len(self.metas) else {}


def build_index(texts: List[str], metas: List[Dict[str, Any]], cfg: Optional[IndexConfig] = None,
                encoder=None) -> ImageTextIndex:
    return ImageTextIndex(cfg, encoder=encoder).build(texts, metas)


__all__ = ["ImageTextIndex", "build_index"]
