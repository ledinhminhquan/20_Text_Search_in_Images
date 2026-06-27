"""Dense text encoder for the OCR-text retriever (the trainable core).

Wraps a sentence-transformers bi-encoder (``BAAI/bge-small-en-v1.5`` default) that embeds both
the query and the per-image OCR text. Lazy imports so the package + the BM25-only offline path
run with no torch. ``load_encoder`` picks the fine-tuned model > the pretrained base > ``None``
(BM25-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..config import ModelConfig
from ..logging_utils import get_logger
from ..models.model_registry import resolve_latest

logger = get_logger(__name__)


class DenseEncoder:
    name = "dense"

    def __init__(self, model, version: str = "enc-1.0"):
        self.model = model
        self.version = version

    @classmethod
    def from_pretrained(cls, model_path: str) -> "DenseEncoder":
        from sentence_transformers import SentenceTransformer  # lazy
        return cls(SentenceTransformer(model_path), version=_read_version(model_path))

    def encode(self, texts: List[str], batch_size: int = 64):
        import numpy as np
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        return self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False).astype("float32")


def _read_version(model_path: str) -> str:
    meta = Path(model_path) / "model_meta.json"
    if meta.exists():
        try:
            import json
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "enc-1.0")
        except Exception:
            pass
    return "enc-base"


def load_encoder(cfg: ModelConfig, *, prefer: str = "dense") -> Optional[DenseEncoder]:
    if prefer != "dense":
        return None
    latest = resolve_latest(cfg.output_dir)
    if latest is not None:
        try:
            return DenseEncoder.from_pretrained(str(latest))
        except Exception as exc:
            logger.info("fine-tuned encoder unavailable (%s); trying base.", exc)
    for mid in (cfg.base_model, cfg.retriever_fallback):
        try:
            return DenseEncoder.from_pretrained(mid)
        except Exception as exc:
            logger.info("encoder %s unavailable (%s)", mid, exc)
    logger.info("no dense encoder available; BM25-only retrieval.")
    return None


__all__ = ["DenseEncoder", "load_encoder"]
