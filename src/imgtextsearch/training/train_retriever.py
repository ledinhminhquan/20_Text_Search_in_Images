"""Fine-tune the dense OCR-text retriever (the trainable core) - sentence-transformers MNRL.

Builds ``(query, OCR-text)`` pairs from the synthetic rendered-text collection (the query is a
term that appears in the image's text - a unique code or a keyword; the positive is the full OCR
text) and fine-tunes a bi-encoder (``BAAI/bge-small-en-v1.5``) with MultipleNegativesRankingLoss
(in-batch negatives), so a query embeds close to the OCR text that contains it. On Colab the real
DocVQA query->document spans add real supervision. bf16/tf32 on Ampere+/H100. (Reuses P18/P19.)
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

from ..config import AppConfig
from ..data import synth_text_images as G
from ..logging_utils import get_logger
from ..models import model_registry as reg

logger = get_logger(__name__)


def _pairs(cfg: AppConfig, cap: int) -> List:
    from sentence_transformers import InputExample
    rng = random.Random(cfg.data.seed)
    out: List = []
    i = 0
    while len(out) < cap and i < cap * 2:
        snip = G.make_snippet(cfg.data.seed + i)
        i += 1
        text = snip["text"]
        # the unique code is a precise positive; a random keyword is a recall positive
        out.append(InputExample(texts=[snip["code"], text]))
        kw = rng.choice(snip["words"])
        out.append(InputExample(texts=[kw, text]))
    return out[:cap]


def train_retriever(cfg: AppConfig, limit: Optional[int] = None, base_model: Optional[str] = None) -> Dict:
    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    mc = cfg.model
    model_id = base_model or mc.base_model
    torch.backends.cuda.matmul.allow_tf32 = bool(mc.tf32)
    cap = limit or 8000

    examples = _pairs(cfg, cap)
    logger.info("Training retriever %s on %d (query, OCR-text) pairs", model_id, len(examples))

    model = SentenceTransformer(model_id)
    model.max_seq_length = mc.max_seq_length
    loader = DataLoader(examples, shuffle=True, batch_size=mc.per_device_train_batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * mc.num_train_epochs * mc.warmup_ratio)
    model.fit(train_objectives=[(loader, loss)], epochs=mc.num_train_epochs, warmup_steps=warmup,
              optimizer_params={"lr": mc.learning_rate}, use_amp=bool(mc.bf16 or mc.fp16),
              show_progress_bar=True)

    version = reg.make_version(model_id)
    final_dir = mc.output_dir / version
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir))
    reg.write_metadata(final_dir, version=version, base_model=model_id,
                       dataset_signature={"pairs": len(examples), "seed": cfg.data.seed},
                       metrics={}, extra={"type": "bi-encoder", "loss": "MNRL"})
    reg.update_latest_pointer(mc.output_dir, final_dir)
    logger.info("retriever training done -> %s", final_dir)
    return {"version": version, "model_dir": str(final_dir), "base_model": model_id, "n_pairs": len(examples)}


__all__ = ["train_retriever"]
