"""Evaluation: text-in-image Recall@k / MRR vs baselines + OCR CER + exact-match precision.

Builds a held-out synthetic collection (different seed), OCRs each image (perfect offline via the
embedded text, optionally with char-noise for a realistic CER), indexes the per-image OCR text
(BM25 + optional dense), and runs each query through the agent - scoring the ranked image ids
against the gold set (images whose text contains the query) with Recall@1/5/10 + MRR. Compares the
hybrid index against BM25-only, dense-only and random, and reports the OCR CER + exact-match
precision. Offline it is BM25-only (no torch); on Colab the fine-tuned dense retriever + Tesseract.
Results -> ``run_dir/eval.json``.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..data import synth_text_images as G
from ..logging_utils import get_logger
from ..ocr.engine import _noisify
from . import metrics as M

logger = get_logger(__name__)


def _eval_collection(cfg: AppConfig, n: int, ocr_noise: float):
    rng = random.Random(cfg.data.seed + 1000)
    gold_texts, ocr_texts, metas, codes, words_per = [], [], [], [], []
    for i in range(n):
        snip = G.make_snippet(cfg.data.seed + 1000 + i)
        gold = snip["text"]
        ocr = _noisify(gold, ocr_noise, cfg.data.seed + 1000 + i)
        gold_texts.append(gold)
        ocr_texts.append(ocr)
        codes.append(snip["code"])
        words_per.append(snip["words"])
        metas.append({"id": f"doc_{i:04d}", "text": ocr})
    ids = [m["id"] for m in metas]
    queries = [{"query": codes[i], "gold_ids": [ids[i]]} for i in range(n)]
    for word in ("invoice", "receipt", "2023", "total"):
        gold = [ids[i] for i in range(n) if word in words_per[i]]
        if gold:
            queries.append({"query": word, "gold_ids": gold})
    return ocr_texts, gold_texts, metas, ids, queries


def evaluate(cfg: AppConfig, *, limit: Optional[int] = None, load_model: bool = True,
             ocr_noise: float = 0.0, save: bool = True) -> Dict[str, Any]:
    from ..agent.search_agent import SearchAgent
    from ..index.text_index import ImageTextIndex
    from ..models.baseline import BM25Only, DenseOnly, RandomRetriever

    n = limit or 120
    ocr_texts, gold_texts, metas, ids, queries = _eval_collection(cfg, n, ocr_noise)
    encoder = None
    if load_model:
        try:
            from ..models.encoder import load_encoder
            encoder = load_encoder(cfg.model, prefer="dense")
        except Exception as exc:
            logger.info("encoder load skipped (%s)", exc)
    index = ImageTextIndex(cfg.index, encoder=encoder).build(ocr_texts, metas)
    agent = SearchAgent(cfg, index=index, encoder=encoder, load_model=False)

    hybrid_ret, golds = [], []
    exact_precisions: List[float] = []
    for q in queries:
        out = agent.search(q["query"])
        ret_ids = [r["id"] for r in out["results"]]
        hybrid_ret.append(ret_ids)
        golds.append(q["gold_ids"])
        ret_texts = [ocr_texts[ids.index(i)] for i in ret_ids if i in ids]
        exact_precisions.append(M.exact_match_precision(ret_texts, q["query"]))

    bm = BM25Only(ocr_texts)
    rand = RandomRetriever(len(ids))
    bm_ret = [[ids[i] for i, _ in bm.search(q["query"], top_k=cfg.index.top_k)] for q in queries]
    rand_ret = [[ids[i] for i, _ in rand.search(q["query"], top_k=cfg.index.top_k)] for q in queries]
    systems = {"hybrid": M.retrieval_metrics(hybrid_ret, golds),
               "bm25_only": M.retrieval_metrics(bm_ret, golds),
               "random": M.retrieval_metrics(rand_ret, golds)}
    if encoder is not None:
        try:
            dense = DenseOnly(ocr_texts, encoder)
            dense_ret = [[ids[i] for i, _ in dense.search(q["query"], top_k=cfg.index.top_k)] for q in queries]
            systems["dense_only"] = M.retrieval_metrics(dense_ret, golds)
        except Exception as exc:
            logger.info("dense-only eval skipped (%s)", exc)

    report = {
        "n_queries": len(queries), "n_collection": len(ids),
        "systems": systems,
        "ocr": {"cer": M.cer(ocr_texts, gold_texts), "noise": ocr_noise},
        "exact_match_precision": round(sum(exact_precisions) / max(1, len(exact_precisions)), 4),
        "encoder": getattr(encoder, "name", "bm25-only") if encoder else "bm25-only",
        "model_version": getattr(encoder, "version", "bm25") if encoder else "bm25",
    }
    report["headline"] = {"hybrid_recall@1": systems["hybrid"]["recall"]["@1"],
                          "hybrid_mrr": systems["hybrid"]["mrr"],
                          "bm25_recall@1": systems["bm25_only"]["recall"]["@1"],
                          "random_recall@1": systems["random"]["recall"]["@1"],
                          "ocr_cer": report["ocr"]["cer"],
                          "exact_match_precision": report["exact_match_precision"]}
    if save:
        out = run_dir() / "eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("eval -> %s (hybrid R@1=%s, random R@1=%s, OCR CER=%s)", out,
                    report["headline"]["hybrid_recall@1"], report["headline"]["random_recall@1"],
                    report["headline"]["ocr_cer"])
    return report


__all__ = ["evaluate"]
