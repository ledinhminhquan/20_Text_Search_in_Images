"""The text-search-within-images agent - a deterministic FSM over the OCR-search pipeline.

    parse (D1 query gate) -> search (D2 BM25 + dense -> RRF) -> coverage (D3 confidence gate)
        -> verify (D4 snippet + literal-term match) -> finalize (D5 abstain gate)

Holds the OCR-text index over a collection (built once: OCR each image -> per-image text -> BM25 +
optional dense). Runs fully offline (BM25 over the seed OCR texts) and upgrades to a fine-tuned
dense retriever + Tesseract when present. The headline value-add: it **verifies the query term
literally appears** in the image's OCR text (fuzzy to OCR errors), returns a highlighted **snippet**,
and **abstains** when no image contains the text. Every step is timed and traced.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from ..config import AppConfig, ensure_dirs
from ..logging_utils import JsonlLogger, get_logger
from . import tools
from .llm_orchestrator import LLMBrain
from .state import JobState, JobStatus, ToolTrace

logger = get_logger(__name__)


class SearchAgent:
    def __init__(self, cfg: Optional[AppConfig] = None, *, index=None, encoder=None, load_model: bool = True):
        self.cfg = cfg or AppConfig()
        if encoder is None and load_model:
            try:
                from ..models.encoder import load_encoder
                encoder = load_encoder(self.cfg.model, prefer="dense")
            except Exception as exc:
                logger.info("encoder load skipped (%s)", exc)
        self.encoder = encoder
        self.index = index if index is not None else self._build_seed_index()
        self.brain = LLMBrain(self.cfg.agent)
        ensure_dirs()
        self._log = JsonlLogger(self.cfg.serving.job_log_path) if self.cfg.serving.log_jobs else None

    def _build_seed_index(self):
        from ..data import samples
        from ..index.text_index import ImageTextIndex
        coll = samples.seed_collection()
        texts = [c["text"] for c in coll]
        metas = [{"id": c["id"], "text": c["text"]} for c in coll]
        return ImageTextIndex(self.cfg.index, encoder=self.encoder).build(texts, metas)

    def _step(self, job: JobState, name: str, fn: Callable[[], JobState], summary: str = "") -> JobState:
        t0 = time.perf_counter()
        try:
            job = fn()
            ok, err = True, None
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            ok, err = False, str(exc)
        job.add_trace(ToolTrace(tool=name, ok=ok, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                summary=summary or name, error=err))
        return job

    def run(self, query: str, *, save: bool = True) -> JobState:
        job = JobState(query=query or "")
        if not (query or "").strip():
            job.status = JobStatus.FAILED
            return job
        t0 = time.perf_counter()
        job = self._step(job, "parse", lambda: tools.tool_parse(job, self.cfg), summary="query gate (D1)")
        if job.status is not JobStatus.FAILED:
            job = self._step(job, "search", lambda: tools.tool_search(job, self.cfg, index=self.index),
                             summary="BM25 + dense -> RRF (D2)")
            job = self._step(job, "coverage", lambda: tools.tool_coverage(job, self.cfg), summary="coverage gate (D3)")
            job = self._step(job, "verify", lambda: tools.tool_verify(job, self.cfg, index=self.index),
                             summary="snippet + literal verify (D4)")
            job = self._step(job, "finalize", lambda: tools.tool_finalize(job, self.cfg), summary="abstain gate (D5)")

        if self.brain.available() and (job.abstained or job.low_confidence):
            note = self.brain.note(job.query, len(job.results), job.n_exact)
            if note:
                job.metrics["brain_note"] = note
                job.metrics["brain_used"] = True

        job.metrics["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        for attr in ("_candidates", "_ranked"):
            if hasattr(job, attr):
                delattr(job, attr)
        if save and self._log is not None:
            try:
                self._log.log("imgtext_search", query=job.query, top_score=job.top_score,
                              n_results=len(job.results), n_exact=job.n_exact, abstained=job.abstained,
                              low_confidence=job.low_confidence, status=job.status.value, metrics=job.metrics)
            except Exception:
                pass
        return job

    def search(self, query: str) -> dict:
        job = self.run(query, save=False)
        return {"query": job.query, "query_kind": job.query_kind, "results": job.results,
                "top_score": job.top_score, "n_exact": job.n_exact, "abstained": job.abstained,
                "low_confidence": job.low_confidence, "model_version": job.model_versions.get("index", "?"),
                "status": job.status.value}


def build_index_from_images(images: List, cfg: AppConfig, encoder=None):
    """OCR each image -> per-image text -> an ImageTextIndex (used to index a real collection)."""
    from ..index.text_index import ImageTextIndex
    from ..ocr.engine import load_ocr_engine, ocr_text
    texts, metas = [], []
    for i, item in enumerate(images):
        img = item.get("image") if isinstance(item, dict) else item
        iid = (item.get("id") if isinstance(item, dict) else None) or f"img_{i:05d}"
        eng = load_ocr_engine(cfg.ocr, image=img)
        t = ocr_text(img, cfg.ocr, engine=eng)
        texts.append(t)
        metas.append({"id": iid, "text": t})
    return ImageTextIndex(cfg.index, encoder=encoder).build(texts, metas)


_AGENT: Optional[SearchAgent] = None


def get_agent(cfg: Optional[AppConfig] = None, **kwargs) -> SearchAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = SearchAgent(cfg, **kwargs)
    return _AGENT


__all__ = ["SearchAgent", "build_index_from_images", "get_agent"]
