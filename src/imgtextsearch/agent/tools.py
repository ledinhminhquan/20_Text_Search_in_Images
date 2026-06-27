"""Agent tools - each operates on the JobState and returns it.

Tools wrap the OCR-text index + the D1-D5 policy. They run against the offline stack (BM25 over
the seed OCR texts) so the whole pipeline runs offline for tests/CI, and upgrade to a fine-tuned
dense retriever + Tesseract when present. The orchestrator wraps each call with timing/trace;
tools never raise.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger
from . import policy
from .state import Decision, JobState, JobStatus

logger = get_logger(__name__)


def tool_parse(job: JobState, cfg: AppConfig) -> JobState:
    """D1 - query gate + classify."""
    job.query = (job.query or "").strip()
    gate = policy.query_gate(job.query, cfg.agent)
    if not gate["ok"]:
        job.status = JobStatus.FAILED
        job.rationale = "Empty or too-short query."
    else:
        job.query_kind = policy.classify_query(job.query)
    job.add_decision(Decision("D1", "query_gate", gate["branch"] if not gate["ok"] else job.query_kind,
                              detail=f"kind={job.query_kind}, len={len(job.query)}"))
    return job


def tool_search(job: JobState, cfg: AppConfig, *, index) -> JobState:
    """D2 - BM25 + dense -> RRF search over the per-image OCR text."""
    candidates, raw_top = index.search(job.query, top_k=cfg.index.top_k)
    job._candidates = candidates  # type: ignore[attr-defined]
    job.n_candidates = len(candidates)
    job.top_score = raw_top
    job.model_versions["index"] = getattr(getattr(index, "encoder", None), "name", "bm25")
    job.add_decision(Decision("D2", "search", "ok" if candidates else "empty", score=raw_top,
                              detail=f"candidates={len(candidates)}, top_score={raw_top}"))
    job.status = JobStatus.SEARCHED
    return job


def tool_coverage(job: JobState, cfg: AppConfig) -> JobState:
    """D3 - coverage / confidence gate."""
    gate = policy.coverage_gate(job.top_score, cfg.agent)
    if not gate["ok"]:
        job.low_confidence = True
    job.add_decision(Decision("D3", "coverage_gate", gate["branch"], score=job.top_score,
                              detail=f"low_confidence={job.low_confidence}"))
    return job


def tool_verify(job: JobState, cfg: AppConfig, *, index) -> JobState:
    """D4 - snippet + literal-term verification (the OCR-search guarantee)."""
    candidates = getattr(job, "_candidates", [])
    verified, others = [], []
    for idx, score in candidates:
        text = index.doc(idx)
        m = index.meta(idx)
        exact = policy.verify_exact(text, job.query, cfg.agent)
        rec = {"id": m.get("id", str(idx)), "score": round(float(score), 4),
               "snippet": policy.make_snippet(text, job.query, cfg.agent.snippet_window),
               "exact_match": exact}
        (verified if exact else others).append(rec)
    # exact-match images first, then the rest (semantic) - stable within each group
    ranked = verified + others
    job._ranked = ranked  # type: ignore[attr-defined]
    job.n_exact = len(verified)
    job.add_decision(Decision("D4", "verify_snippet", "exact" if verified else "no_exact",
                              score=len(verified),
                              detail=f"exact_matches={len(verified)}, semantic_only={len(others)}"))
    job.status = JobStatus.VERIFIED
    return job


def tool_finalize(job: JobState, cfg: AppConfig) -> JobState:
    """D5 - abstain gate; assemble the ranked results + snippets."""
    ranked = getattr(job, "_ranked", [])
    gate = policy.abstain_gate(job.n_exact, job.top_score, len(ranked), cfg.agent)
    if not gate["ok"]:
        job.abstained = True
        job.needs_review = True
        job.results = []
        job.status = JobStatus.ABSTAINED
        if not job.rationale:
            job.rationale = (f"No image contains '{job.query}' ({gate['branch']}); flagged for review.")
    else:
        results = []
        for rank, rec in enumerate(ranked[: cfg.agent.n_results], 1):
            rec = dict(rec); rec["rank"] = rank
            results.append(rec)
        job.results = results
        job.status = JobStatus.NEEDS_REVIEW if (job.needs_review or job.low_confidence) else JobStatus.COMPLETED
        if not job.rationale:
            job.rationale = (f"Found {job.n_exact} image(s) containing '{job.query}' "
                             f"(+{len(ranked) - job.n_exact} semantic) via "
                             f"{job.model_versions.get('index', 'index')}"
                             + (", low-confidence" if job.low_confidence else "") + ".")
    job.add_decision(Decision("D5", "abstain_gate", gate["branch"], score=job.top_score,
                              detail=f"abstained={job.abstained}, results={len(job.results)}"))
    return job


__all__ = ["tool_parse", "tool_search", "tool_coverage", "tool_verify", "tool_finalize"]
