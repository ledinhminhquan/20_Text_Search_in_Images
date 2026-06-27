"""Shared state types for the text-search-within-images agent (deterministic FSM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PARSED = "parsed"
    SEARCHED = "searched"
    VERIFIED = "verified"
    COMPLETED = "completed"
    ABSTAINED = "abstained"            # no image contains the query text
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


@dataclass
class ToolTrace:
    tool: str
    ok: bool
    latency_ms: float
    summary: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, "latency_ms": self.latency_ms,
                "summary": self.summary, "error": self.error}


@dataclass
class Decision:
    id: str
    name: str
    branch: str
    score: Optional[float] = None
    detail: str = ""
    llm_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "branch": self.branch,
                "score": self.score, "detail": self.detail, "llm_used": self.llm_used}


@dataclass
class JobState:
    # ---- input ---------------------------------------------------------------
    query: str = ""
    # ---- derived -------------------------------------------------------------
    status: JobStatus = JobStatus.PENDING
    query_kind: str = "keyword"        # exact_code | keyword | phrase
    n_candidates: int = 0
    top_score: Optional[float] = None
    low_confidence: bool = False
    n_exact: int = 0
    # ---- output --------------------------------------------------------------
    results: List[Dict[str, Any]] = field(default_factory=list)   # [{id, score, snippet, exact_match, rank}]
    abstained: bool = False
    needs_review: bool = False
    rationale: str = ""
    # ---- audit ---------------------------------------------------------------
    decisions: List[Decision] = field(default_factory=list)
    trace: List[ToolTrace] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def add_trace(self, t: ToolTrace) -> None:
        self.trace.append(t)

    def add_decision(self, d: Decision) -> None:
        self.decisions.append(d)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "status": self.status.value, "query_kind": self.query_kind,
            "n_candidates": self.n_candidates, "top_score": self.top_score,
            "low_confidence": self.low_confidence, "n_exact": self.n_exact, "results": self.results,
            "abstained": self.abstained, "needs_review": self.needs_review, "rationale": self.rationale,
            "decisions": [d.to_dict() for d in self.decisions], "trace": [t.to_dict() for t in self.trace],
            "metrics": self.metrics, "model_versions": self.model_versions,
        }


__all__ = ["JobStatus", "ToolTrace", "Decision", "JobState"]
