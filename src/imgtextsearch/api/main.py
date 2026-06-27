"""FastAPI service for the text-search-within-images system.

Endpoints
---------
* ``GET  /healthz`` / ``GET /readyz`` / ``GET /version``
* ``POST /search`` - {query, k} -> ranked image ids + scores + the matching OCR snippet + exact-match flag

The collection is OCR-indexed once at startup. No results ("not found") are returned when no image
contains the query text; low-confidence results are flagged.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..logging_utils import get_logger
from .dependencies import get_agent, get_config
from .schemas import HealthResponse, SearchRequest, SearchResponse

logger = get_logger(__name__)
cfg = get_config()
app = FastAPI(title=cfg.serving.api_title, version=cfg.serving.api_version)


def _resp(out: dict) -> SearchResponse:
    return SearchResponse(query=out.get("query", ""), query_kind=out["query_kind"], results=out["results"],
                          top_score=out["top_score"], n_exact=out["n_exact"], abstained=out["abstained"],
                          low_confidence=out["low_confidence"], model_version=out["model_version"],
                          status=out["status"])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    agent = get_agent()
    return HealthResponse(status="ok", retriever=getattr(getattr(agent, "encoder", None), "name", "bm25"),
                          n_images=len(agent.index), version=__version__)


@app.get("/readyz")
def readyz() -> dict:
    get_agent()
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    agent = get_agent()
    return {"app": __version__, "retriever": getattr(getattr(agent, "encoder", None), "version", "bm25"),
            "model_version": cfg.serving.model_version}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="provide a query")
    agent = get_agent()
    if req.k:
        agent.cfg.agent.n_results = int(req.k)
    out = agent.search(req.query)
    out["query"] = req.query
    return _resp(out)


__all__ = ["app"]
