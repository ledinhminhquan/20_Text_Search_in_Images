"""Pydantic request/response schemas for the text-search-within-images API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="The text to search for inside the images")
    k: int = Field(10, description="Number of results to return")


class SearchResponse(BaseModel):
    query: str = ""
    query_kind: str = ""
    results: List[Dict[str, Any]] = []        # [{id, score, snippet, exact_match, rank}]
    top_score: Optional[float] = None
    n_exact: int = 0
    abstained: bool = False
    low_confidence: bool = False
    model_version: str = ""
    status: str = ""
    disclaimer: str = ("OCR-based text search: results are ranked by the text inside each image, the "
                       "matching snippet is shown, and no results ('not found') are returned when no image "
                       "contains the query text. OCR errors can cause misses - low-confidence results are flagged.")


class HealthResponse(BaseModel):
    status: str
    retriever: str
    n_images: int
    version: str


__all__ = ["SearchRequest", "SearchResponse", "HealthResponse"]
