"""Optional LLM brain (anthropic), with rule fallback.

Advisory only: may expand a short query into synonyms or note low-confidence results. Disabled
by default; validates its own output and on any problem the caller keeps the rule result.
Default deployment makes zero paid API calls and is fully deterministic. **Never changes the
retrieved images or their ranking.**
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import AgentConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


class LLMBrain:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._client = None
        self._tried = False

    def available(self) -> bool:
        return bool(self.cfg.llm_fallback_enabled and os.environ.get(self.cfg.llm_api_key_env))

    def _get_client(self):
        if self._tried:
            return self._client
        self._tried = True
        try:
            import anthropic
            key = os.environ.get(self.cfg.llm_api_key_env)
            self._client = anthropic.Anthropic(api_key=key) if key else None
        except Exception as exc:
            logger.info("anthropic client unavailable (%s)", exc)
            self._client = None
        return self._client

    def note(self, query: str, n_results: int, n_exact: int) -> Optional[str]:
        if not self.available() or not query:
            return None
        client = self._get_client()
        if client is None:
            return None
        prompt = (f"An OCR image-text search for '{query}' returned {n_results} images ({n_exact} with a "
                  f"literal text match). In ONE short sentence, note whether the results look relevant or "
                  f"the query should be refined. Do NOT change the results.")
        try:
            msg = client.messages.create(model=self.cfg.llm_model, max_tokens=80, temperature=0.0,
                                         messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()
            return text or None
        except Exception as exc:
            logger.info("LLM note failed (%s)", exc)
            return None


__all__ = ["LLMBrain"]
