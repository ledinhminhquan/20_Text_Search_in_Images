"""Hugging Face Space / standalone entrypoint for the Text Search within Images demo.

Launches the Gradio UI: type a text query -> the agent searches the OCR-indexed image collection,
verifies the literal term, and returns the matching images with highlighted snippets. Runs offline
(SeedEngine OCR + BM25) when torch/tesseract are absent.
"""

from __future__ import annotations

import os

from imgtextsearch.api.ui import build_demo
from imgtextsearch.config import AppConfig

if __name__ == "__main__":
    cfg = AppConfig()
    load_model = os.environ.get("IMGTEXT_LOAD_MODEL", "1") not in ("0", "false", "False")
    demo = build_demo(cfg, load_model=load_model)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
