"""imgtextsearch - Text Search within Images.

An end-to-end, production-grade system that searches a collection of images by the TEXT they
CONTAIN (via OCR): OCR each image, index the per-image text, and rank images for a query, with
the matching snippet highlighted. A trainable dense text retriever is orchestrated by a
deterministic agent (D1-D5) that fuses BM25 + dense retrieval, verifies the literal term, and
abstains when no image contains the query text. Built for the "NLP in Industry" final assignment
(project #20).
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import AppConfig, load_config, save_config, ensure_dirs  # noqa: E402

__all__ = ["AppConfig", "load_config", "save_config", "ensure_dirs", "__version__"]
