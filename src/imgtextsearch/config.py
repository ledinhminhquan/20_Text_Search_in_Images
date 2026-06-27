"""Typed configuration + YAML loader for the imgtextsearch Text-Search-within-Images system.

Single source of truth for the OCR front-end, the trainable dense text retriever over the OCR
text, the hybrid index (BM25 + dense + RRF), the agent decision thresholds (D1-D5), the
datasets, and serving. Paths come from environment variables.

Pipeline: OCR each image -> per-image OCR text -> index (BM25 + dense) -> query -> RRF rank ->
snippet + literal-term verify. The only trained component is the dense retriever; OCR + BM25 +
the verifier are pretrained/algorithmic.

Environment overrides
---------------------
* ``IMGTEXT_ARTIFACTS_DIR`` - base for data/models/runs (Drive on Colab)
* ``IMGTEXT_DATA_DIR``      - dataset cache / generated synthetic images + index
* ``IMGTEXT_MODEL_DIR``     - trained models (the fine-tuned retriever)
* ``IMGTEXT_RUN_DIR``       - eval/benchmark/analysis JSON
* ``HF_HOME``               - HuggingFace cache
* ``IMGTEXT_LLM_API_KEY``   - optional key for the LLM agent brain

Verified ids (confirmed on the HF Hub during research - keep exact):
  ocr        Tesseract (system, Apache) default; microsoft/trocr-base-printed (MIT) neural upgrade
  retriever  BAAI/bge-small-en-v1.5 (MIT, 384d, default) · all-MiniLM-L6-v2 (Apache) fallback
  ocr data   PleIAs/Post-OCR-Correction (CC0) - see docs/DESIGN_BRIEF.md
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


def artifacts_dir() -> Path:
    return Path(_env("IMGTEXT_ARTIFACTS_DIR", "artifacts")).expanduser()


def data_dir() -> Path:
    return Path(_env("IMGTEXT_DATA_DIR", str(artifacts_dir() / "data"))).expanduser()


def model_dir() -> Path:
    return Path(_env("IMGTEXT_MODEL_DIR", str(artifacts_dir() / "models"))).expanduser()


def run_dir() -> Path:
    return Path(_env("IMGTEXT_RUN_DIR", str(artifacts_dir() / "runs"))).expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """OCR-text source + a synthetic rendered-text-image generator (offline / tests).

    No public benchmark for "search a collection by the text inside the images" exists, so the
    PRIMARY data is a reproducible SYNTHETIC generator (``data/synth_text_images.py``): render
    short text snippets (built from a vocabulary, some words shared so ranking is non-trivial)
    onto images and embed the gold text; the SeedEngine OCR reads it back, optionally with noise.
    """
    ocr_text_dataset: str = "PleIAs/Post-OCR-Correction"   # CC0, real OCR-noise text source
    ocr_text_config: str = "english"
    # Optional REAL collections (verified on HF; off by default, the synthetic generator is the spine):
    #   MiXaiLL76/TextOCR_OCR (MIT, scene-text image+text) · naver-clova-ix/cord-v2 (CC-BY-4.0, receipts) ·
    #   nielsr/docvqa_1200_examples (license-unspecified FLAG; image + OCR words + query + gold answer span).
    collection_dataset: str = "MiXaiLL76/TextOCR_OCR"
    docvqa_dataset: str = "nielsr/docvqa_1200_examples"   # query + gold-span supervision for the retriever
    use_hf: bool = False                    # default to the synthetic generator
    collection_size: int = 200
    image_width: int = 700
    vocab_phrases: int = 60                 # distinct words/phrases to compose snippets from
    seed: int = 42


@dataclass
class OcrConfig:
    """OCR engine for the image text front-end."""
    engine: str = "auto"                    # "auto"|"tesseract"|"seed"|"stub"
    lang: str = "eng"
    psm: int = 6                            # assume a uniform block of text
    min_word_conf: float = 0.0


@dataclass
class ModelConfig:
    """Fine-tuning the dense text retriever (sentence-transformers MNRL on (query, OCR-text) pairs)."""
    base_model: str = "BAAI/bge-small-en-v1.5"
    retriever_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    num_train_epochs: int = 2
    learning_rate: float = 2.0e-5
    per_device_train_batch_size: int = 64
    warmup_ratio: float = 0.1
    max_seq_length: int = 192
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    seed: int = 42
    output_subdir: str = "retriever"
    baseline_filename: str = "bm25_baseline.json"

    @property
    def output_dir(self) -> Path:
        return model_dir() / self.output_subdir

    @property
    def baseline_path(self) -> Path:
        return self.output_dir / self.baseline_filename


@dataclass
class IndexConfig:
    """Hybrid text index over the per-image OCR text (BM25 + dense, fused with RRF)."""
    top_k: int = 20
    use_bm25: bool = True
    use_dense: bool = True
    use_faiss: bool = True
    normalize: bool = True
    rrf_k: int = 60
    index_subdir: str = "index"

    @property
    def index_dir(self) -> Path:
        return data_dir() / self.index_subdir


@dataclass
class AgentConfig:
    """OCR-search agent decision thresholds (D1-D5) + optional LLM brain."""
    # D2 - query parse
    min_query_chars: int = 2
    # D3 - coverage gate
    coverage_min_score: float = 0.05
    # D4 - snippet + literal-term verification
    require_exact: bool = True             # require the query term to literally appear in the OCR text
    exact_match_fuzzy: bool = True         # allow a small edit distance (OCR errors) for the literal check
    snippet_window: int = 6                # words of context around the match
    # D5 - abstain
    abstain_enabled: bool = True
    min_match_score: float = 0.05
    n_results: int = 10
    # optional cloud brain (off by default; the agent runs fully on rules)
    llm_fallback_enabled: bool = False
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_api_key_env: str = "IMGTEXT_LLM_API_KEY"


@dataclass
class ServingConfig:
    model_version: str = "v1"
    api_title: str = "Text Search within Images API"
    api_version: str = "1.0.0"
    log_jobs: bool = True
    job_log_subdir: str = "job_logs"

    @property
    def job_log_path(self) -> Path:
        return run_dir() / self.job_log_subdir / "jobs.jsonl"


@dataclass
class AppConfig:
    project_title: str = "Text Search within Images System"
    author: str = "Le Dinh Minh Quan"
    student_id: str = "23127460"
    data: DataConfig = field(default_factory=DataConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {"data": DataConfig, "ocr": OcrConfig, "model": ModelConfig, "index": IndexConfig,
             "agent": AgentConfig, "serving": ServingConfig}


def _build(cls, raw: Optional[Dict[str, Any]]):
    raw = raw or {}
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: Optional[str | os.PathLike] = None) -> AppConfig:
    raw: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    top = {k: raw[k] for k in ("project_title", "author", "student_id") if k in raw}
    sections = {name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return AppConfig(**top, **sections)


def save_config(cfg: AppConfig, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_dirs() -> Dict[str, Path]:
    dirs = {"artifacts": artifacts_dir(), "data": data_dir(), "models": model_dir(), "runs": run_dir()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


__all__ = ["DataConfig", "OcrConfig", "ModelConfig", "IndexConfig", "AgentConfig", "ServingConfig", "AppConfig",
           "load_config", "save_config", "ensure_dirs", "artifacts_dir", "data_dir", "model_dir", "run_dir"]
