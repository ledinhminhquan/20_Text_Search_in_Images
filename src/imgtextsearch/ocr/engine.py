"""OCR front-end: image -> the text it contains.

* ``tesseract_text`` (default real engine) - Tesseract via pytesseract ``image_to_string``.
* ``SeedEngine`` - the OFFLINE deterministic engine: reads the gold text that the synthetic
  generator embeds in the PNG (``imgtext_spec``), optionally injecting a controllable char-noise
  so the OCR CER is realistic. Lets the index/search/eval/tests run with NO tesseract.
* ``StubEngine`` - empty text (a blank image with no spec).

All heavy imports are lazy; ``ocr_text`` never raises and returns "" on failure. Also exposes
``discover_font`` for the synthetic renderer (a usable TTF for drawing the text).
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from ..config import OcrConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "DejaVuSans.ttf",
]


def discover_font() -> str:
    for c in _FONT_CANDIDATES:
        if os.path.exists(c):
            return c
    return "DejaVuSans.ttf"


def read_spec(image) -> Optional[str]:
    """Pull the gold text embedded by the synthetic generator, if present."""
    info = getattr(image, "info", None) or {}
    raw = info.get("imgtext_spec")
    if raw is None and isinstance(getattr(image, "text", None), dict):
        raw = image.text.get("imgtext_spec")
    if not raw:
        return None
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
        return d.get("text")
    except Exception:
        return None


def has_spec(image) -> bool:
    return read_spec(image) is not None


def _noisify(text: str, rate: float, seed: int) -> str:
    if rate <= 0:
        return text
    rng = random.Random(seed)
    confuse = {"O": "0", "0": "O", "l": "1", "1": "l", "I": "l", "S": "5", "5": "S", "B": "8"}
    out = []
    for c in text:
        if rng.random() < rate:
            r = rng.random()
            if r < 0.5 and c in confuse:
                out.append(confuse[c])
            elif r < 0.75:
                continue
            else:
                out.append(c + c)
        else:
            out.append(c)
    return "".join(out)


class SeedEngine:
    name = "seed"

    def __init__(self, cfg: Optional[OcrConfig] = None, noise: float = 0.0, seed: int = 0):
        self.cfg = cfg
        self.noise = noise
        self.seed = seed

    def text(self, image) -> str:
        t = read_spec(image)
        if t is None:
            return ""
        return _noisify(t, self.noise, self.seed)


class StubEngine:
    name = "stub"

    def __init__(self, cfg: Optional[OcrConfig] = None):
        self.cfg = cfg

    def text(self, image) -> str:
        return ""


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cfg: OcrConfig):
        import pytesseract  # lazy; raises if unavailable
        self._pt = pytesseract
        self.cfg = cfg
        self._pt.get_tesseract_version()

    def text(self, image) -> str:
        from PIL import Image
        img = image if hasattr(image, "size") else Image.fromarray(image)
        return self._pt.image_to_string(img, lang=self.cfg.lang, config=f"--psm {self.cfg.psm}").strip()


def load_ocr_engine(cfg: OcrConfig, engine: Optional[str] = None, image=None):
    requested = engine or cfg.engine
    if requested == "stub":
        return StubEngine(cfg)
    if requested == "seed":
        return SeedEngine(cfg)
    if requested == "auto" and image is not None and has_spec(image):
        return SeedEngine(cfg)
    if requested in ("auto", "tesseract"):
        try:
            return TesseractEngine(cfg)
        except Exception as exc:
            logger.info("tesseract unavailable (%s)", exc)
    if image is not None and has_spec(image):
        return SeedEngine(cfg)
    return StubEngine(cfg)


def ocr_text(image, cfg: OcrConfig, engine=None) -> str:
    try:
        eng = engine or load_ocr_engine(cfg, image=image)
        return eng.text(image)
    except Exception as exc:
        logger.info("ocr_text failed (%s)", exc)
        return ""


__all__ = ["discover_font", "read_spec", "has_spec", "SeedEngine", "StubEngine", "TesseractEngine",
           "load_ocr_engine", "ocr_text"]
