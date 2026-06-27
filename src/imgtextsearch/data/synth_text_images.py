"""Synthetic rendered-text-image generator (the PRIMARY offline data for P20).

No public benchmark for "search a collection by the text inside the images" exists, so the
primary data is a reproducible generator: compose a short text snippet from a vocabulary (common
words + a UNIQUE code per image, so some queries have a single gold image and others have several),
render it onto an image (PIL) and embed the gold text in the PNG. The SeedEngine OCR reads the
embedded text back (optionally with char-noise for a realistic CER); the per-image OCR text is the
"document" the index searches. Queries are terms present in some images; gold = the images whose
text contains the query. Runs the index/search/eval/agent with no tesseract and no torch (a BM25
index stands in for the dense retriever). Mirrors P15/P18/P19.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List

from ..logging_utils import get_logger

logger = get_logger(__name__)

# a small "business document" vocabulary (some words shared across images -> non-trivial ranking)
COMMON = ["invoice", "receipt", "order", "total", "amount", "date", "shipping", "payment",
          "customer", "balance", "due", "tax", "discount", "quantity", "price", "subtotal",
          "refund", "account", "billing", "address", "summary", "report", "contract", "agreement"]
YEARS = ["2021", "2022", "2023", "2024"]


def make_snippet(seed: int) -> Dict:
    rng = random.Random(seed)
    code = f"REF{rng.randint(1000, 9999)}"            # a (usually) unique token -> single-gold queries
    words = rng.sample(COMMON, rng.randint(3, 5)) + [rng.choice(YEARS), code]
    rng.shuffle(words)
    text = " ".join(w.upper() for w in words)
    return {"text": text, "code": code, "words": [w.lower() for w in words]}


def render_text_image(text: str, *, width: int = 700, font_size: int = 34, bg: str = "white"):
    from PIL import Image, ImageDraw, ImageFont
    from ..ocr.engine import discover_font
    fp = discover_font()
    try:
        font = ImageFont.truetype(fp, font_size)
    except Exception:
        font = ImageFont.load_default()
    scratch = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    # greedy wrap to the width
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if scratch.textbbox((0, 0), trial, font=font)[2] <= width - 40 or not cur:
            cur = trial
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    lh = scratch.textbbox((0, 0), "Ahg", font=font)[3] + 14
    height = max(80, 30 + lh * len(lines))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    y = 15
    for ln in lines:
        draw.text((20, y), ln, fill="black", font=font)
        y += lh
    img.info["imgtext_spec"] = json.dumps({"text": text}, ensure_ascii=False)
    return img


def save_png_with_text(img, text: str, path: str) -> str:
    from pathlib import Path

    from PIL.PngImagePlugin import PngInfo
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = PngInfo()
    meta.add_text("imgtext_spec", json.dumps({"text": text}, ensure_ascii=False))
    img.save(str(p), pnginfo=meta)
    return str(p)


def generate_collection(out_dir: str, *, n_images: int = 200, width: int = 700,
                        seed: int = 42, render: bool = True) -> Dict:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    written = 0
    with manifest.open("w", encoding="utf-8") as mf:
        for i in range(n_images):
            snip = make_snippet(seed + i)
            fname = f"doc_{i:04d}.png"
            if render:
                try:
                    img = render_text_image(snip["text"], width=width)
                    save_png_with_text(img, snip["text"], str(out / fname))
                except Exception as exc:
                    logger.info("render skipped (%s)", exc)
            mf.write(json.dumps({"image": fname, "text": snip["text"], "code": snip["code"]},
                                ensure_ascii=False) + "\n")
            written += 1
    logger.info("generated %d synthetic text-images -> %s", written, out)
    return {"images": written, "dir": str(out), "manifest": str(manifest)}


__all__ = ["COMMON", "YEARS", "make_snippet", "render_text_image", "save_png_with_text", "generate_collection"]
