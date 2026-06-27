"""Generate the submission report.pdf for the imgtextsearch (Text Search within Images) system.

A 10-15 page report covering every Section-I deliverable: problem & use cases, data (scene-text +
synthetic), the OCR + dense/BM25 retriever, the agent (D1-D5), the evaluation (Recall@k/MRR + OCR
CER + exact-match), deployment, continual learning & monitoring, privacy & robustness, and ethics.
Live numbers come from ``run_dir()`` artifacts; missing metrics degrade to placeholders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig, run_dir
from ..logging_utils import get_logger, utc_now_iso
from . import charts as charts_mod
from .artifact_loader import (base_model, encoder_name, exact_match, has_eval, latency, load_artifacts,
                              model_version, ocr_cer, read_doc, sys_mrr, sys_recall)

logger = get_logger(__name__)

_SUBTITLE = ("Search a collection of images by the TEXT they CONTAIN: OCR each image, index the per-image "
             "text with a TRAINABLE dense retriever + BM25 (RRF), then for a query rank the images, verify "
             "the term literally appears, and show the matching snippet. A deterministic agent (D1-D5) gates "
             "coverage, verifies the literal match, and abstains when no image contains the text.")

_SECTIONS = [
    ("1. Problem Definition & Use Cases", "problem_definition.md"),
    ("2. Data (Scene-text + Synthetic)", "data_description.md"),
    ("3. System Architecture", "architecture.md"),
    ("4. Model Selection (OCR + Retriever)", "model_selection.md"),
    ("5. Agent Architecture (Decisions D1-D5)", "agent_architecture.md"),
    ("6. Search Evaluation", "search_evaluation.md"),
    ("7. Deployment", "deployment.md"),
    ("8. Continual Learning & Monitoring", "continual_learning_monitoring.md"),
    ("9. Data Privacy & Robustness", "privacy_robustness.md"),
    ("10. Ethics & Responsible AI", "ethics_statement.md"),
]


def _builtin_sections(cfg: AppConfig, arts: Dict[str, Any]) -> Dict[str, str]:
    r1 = sys_recall(arts, "hybrid", "@1")
    rr = sys_recall(arts, "random", "@1")
    cer = ocr_cer(arts)
    if r1 is not None:
        res_line = (f"In the latest eval the hybrid index reaches Recall@1 **{r1*100:.1f}%**"
                    + (f" vs the random floor **{rr*100:.1f}%**" if rr is not None else "")
                    + (f"; OCR CER **{cer:.3f}**." if cer is not None else "."))
    else:
        res_line = "Run `imgtextsearch evaluate` to populate the live numbers here."
    return {
        "problem_definition.md": f"""
## What it does
Search a collection of images by the **text they contain** (via OCR) - "find the images that
contain INVOICE", "images mentioning a 2023 date". This is OCR-based **document search**, distinct
from visual/semantic search. The trainable core is a dense text retriever over the OCR text; the
OCR front-end, BM25 and the literal verifier are pretrained/algorithmic.

## The job-to-be-done
- **Document / receipt / invoice archives**, scanned-contract search, compliance / e-discovery.
- **Screenshot and photo libraries** with signage / captions; finding a specific word across scans.

## Why an agent over a raw index
The value-add: it **verifies the query term literally appears** in the image's OCR text (fuzzy to
OCR errors), returns a highlighted **snippet**, and **abstains** ("not found") when no image
contains the text - instead of returning a semantically-similar but wrong image.

## Success metrics
- **Technical:** text-in-image **Recall@1/5/10 + MRR**; **OCR CER**; **exact-match precision**.
- **Business:** the share of "not found" queries, human review load on flagged results.
{res_line}
""",
        "model_selection.md": f"""
## The OCR front-end (pretrained, NOT trained)
- **Tesseract** via `pytesseract` (**Apache-2.0**) - `image_to_string` per image; born-digital-vs-
  scanned routing; the `SeedEngine` offline reads the gold text embedded in synthetic images.
  Neural upgrade `microsoft/trocr-base-printed` (MIT); docTR/PaddleOCR/EasyOCR (Apache); Surya
  (**CC-BY-NC-SA, flagged**). Ported from P07 dococr / P15 imgtrans.

## The trainable text retriever
- **Default:** `{base_model(arts)}` (`BAAI/bge-small-en-v1.5`, **MIT**, 384d), fine-tuned with
  **MultipleNegativesRankingLoss** on `(query, OCR-text)` pairs. Fallback `all-MiniLM-L6-v2`.
- **Hybrid index:** BM25 (the **exact-term** arm - critical for literal "contains word X") + the
  dense arm (semantic / typo-robust) fused with **RRF**. Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Baselines:** **BM25-only** (the exact-term floor), **dense-only** (semantic, no exact
  guarantee), **random**.
{res_line}
""",
        "agent_architecture.md": f"""
## FSM
A deterministic finite-state machine; every tool returns a uniform dict and every transition is
traced. States: `parse -> search -> coverage -> verify -> finalize`. An optional LLM **brain**
(`{cfg.agent.llm_model}`, OFF by default) only adds an advisory note; rules win and the agent runs
with **zero paid API calls**.

## Five decisions (each acts on an intermediate artifact)
- **D1 - query gate + classify.** Reject empty queries; classify exact-code / keyword / phrase.
- **D2 - search.** BM25 (exact) + dense (semantic) -> **RRF** -> top-{cfg.index.top_k} image
  candidates + scores.
- **D3 - coverage / confidence gate.** Top score below {cfg.agent.coverage_min_score} -> low-confidence.
- **D4 - snippet + literal-term verify.** Extract the matching snippet from each candidate's OCR
  text and verify the query term **literally appears** (fuzzy: edit-distance <= 1, to survive OCR
  errors); exact-match images rank first.
- **D5 - abstain / finalize.** If no image literally contains the query -> **abstain** ("not found"
  + needs_review); else return the ranked images + snippets + the exact-match flag.

The agent emits `{{results[{{id, score, snippet, exact_match, rank}}], n_exact, abstained, decisions[]}}`.
""",
        "search_evaluation.md": f"""
## Metrics
Text-in-image **Recall@1/5/10** (a query hits if ANY image whose OCR text contains it is in the
top-K) + **MRR** + median rank; **OCR CER/WER** (OCR vs the gold rendered text); and **exact-match
precision** (fraction of returned images whose OCR text literally contains the query term).

## Baselines
- **BM25-only** (exact-term floor), **dense-only** (semantic, no exact guarantee), **random**. The
  hybrid (BM25 + dense -> RRF) combines exact + semantic; the fuzzy verify recovers OCR-noise misses.
{res_line}
""",
    }


def _esc(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b\x00").replace("</b>", "\x00/b\x00")
    s = s.replace("<font face='Courier'>", "\x00f\x00").replace("</font>", "\x00/f\x00")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    s = (s.replace("\x00b\x00", "<b>").replace("\x00/b\x00", "</b>")
          .replace("\x00f\x00", "<font face='Courier'>").replace("\x00/f\x00", "</font>"))
    return s


def _md_to_flowables(md: str, styles, max_lines: int = 300):
    from reportlab.platypus import Paragraph, Preformatted, Spacer
    flow, lines, in_code, code, bullet = [], md.splitlines()[:max_lines], False, [], []

    def flush():
        nonlocal bullet
        for b in bullet:
            flow.append(Paragraph("- " + _esc(b), styles["Body"]))
        bullet = []

    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                flow.append(Preformatted("\n".join(code), styles["Code"])); code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(ln); continue
        s = ln.rstrip()
        if not s:
            flush(); flow.append(Spacer(1, 5)); continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            flow.append(Paragraph(_esc(s.lstrip("#").strip()), styles["H2" if level <= 2 else "H3"]))
        elif s.lstrip().startswith(("- ", "* ")):
            bullet.append(s.lstrip()[2:])
        else:
            flush(); flow.append(Paragraph(_esc(s), styles["Body"]))
    flush()
    return flow


def _pct(v):
    return (f"{v*100:.1f}%" if isinstance(v, (int, float)) and not isinstance(v, bool) else "-")


def _results_table(arts: Dict[str, Any], styles):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    flow = [Paragraph("Results - text-in-image retrieval", styles["H3"])]
    if has_eval(arts):
        rows = [["System", "Recall@1", "Recall@5", "Recall@10", "MRR"]]
        for name in ("hybrid", "bm25_only", "random"):
            rows.append([name, _pct(sys_recall(arts, name, "@1")), _pct(sys_recall(arts, name, "@5")),
                         _pct(sys_recall(arts, name, "@10")),
                         f"{sys_mrr(arts, name):.3f}" if sys_mrr(arts, name) is not None else "-"])
    else:
        rows = [["System", "Recall@1", "Recall@5", "Recall@10", "MRR"], ["run `evaluate`", "-", "-", "-", "-"]]
    t = Table(rows, hAlign="LEFT", colWidths=[120, 80, 80, 80, 70])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]))
    cer = ocr_cer(arts); ex = exact_match(arts)
    flow += [t, Spacer(1, 6),
             Paragraph(f"OCR CER: {cer:.3f} | exact-match precision: {_pct(ex)}. BM25 finds exact terms, the "
                       "dense arm adds semantics, and the fuzzy verify recovers OCR-noise misses."
                       if cer is not None else "Run evaluate for OCR CER + exact-match precision.", styles["Body"])]
    lat = latency(arts, "p50")
    if lat is not None:
        flow.append(Paragraph(f"Agent latency: per-query p50 ~ {lat:.0f} ms "
                              f"(p95 ~ {latency(arts, 'p95') or 0:.0f} ms).", styles["Body"]))
    flow.append(Spacer(1, 8))
    return flow


def generate_report(cfg: AppConfig, title: Optional[str] = None, author: Optional[str] = None,
                    out_path: Optional[str] = None) -> str:
    title = title or cfg.project_title
    author = author or cfg.author
    arts = load_artifacts(cfg)
    out = Path(out_path) if out_path else run_dir() / "report" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    builtins = _builtin_sections(cfg, arts)

    def section_md(fname: str) -> str:
        doc = read_doc(fname)
        if doc.strip():
            lines = doc.splitlines()
            return "\n".join(lines[:46]) if len(lines) > 46 else doc
        return builtins.get(fname, "")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer)
    except Exception as exc:
        logger.warning("reportlab unavailable (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)

    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("T", parent=base["Title"], fontSize=22, leading=26),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], textColor="#1a365d", spaceBefore=10),
        "H3": ParagraphStyle("H3", parent=base["Heading3"], textColor="#2b6cb0"),
        "Body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13),
        "Code": ParagraphStyle("C", parent=base["Code"], fontSize=7.5, leading=9, backColor="#f4f6f8"),
        "Meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=11, leading=15),
    }
    try:
        built = dict(charts_mod.build_all(arts, out.parent / "charts"))
    except Exception as exc:
        logger.info("charts skipped (%s)", exc)
        built = {}

    story: List[Any] = [
        Spacer(1, 5 * cm), Paragraph(title, styles["Title"]), Spacer(1, 1 * cm),
        Paragraph(f"<b>{author}</b> - Student {cfg.student_id}", styles["Meta"]),
        Paragraph("NLP in Industry - Final Assignment (P20)", styles["Meta"]),
        Paragraph(_SUBTITLE, styles["Meta"]),
        Paragraph(f"Generated {utc_now_iso()}", styles["Body"]),
        Paragraph(f"Retriever: <b>{model_version(arts)}</b> (base {base_model(arts)}, encoder {encoder_name(arts)})",
                  styles["Body"]),
    ]
    story.append(PageBreak())
    story += _results_table(arts, styles)
    for name in ("recall", "quality", "buckets"):
        if name in built:
            story += [Image(str(built[name]), width=13 * cm, height=7.0 * cm), Spacer(1, 6)]
    story.append(PageBreak())

    for heading, fname in _SECTIONS:
        story.append(Paragraph(heading, styles["H2"]))
        story += _md_to_flowables(section_md(fname), styles)
        story.append(Spacer(1, 10))

    try:
        SimpleDocTemplate(str(out), pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                          leftMargin=1.8 * cm, rightMargin=1.8 * cm, title=title, author=author).build(story)
    except Exception as exc:
        logger.warning("reportlab build failed (%s); writing markdown report", exc)
        md = f"# {title}\n\n{author} (Student {cfg.student_id})\n\n{_SUBTITLE}\n\n"
        for hd, fn in _SECTIONS:
            md += f"\n\n# {hd}\n" + section_md(fn)
        alt = out.with_suffix(".md")
        alt.write_text(md, encoding="utf-8")
        return str(alt)
    logger.info("Report -> %s", out)
    return str(out)


__all__ = ["generate_report"]
