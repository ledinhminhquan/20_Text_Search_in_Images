"""Offline tests for the imgtextsearch pipeline: OCR, index, metrics, agent, eval, report, grading."""

from __future__ import annotations

import json

import pytest


# ---- synthetic generator + SeedEngine OCR -----------------------------------------------------

def test_make_snippet_deterministic():
    from imgtextsearch.data import synth_text_images as G
    a = G.make_snippet(7)
    b = G.make_snippet(7)
    assert a["text"] == b["text"]
    assert a["code"].startswith("REF") and a["code"] in a["text"]


def test_seed_engine_reads_embedded_text(cfg):
    from imgtextsearch.data import synth_text_images as G
    from imgtextsearch.ocr.engine import load_ocr_engine, ocr_text
    snip = G.make_snippet(3)
    image = G.render_text_image(snip["text"])
    eng = load_ocr_engine(cfg.ocr, image=image)
    txt = ocr_text(image, cfg.ocr, engine=eng)
    # the SeedEngine recovers the embedded gold text (case-insensitive token overlap)
    assert snip["code"].lower() in txt.lower()


# ---- BM25 / index ------------------------------------------------------------------------------

def test_bm25_exact_term():
    from imgtextsearch.index.lexical import BM25Index
    docs = ["invoice total 2023 ref0001", "receipt order 2022 ref0002", "contract price ref0003"]
    bm = BM25Index(docs)
    hits = bm.search("ref0002", top_k=3)
    assert hits and hits[0][0] == 1


def test_image_text_index_offline(cfg):
    from imgtextsearch.data import samples
    from imgtextsearch.index.text_index import ImageTextIndex
    coll = samples.seed_collection()
    idx = ImageTextIndex(cfg.index, encoder=None).build([c["text"] for c in coll],
                                                        [{"id": c["id"]} for c in coll])
    assert len(idx) == len(coll)
    ranked, _ = idx.search(coll[0]["text"].split()[1], top_k=5)
    assert ranked


# ---- metrics -----------------------------------------------------------------------------------

def test_retrieval_metrics():
    from imgtextsearch.training import metrics as M
    retrieved = [["a", "b"], ["x", "y"]]
    golds = [["a"], ["y"]]
    rm = M.retrieval_metrics(retrieved, golds)
    assert rm["recall"]["@1"] == 0.5
    assert 0.0 < rm["mrr"] <= 1.0


def test_cer_and_exact_precision():
    from imgtextsearch.training import metrics as M
    assert M.cer(["abcd"], ["abcd"]) == 0.0
    assert M.cer(["abxd"], ["abcd"]) == pytest.approx(0.25, abs=1e-6)
    assert M.exact_match_precision(["has invoice here", "no match"], "invoice") == 0.5


# ---- agent (D1–D5) -----------------------------------------------------------------------------

def test_agent_exact_code_rank1(cfg):
    from imgtextsearch.agent.search_agent import SearchAgent
    from imgtextsearch.data import samples
    agent = SearchAgent(cfg, load_model=False)
    q = next(x for x in samples.seed_queries() if x["kind"] == "exact_code")
    out = agent.search(q["query"])
    assert out["results"] and out["results"][0]["id"] in q["gold_ids"]
    assert out["results"][0]["exact_match"] is True


def test_agent_all_five_decisions(cfg):
    from imgtextsearch.agent.search_agent import SearchAgent
    from imgtextsearch.data import samples
    agent = SearchAgent(cfg, load_model=False)
    job = agent.run(samples.seed_queries()[0]["query"], save=False)
    ids = [d.id for d in job.decisions]
    assert ids == ["D1", "D2", "D3", "D4", "D5"]


def test_agent_abstains_on_absent_term(cfg):
    from imgtextsearch.agent.search_agent import SearchAgent
    agent = SearchAgent(cfg, load_model=False)
    out = agent.search("zzqwx")
    assert out["abstained"] is True


def test_agent_empty_query(cfg):
    from imgtextsearch.agent.search_agent import SearchAgent
    from imgtextsearch.agent.state import JobStatus
    agent = SearchAgent(cfg, load_model=False)
    job = agent.run("   ", save=False)
    assert job.status is JobStatus.FAILED


def test_agent_snippet_contains_term(cfg):
    from imgtextsearch.agent.search_agent import SearchAgent
    from imgtextsearch.data import samples
    agent = SearchAgent(cfg, load_model=False)
    q = next(x for x in samples.seed_queries() if x["kind"] == "exact_code")
    out = agent.search(q["query"])
    assert q["query"].lower() in out["results"][0]["snippet"].lower()


# ---- evaluate / report / grading ---------------------------------------------------------------

def test_evaluate_offline(cfg):
    from imgtextsearch.training.evaluate import evaluate
    rep = evaluate(cfg, save=False, load_model=False, limit=40)
    assert "hybrid" in rep["systems"]
    assert rep["systems"]["hybrid"]["recall"]["@1"] >= rep["systems"]["random"]["recall"]["@1"]


def test_evaluate_ocr_noise_breaks_exact(cfg):
    from imgtextsearch.training.evaluate import evaluate
    clean = evaluate(cfg, save=False, load_model=False, limit=40, ocr_noise=0.0)
    noisy = evaluate(cfg, save=False, load_model=False, limit=40, ocr_noise=0.15)
    assert clean["ocr"]["cer"] == 0.0 and noisy["ocr"]["cer"] > 0.0


def test_search_quality_report(cfg):
    from imgtextsearch.analysis.search_quality import search_quality_report
    r = search_quality_report(cfg, save=False)
    assert "recall" in r and "exact_match_precision" in r


def test_report_and_slides(cfg):
    from imgtextsearch.autoreport.report_pdf import generate_report
    from imgtextsearch.autoreport.slides_pptx import generate_slides
    rp = generate_report(cfg)
    sp = generate_slides(cfg)
    assert rp.endswith((".pdf", ".md"))
    assert sp.endswith((".pptx", ".md"))


def test_grading_runs():
    from pathlib import Path
    from imgtextsearch.grading.checklist import build_checklist
    repo = Path(__file__).resolve().parents[1]
    res = build_checklist(repo)
    assert res["summary"]["FAIL"] == 0, [i for i in res["items"] if i["status"] == "FAIL"]
