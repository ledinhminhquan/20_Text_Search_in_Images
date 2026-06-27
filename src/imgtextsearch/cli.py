"""Command-line interface - the single entrypoint for the imgtextsearch system.

    imgtextsearch <command> [options]

Commands: data, gen-synthetic, train-retriever, train-baseline, tune, evaluate, search,
demo-agent, serve, benchmark, error-analysis, search-quality, monitor-log, generate-report,
generate-slides, autopilot, grade.

All console output is ASCII-only (Windows cp1252 safe); stdout stays pipeable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Text Search within Images System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all
    print(json.dumps(download_all(_load(args), render_synthetic=args.render), indent=2, ensure_ascii=False))


def cmd_gen_synthetic(args):
    from .data.dataset import build_synthetic
    print(json.dumps(build_synthetic(_load(args), split=args.split), indent=2, ensure_ascii=False))


def cmd_train_retriever(args):
    from .training.train_retriever import train_retriever
    print(json.dumps(train_retriever(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_train_baseline(args):
    from .training.train_baseline import build_baseline
    print(json.dumps(build_baseline(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_tune(args):
    from .training.tune import tune
    print(json.dumps(tune(_load(args), load_model=not args.fast), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    rep = evaluate(_load(args), limit=args.limit, load_model=not args.fast, ocr_noise=args.ocr_noise)
    print(json.dumps(rep.get("headline", rep), indent=2, ensure_ascii=False))


def cmd_search(args):
    from .agent.search_agent import SearchAgent
    agent = SearchAgent(_load(args), load_model=not args.fast)
    print(json.dumps(agent.search(args.query), indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.search_agent import SearchAgent
    from .data import samples
    agent = SearchAgent(_load(args), load_model=not args.fast)
    for q in samples.seed_queries()[:12]:
        out = agent.search(q["query"])
        ids = [r["id"] for r in out["results"][:3]]
        ok = "OK" if any(i in q["gold_ids"] for i in ids[:1]) else f"(gold={q['gold_ids'][:2]})"
        print(f"[{out['query_kind']:10s}] '{q['query']:10s}' -> {ids} {ok} exact={out['n_exact']} abst={out['abstained']}")


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["IMGTEXT_INFER_CONFIG"] = str(args.config)
    target = "imgtextsearch.api.app_combined:app" if args.ui else "imgtextsearch.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args)), indent=2, ensure_ascii=False))


def cmd_search_quality(args):
    from .analysis.search_quality import search_quality_report
    print(json.dumps(search_quality_report(_load(args)), indent=2, ensure_ascii=False))


def cmd_monitor_log(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="imgtextsearch", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="prefetch/sanity-check (retriever + collection + seed)")
    sp.add_argument("--render", action="store_true", help="also render the synthetic collection")
    sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("gen-synthetic", help="render a synthetic rendered-text image collection")
    sp.add_argument("--split", default="eval"); sp.set_defaults(func=cmd_gen_synthetic)
    sp = sub.add_parser("train-retriever", help="fine-tune the dense OCR-text retriever (MNRL)")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None)
    sp.set_defaults(func=cmd_train_retriever)
    sp = sub.add_parser("train-baseline", help="persist the BM25 baseline (no GPU)")
    sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_train_baseline)
    sp = sub.add_parser("tune", help="OCR-noise robustness sweep")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="Recall@k/MRR vs baselines + OCR CER + exact-match")
    sp.add_argument("--limit", type=int, default=None); sp.add_argument("--ocr-noise", type=float, default=0.0)
    sp.add_argument("--fast", action="store_true", help="BM25-only (no model download)")
    sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("search", help="search the collection for images containing a query")
    sp.add_argument("--query", required=True); sp.add_argument("--fast", action="store_true")
    sp.set_defaults(func=cmd_search)
    sp = sub.add_parser("demo-agent", help="run the agent on the seed queries")
    sp.add_argument("--fast", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server (+ --ui for the Gradio demo)")
    sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--ui", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark of the agent")
    sp.add_argument("--n", type=int, default=10); sp.add_argument("--warmup", type=int, default=2)
    sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="per-query error analysis + abstention buckets")
    sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("search-quality", help="Recall@k + exact-match + abstention report")
    sp.set_defaults(func=cmd_search_quality)
    sp = sub.add_parser("monitor-log", help="production monitoring report from the job log")
    sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor_log)
    sp = sub.add_parser("generate-report", help="generate the PDF report")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides")
    sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR)
    sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check")
    sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
