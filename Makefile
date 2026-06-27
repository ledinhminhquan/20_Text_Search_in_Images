.PHONY: help install install-all test lint data synthetic train evaluate tune search demo serve report slides autopilot grade docker clean

PY ?= python

help:
	@echo "Text Search within Images (imgtextsearch) — common targets:"
	@echo "  install      core deps + package (no torch/tesseract)"
	@echo "  install-all  package with [all] extras (ml+ocr+api+report)"
	@echo "  test         run the offline pytest suite"
	@echo "  data         prefetch/sanity-check (retriever + collection + seed)"
	@echo "  synthetic    render a synthetic rendered-text image collection"
	@echo "  train        fine-tune the dense OCR-text retriever (needs a GPU)"
	@echo "  evaluate     Recall@k/MRR vs baselines + OCR CER + exact-match (--fast = BM25-only)"
	@echo "  search Q=..  search the collection for images containing a query"
	@echo "  demo         run the agent on the seed queries"
	@echo "  serve        run the FastAPI server + Gradio UI"
	@echo "  autopilot    one-button: train->eval->analysis->report+slides+grade+bundle"
	@echo "  grade        rubric completeness self-check"

install:
	$(PY) -m pip install -e .

install-all:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests

data:
	imgtextsearch data

synthetic:
	imgtextsearch gen-synthetic

train:
	imgtextsearch train-retriever

evaluate:
	imgtextsearch evaluate --fast

search:
	imgtextsearch search --query "$(Q)" --fast

demo:
	imgtextsearch demo-agent --fast

serve:
	imgtextsearch serve --ui

report:
	imgtextsearch generate-report

slides:
	imgtextsearch generate-slides

autopilot:
	imgtextsearch autopilot

grade:
	imgtextsearch grade

docker:
	docker build -t imgtextsearch:latest .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
