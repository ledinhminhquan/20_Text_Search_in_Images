"""Generate the H100/Colab training notebook (writes a valid .ipynb JSON).

Run:  python notebooks/_build_notebook.py
Produces: notebooks/Text_Search_in_Images_Colab_H100.ipynb

The notebook: controls (#@param) -> GPU check -> Drive mount + env paths -> git clone/upload ->
Colab-safe install (requirements_colab.txt, then `pip install -e . --no-deps`) -> GPU auto-profile
-> write train_colab.yaml -> gen-synthetic -> ONE-BUTTON autopilot -> individual steps ->
diagnostics -> test the trained retriever -> locate deliverables. Auto-adapts H100/A100/L4/T4.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "Text_Search_in_Images_Colab_H100.ipynb"
REPO = "https://github.com/ledinhminhquan/20_Text_Search_in_Images.git"  # <-- edit after you push


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
    "# Text Search within Images — Colab / H100 Training\n"
    "\n"
    "**Search a collection of images by the TEXT they CONTAIN (OCR).** Pipeline: OCR each image → "
    "per-image text → hybrid index (BM25 exact + dense semantic → RRF) → query → rank images + "
    "verify the literal term + highlight the matching snippet, and **abstain** when no image "
    "contains the text. The trainable core is a dense text retriever fine-tuned with MNRL.\n"
    "\n"
    "**One button:** run the *Setup* cells, then **ONE-BUTTON AUTOPILOT**. It auto-detects the GPU "
    "(H100 → A100 → L4 → T4), fine-tunes the retriever, evaluates vs baselines, runs analysis, and "
    "writes `report.pdf` + `slides.pptx` + a submission bundle to your Drive.\n"
    "\n"
    "_Author: Le Dinh Minh Quan (23127460) — NLP in Industry, Final Assignment (P20)._"))

CELLS.append(md("## 0. Controls"))
CELLS.append(code(
    "#@title Controls { run: 'auto' }\n"
    "USE_DRIVE = True           #@param {type:'boolean'}\n"
    "CLONE_FROM_GIT = True      #@param {type:'boolean'}\n"
    "GIT_URL = 'https://github.com/ledinhminhquan/20_Text_Search_in_Images.git'  #@param {type:'string'}\n"
    "TRAIN_RETRIEVER = True     #@param {type:'boolean'}\n"
    "USE_REAL_DATA = False      #@param {type:'boolean'}   # index a real HF collection instead of synthetic\n"
    "EPOCHS = 2                 #@param {type:'integer'}\n"
    "PAIR_LIMIT = 8000          #@param {type:'integer'}\n"
    "RUN_AUTOPILOT = True       #@param {type:'boolean'}\n"
    "print('controls set')"))

CELLS.append(md("## 1. GPU check"))
CELLS.append(code(
    "!nvidia-smi -L || echo 'No GPU — runtime > Change runtime type > GPU (H100/A100/L4/T4)'\n"
    "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),\n"
    "                    '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"))

CELLS.append(md("## 2. Drive mount + paths"))
CELLS.append(code(
    "import os\n"
    "if USE_DRIVE:\n"
    "    from google.colab import drive; drive.mount('/content/drive')\n"
    "    BASE = '/content/drive/MyDrive/imgtextsearch'\n"
    "else:\n"
    "    BASE = '/content/imgtextsearch'\n"
    "os.makedirs(BASE, exist_ok=True)\n"
    "os.environ['IMGTEXT_ARTIFACTS_DIR'] = BASE + '/artifacts'\n"
    "os.environ['HF_HOME'] = BASE + '/hf'\n"
    "os.environ['IMGTEXT_USE_HF'] = '1' if USE_REAL_DATA else '0'\n"
    "print('artifacts ->', os.environ['IMGTEXT_ARTIFACTS_DIR'])"))

CELLS.append(md("## 3. Get the code"))
CELLS.append(code(
    "import os\n"
    "if CLONE_FROM_GIT:\n"
    "    if not os.path.isdir('/content/repo'):\n"
    "        !git clone $GIT_URL /content/repo\n"
    "    else:\n"
    "        !cd /content/repo && git pull --ff-only || true\n"
    "    PROJ = '/content/repo'\n"
    "else:\n"
    "    # Upload/extract the repo zip to Drive and set the path below.\n"
    "    PROJ = BASE + '/20_Text_Search_in_Images'\n"
    "os.environ['PROJ'] = PROJ\n"
    "assert os.path.isdir(PROJ + '/src/imgtextsearch'), 'repo not found at ' + PROJ\n"
    "print('repo at', PROJ)"))

CELLS.append(md("## 4. Install (Colab-safe)\n"
                "Install the ML/serving/report deps from `requirements_colab.txt` (torch is "
                "preinstalled on Colab), then the package with `--no-deps` so it does not perturb "
                "Colab's resolved torch/CUDA. Tesseract is apt-installed for real OCR."))
CELLS.append(code(
    "!apt-get -qq install -y tesseract-ocr > /dev/null && echo 'tesseract installed'\n"
    "!pip -q install -r $PROJ/requirements_colab.txt\n"
    "!pip -q install -e $PROJ --no-deps\n"
    "import importlib, imgtextsearch; importlib.reload(imgtextsearch)\n"
    "print('imgtextsearch', imgtextsearch.__version__)"))

CELLS.append(md("## 5. GPU auto-profile → write `train_colab.yaml`\n"
                "Batch size auto-scales by GPU tier (H100 → A100 → L4 → T4)."))
CELLS.append(code(
    "import torch, yaml, os\n"
    "name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'\n"
    "mem = (torch.cuda.get_device_properties(0).total_memory/1e9) if torch.cuda.is_available() else 0\n"
    "if 'H100' in name: bs, tf32, bf16 = 192, True, True\n"
    "elif 'A100' in name: bs, tf32, bf16 = 128, True, True\n"
    "elif 'L4' in name:  bs, tf32, bf16 = 64, True, True\n"
    "elif 'T4' in name:  bs, tf32, bf16 = 32, False, False\n"
    "else:               bs, tf32, bf16 = 16, False, False\n"
    "cfg = {'data': {'use_hf': bool(USE_REAL_DATA), 'collection_size': 400, 'seed': 42},\n"
    "       'model': {'base_model': 'BAAI/bge-small-en-v1.5', 'num_train_epochs': int(EPOCHS),\n"
    "                 'per_device_train_batch_size': bs, 'bf16': bf16, 'tf32': tf32, 'max_seq_length': 256},\n"
    "       'index': {'use_bm25': True, 'use_dense': True, 'use_faiss': True},\n"
    "       'agent': {'llm_fallback_enabled': False}}\n"
    "os.makedirs(PROJ + '/configs', exist_ok=True)\n"
    "open(PROJ + '/configs/train_colab.yaml','w').write(yaml.safe_dump(cfg, sort_keys=False))\n"
    "print(f'GPU={name} mem={mem:.0f}GB -> batch_size={bs} bf16={bf16} tf32={tf32}')\n"
    "print(open(PROJ + '/configs/train_colab.yaml').read())"))

CELLS.append(md("## 6. Sanity-check + render a synthetic collection"))
CELLS.append(code(
    "!cd $PROJ && imgtextsearch --config configs/train_colab.yaml data\n"
    "!cd $PROJ && imgtextsearch --config configs/train_colab.yaml gen-synthetic"))

CELLS.append(md("## 7. ONE-BUTTON AUTOPILOT 🚀\n"
                "data → baseline → **train retriever** → evaluate → tune → error-analysis → "
                "search-quality → benchmark → demo → monitoring → **report.pdf + slides.pptx** → "
                "grade → zipped bundle. Each step is isolated; training is skipped automatically if "
                "no GPU is present."))
CELLS.append(code(
    "import os\n"
    "flag = '' if TRAIN_RETRIEVER else '--no-train'\n"
    "lim = f'--limit {PAIR_LIMIT}' if PAIR_LIMIT else ''\n"
    "if RUN_AUTOPILOT:\n"
    "    !cd $PROJ && imgtextsearch --config configs/train_colab.yaml autopilot $flag $lim\n"
    "else:\n"
    "    print('RUN_AUTOPILOT is off — use the individual steps below.')"))

CELLS.append(md("## 8. Individual steps (optional)"))
CELLS.append(code(
    "# Fine-tune only:\n"
    "# !cd $PROJ && imgtextsearch --config configs/train_colab.yaml train-retriever --limit $PAIR_LIMIT\n"
    "# Evaluate (full, with the trained dense retriever):\n"
    "# !cd $PROJ && imgtextsearch --config configs/train_colab.yaml evaluate\n"
    "# OCR-noise robustness sweep:\n"
    "# !cd $PROJ && imgtextsearch --config configs/train_colab.yaml tune\n"
    "# Report + slides only:\n"
    "# !cd $PROJ && imgtextsearch --config configs/train_colab.yaml generate-report\n"
    "# !cd $PROJ && imgtextsearch --config configs/train_colab.yaml generate-slides"))

CELLS.append(md("## 9. Diagnostics"))
CELLS.append(code(
    "!cd $PROJ && imgtextsearch --config configs/train_colab.yaml grade\n"
    "!cd $PROJ && imgtextsearch --config configs/train_colab.yaml demo-agent"))

CELLS.append(md("## 10. Test the trained retriever"))
CELLS.append(code(
    "from imgtextsearch.config import load_config\n"
    "from imgtextsearch.agent.search_agent import SearchAgent\n"
    "cfg = load_config(PROJ + '/configs/train_colab.yaml')\n"
    "agent = SearchAgent(cfg, load_model=True)  # loads the fine-tuned dense retriever if trained\n"
    "for q in ['invoice', 'REF3386', '2023', 'zzqwx']:\n"
    "    out = agent.search(q)\n"
    "    print(q, '->', [(r['id'], round(r['score'],2), r['exact_match']) for r in out['results'][:3]],\n"
    "          '| abstained=', out['abstained'])"))

CELLS.append(md("## 11. Locate deliverables"))
CELLS.append(code(
    "import glob, os\n"
    "root = os.environ['IMGTEXT_ARTIFACTS_DIR']\n"
    "for pat in ['submission/*/report.pdf','submission/*/slides.pptx','submission/*/submission_bundle.zip',\n"
    "            'runs/*/eval.json','models/*']:\n"
    "    for p in glob.glob(os.path.join(root, pat)):\n"
    "        print(p)\n"
    "print('\\nDownload report.pdf + slides.pptx + submission_bundle.zip from the path above (in your Drive).')"))


def main():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
