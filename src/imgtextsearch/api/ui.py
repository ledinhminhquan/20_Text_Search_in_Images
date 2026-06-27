"""Gradio demo UI for text search within images.

Type a text query -> the agent searches the OCR-indexed collection, verifies the literal term,
and returns the matching images with a highlighted snippet (rendered from the synthetic text-images
offline, or the real images on Colab). Heavy deps imported lazily.
"""

from __future__ import annotations

from ..config import AppConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def build_demo(cfg: AppConfig = None, load_model: bool = True):
    import gradio as gr
    from ..agent.search_agent import SearchAgent
    from ..data import samples

    cfg = cfg or AppConfig()
    agent = SearchAgent(cfg, load_model=load_model)
    coll = {c["id"]: c for c in samples.seed_collection()}

    def run(query):
        if not (query or "").strip():
            return [], "Please type a query."
        out = agent.search(query)
        gallery = []
        for r in out["results"]:
            tag = "[exact] " if r.get("exact_match") else "[semantic] "
            cap = f"{tag}{r['id']} ({r['score']:.2f}): {r['snippet']}"
            text = coll.get(r["id"], {}).get("text")
            img = None
            if text is not None:
                try:
                    from ..data.synth_text_images import render_text_image
                    img = render_text_image(text)
                except Exception:
                    img = None
            if img is not None:
                gallery.append((img, cap))
        info = (f"kind={out['query_kind']} | top_score={out['top_score']} | exact_matches={out['n_exact']} | "
                f"abstained={out['abstained']} | low_confidence={out['low_confidence']}")
        if out["abstained"]:
            info = f"No image contains '{query}' (not found). " + info
        return gallery, info

    with gr.Blocks(title=cfg.serving.api_title) as demo:
        gr.Markdown(f"# {cfg.serving.api_title}\n"
                    "Search the image collection by the **text inside the images** (OCR). The agent verifies "
                    "the query term actually appears, shows the matching snippet, and returns **nothing** when "
                    "no image contains the text.")
        q = gr.Textbox(label="Query (a word or phrase to find inside the images)", value="invoice")
        btn = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Matching images", columns=3, height=380)
        info = gr.Textbox(label="Trace", lines=2)
        btn.click(run, inputs=[q], outputs=[gallery, info])
        gr.Markdown("_OCR-based search; OCR errors can cause misses - low-confidence results are flagged._")
    return demo


def launch(cfg: AppConfig = None, share: bool = False, **kwargs):
    build_demo(cfg).launch(share=share, **kwargs)


__all__ = ["build_demo", "launch"]
