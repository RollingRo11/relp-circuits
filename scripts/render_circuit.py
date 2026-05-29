"""Render an interactive HTML view of an attribution + (optional) dataset-acts index.

Usage:
  uv run python -m scripts.render_circuit \
      --attr   /data/.../attribution/mh-relp-smoke \
      --acts   /data/.../activations/acts-10m \
      --output /data/.../viz/multihop.html \
      --title  "city -> state -> capital"

The renderer reads only attribution + acts artifacts (no torch / no model load), so
it's CPU-only. It loads the OLMo-3 tokenizer via the rust `tokenizers` package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relp_circuits.viz import render_circuit_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr", type=Path, required=True, help="attribution output dir")
    p.add_argument("--acts", type=Path, default=None,
                   help="optional dataset-acts index dir (enables hover examples)")
    p.add_argument("--output", type=Path, required=True, help="path to output .html")
    p.add_argument("--title", default=None)
    p.add_argument("--tokenizer", default="allenai/Olmo-3-7B-Think")
    p.add_argument("--topk-neurons", type=int, default=128)
    p.add_argument("--topk-examples", type=int, default=8)
    p.add_argument("--context-tokens", type=int, default=12)
    p.add_argument("--prompts-json", type=Path, default=None,
                   help="optional path to prompts.json. If absent, falls back to attr/prompts.json.")
    p.add_argument("--selection", choices=["auto", "tau", "topk"], default="auto",
                   help="which neuron set to render: 'tau' (paper's filter), 'topk' (legacy), or 'auto' (prefer tau).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompts: list[dict] | None = None
    pj = args.prompts_json or (args.attr / "prompts.json")
    if pj.exists():
        prompts = json.loads(pj.read_text())

    render_circuit_html(
        attr_dir=args.attr,
        output=args.output,
        acts_dir=args.acts,
        tokenizer_id=args.tokenizer,
        topk_neurons=args.topk_neurons,
        topk_examples=args.topk_examples,
        context_tokens=args.context_tokens,
        title=args.title,
        prompts=prompts,
        selection=args.selection,
    )
    print(f"[render] wrote {args.output} ({args.output.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
