"""Query a top-K max-activating-examples index produced by collect_acts.

Two query modes:

  --neuron LAYER:NEURON   show top-K dataset spans where this neuron fires hardest
  --token-search TEXT     find the token id(s) of TEXT, scan loaded docs for occurrences,
                          then print which (layer, neuron)s ranked those positions in their
                          top-K — the inverse view used to ask "which neurons fired here?"

Run interactively (uv run) — this is for inspection, not Slurm.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True, help="dir containing topk_*.npy + docs.parquet")
    p.add_argument("--tokenizer", default="allenai/Olmo-3-7B-Think")
    p.add_argument("--neuron", default=None, help="LAYER:NEURON")
    p.add_argument("--token-search", default=None,
                   help="text to search for; reports neurons whose top-K hit those positions")
    p.add_argument("--context", type=int, default=12, help="tokens of left/right context to print")
    p.add_argument("--show", type=int, default=8, help="how many top-K results to render")
    return p.parse_args()


def _load(idx_dir: Path):
    vals = np.load(idx_dir / "topk_vals.npy")               # (L, K, D)
    docs = np.load(idx_dir / "topk_doc_ids.npy")            # (L, K, D)
    pos = np.load(idx_dir / "topk_positions.npy")           # (L, K, D)
    table = pq.read_table(idx_dir / "docs.parquet")
    doc_lookup: dict[int, list[int]] = {}
    for d, t in zip(table["doc_id"].to_pylist(), table["tokens"].to_pylist(), strict=True):
        doc_lookup[int(d)] = list(t)
    return vals, docs, pos, doc_lookup


def _render_span(tokenizer, tokens: list[int], pos: int, ctx: int) -> str:
    lo = max(0, pos - ctx)
    hi = min(len(tokens), pos + ctx + 1)
    parts = []
    for i in range(lo, hi):
        text = tokenizer.decode([tokens[i]], skip_special_tokens=False)
        if i == pos:
            parts.append(f"\x1b[1;33m{text}\x1b[0m")
        else:
            parts.append(text)
    return "".join(parts)


def cmd_neuron(args, vals, docs, pos, doc_lookup, tokenizer):
    layer_str, neuron_str = args.neuron.split(":")
    li, ni = int(layer_str), int(neuron_str)
    L, K, D = vals.shape
    if li >= L or ni >= D:
        print(f"out of range: layer must be <{L}, neuron <{D}")
        return 2
    col_vals = vals[li, :, ni]                # (K,)
    col_docs = docs[li, :, ni]
    col_pos = pos[li, :, ni]
    order = np.argsort(-col_vals)
    print(f"layer {li}, neuron {ni}: top {args.show} of K={K}")
    shown = 0
    for r in order:
        if shown >= args.show:
            break
        v = float(col_vals[r])
        if not np.isfinite(v):
            continue
        d = int(col_docs[r])
        p = int(col_pos[r])
        toks = doc_lookup.get(d)
        if toks is None or p >= len(toks):
            print(f"  rank={shown}  act={v:+.3f}  doc={d}  pos={p}  (doc not in manifest)")
        else:
            span = _render_span(tokenizer, toks, p, args.context)
            print(f"  rank={shown}  act={v:+.3f}  doc={d}  pos={p}")
            print(f"    {span}")
        shown += 1
    return 0


def cmd_token_search(args, vals, docs, pos, doc_lookup, tokenizer):
    target_ids = tokenizer.encode(args.token_search, add_special_tokens=False)
    if not target_ids:
        print("empty tokenization")
        return 2
    print(f"searching for token ids {target_ids} (text={args.token_search!r})")

    # Find (doc, pos) hits in the manifest.
    hits: list[tuple[int, int]] = []
    for d, toks in doc_lookup.items():
        for i, t in enumerate(toks):
            if t == target_ids[0]:
                hits.append((d, i))
    print(f"found {len(hits)} occurrences in {len(doc_lookup)} docs")
    if not hits:
        return 0

    # For each hit, scan all (layer, neuron) entries to find which had this in their top-K.
    L, K, D = vals.shape
    counts: dict[tuple[int, int], list[float]] = defaultdict(list)
    docs_K = docs   # (L, K, D)
    pos_K = pos
    vals_K = vals
    for d, p in hits[: min(50, len(hits))]:  # scan up to 50 hits
        # boolean mask of where this (doc, pos) appears in the top-K index
        mask = (docs_K == d) & (pos_K == p)   # (L, K, D)
        if not mask.any():
            continue
        idx = np.argwhere(mask)
        for li, _ki, ni in idx:
            counts[(int(li), int(ni))].append(float(vals_K[li, _ki, ni]))

    if not counts:
        print("no neurons rank these tokens in their top-K")
        return 0
    ranked = sorted(counts.items(), key=lambda kv: max(kv[1]), reverse=True)
    print(f"top {args.show} neurons that ranked these tokens highly:")
    for (li, ni), vs in ranked[: args.show]:
        print(f"  layer {li:2d}  neuron {ni:5d}  hits={len(vs):3d}  max_act={max(vs):+.3f}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.neuron and not args.token_search:
        print("specify --neuron LAYER:NEURON or --token-search TEXT")
        return 2

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vals, docs, pos, doc_lookup = _load(args.index)
    print(f"[query] loaded index: layers={vals.shape[0]} K={vals.shape[1]} d_ffn={vals.shape[2]} "
          f"docs_in_manifest={len(doc_lookup)}")

    if args.neuron:
        return cmd_neuron(args, vals, docs, pos, doc_lookup, tokenizer)
    return cmd_token_search(args, vals, docs, pos, doc_lookup, tokenizer)


if __name__ == "__main__":
    raise SystemExit(main())
