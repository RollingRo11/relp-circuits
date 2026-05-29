"""Compute CLJA cross-layer edges between τ-circuit features.

Reads an existing attribution run (with tau_circuit.json) and writes
`edges.npz` with shape (P, F, F) — entry [p, src, tgt] is the absolute
JVP h_src · ∂h_tgt/∂h_src summed over source token positions, in the
linearized RelP graph.

This complements run_adag.py — the attribution profiles give us
node-level Attr/Contrib, and this gives the inter-feature edges that
the circuit-tracer-style frontend needs to render an actual graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from relp_circuits.edges import compute_clja_edges
from relp_circuits.model import HookedModel
from relp_circuits.tasks import (
    build_blimp_sva,
    build_eval_awareness_flat,
    build_eval_awareness_pairs,
    build_eval_awareness_paired,
    build_multihop_pairs,
    build_sva_pairs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--task",
        choices=["sva", "multihop", "blimp_sva", "eval_awareness",
                 "eval_awareness_paired", "eval_awareness_flat"],
        required=True,
    )
    p.add_argument("--num-pairs", type=int, default=20)
    p.add_argument("--attr-dir", type=Path, required=True,
                   help="path to a completed attribution run with tau_circuit.json")
    p.add_argument("--max-features", type=int, default=300,
                   help="cap τ-circuit; CLJA is O(F²) memory and roughly O(F·layers) compute.")
    p.add_argument("--chunk-size", type=int, default=20,
                   help="batched-Jacobian chunk size over tgt features (memory ↔ speed)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[clja] loading {args.model} (device={args.device})", flush=True)
    t0 = time.time()
    hooked = HookedModel.load(args.model, device=args.device, dtype=torch.bfloat16)
    print(f"[clja] loaded in {time.time()-t0:.1f}s; layers={hooked.n_layers} d_ffn={hooked.d_ffn}",
          flush=True)

    # Build the prompt set (must match the original attribution run).
    if args.task == "sva":
        pairs = build_sva_pairs(hooked.tokenizer, num_pairs=args.num_pairs)
    elif args.task == "multihop":
        pairs = build_multihop_pairs(hooked.tokenizer, num_pairs=args.num_pairs)
    elif args.task == "blimp_sva":
        pairs = build_blimp_sva(hooked.tokenizer, num_pairs=args.num_pairs)
    elif args.task == "eval_awareness":
        pairs = build_eval_awareness_pairs(hooked.tokenizer, num_pairs=args.num_pairs)
    elif args.task == "eval_awareness_paired":
        pairs = build_eval_awareness_paired(hooked.tokenizer, num_pairs=args.num_pairs)
    elif args.task == "eval_awareness_flat":
        pairs = build_eval_awareness_flat(hooked.tokenizer, num_pairs=args.num_pairs)
    else:
        raise ValueError(args.task)
    print(f"[clja] {len(pairs)} pairs", flush=True)

    tau_path = args.attr_dir / "tau_circuit.json"
    if not tau_path.exists():
        print(f"[clja] FATAL: {tau_path} not found", file=sys.stderr)
        return 2
    circuit_full = json.loads(tau_path.read_text())["neurons"]
    if len(circuit_full) > args.max_features:
        circuit_full.sort(key=lambda r: (
            -int(r.get("frequency", 0)),
            -abs(float(r.get("max_norm_score", 0))),
        ))
        circuit = circuit_full[: args.max_features]
        print(f"[clja] τ-circuit: {len(circuit_full)} → cap to {len(circuit)}", flush=True)
    else:
        circuit = circuit_full
        print(f"[clja] τ-circuit: {len(circuit)} neurons", flush=True)

    print(f"[clja] computing edges (chunk_size={args.chunk_size})", flush=True)
    t0 = time.time()
    result = compute_clja_edges(
        hooked, pairs, circuit, chunk_size=args.chunk_size,
    )
    elapsed = time.time() - t0
    print(f"[clja] edges done in {elapsed:.1f}s; "
          f"shape={result.edges.shape} layer_pairs={result.layer_pairs_seen}",
          flush=True)

    # Per-prompt diagnostics
    abs_edges = np.abs(result.edges)
    print(f"[clja] |edge| stats: max={abs_edges.max():.4g} "
          f"mean={abs_edges.mean():.4g} "
          f"frac>{1e-4}={(abs_edges > 1e-4).mean():.4g}",
          flush=True)

    out = args.attr_dir / "edges.npz"
    np.savez_compressed(
        out,
        edges=result.edges.astype(np.float32),
        feature_idx=np.array(result.feature_idx, dtype=np.int32),
        target_pos=result.target_pos,
        chunk_size=np.array([args.chunk_size]),
        elapsed_seconds=np.array([elapsed]),
    )
    print(f"[clja] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
