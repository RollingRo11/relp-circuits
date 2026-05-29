"""ADAG end-to-end driver: profiles → clustering → cluster contexts.

Reads an existing attribution run (must be `--method relp_unpaired` so we
have per_pair_token_scores.npy + tau_circuit.json) and emits, into the same
directory:

  per_pair_input_attr.npy        (P, F, T_max)   – Attr(f, x)_i
  per_pair_output_contrib.npy    (P, F, K)        – Contrib(f, x)_j
  top_k_logit_ids.npy            (P, K)
  top_k_logit_vals.npy           (P, K)
  clusters.json                  feature i → cluster id, plus quality metrics
  cluster_contexts.json          per-cluster prompt for the LLM describer

The describer itself is a separate script (run_describe.py) so we don't
mix expensive backward-pass compute with vLLM serving on the same job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from relp_circuits.clustering import cluster_features
from relp_circuits.describe import build_cluster_contexts
from relp_circuits.model import HookedModel
from relp_circuits.profiles import compute_attribution_profiles
from relp_circuits.tasks import (
    build_blimp_sva,
    build_eval_awareness_flat,
    build_eval_awareness_pairs,
    build_eval_awareness_paired,
    build_eval_boundaries,
    build_multihop_pairs,
    build_sva_pairs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--task",
        choices=["sva", "multihop", "blimp_sva", "eval_awareness",
                 "eval_awareness_paired", "eval_awareness_flat",
                 "eval_boundaries"],
        required=True,
    )
    p.add_argument("--num-pairs", type=int, default=20)
    p.add_argument("--attr-dir", type=Path, required=True,
                   help="path to a completed relp_unpaired attribution run "
                        "(must contain tau_circuit.json + per_pair_*.npy)")
    p.add_argument("--acts-dir", type=Path, default=None,
                   help="optional acts index; only needed for cluster_contexts.json")
    p.add_argument("--top-k-logits", type=int, default=5)
    p.add_argument("--n-clusters", type=int, default=16)
    p.add_argument("--max-features", type=int, default=1000,
                   help="cap the τ-circuit at this many features (sorted by τ frequency × |max_norm_score|). "
                        "spectral clustering scales O(F²) memory; for huge circuits (e.g. 26K-neuron eval-awareness paired) "
                        "set lower.")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[adag] loading model {args.model} (device={args.device})", flush=True)
    t0 = time.time()
    hooked = HookedModel.load(args.model, device=args.device, dtype=torch.bfloat16)
    print(f"[adag] loaded in {time.time()-t0:.1f}s; layers={hooked.n_layers} d_ffn={hooked.d_ffn}",
          flush=True)

    # Build the same prompt set as run_attribution.py.
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
    elif args.task == "eval_boundaries":
        pairs = build_eval_boundaries(hooked.tokenizer, num_pairs=args.num_pairs)
    else:
        raise ValueError(args.task)
    print(f"[adag] {len(pairs)} pairs", flush=True)

    tau_path = args.attr_dir / "tau_circuit.json"
    if not tau_path.exists():
        print(f"[adag] FATAL: {tau_path} not found", file=sys.stderr)
        return 2
    circuit_full = json.loads(tau_path.read_text())["neurons"]
    if len(circuit_full) > args.max_features:
        # Rank by τ-frequency first, then by |max_norm_score| as tiebreak.
        circuit_full.sort(key=lambda r: (
            -int(r.get("frequency", 0)),
            -abs(float(r.get("max_norm_score", 0))),
        ))
        circuit = circuit_full[: args.max_features]
        print(f"[adag] τ-circuit: {len(circuit_full)} neurons → capped to top {len(circuit)} by τ-frequency",
              flush=True)
    else:
        circuit = circuit_full
        print(f"[adag] τ-circuit: {len(circuit)} neurons", flush=True)

    # ─── Step 1: attribution profiles ───
    print(f"[adag] computing attribution profiles (top_k={args.top_k_logits})", flush=True)
    t0 = time.time()
    profiles = compute_attribution_profiles(
        hooked, pairs, circuit, top_k=args.top_k_logits,
    )
    print(f"[adag] profiles done in {time.time()-t0:.1f}s; "
          f"input_attr {profiles.input_attr.shape}, output_contrib {profiles.output_contrib.shape}",
          flush=True)
    np.save(args.attr_dir / "per_pair_input_attr.npy", profiles.input_attr)
    np.save(args.attr_dir / "per_pair_output_contrib.npy", profiles.output_contrib)
    np.save(args.attr_dir / "top_k_logit_ids.npy", profiles.top_k_logit_ids)
    np.save(args.attr_dir / "top_k_logit_vals.npy", profiles.top_k_logit_vals)

    # ─── Step 2: clustering ───
    print(f"[adag] spectral clustering, k={args.n_clusters}", flush=True)
    t0 = time.time()
    result = cluster_features(
        profiles.input_attr, profiles.output_contrib,
        n_clusters=args.n_clusters,
    )
    print(f"[adag] clustering in {time.time()-t0:.1f}s; "
          f"k={result.n_clusters}  silhouette={result.silhouette:.3f}  "
          f"CV={result.coef_of_variation:.3f}  opp_sign={result.pct_opposing_sign:.3f}",
          flush=True)
    cluster_payload = {
        "feature_keys": [list(k) for k in profiles.feature_idx],
        "cluster_ids": result.cluster_ids.tolist(),
        "n_clusters": result.n_clusters,
        "metrics": {
            "silhouette": result.silhouette,
            "coef_of_variation": result.coef_of_variation,
            "pct_opposing_sign": result.pct_opposing_sign,
        },
    }
    (args.attr_dir / "clusters.json").write_text(json.dumps(cluster_payload, indent=2))

    # ─── Step 3: cluster contexts (input for LLM describer) ───
    print("[adag] building cluster contexts for the describer", flush=True)
    pair_token_strs = [
        [hooked.tokenizer.decode([tid]) for tid in pair.clean_ids.tolist()]
        for pair in pairs
    ]

    neuron_examples: dict[str, list[dict]] = {}
    if args.acts_dir is not None:
        # Reuse viz._build_examples_for_neurons for consistency.
        from tokenizers import Tokenizer
        from relp_circuits.viz import _build_examples_for_neurons
        tok = Tokenizer.from_pretrained(args.model)
        wanted = [(int(c["layer"]), int(c["neuron"])) for c in circuit]
        ex_map = _build_examples_for_neurons(
            args.acts_dir, tok, wanted, topk_examples=8, context_tokens=12,
        )
        for (li, ni), recs in ex_map.items():
            neuron_examples[f"{li}:{ni}"] = recs

    contexts = build_cluster_contexts(
        circuit, result.cluster_ids,
        profiles.input_attr, profiles.output_contrib,
        profiles.top_k_logit_ids, profiles.per_pair_lengths,
        pair_token_strs, hooked.tokenizer, neuron_examples,
    )

    # ─── Per-cluster eval-vs-deploy differential ───
    # When the prompt set carries an `is_eval` boolean in pair.meta, compute
    # the mean cluster-activation magnitude on eval-framed vs deploy-framed
    # prompts. Δ > 0 means the cluster is more active on eval prompts (i.e.
    # an eval-aware feature); Δ < 0 means deploy-aware. We use the absolute
    # output-contribution magnitude per (prompt, feature) summed within the
    # cluster as the activation proxy (it's already per-prompt and accounts
    # for sign ambiguity from the half-rule attention linearization).
    differentials: dict[int, dict] = {}
    has_groups = all(("is_eval" in p.meta) for p in pairs)
    if has_groups:
        is_eval = np.array([bool(p.meta["is_eval"]) for p in pairs])
        n_eval = int(is_eval.sum())
        n_deploy = int((~is_eval).sum())
        # |contrib| over (P, F, K) → (P, F) total magnitude per prompt+feature.
        per_pair_feature_mag = np.abs(profiles.output_contrib).sum(axis=2)
        cluster_ids_arr = np.asarray(result.cluster_ids)
        for c in sorted(set(int(x) for x in cluster_ids_arr)):
            members = np.where(cluster_ids_arr == c)[0]
            if members.size == 0:
                continue
            # (P,) mean over features in cluster
            per_pair_mag = per_pair_feature_mag[:, members].mean(axis=1)
            mean_eval = float(per_pair_mag[is_eval].mean()) if n_eval else 0.0
            mean_deploy = float(per_pair_mag[~is_eval].mean()) if n_deploy else 0.0
            differentials[int(c)] = {
                "mean_eval": mean_eval,
                "mean_deploy": mean_deploy,
                "delta": mean_eval - mean_deploy,
                "n_eval": n_eval,
                "n_deploy": n_deploy,
            }
        print(f"[adag] computed eval/deploy differentials for {len(differentials)} clusters "
              f"(n_eval={n_eval}, n_deploy={n_deploy})", flush=True)
    else:
        print("[adag] prompts lack is_eval metadata; skipping group differential", flush=True)
    cc_payload = []
    for c in contexts:
        entry = {
            "cluster_id": c.cluster_id,
            "n_features": len(c.feature_idxs),
            "aggregate_score": c.aggregate_score,
            "feature_keys": [list(k) for k in c.feature_keys],
            "input_token_block": c.input_token_block,
            "output_token_block": c.output_token_block,
            "neuron_block": c.neuron_block,
        }
        if differentials and c.cluster_id in differentials:
            d = differentials[c.cluster_id]
            entry["group_differential"] = d
            sign = "more on eval-frame" if d["delta"] > 0 else "more on deploy-frame"
            entry["differential_summary"] = (
                f"Activates {sign} prompts: "
                f"|contrib|_eval={d['mean_eval']:.3f}, "
                f"|contrib|_deploy={d['mean_deploy']:.3f}, "
                f"Δ={d['delta']:+.3f}."
            )
        cc_payload.append(entry)
    (args.attr_dir / "cluster_contexts.json").write_text(json.dumps(cc_payload, indent=2))

    # Persist the artifacts the describer/simulator needs to rebuild full
    # `ClusterContext` objects with their per-prompt profiles.
    (args.attr_dir / "pair_token_strs.json").write_text(json.dumps(pair_token_strs))
    np.save(args.attr_dir / "per_pair_lengths.npy", profiles.per_pair_lengths)
    if neuron_examples:
        (args.attr_dir / "neuron_examples.json").write_text(json.dumps(neuron_examples))

    print(f"[adag] OK. {len(contexts)} cluster contexts written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
