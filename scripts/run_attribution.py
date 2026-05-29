"""Entry point: load the model, run attribution on a task, validate via ablation, dump artifacts.

Submitted to Slurm via slurm/attribution.sbatch. Do not run on the login node.

Default method `relp_paper` matches the ADAG paper (arXiv 2604.07615): bare h · grad
against the sum of the top-K logits (K=5) at the answer position, with the RelP
backward-replacement model installed.

Outputs (under --output):
  scores.npy               # (n_layers, d_ffn) float32 attribution scores
  topk.json                # global top-k (layer, neuron, score) entries
  ablation.json            # baseline, ablated, random-baseline, drops
  config.json              # arg dump for reproducibility
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from relp_circuits.ablation import ablate_specific, ablate_topk
from relp_circuits.attribution import (
    atp_attribution,
    ig_attribution,
    paper_relp_attribution,
    relp_attribution,
    unpaired_relp_attribution,
)
from relp_circuits.baseline import baseline_cache_path
from relp_circuits.model import HookedModel
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
        default="sva",
    )
    p.add_argument(
        "--method",
        choices=["atp", "ig", "relp", "relp_unpaired", "relp_paper"],
        default="relp_paper",
        help="default `relp_paper` matches arXiv 2604.07615 exactly: bare h·grad against "
             "the sum of top-K logits at the answer position, no counterfactual.",
    )
    p.add_argument("--baseline", type=Path, default=None,
                   help="Path to a precomputed (n_layers, d_ffn) baseline_mean.npy from "
                        "scripts/compute_baseline.py. Required for --method relp_unpaired; "
                        "ignored otherwise. If omitted, the script tries the standard cache "
                        "location.")
    p.add_argument("--num-pairs", type=int, default=30)
    p.add_argument("--ig-steps", type=int, default=10)
    p.add_argument("--top-k-logits", type=int, default=5,
                   help="K for the relp_paper target = Σ top-K logits at the answer position. "
                        "Paper uses K=5.")
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.005,
                   help="paper's per-pair τ filter: keep neurons where |RelP|/|m| ≥ τ on at least one pair")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda",
                   help="\"cuda\" picks the freest single GPU; \"auto\" spreads across all "
                        "visible GPUs (model parallel — needed for 32B+ with backward).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[run] loading {args.model} on {args.device}", flush=True)
    t0 = time.time()
    hooked = HookedModel.load(args.model, device=args.device, dtype=torch.bfloat16)
    print(f"[run] model loaded in {time.time()-t0:.1f}s; layers={hooked.n_layers} d_ffn={hooked.d_ffn}", flush=True)

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
    print(f"[run] task={args.task} pairs={len(pairs)}", flush=True)
    if not pairs:
        print("[run] FATAL: no usable pairs (likely tokenizer length mismatches)", file=sys.stderr)
        return 2

    t0 = time.time()
    if args.method == "atp":
        attr = atp_attribution(hooked, pairs)
    elif args.method == "ig":
        attr = ig_attribution(hooked, pairs, n_steps=args.ig_steps)
    elif args.method == "relp":
        attr = relp_attribution(hooked, pairs)
    elif args.method == "relp_paper":
        attr = paper_relp_attribution(hooked, pairs, top_k=args.top_k_logits)
    elif args.method == "relp_unpaired":
        baseline_path = args.baseline
        if baseline_path is None:
            # Try the standard cache; pick the first match for this model+seq_len.
            cache_root = Path("/data/artifacts/rohan/relp-circuits/baselines")
            safe = args.model.replace("/", "__")
            candidates = sorted(cache_root.glob(f"{safe}_seq*_n*.npy"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                print(f"[run] FATAL: --method relp_unpaired requires a baseline. "
                      f"None found under {cache_root}. Run scripts/compute_baseline.py first.",
                      file=sys.stderr)
                return 2
            baseline_path = candidates[0]
            print(f"[run] auto-picked baseline {baseline_path}", flush=True)
        baseline_mean = np.load(baseline_path)
        attr = unpaired_relp_attribution(hooked, pairs, baseline_mean=baseline_mean)
    else:
        raise ValueError(args.method)
    print(f"[run] attribution method={args.method} done in {time.time()-t0:.1f}s", flush=True)

    np.save(args.output / "scores.npy", attr.scores)

    layer_idx, neuron_idx = attr.topk_global(args.topk)
    topk_payload = [
        {"rank": i, "layer": int(layer_idx[i]), "neuron": int(neuron_idx[i]),
         "score": float(attr.scores[layer_idx[i], neuron_idx[i]])}
        for i in range(len(layer_idx))
    ]
    (args.output / "topk.json").write_text(json.dumps(topk_payload, indent=2))

    # Pick the ablation metric that matches the attribution form so the ablation drop
    # is measured in the same units as the attribution target:
    #   relp_paper      → Σ top-K logits at the answer position (paper-faithful)
    #   relp_unpaired   → logit(correct), absolute (no contrast token)
    #   atp / ig / relp → logit_diff(correct - incorrect), paired contrast
    if args.method == "relp_paper":
        metric_mode = "top_k_logit_sum"
    elif args.method == "relp_unpaired":
        metric_mode = "logit_value"
    else:
        metric_mode = "logit_diff"
    per_pair_top_k_ids: list[list[int]] | None = None
    if args.method == "relp_paper" and attr.extras and "per_pair_top_logit_ids" in attr.extras:
        per_pair_top_k_ids = attr.extras["per_pair_top_logit_ids"]
    print(f"[run] running top-{args.topk} ablation validation (metric={metric_mode})", flush=True)
    t0 = time.time()
    report = ablate_topk(hooked, pairs, attr, k=args.topk, metric_mode=metric_mode,
                         per_pair_top_k_ids=per_pair_top_k_ids)
    print(f"[run] top-k ablation in {time.time()-t0:.1f}s", flush=True)

    (args.output / "ablation.json").write_text(json.dumps({
        "baseline_metric": report.baseline_metric,
        "ablated_metric": report.ablated_metric,
        "random_baseline_metric": report.random_baseline_metric,
        "metric_drop": report.metric_drop,
        "random_drop": report.random_drop,
        "k": report.k,
        "n_pairs": report.n_pairs,
    }, indent=2))

    # τ-filter circuit: paper's per-example selection rule (RelP variants only).
    if args.method in ("relp", "relp_unpaired", "relp_paper") and attr.per_pair_scores is not None:
        tau_circuit = attr.tau_circuit(args.tau)
        (args.output / "tau_circuit.json").write_text(json.dumps({
            "tau": args.tau,
            "neurons": tau_circuit,
        }, indent=2))
        print(f"[run] τ={args.tau} kept {len(tau_circuit)} neurons across {attr.n_pairs} pairs", flush=True)
        # Validate by ablating the τ-circuit (different size than top-k).
        t0 = time.time()
        layer_neurons = [(r["layer"], r["neuron"]) for r in tau_circuit]
        tau_report = ablate_specific(
            hooked, pairs, attr.n_layers, attr.d_ffn, layer_neurons,
            metric_mode=metric_mode,
            per_pair_top_k_ids=per_pair_top_k_ids,
        )
        (args.output / "ablation_tau.json").write_text(json.dumps({
            "baseline_metric": tau_report.baseline_metric,
            "ablated_metric": tau_report.ablated_metric,
            "random_baseline_metric": tau_report.random_baseline_metric,
            "metric_drop": tau_report.metric_drop,
            "random_drop": tau_report.random_drop,
            "k": tau_report.k,
            "n_pairs": tau_report.n_pairs,
            "tau": args.tau,
        }, indent=2))
        print(f"[run] τ ablation in {time.time()-t0:.1f}s — drop={tau_report.metric_drop:.3f} "
              f"(random {tau_report.random_drop:.3f})", flush=True)
        # Save per-pair tensors so analyses can re-derive any τ + the graph viz can use them.
        np.save(args.output / "per_pair_scores.npy", attr.per_pair_scores)
        np.save(args.output / "per_pair_metric.npy", attr.per_pair_metric)
        if attr.per_pair_token_scores is not None:
            np.save(args.output / "per_pair_token_scores.npy", attr.per_pair_token_scores)
            np.save(args.output / "per_pair_lengths.npy", attr.per_pair_lengths)
            (args.output / "per_pair_token_ids.json").write_text(json.dumps(attr.per_pair_token_ids))

    (args.output / "config.json").write_text(json.dumps({
        "model": args.model, "task": args.task, "method": args.method,
        "num_pairs": args.num_pairs, "ig_steps": args.ig_steps, "topk": args.topk,
    }, indent=2))

    # Save prompts so the HTML viz can render them in the prompts panel later.
    prompts_payload = []
    for pair in pairs:
        prompts_payload.append({
            "clean": pair.clean_text,
            "cf": pair.cf_text,
            "correct": hooked.tokenizer.decode([pair.metric_correct_id()]),
            "incorrect": hooked.tokenizer.decode([pair.metric_incorrect_id()]),
        })
    (args.output / "prompts.json").write_text(json.dumps(prompts_payload, indent=2))

    print(f"[run] OK. baseline={report.baseline_metric:.3f} ablated={report.ablated_metric:.3f} "
          f"random={report.random_baseline_metric:.3f} drop={report.metric_drop:.3f} "
          f"random_drop={report.random_drop:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
