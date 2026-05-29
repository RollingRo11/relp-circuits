"""ADAG describer + simulator: turns cluster_contexts.json into
cluster_descriptions.json by running the paper's two-explainer pipeline plus
the held-out simulator.

For each cluster, this script produces:
  - input_description       (from the input-attribution explainer, paper's
                             Transluce/llama_8b_explainer by default)
  - output_description      (from the output-contribution explainer, paper's
                             Claude Haiku by default)
  - input_simulator_pearson_r
  - output_simulator_pearson_r

Backends can be picked independently for the three roles. Defaults follow the
paper: HF/Transluce for input, Anthropic API for output, HF for the simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from relp_circuits.describe import (
    build_cluster_contexts,
    dump_explanations,
    explain_and_score_clusters,
    make_backend,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr-dir", type=Path, required=True,
                   help="path to an attribution run that already has clusters.json + the "
                        "per_pair_* arrays written by scripts/run_adag.py")
    p.add_argument("--task-name", required=True)
    p.add_argument("--tokenizer", required=True,
                   help="HF id (or path) of the tokenizer used to decode top-K logit ids "
                        "and input token strings. Must match the model that produced the "
                        "attribution run (e.g. meta-llama/Llama-3.1-8B-Instruct).")

    # Independent backend choice for each role; paper defaults are baked in.
    p.add_argument("--input-backend", choices=["hf", "vllm", "api"], default="hf")
    p.add_argument("--input-model", default="Transluce/llama_8b_explainer",
                   help="Paper uses Transluce/llama_8b_explainer for input attribution.")
    p.add_argument("--output-backend", choices=["hf", "vllm", "api"], default="api")
    p.add_argument("--output-model", default="claude-3-5-haiku-latest",
                   help="Paper uses Claude Haiku for output contribution.")
    p.add_argument("--simulator-backend", choices=["hf", "vllm", "api"], default="hf",
                   help="Backend used to predict scores from a description on held-out "
                        "prompts. Defaults to the same Transluce/llama_8b_explainer HF "
                        "model as input.")
    p.add_argument("--simulator-model", default="Transluce/llama_8b_explainer")

    p.add_argument("--holdout-frac", type=float, default=0.5,
                   help="Fraction of prompts reserved for the Pearson-r simulator score.")
    p.add_argument("--n-neurons", type=int, default=8)
    p.add_argument("--n-input-tokens", type=int, default=12)
    p.add_argument("--n-output-tokens", type=int, default=8)
    p.add_argument("--max-input-tokens-per-prompt", type=int, default=64,
                   help="Cap on tokens shown to the simulator per held-out prompt.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cc_path = args.attr_dir / "cluster_contexts.json"
    if not cc_path.exists():
        raise SystemExit(f"missing {cc_path} — run run_adag.py first")

    tau_path = args.attr_dir / "tau_circuit.json"
    if not tau_path.exists():
        raise SystemExit(f"missing {tau_path} — needed for feature ordering")
    circuit = json.loads(tau_path.read_text())["neurons"]

    clusters_path = args.attr_dir / "clusters.json"
    if not clusters_path.exists():
        raise SystemExit(f"missing {clusters_path}")
    clusters = json.loads(clusters_path.read_text())
    cluster_ids = np.asarray(clusters["cluster_ids"], dtype=np.int32)

    # ── load per-pair raw arrays produced by run_adag.py ──
    input_attr = np.load(args.attr_dir / "per_pair_input_attr.npy")
    output_contrib = np.load(args.attr_dir / "per_pair_output_contrib.npy")
    top_k_logit_ids = np.load(args.attr_dir / "top_k_logit_ids.npy")
    per_pair_lengths = np.load(args.attr_dir / "per_pair_lengths.npy")

    pair_token_strs_path = args.attr_dir / "pair_token_strs.json"
    pair_token_strs = json.loads(pair_token_strs_path.read_text())

    neuron_examples_path = args.attr_dir / "neuron_examples.json"
    neuron_examples = (
        json.loads(neuron_examples_path.read_text())
        if neuron_examples_path.exists() else {}
    )

    # Cap the circuit at the number of cluster assignments; clusters.json was
    # built from the same (possibly-capped) circuit, so they should match.
    if len(circuit) != len(cluster_ids):
        circuit = circuit[: len(cluster_ids)]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    contexts = build_cluster_contexts(
        circuit=circuit,
        cluster_ids=cluster_ids,
        input_attr=input_attr,
        output_contrib=output_contrib,
        top_k_logit_ids=top_k_logit_ids,
        per_pair_lengths=per_pair_lengths,
        pair_token_strs=pair_token_strs,
        tokenizer=tokenizer,
        neuron_examples=neuron_examples,
        top_neurons_per_cluster=args.n_neurons,
        top_input_tokens=args.n_input_tokens,
        top_output_tokens=args.n_output_tokens,
    )
    n_total = len(contexts)
    print(
        f"[describe] {n_total} clusters; input={args.input_backend}:{args.input_model} "
        f"output={args.output_backend}:{args.output_model} "
        f"simulator={args.simulator_backend}:{args.simulator_model}",
        flush=True,
    )

    input_backend = make_backend(args.input_backend, args.input_model)
    output_backend = make_backend(args.output_backend, args.output_model)
    simulator_backend = make_backend(args.simulator_backend, args.simulator_model)

    explanations = explain_and_score_clusters(
        contexts,
        input_backend=input_backend,
        output_backend=output_backend,
        simulator_backend=simulator_backend,
        tokenizer=tokenizer,
        per_pair_lengths=per_pair_lengths.tolist(),
        holdout_frac=args.holdout_frac,
        max_input_tokens_per_prompt=args.max_input_tokens_per_prompt,
        seed=args.seed,
    )

    out_path = args.attr_dir / "cluster_descriptions.json"
    dump_explanations(explanations, out_path)
    print(f"[describe] wrote {out_path}", flush=True)

    # Also report mean Pearson r across clusters as the headline faithfulness score.
    if explanations:
        mean_in = float(np.mean([e.input_simulator_pearson_r for e in explanations]))
        mean_out = float(np.mean([e.output_simulator_pearson_r for e in explanations]))
        print(f"[describe] mean Pearson r — input={mean_in:+.3f}  output={mean_out:+.3f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
