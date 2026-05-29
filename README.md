# relp-circuits

Implementation of *"ADAG: Automatically Describing Attribution Graphs"* (Arora, Wu,
Steinhardt, Schwettmann; [arXiv 2604.07615](https://arxiv.org/abs/2604.07615),
Apr 2026). Method is paper-faithful; default model is **`allenai/Olmo-3.1-32B-Think`**
(the paper's primary target was Llama 3.1 8B Instruct — we run the same method on
OLMo-3.1 32B as a larger same-method probe; pass `RELP_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct`
to reproduce the paper's exact model).

The pipeline does gradient-based circuit attribution directly over MLP post-activation
neurons (the d_ffn vector that feeds `down_proj` in a SwiGLU MLP), then clusters
neurons by their attribution profiles and runs the paper's two-explainer +
simulator description loop.

## Components

- `relp_circuits.model.HookedModel` — loads OLMo-3 (or Llama 3.1) and exposes per-layer
  MLP-neuron tensors. Inside `relp_active()` the model uses the RelP backward
  replacement: SiLU as a constant multiplier (`silu(g) = sigmoid(g)·g`), frozen
  RMSNorm scale, frozen softmax, half-rule on bilinear matmuls. This satisfies
  the conservation property `Σ h·grad = target`.
- `relp_circuits.attribution.paper_relp_attribution` — **default method**.
  Paper-faithful: bare `h · ∂target/∂h` against `target = Σ top-K logits` at the
  answer position, K=5. No counterfactual, no baseline subtraction.
- `relp_circuits.attribution.relp_attribution` — paired Relevance-Patching variant
  (Jafari et al. 2025 form): `(h_clean − h_cf) · grad` against logit-diff.
- `relp_circuits.attribution.unpaired_relp_attribution` — `(h − mean) · grad` with
  a corpus-mean baseline. Comparison/ablation variant.
- `relp_circuits.attribution.atp_attribution`, `ig_attribution` — baselines.
- `relp_circuits.ablation.ablate_topk` — top-k zero-ablation validation w/ random
  control. Supports the `top_k_logit_sum` metric so ablation drops are reported
  in the same units as the attribution target.
- `relp_circuits.profiles` — input-attribution and output-contribution profiles
  used by the clusterer (Sec. 3.2 of the paper).
- `relp_circuits.clustering.cluster_features` — multi-view spectral clustering
  with harmonic-mean of ReLU'd cosine similarities (Sec. 3.3).
- `relp_circuits.describe.explain_and_score_clusters` — paper's two-explainer
  pipeline: HF Transluce/llama_8b_explainer for input attribution + Anthropic
  Claude Haiku for output contribution, then the simulator's Pearson r on a
  held-out split (Sec. 4).
- `relp_circuits.tasks.*` — paired-prompt task builders (SVA, multihop, BLiMP,
  eval-awareness).

## Discipline (this cluster)

- Never run on the login node. All execution goes through `sbatch`.
- All Python work uses `uv`. Project venv is at `./.venv` (default uv location).
- Large *data* (HF cache, model weights, activations, attribution outputs) goes under
  `/data/artifacts/rohan/relp-circuits/`. Source/config stays in the repo.

## First-time setup

```
sbatch slurm/setup.sbatch          # uv sync into ./.venv
sbatch slurm/download_model.sbatch # prefetch Olmo-3.1-32B-Think into HF cache
```

## Run paper-faithful RelP attribution

```
RELP_METHOD=relp_paper RELP_NUM_PAIRS=30 RELP_TOPK=256 \
  sbatch slurm/attribution.sbatch
```

`relp_paper` is the default. Outputs land at
`/data/artifacts/rohan/relp-circuits/attribution/<run_name>/`:

- `scores.npy`              — `(n_layers, d_ffn)` averaged attribution
- `per_pair_scores.npy`     — `(n_pairs, n_layers, d_ffn)` for τ-thresholding
- `per_pair_metric.npy`     — per-pair target (Σ top-K logits)
- `tau_circuit.json`        — neurons that pass `|score| ≥ τ·|target|` (τ=0.005)
- `ablation.json`           — top-k ablation drop in top-K-logit-sum units
- `ablation_tau.json`       — τ-circuit ablation drop

## Run ADAG end-to-end

```
# 1) attribution (paper-faithful)
RELP_TASK=eval_awareness RELP_METHOD=relp_paper RELP_RUN_NAME=eval-paper \
  sbatch slurm/attribution.sbatch

# 2) profiles + clustering
RELP_ATTR=/data/artifacts/rohan/relp-circuits/attribution/eval-paper \
  sbatch slurm/run_adag.sbatch

# 3) two-explainer + simulator describer
ANTHROPIC_API_KEY=... \
RELP_ATTR=/data/artifacts/rohan/relp-circuits/attribution/eval-paper \
RELP_TASK_NAME=eval_awareness \
  sbatch slurm/run_describe.sbatch
```

`cluster_descriptions.json` will contain, per cluster: `input_description`,
`output_description`, `input_simulator_pearson_r`, `output_simulator_pearson_r`.
