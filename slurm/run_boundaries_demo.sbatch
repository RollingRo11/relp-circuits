#!/bin/bash
#SBATCH --job-name=relp-boundaries-demo
#SBATCH --partition=compute
# 32B + backward state ~75GB; spread across 4 H100s via HF device_map=auto.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/boundaries-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/boundaries-%j.err
# Forward the submitting shell's env (including ANTHROPIC_API_KEY) into the job.
# Some slurm configs default to --export=NONE; we want ALL plus an explicit
# mention of the API key so the describer step can reach Anthropic.
#SBATCH --export=ALL,ANTHROPIC_API_KEY,HF_TOKEN

# End-to-end ADAG demo on a single prompt whose private-reasoning trace
# contains the literal phrase "maybe the user is testing the boundaries?".
# Produces a single self-contained HTML dashboard at
#   ${OUT_DIR}/dashboard.html
# that can be downloaded and opened locally without a server.

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
RUN_NAME="${RELP_RUN_NAME:-boundaries-demo-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${RELP_ARTIFACTS}/attribution/${RUN_NAME}"
DEVICE="${RELP_DEVICE:-auto}"
TAU="${RELP_TAU:-0.0005}"     # lower than the default 0.005 so 1 prompt yields a non-empty τ-circuit
TOPK="${RELP_TOPK:-64}"        # top-k neurons reported for top-k ablation
MAX_FEATURES="${RELP_MAX_FEATURES:-256}"
N_CLUSTERS="${RELP_N_CLUSTERS:-6}"

mkdir -p "${OUT_DIR}"
echo "[boundaries] model=${MODEL_ID} out=${OUT_DIR}"
nvidia-smi -L || true

# ── Step 1: paper-faithful RelP attribution on the 1-prompt task ────────────
echo "[boundaries] step 1/4 — attribution"
uv run python -m scripts.run_attribution \
    --task eval_boundaries \
    --method relp_paper \
    --model "${MODEL_ID}" \
    --num-pairs 1 \
    --topk "${TOPK}" \
    --tau "${TAU}" \
    --device "${DEVICE}" \
    --output "${OUT_DIR}"

# ── Step 2: ADAG profiles + multi-view spectral clustering ──────────────────
echo "[boundaries] step 2/4 — adag profiles + clustering"
uv run python -m scripts.run_adag \
    --model "${MODEL_ID}" \
    --task eval_boundaries \
    --num-pairs 1 \
    --attr-dir "${OUT_DIR}" \
    --top-k-logits 5 \
    --n-clusters "${N_CLUSTERS}" \
    --max-features "${MAX_FEATURES}" \
    --device "${DEVICE}"

# ── Step 3: two-explainer describer + held-out Pearson-r simulator ──────────
# Use Anthropic API for all three roles so no extra HF weights need
# downloading. Requires ANTHROPIC_API_KEY to be in the env (sbatch carries
# it through from the submitting shell).
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "[boundaries] WARNING: ANTHROPIC_API_KEY not set; describer will fail." >&2
fi
echo "[boundaries] step 3/4 — describer"
uv run --with anthropic python -m scripts.run_describe \
    --attr-dir "${OUT_DIR}" \
    --task-name eval_boundaries \
    --tokenizer "${MODEL_ID}" \
    --input-backend api \
    --input-model claude-3-5-haiku-latest \
    --output-backend api \
    --output-model claude-3-5-haiku-latest \
    --simulator-backend api \
    --simulator-model claude-3-5-haiku-latest \
    --holdout-frac 1.0

# ── Step 4: bake the self-contained dashboard HTML ──────────────────────────
# circuit_tracer_export.py handles a missing edges.npz gracefully — we skip
# run_clja_edges for this single-prompt demo. The resulting HTML still shows
# embed→feature and feature→logit links and cluster annotations.
echo "[boundaries] step 4/4 — bake dashboard"
uv run python -m scripts.run_bake_circuit_html \
    --attr-dir "${OUT_DIR}" \
    --model "${MODEL_ID}" \
    --out "${OUT_DIR}/dashboard.html" \
    --num-graphs 1 \
    --slug-prefix boundaries \
    --title "ADAG — 'maybe the user is testing the boundaries?'"

# Drop a short README in the output dir for whoever downloads the bundle.
cat > "${OUT_DIR}/README.txt" <<EOF
Self-contained ADAG dashboard for the 1-prompt eval_boundaries demo.

Open dashboard.html in any modern browser — no server required. All graph
data, neuron exemplars, and frontend assets are inlined.

Prompt: the assistant's private-reasoning trace literally contains
        "maybe the user is testing the boundaries?"
Method: paper-faithful RelP (bare h·∂target/∂h on Σ top-K logits, K=5).
Model:  ${MODEL_ID}
Tau:    ${TAU}

Other files in this directory (kept for reproducibility):
  scores.npy, per_pair_*.npy, tau_circuit.json, ablation*.json,
  clusters.json, cluster_contexts.json, cluster_descriptions.json,
  prompts.json, config.json
EOF

VIZ_COPY="${RELP_REPO}/viz/${RUN_NAME}.html"
cp "${OUT_DIR}/dashboard.html" "${VIZ_COPY}"

echo "[boundaries] done."
echo "[boundaries]   primary:  ${OUT_DIR}/dashboard.html"
echo "[boundaries]   viz copy: ${VIZ_COPY}"
