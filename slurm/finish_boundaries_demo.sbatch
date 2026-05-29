#!/bin/bash
#SBATCH --job-name=relp-boundaries-finish
#SBATCH --partition=compute
# Steps 3 (describer, API-only) and 4 (bake, JSON-only) — no GPU needed.
# Local 7B describer (OLMo-3-7B-Think) — 1 H100 is plenty. Avoids the network
# call to Anthropic, so this run doesn't depend on ANTHROPIC_API_KEY being set
# inside the slurm environment.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/boundaries-finish-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/boundaries-finish-%j.err
#SBATCH --export=ALL,ANTHROPIC_API_KEY,HF_TOKEN

# Resumes a partial boundaries-demo run by executing only steps 3-4 against an
# existing attribution+ADAG output directory (set RELP_ATTR). Useful when the
# heavy GPU steps already succeeded but the describer failed (e.g. missing API
# key) — re-running everything would waste ~30 min of 4×H100 time.

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

OUT_DIR="${RELP_ATTR:?must set RELP_ATTR to an existing run dir}"
MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"

echo "[finish] resuming run at ${OUT_DIR}"

# Default to the non-Think instruct sibling — Think variants emit raw CoT
# ("Okay, let's see...") that pollutes descriptions and breaks the simulator's
# Pearson r. The Instruct variant produces direct 1-2 sentence answers.
DESCRIBER_MODEL="${RELP_DESCRIBER_MODEL:-allenai/Olmo-3-7B-Instruct}"
echo "[boundaries] step 3/4 — describer (local HF: ${DESCRIBER_MODEL})"
uv run python -m scripts.run_describe \
    --attr-dir "${OUT_DIR}" \
    --task-name eval_boundaries \
    --tokenizer "${MODEL_ID}" \
    --input-backend hf \
    --input-model "${DESCRIBER_MODEL}" \
    --output-backend hf \
    --output-model "${DESCRIBER_MODEL}" \
    --simulator-backend hf \
    --simulator-model "${DESCRIBER_MODEL}" \
    --holdout-frac 1.0

echo "[boundaries] step 4/4 — bake dashboard"
uv run python -m scripts.run_bake_circuit_html \
    --attr-dir "${OUT_DIR}" \
    --model "${MODEL_ID}" \
    --out "${OUT_DIR}/dashboard.html" \
    --num-graphs 1 \
    --slug-prefix boundaries \
    --title "ADAG — 'maybe the user is testing the boundaries?'"

cat > "${OUT_DIR}/README.txt" <<EOF
Self-contained ADAG dashboard for the 1-prompt eval_boundaries demo.

Open dashboard.html in any modern browser — no server required. All graph
data, neuron exemplars, and frontend assets are inlined.

Prompt: the assistant's private-reasoning trace literally contains
        "maybe the user is testing the boundaries?"
Method: paper-faithful RelP (bare h·∂target/∂h on Σ top-K logits, K=5).
Model:  ${MODEL_ID}

Other files in this directory (kept for reproducibility):
  scores.npy, per_pair_*.npy, tau_circuit.json, ablation*.json,
  clusters.json, cluster_contexts.json, cluster_descriptions.json,
  prompts.json, config.json
EOF

# Also drop a copy under the in-repo viz/ folder so it can be downloaded
# straight from the editor's tree view without an scp.
RUN_NAME="$(basename "${OUT_DIR}")"
VIZ_COPY="${RELP_REPO}/viz/${RUN_NAME}.html"
cp "${OUT_DIR}/dashboard.html" "${VIZ_COPY}"

echo "[boundaries] done."
echo "[boundaries]   primary:  ${OUT_DIR}/dashboard.html"
echo "[boundaries]   viz copy: ${VIZ_COPY}"
