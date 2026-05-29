#!/bin/bash
#SBATCH --job-name=relp-adag
#SBATCH --partition=compute
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/adag-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/adag-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
TASK="${RELP_TASK:-eval_awareness}"
NUM_PAIRS="${RELP_NUM_PAIRS:-10}"
ATTR_DIR="${RELP_ATTR:?must set RELP_ATTR}"
ACTS_DIR="${RELP_ACTS:-}"
TOP_K="${RELP_TOP_K_LOGITS:-5}"
N_CLUSTERS="${RELP_N_CLUSTERS:-16}"
MAX_FEATURES="${RELP_MAX_FEATURES:-1000}"
DEVICE="${RELP_DEVICE:-auto}"

EXTRA=()
[[ -n "${ACTS_DIR}" ]] && EXTRA+=(--acts-dir "${ACTS_DIR}")

uv run python -m scripts.run_adag \
    --model "${MODEL_ID}" \
    --task "${TASK}" \
    --num-pairs "${NUM_PAIRS}" \
    --attr-dir "${ATTR_DIR}" \
    --top-k-logits "${TOP_K}" \
    --n-clusters "${N_CLUSTERS}" \
    --max-features "${MAX_FEATURES}" \
    --device "${DEVICE}" \
    "${EXTRA[@]}"

echo "[adag] done. Artifacts in ${ATTR_DIR}"
