#!/bin/bash
#SBATCH --job-name=relp-clja
#SBATCH --partition=compute
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/clja-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/clja-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
TASK="${RELP_TASK:-eval_awareness_flat}"
NUM_PAIRS="${RELP_NUM_PAIRS:-100}"
ATTR_DIR="${RELP_ATTR:?must set RELP_ATTR}"
MAX_FEATURES="${RELP_MAX_FEATURES:-300}"
CHUNK_SIZE="${RELP_CHUNK_SIZE:-20}"
DEVICE="${RELP_DEVICE:-auto}"

uv run python -m scripts.run_clja_edges \
    --model "${MODEL_ID}" \
    --task "${TASK}" \
    --num-pairs "${NUM_PAIRS}" \
    --attr-dir "${ATTR_DIR}" \
    --max-features "${MAX_FEATURES}" \
    --chunk-size "${CHUNK_SIZE}" \
    --device "${DEVICE}"

echo "[clja] done. edges.npz in ${ATTR_DIR}"
