#!/bin/bash
#SBATCH --job-name=relp-acts-sgl
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/collect_acts_sglang-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/collect_acts_sglang-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh

cd "${RELP_REPO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
DATASET="${RELP_DATASET:-allenai/dolma3_mix-6T-1025-7B}"
SEQ_LEN="${RELP_SEQ_LEN:-512}"
BATCH="${RELP_BATCH:-64}"
TOPK="${RELP_TOPK:-32}"
MAX_TOKENS="${RELP_MAX_TOKENS:-10000000}"
TP_SIZE="${RELP_TP_SIZE:-1}"
RUN_NAME="${RELP_RUN_NAME:-acts-sgl-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${RELP_ARTIFACTS}/activations/${RUN_NAME}"
LAYERS="${RELP_LAYERS:-}"
LOCAL_SHARDS="${RELP_LOCAL_SHARDS:-}"
mkdir -p "${OUT_DIR}"

echo "[acts-sgl] model=${MODEL_ID} dataset=${DATASET} max_tokens=${MAX_TOKENS}"
echo "[acts-sgl] tp=${TP_SIZE} seq_len=${SEQ_LEN} batch=${BATCH} K=${TOPK}"
echo "[acts-sgl] out=${OUT_DIR}"
nvidia-smi -L || true

EXTRA=()
if [[ -n "${LAYERS}" ]]; then EXTRA+=(--layers "${LAYERS}"); fi
if [[ -n "${LOCAL_SHARDS}" ]]; then EXTRA+=(--local-shards "${LOCAL_SHARDS}"); fi

uv run --extra sglang python -m scripts.collect_acts_sglang \
    --model "${MODEL_ID}" \
    --dataset "${DATASET}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH}" \
    --topk "${TOPK}" \
    --max-tokens "${MAX_TOKENS}" \
    --tp-size "${TP_SIZE}" \
    --output "${OUT_DIR}" \
    "${EXTRA[@]}"

echo "[acts-sgl] done. Artifacts: ${OUT_DIR}"
