#!/bin/bash
#SBATCH --job-name=relp-acts
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/collect_acts-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/collect_acts-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh

cd "${RELP_REPO}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
DATASET="${RELP_DATASET:-allenai/dolma3_mix-6T-1025-7B}"
SEQ_LEN="${RELP_SEQ_LEN:-512}"
BATCH="${RELP_BATCH:-8}"
TOPK="${RELP_TOPK:-16}"
MAX_TOKENS="${RELP_MAX_TOKENS:-10000000}"
RUN_NAME="${RELP_RUN_NAME:-acts-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${RELP_ARTIFACTS}/activations/${RUN_NAME}"
LOCAL_SHARDS="${RELP_LOCAL_SHARDS:-}"
mkdir -p "${OUT_DIR}"

echo "[acts] model=${MODEL_ID} max_tokens=${MAX_TOKENS}"
if [[ -n "${LOCAL_SHARDS}" ]]; then
    echo "[acts] source=local: ${LOCAL_SHARDS}"
else
    echo "[acts] source=hf-stream: ${DATASET}"
fi
echo "[acts] seq_len=${SEQ_LEN} batch=${BATCH} K=${TOPK} out=${OUT_DIR}"
nvidia-smi -L || true

EXTRA=()
if [[ -n "${LOCAL_SHARDS}" ]]; then EXTRA+=(--local-shards "${LOCAL_SHARDS}"); fi

uv run python -m scripts.collect_acts \
    --model "${MODEL_ID}" \
    --dataset "${DATASET}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH}" \
    --topk "${TOPK}" \
    --max-tokens "${MAX_TOKENS}" \
    --output "${OUT_DIR}" \
    "${EXTRA[@]}"

echo "[acts] done. Artifacts: ${OUT_DIR}"
