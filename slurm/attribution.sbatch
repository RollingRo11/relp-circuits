#!/bin/bash
#SBATCH --job-name=relp-attr
#SBATCH --partition=compute
# Default: 1 GPU. For 32B+ with backward, override with --gres=gpu:4 at submit time
# (32B + autograd state ~75GB peak — doesn't fit on one H100, needs HF device_map=auto).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/attribution-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/attribution-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh

cd "${RELP_REPO}"

# Help PyTorch handle GPUs that are shared with other users' processes.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Args are passed via environment variables so the sbatch file stays generic.
TASK="${RELP_TASK:-sva}"
# Paper default: relp_paper (bare h·grad on Σ top-K logits at the answer position).
METHOD="${RELP_METHOD:-relp_paper}"   # relp_paper | relp | relp_unpaired | atp | ig
NUM_PAIRS="${RELP_NUM_PAIRS:-30}"
TOPK="${RELP_TOPK:-256}"
# Project default: OLMo-3.1-32B-Think. The paper's primary target was Llama 3.1 8B
# Instruct (arXiv 2604.07615 Sec. 5); we use OLMo-3.1 32B as a larger same-method probe.
# 32B + backward state ~75GB; override --gres=gpu:4 with device=auto at submit time.
MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
RUN_NAME="${RELP_RUN_NAME:-${TASK}-${METHOD}-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${RELP_ARTIFACTS}/attribution/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "[attribution] task=${TASK} method=${METHOD} pairs=${NUM_PAIRS} topk=${TOPK}"
echo "[attribution] model=${MODEL_ID} out=${OUT_DIR}"
nvidia-smi -L || true

DEVICE="${RELP_DEVICE:-cuda}"

uv run python -m scripts.run_attribution \
    --task "${TASK}" \
    --method "${METHOD}" \
    --model "${MODEL_ID}" \
    --num-pairs "${NUM_PAIRS}" \
    --topk "${TOPK}" \
    --device "${DEVICE}" \
    --output "${OUT_DIR}"

echo "[attribution] done. Artifacts: ${OUT_DIR}"
