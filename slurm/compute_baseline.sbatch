#!/bin/bash
#SBATCH --job-name=relp-baseline
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/baseline-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/baseline-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
LOCAL_SHARDS="${RELP_LOCAL_SHARDS:-/data/artifacts/rohan/relp-circuits/dolma3/datasets--allenai--dolma3_mix-6T-1025-7B/snapshots/2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b/data}"
SEQ_LEN="${RELP_SEQ_LEN:-512}"
MAX_TOKENS="${RELP_MAX_TOKENS:-200000}"
BATCH="${RELP_BATCH:-8}"
DEVICE="${RELP_DEVICE:-cuda}"

uv run python -m scripts.compute_baseline \
    --model "${MODEL_ID}" \
    --local-shards "${LOCAL_SHARDS}" \
    --seq-len "${SEQ_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --batch-size "${BATCH}" \
    --device "${DEVICE}"
