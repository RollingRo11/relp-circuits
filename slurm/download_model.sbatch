#!/bin/bash
#SBATCH --job-name=relp-dl
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/download-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/download-%j.err

# Pre-fetch the model weights + tokenizer into the shared HF cache so the GPU
# job that runs attribution doesn't pay download time.

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh

MODEL_ID="${RELP_MODEL_ID:-allenai/Olmo-3.1-32B-Think}"
echo "[download] fetching ${MODEL_ID} into ${HF_HUB_CACHE}"

cd "${RELP_REPO}"
RELP_MODEL_ID_RUNTIME="${MODEL_ID}" uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download
mid = os.environ["RELP_MODEL_ID_RUNTIME"]
path = snapshot_download(
    repo_id=mid,
    cache_dir=os.environ["HF_HUB_CACHE"],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
)
print(f"[download] cached at {path}")
PY

echo "[download] done."
