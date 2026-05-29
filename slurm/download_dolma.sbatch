#!/bin/bash
#SBATCH --job-name=relp-dl-dolma
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/download_dolma-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/download_dolma-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

SHARDS_PER_SOURCE="${RELP_SHARDS_PER_SOURCE:-2}"
echo "[dolma] fetching ${SHARDS_PER_SOURCE} shards per source (round-robin) from allenai/dolma3_mix-6T-1025-7B"

uv run python -m scripts.download_dolma_shards --shards-per-source "${SHARDS_PER_SOURCE}"
echo "[dolma] done"
