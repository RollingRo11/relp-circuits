#!/bin/bash
#SBATCH --job-name=relp-setup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/setup-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/setup-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh

cd "${RELP_REPO}"
echo "[setup] uv version: $(uv --version)"
echo "[setup] syncing project env into ${RELP_REPO}/.venv"
uv sync --frozen 2>/dev/null || uv sync

echo "[setup] verifying core imports"
uv run python -c "import torch, transformers; print('torch=', torch.__version__, 'transformers=', transformers.__version__)"

echo "[setup] done."
