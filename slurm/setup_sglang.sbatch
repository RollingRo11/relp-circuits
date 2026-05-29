#!/bin/bash
#SBATCH --job-name=relp-setup-sgl
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/setup_sglang-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/setup_sglang-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

echo "[setup-sgl] uv version: $(uv --version)"
echo "[setup-sgl] syncing project env with --extra sglang"
uv sync --extra sglang

echo "[setup-sgl] verifying imports"
uv run --extra sglang python -c "
import sglang, torch, transformers
print('sglang=', sglang.__version__)
print('torch=', torch.__version__)
print('transformers=', transformers.__version__)
from sglang.srt.models.olmo2 import Olmo2ForCausalLM, Olmo2DecoderLayer, Olmo2MLP
print('Olmo2 classes imported OK')
"
echo "[setup-sgl] done"
