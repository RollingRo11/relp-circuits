#!/bin/bash
#SBATCH --job-name=relp-describe
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/describe-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/describe-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

ATTR="${RELP_ATTR:?must set RELP_ATTR}"
TASK_NAME="${RELP_TASK_NAME:-circuit}"
# Tokenizer must match the model that produced the attribution run.
TOKENIZER="${RELP_TOKENIZER:-allenai/Olmo-3.1-32B-Think}"

# Paper defaults: HF Transluce explainer for input attribution descriptions,
# Anthropic Claude Haiku for output contribution descriptions, HF explainer
# as the simulator.
INPUT_BACKEND="${RELP_INPUT_BACKEND:-hf}"
INPUT_MODEL="${RELP_INPUT_MODEL:-Transluce/llama_8b_explainer}"
OUTPUT_BACKEND="${RELP_OUTPUT_BACKEND:-api}"
OUTPUT_MODEL="${RELP_OUTPUT_MODEL:-claude-3-5-haiku-latest}"
SIMULATOR_BACKEND="${RELP_SIM_BACKEND:-hf}"
SIMULATOR_MODEL="${RELP_SIM_MODEL:-Transluce/llama_8b_explainer}"

HOLDOUT="${RELP_HOLDOUT_FRAC:-0.5}"

uv run python -m scripts.run_describe \
    --attr-dir "${ATTR}" \
    --task-name "${TASK_NAME}" \
    --tokenizer "${TOKENIZER}" \
    --input-backend "${INPUT_BACKEND}" \
    --input-model "${INPUT_MODEL}" \
    --output-backend "${OUTPUT_BACKEND}" \
    --output-model "${OUTPUT_MODEL}" \
    --simulator-backend "${SIMULATOR_BACKEND}" \
    --simulator-model "${SIMULATOR_MODEL}" \
    --holdout-frac "${HOLDOUT}"
