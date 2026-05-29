#!/bin/bash
#SBATCH --job-name=relp-render
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/render-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/render-%j.err

# Args (env):
#   RELP_ATTR        attribution output dir (required)
#   RELP_ACTS        optional acts index dir
#   RELP_OUT_HTML    output .html path (required)
#   RELP_TITLE       optional display title
#   RELP_TOPK_NEU    optional, default 128
#   RELP_TOPK_EX     optional, default 8

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

ATTR="${RELP_ATTR:?must set RELP_ATTR}"
OUT="${RELP_OUT_HTML:?must set RELP_OUT_HTML}"
TITLE="${RELP_TITLE:-}"
ACTS="${RELP_ACTS:-}"
TOPK_NEU="${RELP_TOPK_NEU:-128}"
TOPK_EX="${RELP_TOPK_EX:-8}"

mkdir -p "$(dirname "${OUT}")"
ARGS=(--attr "${ATTR}" --output "${OUT}" --topk-neurons "${TOPK_NEU}" --topk-examples "${TOPK_EX}")
[[ -n "${TITLE}" ]] && ARGS+=(--title "${TITLE}")
[[ -n "${ACTS}" ]] && ARGS+=(--acts "${ACTS}")

uv run python -m scripts.render_circuit "${ARGS[@]}"
