#!/bin/bash
#SBATCH --job-name=relp-dump
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/dump-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/dump-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

uv run python - <<'PY'
from transformers import AutoTokenizer
from relp_circuits.tasks.multihop import build_multihop_pairs
import os
tok = AutoTokenizer.from_pretrained(os.environ.get("RELP_TOKENIZER", "allenai/Olmo-3.1-32B-Think"))
pairs = build_multihop_pairs(tok, num_pairs=20)
for i, p in enumerate(pairs):
    correct = tok.decode([p.clean_correct_id])
    incorrect = tok.decode([p.cf_correct_id])
    print(f"[{i+1:2d}] clean={p.clean_text!r}")
    print(f"     cf   ={p.cf_text!r}")
    print(f"     correct={correct!r}  incorrect={incorrect!r}")
PY
