#!/bin/bash
#SBATCH --job-name=relp-prompts
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/prompts-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/prompts-%j.err

# Args (env):
#   RELP_TASK   one of {sva, multihop, blimp_sva}
#   RELP_NUM    num pairs (must match what was used at attribution time)
#   RELP_OUT    path to write prompts.json

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

uv run python - <<PY
import json
import os
from transformers import AutoTokenizer
from relp_circuits.tasks import (
    build_sva_pairs, build_multihop_pairs, build_blimp_sva,
    build_eval_awareness_pairs, build_eval_awareness_paired,
)

tok = AutoTokenizer.from_pretrained(os.environ.get("RELP_TOKENIZER", "allenai/Olmo-3.1-32B-Think"))
task = os.environ["RELP_TASK"]
num = int(os.environ["RELP_NUM"])
out = os.environ["RELP_OUT"]

if task == "sva":
    pairs = build_sva_pairs(tok, num_pairs=num)
elif task == "multihop":
    pairs = build_multihop_pairs(tok, num_pairs=num)
elif task == "blimp_sva":
    pairs = build_blimp_sva(tok, num_pairs=num)
elif task == "eval_awareness":
    pairs = build_eval_awareness_pairs(tok, num_pairs=num)
elif task == "eval_awareness_paired":
    pairs = build_eval_awareness_paired(tok, num_pairs=num)
else:
    raise SystemExit(f"unknown task: {task}")

payload = []
for p in pairs:
    payload.append({
        "clean": p.clean_text,
        "cf": p.cf_text,
        "correct": tok.decode([p.metric_correct_id()]),
        "incorrect": tok.decode([p.metric_incorrect_id()]),
    })
with open(out, "w") as f:
    json.dump(payload, f, indent=2)
print(f"[prompts] wrote {len(payload)} pairs to {out}")
PY
