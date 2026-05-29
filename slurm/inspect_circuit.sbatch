#!/bin/bash
#SBATCH --job-name=relp-inspect
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/inspect-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/inspect-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

ATTR="${RELP_ATTR:?must set RELP_ATTR}"
ACTS="${RELP_ACTS:?must set RELP_ACTS}"
TOPN="${RELP_TOPN:-12}"

uv run python - <<PY
import json, numpy as np
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from tokenizers import Tokenizer

attr = Path("${ATTR}")
acts = Path("${ACTS}")
topn = int("${TOPN}")

# load circuit
tau = json.loads((attr / "tau_circuit.json").read_text())["neurons"]
ablation = json.loads((attr / "ablation_tau.json").read_text())
config = json.loads((attr / "config.json").read_text())

print(f"=== {attr.name} ({config['model']} / {config['method']}) ===")
print(f"τ-circuit: {len(tau)} neurons | drop {ablation['metric_drop']:+.2f} of {ablation['baseline_metric']:+.2f}")
# layer histogram
hist = Counter(n['layer'] for n in tau)
n_layers = config.get('n_layers', 64)
peak = max(hist.values())
print()
print('layer distribution (each # = 1 neuron):')
for l in sorted(hist):
    print(f'  L{l:2d} {"#"*hist[l]} ({hist[l]})')
print()

# load acts
vals = np.load(acts / "topk_vals.npy")
docs = np.load(acts / "topk_doc_ids.npy")
poss = np.load(acts / "topk_positions.npy")
table = pq.read_table(acts / "docs.parquet")
doc_lookup = {int(d): list(t) for d, t in zip(table["doc_id"].to_pylist(), table["tokens"].to_pylist())}
print(f'acts index: vals={vals.shape}  docs={len(doc_lookup)}')
print()

tok = Tokenizer.from_pretrained(config['model'])
def render_span(toks, p, ctx=10):
    lo, hi = max(0, p-ctx), min(len(toks), p+ctx+1)
    out = []
    for i in range(lo, hi):
        s = tok.decode([int(toks[i])])
        s = s.replace('\n', '↵')
        if i == p:
            out.append(f'<<{s}>>')
        else:
            out.append(s)
    return ''.join(out)

# top-N neurons by frequency × max_norm_score
print(f'=== top-{topn} neurons by τ-frequency, with their max-activating Dolma spans ===')
for entry in tau[:topn]:
    li, ni = entry['layer'], entry['neuron']
    fr = entry['frequency']
    mns = entry['max_norm_score']
    agg = entry['agg_score']
    print(f'\n--- L{li}/N{ni}  freq={fr}/{config["num_pairs"]}  max_norm={mns:+.3f}  agg={agg:+.3f} ---')
    col_v = vals[li, :, ni]
    col_d = docs[li, :, ni]
    col_p = poss[li, :, ni]
    order = np.argsort(-col_v)
    shown = 0
    for r in order:
        if shown >= 5: break
        v = float(col_v[r])
        if not np.isfinite(v) or v <= 0: continue
        d, p = int(col_d[r]), int(col_p[r])
        toks_doc = doc_lookup.get(d)
        if not toks_doc or p >= len(toks_doc): continue
        span = render_span(toks_doc, p, ctx=8)
        # Truncate long spans for readability
        span = span[:160] + ('…' if len(span) > 160 else '')
        print(f'  act={v:>6.1f}  {span!r}')
        shown += 1
    if shown == 0:
        print('  (no examples in 100M Dolma slice)')
PY
