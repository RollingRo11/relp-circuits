#!/bin/bash
#SBATCH --job-name=relp-show
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/show-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/show-%j.err

set -euo pipefail
source /home/rkathuria/relp-circuits/slurm/common.sh
cd "${RELP_REPO}"

uv run python - <<'PY'
import json, numpy as np
from collections import Counter

ATTR = "/data/artifacts/rohan/relp-circuits/attribution/mh-relp-smoke"
ACTS = "/data/artifacts/rohan/relp-circuits/activations/acts-10m"

scores = np.load(f"{ATTR}/scores.npy")
topk = json.load(open(f"{ATTR}/topk.json"))
n_layers, d_ffn = scores.shape

# === circuit summary ===
ablation = json.load(open(f"{ATTR}/ablation.json"))
print("="*72)
print("CITY -> STATE -> CAPITAL CIRCUIT (Olmo-3.1-32B-Think, RelP, 20 pairs)")
print("="*72)
print(f"  baseline metric (logit_diff correct - incorrect):  {ablation['baseline_metric']:+.3f}")
print(f"  ablate top-128 RelP-attributed neurons:            {ablation['ablated_metric']:+.3f}   drop {ablation['metric_drop']:+.3f}")
print(f"  ablate 128 random neurons (control):               {ablation['random_baseline_metric']:+.3f}   drop {ablation['random_drop']:+.3f}")
print(f"  signal/noise: {ablation['metric_drop'] / max(abs(ablation['random_drop']), 1e-3):.0f}x")
print(f"  top-128 = 0.036% of {n_layers*d_ffn:,} total MLP neurons")
print()

print("Top-20 neurons by |attribution|:")
print(f"  {'rank':>4} {'L':>3} {'neuron':>6} {'score':>9}  sign-meaning")
for e in topk[:20]:
    sign = "(+ on clean)" if e['score'] > 0 else "(+ on cf)"
    print(f"  {e['rank']:>4} L{e['layer']:>2} {e['neuron']:>6} {e['score']:+9.4f}  {sign}")
print()

print("Layer distribution of top-128 (each # = 1 neuron):")
c = Counter(e['layer'] for e in topk)
for l in range(n_layers):
    n = c.get(l, 0)
    if n:
        print(f"  L{l:>2}  {'#'*n}  ({n})")
print()
buckets = [(0,13,"L0-13   embed/syntactic   "),
           (14,21,"L14-21  early-middle      "),
           (22,29,"L22-29  middle-late       "),
           (30,31,"L30-31  final logit lens  ")]
for lo, hi, name in buckets:
    n = sum(c.get(l, 0) for l in range(lo, hi+1))
    print(f"  {name}  {n:3d} of 128  ({100*n/128:.0f}%)")
print()

# === query top-5 circuit neurons against the streaming acts index ===
print("="*72)
print(f"What the top-5 circuit neurons fire on in Dolma (smoke-pass, 100k tokens, K=16)")
print("="*72)

vals = np.load(f"{ACTS}/topk_vals.npy")        # (L, K, D)
docs = np.load(f"{ACTS}/topk_doc_ids.npy")
poss = np.load(f"{ACTS}/topk_positions.npy")
import pyarrow.parquet as pq
table = pq.read_table(f"{ACTS}/docs.parquet")
doc_lookup = {int(d): list(t) for d, t in zip(table["doc_id"].to_pylist(), table["tokens"].to_pylist(), strict=True)}
print(f"acts index: layers={vals.shape[0]} K={vals.shape[1]} d_ffn={vals.shape[2]} docs={len(doc_lookup)}")
print()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Think")

def render(toks, p, ctx=10):
    lo, hi = max(0, p-ctx), min(len(toks), p+ctx+1)
    parts = []
    for i in range(lo, hi):
        s = tok.decode([toks[i]], skip_special_tokens=False)
        s = s.replace("\n", "\\n")
        if i == p:
            parts.append(f"<<{s}>>")
        else:
            parts.append(s)
    return "".join(parts)

for e in topk[:10]:
    li, ni = e['layer'], e['neuron']
    print(f"--- L{li}/N{ni} (RelP score {e['score']:+.4f}) ---")
    col_v = vals[li, :, ni]
    col_d = docs[li, :, ni]
    col_p = poss[li, :, ni]
    order = np.argsort(-col_v)
    shown = 0
    for r in order:
        v = float(col_v[r])
        if not np.isfinite(v):
            continue
        d, p = int(col_d[r]), int(col_p[r])
        toks_doc = doc_lookup.get(d)
        if toks_doc is None or p >= len(toks_doc):
            continue
        ctx = render(toks_doc, p)
        print(f"  act={v:+8.3f}  {ctx!r}")
        shown += 1
        if shown >= 6:
            break
    if shown == 0:
        print("  (no examples in 100k-token slice — neuron may be rare; need bigger corpus)")
    print()
PY
