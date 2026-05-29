"""Single-page circuit viz, redesigned for clarity.

Layout (per prompt):

    ┌──────────────────────── header (sticky) ──────────────────────────┐
    │  task / model / method      [‹ prompt k/N ›] [jump ▾]               │
    │  prediction: ▮Austin  baseline=15.8  drop -13.0  signal 330×      │
    └──────┬────────────────────────────────────────────────┬───────────┘
           │                                                │
       L63 │ • • •         ●●     ●         ●               │  side panel
       L60 │   • •  ●        •      •                       │  (focused
       L55 │ •         •      ●                             │   neuron's
       …   │                                                │   examples)
       L08 │                                                │
       Emb │                                                │
           ├────────────────────────────────────────────────┤
           │ Dallas  is  in  Texas  .  The  capital  of …  │  bottom (sticky)
           └────────────────────────────────────────────────┘

Each neuron is a small constant-size dot placed at (peak_token_position, layer).
Color carries sign (blue=+, red=−); opacity carries |score| relative to the
strongest neuron in the prompt. Hover spotlights one neuron and dims everything
else; click pins it (magenta stroke) and populates the right rail with its
top-K Dolma activation spans, with the trigger token highlighted yellow. Pins
persist across hovers and across prompt switches.

Design influences: Anthropic's `attribution-graphs-frontend` (the layout
skeleton, sticky rails, opacity-only importance, magenta pin, side rail for
feature details) and Goodfire's `param-decomp` viz (canvas-style layered
rendering, the dataset-token highlighting pattern). We do not currently
attribute feature→feature edges so we omit the canvas edge layer; if/when we
add that the SVG node layer slots straight on top.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Tokenizer + dataset-example helpers (unchanged from prior version).

def _load_tokenizer(tokenizer_id: str):
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(tokenizer_id)


def _decode_one(tok, token_id: int) -> str:
    s = tok.decode([int(token_id)])
    if not s:
        s = tok.id_to_token(int(token_id)) or ""
        s = s.replace("Ġ", " ").replace("Ċ", "\n")
    return s


def _render_example_html(tok, doc_tokens, pos: int, ctx: int) -> str:
    lo = max(0, pos - ctx)
    hi = min(len(doc_tokens), pos + ctx + 1)
    pieces: list[str] = []
    for i in range(lo, hi):
        s = _decode_one(tok, doc_tokens[i])
        s_esc = html_lib.escape(s).replace("\n", "↵")
        if i == pos:
            pieces.append(f"<mark>{s_esc}</mark>")
        else:
            pieces.append(s_esc)
    return "".join(pieces)


def _build_examples_for_neurons(
    acts_dir: Path,
    tok,
    neurons: list[tuple[int, int]],
    topk_examples: int,
    context_tokens: int,
) -> dict[tuple[int, int], list[dict]]:
    import pyarrow.parquet as pq
    vals = np.load(acts_dir / "topk_vals.npy")
    docs_arr = np.load(acts_dir / "topk_doc_ids.npy")
    pos_arr = np.load(acts_dir / "topk_positions.npy")
    table = pq.read_table(acts_dir / "docs.parquet")
    doc_lookup: dict[int, list[int]] = {
        int(d): list(t)
        for d, t in zip(table["doc_id"].to_pylist(), table["tokens"].to_pylist(), strict=True)
    }
    out: dict[tuple[int, int], list[dict]] = {}
    for li, ni in neurons:
        col_v = vals[li, :, ni]
        col_d = docs_arr[li, :, ni]
        col_p = pos_arr[li, :, ni]
        order = np.argsort(-col_v)
        recs: list[dict] = []
        for r in order:
            if len(recs) >= topk_examples:
                break
            v = float(col_v[r])
            if not np.isfinite(v):
                continue
            d, p = int(col_d[r]), int(col_p[r])
            doc_toks = doc_lookup.get(d)
            if not doc_toks or p >= len(doc_toks):
                continue
            recs.append({
                "act": v,
                "doc_id": d,
                "pos": p,
                "html": _render_example_html(tok, doc_toks, p, context_tokens),
            })
        out[(li, ni)] = recs
    return out


# ---------------------------------------------------------------------------
# Per-prompt node layout.

def _build_prompts_payload(
    per_prompt_token_scores: np.ndarray,    # (P, L, T_max, D)
    per_prompt_lengths: np.ndarray,         # (P,)
    per_prompt_token_ids: list[list[int]],
    per_prompt_metric: np.ndarray,
    prompts: list[dict] | None,
    tok,
    tau_circuit: list[dict] | None,       # if available, restrict to τ neurons
    max_neurons_per_prompt: int,
) -> tuple[list[dict], set[tuple[int, int]]]:
    """One node per (layer, neuron) per prompt, positioned at the token where its
    per-token attribution magnitude peaks. We restrict to the top
    `max_neurons_per_prompt` per prompt after sorting by |aggregate score| (or by
    τ-frequency if a tau_circuit was provided) — keeps the SVG grid readable.
    """
    P, L, _T_max, D = per_prompt_token_scores.shape

    # If we have a τ-circuit, restrict to its members. Otherwise pick top-k per prompt.
    tau_set: set[tuple[int, int]] | None = None
    if tau_circuit is not None:
        tau_set = {(int(r["layer"]), int(r["neuron"])) for r in tau_circuit}

    needed_neurons: set[tuple[int, int]] = set()
    prompts_out: list[dict] = []
    for p in range(P):
        T = int(per_prompt_lengths[p])
        scores = np.nan_to_num(per_prompt_token_scores[p, :, :T, :], nan=0.0)  # (L, T, D)
        abs_scores = np.abs(scores)

        # Per-(l, n) score (summed over T) and peak token position.
        per_ln_score = scores.sum(axis=1)                  # (L, D) signed
        per_ln_abs_peak = abs_scores.max(axis=1)           # (L, D)
        per_ln_peak_pos = abs_scores.argmax(axis=1)        # (L, D) which token

        # Pick which (l, n) to place. tau_set if provided; else top-k by |sum|.
        candidates: list[tuple[int, int]] = []
        if tau_set is not None:
            for li, ni in tau_set:
                candidates.append((li, ni))
        else:
            flat = np.abs(per_ln_score).reshape(-1)
            order = np.argsort(-flat)[:max_neurons_per_prompt]
            for idx in order:
                li, ni = int(idx // D), int(idx % D)
                candidates.append((li, ni))

        # Sort candidates for this prompt by |aggregate score|, cap to budget.
        candidates.sort(key=lambda ln: -abs(per_ln_score[ln[0], ln[1]]))
        candidates = candidates[:max_neurons_per_prompt]

        nodes = []
        for li, ni in candidates:
            score = float(per_ln_score[li, ni])
            abs_peak = float(per_ln_abs_peak[li, ni])
            if abs_peak <= 0:
                continue
            peak_t = int(per_ln_peak_pos[li, ni])
            nodes.append({
                "l": li, "n": ni,
                "score": score,
                "peak_abs": abs_peak,
                "t": peak_t,
            })
            needed_neurons.add((li, ni))

        token_strs = [_decode_one(tok, tid) for tid in per_prompt_token_ids[p]]
        token_mass = abs_scores.sum(axis=(0, 2))             # (T,) per-token mass
        prompt_meta = (prompts[p] if prompts and p < len(prompts) else {}) or {}
        prompts_out.append({
            "tokens": token_strs,
            "token_mass": [float(x) for x in token_mass.tolist()],
            "nodes": nodes,
            "answer": prompt_meta.get("correct", "?"),
            "wrong": prompt_meta.get("incorrect", "?"),
            "clean_text": prompt_meta.get("clean", ""),
            "cf_text": prompt_meta.get("cf", ""),
            "metric": float(per_prompt_metric[p]),
        })
    return prompts_out, needed_neurons


# ---------------------------------------------------------------------------
# HTML template.

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --pos: #1976d2;
    --neg: #e53935;
    --pin: #f0f;
    --highlight: #ffeb3b;
    --rail-bg: #fafafa;
    --rail-border: #e8e8e8;
    --text: #222;
    --text-muted: #888;
    --text-dim: #aaa;
    --border: #ddd;
    --header-h: 70px;
    --token-row-h: 50px;
    --rail-w: 52px;
    --side-w: 340px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%;
               font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                            Roboto, sans-serif; color: var(--text);
               font-size: 13px; background: #fff; }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; }

  /* ───────── header ───────── */
  header { position: sticky; top: 0; z-index: 30; background: white;
           border-bottom: 1px solid var(--border); padding: 10px 18px; }
  header .row { display: flex; align-items: center; gap: 12px;
                flex-wrap: wrap; }
  header .title { font-weight: 600; font-size: 14px; }
  header .meta { color: var(--text-muted); font-size: 11px; }
  header .prompt-controls { display: flex; align-items: center; gap: 6px; }
  header button { padding: 1px 8px; border: 1px solid #ccc; border-radius: 3px;
                  background: white; cursor: pointer; font-size: 13px; }
  header button:hover { background: #eee; }
  header select { padding: 1px 4px; border: 1px solid #ccc; border-radius: 3px;
                  font-size: 11px; max-width: 320px;
                  font-family: ui-monospace, monospace; }
  header .prompt-num { color: var(--text-muted); font-size: 11px; }
  header .metrics { display: flex; gap: 16px; font-size: 11px; margin-top: 6px; }
  header .metrics .k { color: var(--text-muted); }
  header .metrics .v { font-weight: 600; }
  header .pred { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  header .pill { padding: 2px 9px; border-radius: 10px; font-weight: 600;
                 font-family: ui-monospace, monospace; font-size: 12px; }
  header .pill.correct { background: #d6f3df; color: #167a3a;
                          border: 1px solid #b6e3c1; }
  header .pill.cf { background: #fde2e2; color: #b94a48;
                     border: 1px solid #f3c4c4; padding: 1px 7px; font-size: 11px; }

  /* ───────── main grid ───────── */
  .stage { display: grid;
           grid-template-columns: var(--rail-w) 1fr var(--side-w);
           grid-template-rows: 1fr var(--token-row-h);
           height: calc(100vh - var(--header-h)); }

  .layer-rail { grid-column: 1; grid-row: 1;
                background: var(--rail-bg); border-right: 1px solid var(--rail-border);
                position: relative; overflow: hidden; }
  .layer-rail .label { position: absolute; right: 6px;
                       transform: translateY(-50%);
                       font-family: ui-monospace, monospace;
                       font-size: 10px; color: var(--text-muted); }
  .layer-rail .gridline { position: absolute; left: 0; right: 0;
                          height: 1px; background: var(--rail-border); }

  .circuit { grid-column: 2; grid-row: 1;
             position: relative; overflow: auto; background: #fff; }
  .circuit svg { display: block; }

  .side { grid-column: 3; grid-row: 1 / 3;
          background: var(--rail-bg); border-left: 1px solid var(--rail-border);
          padding: 14px 16px; overflow-y: auto; }
  .side h3 { margin: 0 0 6px; font-size: 11px; color: var(--text-muted);
             text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
  .side .empty { color: var(--text-dim); font-size: 12px; font-style: italic;
                 padding: 8px 0; }
  .side .neuron-card { border-bottom: 1px solid var(--rail-border);
                       padding: 10px 0; }
  .side .neuron-card .head { display: flex; align-items: baseline;
                              gap: 8px; margin-bottom: 6px; }
  .side .neuron-card .id { font-family: ui-monospace, monospace; font-size: 12px;
                            font-weight: 600; }
  .side .neuron-card .score { font-family: ui-monospace, monospace; font-size: 11px;
                               color: var(--text-muted); margin-left: auto; }
  .side .neuron-card .ex { font-family: ui-monospace, monospace; font-size: 11px;
                            margin: 4px 0; padding: 4px 7px;
                            background: white; border: 1px solid var(--rail-border);
                            border-radius: 3px; line-height: 1.5;
                            word-break: break-word; }
  .side .neuron-card .ex .act { color: var(--text-dim); margin-right: 6px; font-size: 10px; }
  .side .neuron-card .ex mark { background: var(--highlight); padding: 0 2px;
                                  border-radius: 2px; font-weight: 600; }

  .toplist { padding-bottom: 4px; }
  .toplist .row { display: flex; gap: 8px; align-items: center;
                  font-family: ui-monospace, monospace; font-size: 11px;
                  padding: 3px 0; cursor: pointer; }
  .toplist .row:hover { background: #fff; }
  .toplist .row .swatch { width: 8px; height: 8px; border-radius: 4px; flex-shrink: 0; }
  .toplist .row .lab { flex: 1; }
  .toplist .row .v { color: var(--text-muted); }

  .token-row { grid-column: 2; grid-row: 2;
               background: var(--rail-bg); border-top: 1px solid var(--rail-border);
               position: relative; overflow: hidden; }
  .token-row .col { position: absolute; top: 4px; bottom: 4px;
                    border-left: 1px dotted #efefef; padding: 6px 4px 0;
                    text-align: center;
                    font-family: ui-monospace, monospace; font-size: 10.5px; }
  .token-row .col .t { display: block; white-space: nowrap; overflow: hidden;
                       text-overflow: ellipsis; max-width: 90px;
                       margin: 0 auto; color: var(--text); }
  .token-row .col .bar { width: 70%; height: 3px; margin: 4px auto 0;
                          border-radius: 1px; background: rgba(245,124,0,0.4); }
  .token-row .col .pos { color: var(--text-dim); font-size: 9px; }

  .corner { grid-column: 1; grid-row: 2;
            background: var(--rail-bg); border-top: 1px solid var(--rail-border);
            border-right: 1px solid var(--rail-border); }

  /* ───────── cluster cards (ADAG) ───────── */
  .cards-stage { display: grid; grid-template-columns: 1fr var(--side-w);
                 grid-template-rows: 1fr; height: calc(100vh - var(--header-h)); }
  .cards-area { grid-column: 1; padding: 16px 18px; overflow-y: auto;
                background: #fcfcfc; }
  .cards-grid { display: grid; gap: 12px;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  .cluster-card { background: white; border: 1px solid #e0e0e0;
                  border-radius: 6px; padding: 12px 14px; cursor: pointer;
                  transition: transform 0.06s, box-shadow 0.08s; }
  .cluster-card:hover { transform: translateY(-1px);
                        box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
  .cluster-card.selected { border: 2px solid var(--pin); padding: 11px 13px; }
  .cluster-card .card-head { display: flex; align-items: center;
                              gap: 8px; margin-bottom: 8px; }
  .cluster-card .swatch { width: 12px; height: 12px; border-radius: 3px; }
  .cluster-card .cid { font-family: ui-monospace, monospace; font-size: 12px;
                        font-weight: 600; }
  .cluster-card .nfeat { color: var(--text-muted); font-size: 11px; }
  .cluster-card .score-bar { flex: 1; height: 6px; background: #f0f0f0;
                              border-radius: 3px; overflow: hidden; }
  .cluster-card .score-fill { height: 100%; background: rgba(245,124,0,0.7); }
  .cluster-card .score-num { color: var(--text-muted); font-size: 11px;
                              font-family: ui-monospace, monospace; }
  .cluster-card .descr { font-size: 12.5px; line-height: 1.45; color: #222;
                          margin-bottom: 10px;
                          display: -webkit-box; -webkit-line-clamp: 4;
                          -webkit-box-orient: vertical; overflow: hidden; }
  .cluster-card .descr.placeholder { color: var(--text-dim); font-style: italic; }
  .cluster-card .row-label { color: var(--text-muted); font-size: 10px;
                              text-transform: uppercase; letter-spacing: 0.05em;
                              margin: 6px 0 3px; font-weight: 500; }
  .cluster-card .chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .cluster-card .chip { padding: 2px 7px; border-radius: 9px; font-size: 11px;
                         font-family: ui-monospace, monospace;
                         border: 1px solid #ddd; background: #fafafa; color: #333; }
  .cluster-card .chip.in  { background: rgba(245,124,0,0.10); border-color: rgba(245,124,0,0.4); }
  .cluster-card .chip.out { background: rgba(25,118,210,0.10); border-color: rgba(25,118,210,0.4); }
  .cluster-card .chip .v  { color: var(--text-muted); margin-left: 4px; font-size: 10px; }
  .cluster-card .layer-strip { height: 10px; background: #f4f4f4;
                                 border-radius: 2px; margin: 8px 0;
                                 position: relative; }
  .cluster-card .layer-bar { position: absolute; top: 1px; bottom: 1px;
                              border-radius: 2px; }
  .cluster-card .layer-strip .scale { color: var(--text-dim); font-size: 9px;
                                        font-family: ui-monospace, monospace;
                                        position: absolute; bottom: -12px; }

  .cards-detail { grid-column: 2; background: var(--rail-bg);
                  border-left: 1px solid var(--rail-border);
                  padding: 14px 16px; overflow-y: auto; }
  .cards-detail h3 { margin: 0 0 6px; font-size: 11px; color: var(--text-muted);
                     text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
  .cards-detail .full-descr { font-size: 12.5px; line-height: 1.5;
                                background: white; padding: 10px 12px;
                                border-radius: 4px; border: 1px solid var(--rail-border);
                                margin-bottom: 14px; }
  .cards-detail .neuron-card { border: 1px solid var(--rail-border);
                                 background: white; border-radius: 4px;
                                 padding: 8px 10px; margin-bottom: 8px; }
  .cards-detail .neuron-card .head { display: flex; align-items: baseline;
                                       gap: 8px; margin-bottom: 6px;
                                       font-family: ui-monospace, monospace; font-size: 12px; }
  .cards-detail .neuron-card .id { font-weight: 600; }
  .cards-detail .neuron-card .score { color: var(--text-muted);
                                        margin-left: auto; font-size: 11px; }
  .cards-detail .ex { font-family: ui-monospace, monospace; font-size: 11px;
                       margin: 3px 0; padding: 4px 6px;
                       background: #f8f8f8; border-radius: 3px;
                       line-height: 1.45; word-break: break-word; }
  .cards-detail .ex .act { color: var(--text-dim); margin-right: 6px; font-size: 10px; }
  .cards-detail .ex mark { background: var(--highlight); padding: 0 2px;
                             border-radius: 2px; font-weight: 600; }
  .cards-detail .empty-state { color: var(--text-dim); font-size: 12px;
                                font-style: italic; }
  .cards-detail .metrics-block { color: var(--text-muted); font-size: 11px;
                                  margin-top: 16px;
                                  font-family: ui-monospace, monospace; }

  /* ───────── nodes (legacy dot view) ───────── */
  .node { transition: opacity 0.08s; cursor: pointer; }
  .node.dimmed { opacity: 0.08 !important; }
  .node.pinned { stroke: var(--pin); stroke-width: 1.5; }
  .node.hovered { stroke: #000; stroke-width: 1; }

  .col-highlight { fill: rgba(0,0,0,0.025); }

  /* ───────── floating tooltip ───────── */
  #tooltip { position: fixed; background: white; border: 1px solid #888;
             padding: 6px 9px; max-width: 320px;
             box-shadow: 0 4px 14px rgba(0,0,0,0.15); font-size: 11px;
             border-radius: 4px; z-index: 100; display: none;
             pointer-events: none; font-family: ui-monospace, monospace; }
</style>
</head>
<body>
  <header>
    <div class="row">
      <span class="title" id="title"></span>
      <span class="meta" id="meta"></span>
      <span class="prompt-controls" style="margin-left:auto">
        <button id="prev">‹</button>
        <span class="prompt-num" id="prompt-num"></span>
        <button id="next">›</button>
        <select id="jump"></select>
      </span>
    </div>
    <div class="row">
      <div class="pred">
        <span class="meta">target</span>
        <span class="pill correct" id="pred"></span>
      </div>
      <div class="metrics" id="metrics"></div>
    </div>
  </header>
  <!-- Cards view: shown when ADAG clusters are present -->
  <div class="cards-stage" id="cards-stage" style="display:none">
    <div class="cards-area">
      <div class="cards-grid" id="cards-grid"></div>
    </div>
    <aside class="cards-detail" id="cards-detail"></aside>
  </div>

  <!-- Legacy dot-grid view: shown when no ADAG clusters -->
  <div class="stage" id="dots-stage">
    <div class="layer-rail" id="layer-rail"></div>
    <div class="circuit" id="circuit"></div>
    <aside class="side" id="side"></aside>
    <div class="corner"></div>
    <div class="token-row" id="token-row"></div>
  </div>
  <div id="tooltip"></div>

<script>
const PAYLOAD = __PAYLOAD__;

const $ = (id) => document.getElementById(id);
const tooltip = $("tooltip");

const COL_W = 70;        // px per token column
const ROW_H = 13;         // px per layer row (compact)
const NODE_R = 3.5;       // node radius
const PAD = 6;            // edge padding inside grid
const SIDE_TOP_K = 12;    // how many neurons in side panel "top"
const PALETTE_POS = [25, 118, 210];
const PALETTE_NEG = [229, 57, 53];

let curPrompt = 0;
let pinned = new Set();   // {layer:neuron} keys
let hovered = null;       // {l, n} or null

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function nodeKey(l, n) { return `${l}:${n}`; }
function rgba(rgb, a) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a.toFixed(3)})`; }
function nodeColor(score, maxAbs) {
  const a = Math.pow(Math.min(1, Math.abs(score) / Math.max(maxAbs, 1e-9)), 0.55);
  const opacity = 0.18 + 0.82 * a;
  return rgba(score >= 0 ? PALETTE_POS : PALETTE_NEG, opacity);
}

// 16-cluster categorical palette (Tol-vibrant-ish; readable on white).
const CLUSTER_PALETTE = [
  [0,114,178], [213,94,0], [0,158,115], [204,121,167],
  [240,228,66], [86,180,233], [230,159,0], [148,103,189],
  [180,40,40], [80,140,80], [40,40,180], [200,80,200],
  [120,180,160], [180,120,40], [70,70,70], [40,170,200],
];
function clusterColor(cid, score, maxAbs) {
  const rgb = CLUSTER_PALETTE[((cid % CLUSTER_PALETTE.length) + CLUSTER_PALETTE.length) % CLUSTER_PALETTE.length];
  const a = Math.pow(Math.min(1, Math.abs(score) / Math.max(maxAbs, 1e-9)), 0.55);
  const opacity = 0.25 + 0.75 * a;
  return rgba(rgb, opacity);
}

function renderHeader() {
  const c = PAYLOAD.config || {};
  const m = PAYLOAD.metrics || {};
  $("title").textContent = PAYLOAD.title;
  $("meta").textContent =
    `${c.model || ""}  •  task=${c.task || ""}  •  method=${c.method || ""}` +
    (c.tau ? `  •  τ=${c.tau}` : "");
  const sig = m.metric_drop !== undefined
    ? (m.metric_drop / Math.max(Math.abs(m.random_drop || 0), 1e-3)).toFixed(0) + "×"
    : "—";
  $("metrics").innerHTML =
    `<span><span class="k">baseline</span> <span class="v mono">${(m.baseline_metric ?? 0).toFixed(2)}</span></span>` +
    `<span><span class="k">ablated (${m.k})</span> <span class="v mono">${(m.ablated_metric ?? 0).toFixed(2)}</span></span>` +
    `<span><span class="k">drop</span> <span class="v mono">${(m.metric_drop ?? 0).toFixed(2)}</span></span>` +
    `<span><span class="k">random drop</span> <span class="v mono">${(m.random_drop ?? 0).toFixed(2)}</span></span>` +
    `<span><span class="k">signal</span> <span class="v mono">${sig}</span></span>` +
    `<span><span class="k">prompts</span> <span class="v mono">${PAYLOAD.prompts.length}</span></span>`;
  // Prompt selector
  const jump = $("jump");
  jump.innerHTML = "";
  PAYLOAD.prompts.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    const label = (p.clean_text || p.tokens.slice(0, 8).join("")).slice(0, 80);
    opt.textContent = `${i + 1}. ${label} → ${p.answer}`;
    jump.appendChild(opt);
  });
  jump.addEventListener("change", e => goPrompt(parseInt(e.target.value, 10)));
  $("prev").addEventListener("click", () => goPrompt(curPrompt - 1));
  $("next").addEventListener("click", () => goPrompt(curPrompt + 1));
}

function goPrompt(i) {
  const N = PAYLOAD.prompts.length;
  curPrompt = ((i % N) + N) % N;
  $("jump").value = curPrompt;
  $("prompt-num").textContent = `prompt ${curPrompt + 1} / ${N}`;
  hovered = null;
  // Pins persist across prompt-switches.
  drawPrompt();
  drawSide();
}

function drawPrompt() {
  const prompt = PAYLOAD.prompts[curPrompt];
  const N_TOK = prompt.tokens.length;

  // Only show layers that have at least one node — collapses empty regions.
  // The user still sees actual layer numbers in the rail, so they can spot
  // gaps (e.g. "L19, L24, L25, L27…" instead of "L0..L63" with most empty).
  const activeLayers = [...new Set(prompt.nodes.map(n => n.l))].sort((a, b) => b - a);
  const layerYIdx = new Map();
  activeLayers.forEach((li, i) => layerYIdx.set(li, i));
  const N_ROWS = activeLayers.length;

  $("pred").textContent = (prompt.answer || "").trim() || prompt.answer || "?";

  // Layer rail — one compact label per active layer.
  const rail = $("layer-rail");
  rail.innerHTML = "";
  activeLayers.forEach((li, idx) => {
    const lab = document.createElement("div");
    lab.className = "label";
    lab.style.top = `${PAD + idx * ROW_H + ROW_H / 2}px`;
    lab.textContent = `L${li}`;
    rail.appendChild(lab);
  });

  // Token row
  const trow = $("token-row");
  trow.innerHTML = "";
  const maxMass = Math.max(...prompt.token_mass, 1e-9);
  for (let t = 0; t < N_TOK; t++) {
    const col = document.createElement("div");
    col.className = "col";
    col.style.left = `${PAD + t * COL_W}px`;
    col.style.width = `${COL_W}px`;
    col.dataset.t = t;
    let txt = prompt.tokens[t]
      .replace(/\n/g, "↵").replace(/^ /, "·") || "·";
    if (txt.length > 14) txt = txt.slice(0, 13) + "…";
    const massN = prompt.token_mass[t] / maxMass;
    col.innerHTML =
      `<span class="t">${escapeHtml(txt)}</span>` +
      `<div class="bar" style="background:rgba(245,124,0,${(0.15 + 0.85 * massN).toFixed(3)})"></div>` +
      `<div class="pos">[${t}]</div>`;
    trow.appendChild(col);
  }

  // Circuit SVG: tall = active-layer count × ROW_H, not n_layers — collapses
  // dead vertical space when most layers have no circuit footprint.
  const circuit = $("circuit");
  circuit.innerHTML = "";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const W = PAD * 2 + N_TOK * COL_W;
  const H = PAD * 2 + N_ROWS * ROW_H;
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // Background column shading on hover-anchor token (drawn after).
  // Compute max abs node score for opacity normalisation in this prompt.
  let maxAbs = 0;
  prompt.nodes.forEach(n => {
    if (Math.abs(n.score) > maxAbs) maxAbs = Math.abs(n.score);
  });

  // Bucket nodes per (l, t) for cluster-jittering.
  const buckets = new Map();
  prompt.nodes.forEach(n => {
    const k = `${n.l}:${n.t}`;
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(n);
  });
  buckets.forEach(arr => {
    arr.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  });

  // Draw nodes.
  buckets.forEach((arr, key) => {
    const [li, t] = key.split(":").map(Number);
    const cx0 = PAD + t * COL_W + COL_W / 2;
    const cy0 = PAD + layerYIdx.get(li) * ROW_H + ROW_H / 2;
    arr.forEach((n, idx) => {
      // Spread within cell horizontally so they don't overlap.
      const offset = (idx - (arr.length - 1) / 2) * (NODE_R * 2 + 1);
      const cx = cx0 + offset;
      const cy = cy0;
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.classList.add("node");
      c.dataset.l = li;
      c.dataset.n = n.n;
      c.dataset.t = t;
      c.setAttribute("cx", cx);
      c.setAttribute("cy", cy);
      c.setAttribute("r", NODE_R);
      // If clusters are loaded, color nodes by cluster id; otherwise use sign.
      let fill = nodeColor(n.score, maxAbs);
      if (PAYLOAD.clusters && PAYLOAD.clusters.feat_to_cluster) {
        const cid = PAYLOAD.clusters.feat_to_cluster[`${li}:${n.n}`];
        if (cid !== undefined) fill = clusterColor(cid, n.score, maxAbs);
      }
      c.setAttribute("fill", fill);
      if (pinned.has(nodeKey(li, n.n))) c.classList.add("pinned");
      c.addEventListener("mousemove", (e) => onHover(e, n, li, t));
      c.addEventListener("mouseleave", onLeave);
      c.addEventListener("click", (e) => onClick(e, n, li));
      svg.appendChild(c);
    });
  });

  circuit.appendChild(svg);
}

function onHover(e, n, li, t) {
  hovered = { l: li, n: n.n };
  updateNodeStyles();
  showTooltip(e, n, li, t);
}
function onLeave() {
  hovered = null;
  hideTooltip();
  updateNodeStyles();
}
function onClick(e, n, li) {
  e.stopPropagation();
  const k = nodeKey(li, n.n);
  if (pinned.has(k)) pinned.delete(k);
  else pinned.add(k);
  updateNodeStyles();
  drawSide();
}
document.body.addEventListener("click", (e) => {
  // Background click: clear pins.
  if (e.target.closest(".node")) return;
  if (e.target.closest("aside")) return;
  if (pinned.size === 0) return;
  pinned.clear();
  updateNodeStyles();
  drawSide();
});

function updateNodeStyles() {
  // Spotlight: when hovering, dim everyone else (except pins). When pinned-only,
  // pins glow magenta and others normal.
  const all = document.querySelectorAll(".node");
  all.forEach(c => {
    c.classList.remove("dimmed", "hovered");
    const k = nodeKey(c.dataset.l, c.dataset.n);
    if (hovered && (parseInt(c.dataset.l, 10) !== hovered.l || parseInt(c.dataset.n, 10) !== hovered.n)) {
      if (!pinned.has(k)) c.classList.add("dimmed");
    }
    if (hovered && parseInt(c.dataset.l, 10) === hovered.l && parseInt(c.dataset.n, 10) === hovered.n) {
      c.classList.add("hovered");
    }
    if (pinned.has(k)) c.classList.add("pinned");
    else c.classList.remove("pinned");
  });
}

function showTooltip(e, n, li, t) {
  const sign = n.score >= 0 ? "+" : "−";
  const tokStr = (PAYLOAD.prompts[curPrompt].tokens[t] || "").replace(/\n/g, "↵");
  tooltip.innerHTML =
    `<b>L${li}/N${n.n}</b>  ${sign}${Math.abs(n.score).toFixed(3)}` +
    `<br><span style="color:#888">peak @ tok[${t}] "${escapeHtml(tokStr)}"</span>`;
  tooltip.style.display = "block";
  const rect = tooltip.getBoundingClientRect();
  let x = e.clientX + 12, y = e.clientY + 12;
  if (x + rect.width > window.innerWidth - 12) x = e.clientX - rect.width - 12;
  if (y + rect.height > window.innerHeight - 12) y = e.clientY - rect.height - 12;
  tooltip.style.left = Math.max(8, x) + "px";
  tooltip.style.top = Math.max(8, y) + "px";
}
function hideTooltip() { tooltip.style.display = "none"; }

function drawSide() {
  const side = $("side");
  side.innerHTML = "";

  // 0. Cluster summaries (ADAG output) — at the very top if present.
  if (PAYLOAD.clusters && Object.keys(PAYLOAD.clusters.feat_to_cluster || {}).length) {
    const head = document.createElement("h3");
    head.textContent = `clusters (k=${PAYLOAD.clusters.n_clusters})`;
    side.appendChild(head);
    // Count features per cluster from feat_to_cluster.
    const counts = {};
    Object.values(PAYLOAD.clusters.feat_to_cluster).forEach(cid => {
      counts[cid] = (counts[cid] || 0) + 1;
    });
    const ids = Object.keys(counts).map(Number).sort((a, b) => a - b);
    const wrap = document.createElement("div");
    wrap.className = "toplist";
    ids.forEach(cid => {
      const row = document.createElement("div");
      row.className = "row";
      const desc = (PAYLOAD.clusters.descriptions || {})[cid] || `cluster ${cid}`;
      const bg = clusterColor(cid, 1, 1);
      row.innerHTML =
        `<span class="swatch" style="background:${bg}"></span>` +
        `<span class="lab" style="font-family:inherit; font-size:11px; line-height:1.35">` +
          `<b>${cid}</b> · ${counts[cid]} feat — ` +
          `${escapeHtml(desc.length > 90 ? desc.slice(0, 87) + '…' : desc)}` +
        `</span>`;
      wrap.appendChild(row);
    });
    side.appendChild(wrap);
    if (PAYLOAD.clusters.metrics) {
      const m = PAYLOAD.clusters.metrics;
      const sm = document.createElement("div");
      sm.style.cssText = "color:#888; font-size:10px; margin: 4px 0 8px; font-family: ui-monospace, monospace";
      sm.textContent =
        `silhouette ${(m.silhouette ?? 0).toFixed(2)} · CV ${(m.coef_of_variation ?? 0).toFixed(2)} · opp-sign ${((m.pct_opposing_sign ?? 0)*100).toFixed(0)}%`;
      side.appendChild(sm);
    }
  }

  // 1. Pinned/hovered neuron details (dataset examples).
  const focus = [...pinned].map(k => k.split(":").map(Number));
  if (hovered) focus.push([hovered.l, hovered.n]);
  // de-dup
  const seen = new Set();
  const uniq = focus.filter(([l, n]) => {
    const k = `${l}:${n}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });

  if (uniq.length > 0) {
    const head = document.createElement("h3");
    head.textContent = uniq.length === 1 ? "selected neuron" : `${uniq.length} pinned`;
    side.appendChild(head);
    uniq.forEach(([li, ni]) => {
      const card = document.createElement("div");
      card.className = "neuron-card";
      const k = `${li}:${ni}`;
      const ndata = (PAYLOAD.neurons || {})[k] || {};
      const examples = ndata.examples || [];
      // Look up score/peak from current prompt
      const pn = PAYLOAD.prompts[curPrompt].nodes.find(x => x.l === li && x.n === ni);
      const scoreStr = pn ? `${pn.score >= 0 ? "+" : "−"}${Math.abs(pn.score).toFixed(3)}` : "—";
      let inner = `<div class="head">` +
                  `<span class="id">L${li}/N${ni}</span>` +
                  `<span class="score">${scoreStr}</span></div>`;
      if (examples.length === 0) {
        inner += `<div class="empty">no Dolma examples for this neuron</div>`;
      } else {
        examples.slice(0, 6).forEach(r => {
          inner += `<div class="ex"><span class="act">${r.act.toFixed(0)}</span>${r.html}</div>`;
        });
      }
      card.innerHTML = inner;
      side.appendChild(card);
    });
  }

  // 2. Top neurons in the current prompt, ordered by |score|.
  const head2 = document.createElement("h3");
  head2.textContent = `top neurons in prompt ${curPrompt + 1}`;
  head2.style.marginTop = uniq.length ? "16px" : "0";
  side.appendChild(head2);
  const tl = document.createElement("div");
  tl.className = "toplist";
  const sortedNodes = [...PAYLOAD.prompts[curPrompt].nodes].sort(
    (a, b) => Math.abs(b.score) - Math.abs(a.score)
  ).slice(0, SIDE_TOP_K);
  let maxAbs = 0;
  PAYLOAD.prompts[curPrompt].nodes.forEach(n => {
    if (Math.abs(n.score) > maxAbs) maxAbs = Math.abs(n.score);
  });
  sortedNodes.forEach(n => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<span class="swatch" style="background:${nodeColor(n.score, maxAbs)}"></span>` +
      `<span class="lab">L${n.l}/N${n.n}</span>` +
      `<span class="v">${n.score >= 0 ? "+" : "−"}${Math.abs(n.score).toFixed(3)}</span>`;
    row.addEventListener("click", () => {
      const k = nodeKey(n.l, n.n);
      if (pinned.has(k)) pinned.delete(k); else pinned.add(k);
      updateNodeStyles();
      drawSide();
    });
    tl.appendChild(row);
  });
  side.appendChild(tl);

  // 3. Empty-state cue.
  if (uniq.length === 0 && sortedNodes.length === 0) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "click a node in the circuit to inspect";
    side.appendChild(e);
  }
}

// ─── Cards-view (ADAG primary) ──────────────────────────────────────────

let selectedCluster = null;

function _parseTokenList(text) {
  // Lines look like:  "  - ' Texas'  (mass 143.20)"  or "  - ' Austin'  (contrib 36.85)"
  const out = [];
  for (const raw of (text || "").split("\n")) {
    const m = raw.match(/^\s*-\s*('([^']*)'|\"([^\"]*)\")\s*\((mass|contrib)\s+([0-9.]+)\)/);
    if (!m) continue;
    out.push({ token: m[2] !== undefined ? m[2] : m[3], value: parseFloat(m[5]) });
  }
  return out;
}

function clusterColorBg(cid, alpha) {
  const rgb = CLUSTER_PALETTE[((cid % CLUSTER_PALETTE.length) + CLUSTER_PALETTE.length) % CLUSTER_PALETTE.length];
  return rgba(rgb, alpha);
}

function renderCards() {
  const cards = (PAYLOAD.clusters && PAYLOAD.clusters.cards) || [];
  if (cards.length === 0) return false;

  const grid = $("cards-grid");
  grid.innerHTML = "";

  // Compute layer range per cluster from PAYLOAD.clusters.feat_to_cluster
  // (we already have that mapping; enumerate to find layer ids per cluster).
  const layersByCluster = new Map();
  Object.entries(PAYLOAD.clusters.feat_to_cluster).forEach(([key, cid]) => {
    const li = parseInt(key.split(":")[0], 10);
    if (!layersByCluster.has(cid)) layersByCluster.set(cid, []);
    layersByCluster.get(cid).push(li);
  });
  const nLayers = (PAYLOAD.config && PAYLOAD.config.n_layers) || 32;

  const maxScore = cards[0].aggregate_score || 1;
  cards.forEach(card => {
    const div = document.createElement("div");
    div.className = "cluster-card";
    div.dataset.cid = card.cluster_id;

    const scorePct = (card.aggregate_score / maxScore) * 100;
    const swatchBg = clusterColorBg(card.cluster_id, 0.95);

    const layers = layersByCluster.get(card.cluster_id) || [];
    const lMin = layers.length ? Math.min(...layers) : 0;
    const lMax = layers.length ? Math.max(...layers) : 0;
    // Layer histogram: bucketize layers into 16 bins across the full layer range.
    const bins = 16;
    const binCounts = new Array(bins).fill(0);
    layers.forEach(l => {
      const bi = Math.min(bins - 1, Math.floor((l / (nLayers - 1)) * bins));
      binCounts[bi]++;
    });
    const maxBin = Math.max(1, ...binCounts);
    const binsHtml = binCounts.map((c, i) => {
      if (c === 0) return "";
      const left = (i / bins) * 100;
      const w = 100 / bins;
      const h = 8 * (c / maxBin);
      return `<div class="layer-bar" style="left:${left.toFixed(2)}%; width:${w.toFixed(2)}%; height:${h.toFixed(1)}px; background:${swatchBg}"></div>`;
    }).join("");

    // Top input/output token chips
    const inToks = _parseTokenList(card.input_token_summary).slice(0, 6);
    const outToks = _parseTokenList(card.output_token_summary).slice(0, 5);
    const inMaxV = inToks.length ? inToks[0].value : 1;
    const outMaxV = outToks.length ? outToks[0].value : 1;
    const inChipsHtml = inToks.map(t =>
      `<span class="chip in">${escapeHtml((t.token || "").replace(/^ /, "·") || "·")}<span class="v">${t.value.toFixed(0)}</span></span>`
    ).join("") || `<span style="color:#bbb;font-size:11px">—</span>`;
    const outChipsHtml = outToks.map(t =>
      `<span class="chip out">→ ${escapeHtml((t.token || "").replace(/^ /, "·") || "·")}<span class="v">${t.value.toFixed(0)}</span></span>`
    ).join("") || `<span style="color:#bbb;font-size:11px">—</span>`;

    const desc = card.description && card.description.length
      ? `<div class="descr">${escapeHtml(card.description)}</div>`
      : `<div class="descr placeholder">no description (run scripts/run_describe.py)</div>`;

    // Optional eval-vs-deploy differential badge.
    let diffBadge = "";
    if (card.group_differential) {
      const d = card.group_differential;
      const lean = d.delta > 0 ? "eval" : "deploy";
      const color = d.delta > 0 ? "#1976d2" : "#f57c00";
      const sign = d.delta >= 0 ? "+" : "−";
      diffBadge =
        `<span style="margin-left:8px; padding:1px 7px; border-radius:9px; ` +
        `background:${color}; color:white; font-size:10px; font-family:ui-monospace,monospace; ` +
        `font-weight:600">${lean} ${sign}${Math.abs(d.delta).toFixed(2)}</span>`;
    }

    div.innerHTML =
      `<div class="card-head">` +
        `<span class="swatch" style="background:${swatchBg}"></span>` +
        `<span class="cid">cluster ${card.cluster_id}</span>` +
        `<span class="nfeat">${card.n_features} feat</span>` +
        diffBadge +
        `<span class="score-bar"><span class="score-fill" style="width:${scorePct}%"></span></span>` +
        `<span class="score-num">${card.aggregate_score.toFixed(2)}</span>` +
      `</div>` +
      desc +
      `<div class="layer-strip">${binsHtml}` +
        `<span class="scale" style="left:0">L${lMin}</span>` +
        `<span class="scale" style="right:0">L${lMax}</span>` +
      `</div>` +
      `<div class="row-label">input tokens (mass)</div>` +
      `<div class="chips">${inChipsHtml}</div>` +
      `<div class="row-label">output tokens (contribution)</div>` +
      `<div class="chips">${outChipsHtml}</div>`;

    div.addEventListener("click", () => selectCluster(card.cluster_id));
    grid.appendChild(div);
  });

  if (selectedCluster === null && cards.length) {
    selectCluster(cards[0].cluster_id);
  } else {
    drawDetail();
  }
  return true;
}

function selectCluster(cid) {
  selectedCluster = cid;
  document.querySelectorAll(".cluster-card").forEach(el => {
    el.classList.toggle("selected", parseInt(el.dataset.cid, 10) === cid);
  });
  drawDetail();
}

function drawDetail() {
  const detail = $("cards-detail");
  detail.innerHTML = "";
  if (selectedCluster === null) return;
  const card = (PAYLOAD.clusters.cards || []).find(c => c.cluster_id === selectedCluster);
  if (!card) return;

  const head = document.createElement("h3");
  head.textContent = `cluster ${card.cluster_id}  ·  ${card.n_features} features`;
  detail.appendChild(head);

  const fullDescr = document.createElement("div");
  fullDescr.className = "full-descr";
  if (card.description && card.description.length) {
    fullDescr.textContent = card.description;
  } else {
    fullDescr.innerHTML = `<i style="color:#aaa">No LLM description for this cluster yet. Run <code>scripts/run_describe.py</code>.</i>`;
  }
  detail.appendChild(fullDescr);

  const featureSection = document.createElement("h3");
  featureSection.textContent = "member neurons (top 10)";
  detail.appendChild(featureSection);

  // Show top 10 member neurons by aggregate score.
  const memberKeys = card.feature_keys || [];
  // Look up each feature's score from any prompt's nodes payload (use prompt 0).
  const scoreLookup = {};
  if (PAYLOAD.prompts && PAYLOAD.prompts[0]) {
    PAYLOAD.prompts[0].nodes.forEach(n => {
      scoreLookup[`${n.l}:${n.n}`] = n.score;
    });
  }
  const sorted = memberKeys
    .map(([li, ni]) => ({ li, ni, score: scoreLookup[`${li}:${ni}`] || 0 }))
    .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
    .slice(0, 10);

  if (sorted.length === 0) {
    const e = document.createElement("div");
    e.className = "empty-state";
    e.textContent = "no member neurons";
    detail.appendChild(e);
  } else {
    sorted.forEach(({ li, ni, score }) => {
      const k = `${li}:${ni}`;
      const ndata = (PAYLOAD.neurons || {})[k] || {};
      const examples = (ndata.examples || []).slice(0, 4);
      const sign = score >= 0 ? "+" : "−";
      const card = document.createElement("div");
      card.className = "neuron-card";
      let inner =
        `<div class="head">` +
          `<span class="id">L${li}/N${ni}</span>` +
          `<span class="score">${sign}${Math.abs(score).toFixed(3)}</span>` +
        `</div>`;
      if (examples.length === 0) {
        inner += `<div class="empty-state">no Dolma examples</div>`;
      } else {
        examples.forEach(r => {
          inner += `<div class="ex"><span class="act">${r.act.toFixed(0)}</span>${r.html}</div>`;
        });
      }
      card.innerHTML = inner;
      detail.appendChild(card);
    });
  }

  // Quality metrics footer.
  const m = (PAYLOAD.clusters.metrics || {});
  if (Object.keys(m).length) {
    const mb = document.createElement("div");
    mb.className = "metrics-block";
    mb.innerHTML =
      `clustering: silhouette ${(m.silhouette ?? 0).toFixed(3)} · ` +
      `CV ${(m.coef_of_variation ?? 0).toFixed(3)} · ` +
      `opp-sign ${((m.pct_opposing_sign ?? 0) * 100).toFixed(1)}%`;
    detail.appendChild(mb);
  }
}

renderHeader();
const hasClusters = PAYLOAD.clusters && (PAYLOAD.clusters.cards || []).length > 0;
if (hasClusters) {
  $("cards-stage").style.display = "grid";
  $("dots-stage").style.display = "none";
  renderCards();
  // Hide the prompt selector since cluster cards aggregate across all prompts.
  document.querySelector(".pair-controls").style.display = "none";
  document.querySelector("header .pred").style.display = "none";
} else {
  goPrompt(0);
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------

def render_circuit_html(
    attr_dir: Path,
    output: Path,
    acts_dir: Path | None = None,
    tokenizer_id: str = "allenai/Olmo-3-7B-Think",
    topk_neurons: int = 256,
    topk_examples: int = 8,
    context_tokens: int = 12,
    title: str | None = None,
    prompts: list[dict] | None = None,
    selection: str = "auto",
    n_bands: int = 4,           # legacy arg; ignored by the new layout
    top_per_cell: int = 6,       # legacy arg; ignored
    clusters_path: Path | None = None,
    cluster_descriptions_path: Path | None = None,
) -> None:
    """Single-page circuit HTML. Each prompt shows neurons as positioned dots in
    a (token × layer) grid, with the focused neuron's dataset spans on the
    right. Reads either tau_circuit.json (preferred) or topk.json."""
    scores = np.load(attr_dir / "scores.npy")
    n_layers, d_ffn = scores.shape
    cfg = json.loads((attr_dir / "config.json").read_text())
    cfg["n_layers"] = n_layers
    cfg["d_ffn"] = d_ffn

    tau_path = attr_dir / "tau_circuit.json"
    has_tau = tau_path.exists()
    if selection == "auto":
        selection = "tau" if has_tau else "topk"
    if selection == "tau" and not has_tau:
        raise FileNotFoundError(f"selection=tau but no tau_circuit.json under {attr_dir}")

    if selection == "tau":
        tau_payload = json.loads(tau_path.read_text())
        cfg["tau"] = tau_payload.get("tau")
        cfg["selection"] = "tau"
        tau_circuit = tau_payload["neurons"]
        ablation_file = attr_dir / "ablation_tau.json"
        if not ablation_file.exists():
            ablation_file = attr_dir / "ablation.json"
        ablation = json.loads(ablation_file.read_text())
    else:
        tau_circuit = json.loads((attr_dir / "topk.json").read_text())[:topk_neurons]
        cfg["selection"] = "topk"
        # topk.json items already have layer/neuron/score; treat as ad-hoc τ-list
        tau_circuit = [{"layer": e["layer"], "neuron": e["neuron"]} for e in tau_circuit]
        ablation = json.loads((attr_dir / "ablation.json").read_text())

    pp_token_path = attr_dir / "per_pair_token_scores.npy"
    pp_lengths_path = attr_dir / "per_pair_lengths.npy"
    pp_metric_path = attr_dir / "per_pair_metric.npy"
    pp_token_ids_path = attr_dir / "per_pair_token_ids.json"
    if not all(p.exists() for p in (pp_token_path, pp_lengths_path, pp_metric_path, pp_token_ids_path)):
        raise FileNotFoundError(
            f"{attr_dir} is missing per-prompt token attribution outputs. "
            f"This viz requires per_prompt_token_scores.npy, per_prompt_lengths.npy, "
            f"per_prompt_metric.npy, per_prompt_token_ids.json — all written by "
            f"run_attribution.py for prompted RelP and unprompted RelP only."
        )

    tok = _load_tokenizer(tokenizer_id)
    ppts = np.load(pp_token_path)
    ppl = np.load(pp_lengths_path)
    ppm = np.load(pp_metric_path)
    pptid = json.loads(pp_token_ids_path.read_text())
    prompts_data, needed = _build_prompts_payload(
        ppts, ppl, pptid, ppm, prompts, tok,
        tau_circuit=tau_circuit,
        max_neurons_per_prompt=topk_neurons,
    )

    neurons_payload: dict[str, dict] = {}
    if acts_dir is not None and needed:
        ex_map = _build_examples_for_neurons(
            acts_dir, tok, sorted(needed),
            topk_examples=topk_examples, context_tokens=context_tokens,
        )
        for (li, ni), recs in ex_map.items():
            neurons_payload[f"{li}:{ni}"] = {
                "layer": li,
                "neuron": ni,
                "examples": recs,
            }

    # Optional ADAG cluster overlay. When clusters.json + cluster_descriptions.json
    # + cluster_contexts.json are all present, we promote the cards-view to be
    # the primary layout.
    clusters_blob: dict | None = None
    cluster_path = clusters_path or (attr_dir / "clusters.json")
    if cluster_path.exists():
        clusters_data = json.loads(cluster_path.read_text())
        feat_to_cluster: dict[str, int] = {}
        for (li, ni), cid in zip(
            clusters_data["feature_keys"], clusters_data["cluster_ids"], strict=True
        ):
            feat_to_cluster[f"{int(li)}:{int(ni)}"] = int(cid)
        descriptions: dict[int, str] = {}
        desc_path = cluster_descriptions_path or (attr_dir / "cluster_descriptions.json")
        if desc_path.exists():
            for entry in json.loads(desc_path.read_text()):
                descriptions[int(entry["cluster_id"])] = entry.get("description", "")
        cards: list[dict] = []
        ctx_path = attr_dir / "cluster_contexts.json"
        if ctx_path.exists():
            for ctx in json.loads(ctx_path.read_text()):
                cid = int(ctx["cluster_id"])
                cards.append({
                    "cluster_id": cid,
                    "n_features": ctx["n_features"],
                    "aggregate_score": ctx["aggregate_score"],
                    "feature_keys": [list(k) for k in ctx["feature_keys"]],
                    "input_token_summary": ctx["input_token_summary"],
                    "output_token_summary": ctx["output_token_summary"],
                    "description": descriptions.get(cid, ""),
                    "group_differential": ctx.get("group_differential"),
                    "differential_summary": ctx.get("differential_summary", ""),
                })
            cards.sort(key=lambda c: -c["aggregate_score"])
        clusters_blob = {
            "n_clusters": clusters_data["n_clusters"],
            "feat_to_cluster": feat_to_cluster,
            "metrics": clusters_data.get("metrics", {}),
            "descriptions": descriptions,
            "cards": cards,
        }

    payload = {
        "title": title or f"{cfg.get('task', 'circuit')} / {cfg.get('method', '?')}",
        "config": cfg,
        "metrics": ablation,
        "prompts": prompts_data,
        "neurons": neurons_payload,
        "clusters": clusters_blob,
    }
    title_safe = html_lib.escape(payload["title"])
    out = _TEMPLATE.replace("__TITLE__", title_safe).replace("__PAYLOAD__", json.dumps(payload))
    output.write_text(out)
