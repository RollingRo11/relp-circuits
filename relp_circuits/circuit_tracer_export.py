"""Translate our (RelP attribution + CLJA edges + clustering) artifacts into
the JSON schema expected by Transluce's circuit-tracer frontend (adapted
from Anthropic's `circuit-tracer` repo).

Produces:
  * One graph_data/<slug>__<i>.json per prompt
  * One graph-metadata.json indexing all of them

The schema (see circuits/frontend/graph_models.py upstream):

  Metadata: slug, scan, transcoder_list, prompt_tokens, prompt,
            node_threshold, schema_version, target_logit_tokens
  QParams:  pinnedIds, supernodes, linkType, clickedId, sg_pos
  Node:     node_id (string), feature (int), layer (string), ctx_idx,
            feature_type ∈ {"embedding","cross layer transcoder","logit",
            "mlp reconstruction error"}, jsNodeId, clerp, influence,
            activation, attr_map, contrib_map, ...
  Link:     source, target, weight

We map our concepts as follows:

  * τ-circuit feature (layer L, neuron N) at the target token position →
    a "cross layer transcoder" node with node_id = f"{L}_{N}_{target_pos}"
    and clerp = the cluster description (if any).
  * Each input token position → an "embedding" node E_{vocab}_{pos}.
  * Each top-K target logit → a "logit" node {num_layers+1}_{vocab}_{tp}.
  * CLJA edge (src_feature → tgt_feature) → Link with absolute JVP weight.
  * Embed→feature link weight: input_attr[p, f, t]
  * Feature→logit link weight:  output_contrib[p, f, k]

Per-prompt graphs are pruned: only the top-K incoming and outgoing edges
per node are kept (configurable), plus a global threshold on |weight|.
This keeps each JSON manageable (~100-500 KB per prompt) and the graph
readable in the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExportConfig:
    out_dir: Path
    slug_prefix: str = "circuit"
    scan: str = "model"
    edge_top_k_per_target: int = 20
    embed_top_k_per_feature: int = 8
    logit_edges: bool = True
    feature_threshold: float = 1e-3
    edge_threshold: float = 1e-3


def _cantor(x: int, y: int) -> int:
    return (x + y) * (x + y + 1) // 2 + y


def _synth_neuron_exemplars_into(attr_dir: Path, out_dir: Path, tokenizer) -> None:
    """Synthesize a `data/neuron_exemplars.json` payload from the per-pair
    attribution profile, so the Transluce frontend can render feature-detail
    panels without a remote `./features/<scan>/<fidx>.json` fetch (which fails
    on `file://` opens and isn't relevant for our setup anyway).

    For each τ-circuit feature (layer L, neuron N) we emit two entries keyed
    `L_N_+` and `L_N_-` (matching the `/api/neuron_exemplars?sign=±` requests
    the frontend issues). Each entry has the schema the frontend expects:

      {
        "act_min": 0, "act_max": <max abs act on prompt>,
        "examples_quantiles": [
          {"quantile_name": "Demo prompt",
           "examples": [{"tokens": [...], "tokens_acts_list": [...],
                          "train_token_ind": <argmax>,
                          "is_repeated_datapoint": false}]}
        ],
        "top_logits": [...token strings ranked by +contrib...],
        "bottom_logits": [...token strings ranked by -contrib...]
      }

    Activations come from `per_pair_input_attr` (input attribution per token);
    we split sign so the `+` entry shows tokens where the feature responds
    positively, and `-` shows the negative side. Top/bottom logits come from
    `per_pair_output_contrib` (× `top_k_logit_ids`).

    Skips silently if any required input file is missing — this keeps the
    pipeline robust when run on an attribution dir produced before the
    ADAG-profiles step.
    """
    needed = [
        attr_dir / "per_pair_input_attr.npy",
        attr_dir / "per_pair_output_contrib.npy",
        attr_dir / "top_k_logit_ids.npy",
        attr_dir / "per_pair_lengths.npy",
        attr_dir / "tau_circuit.json",
    ]
    if not all(p.exists() for p in needed):
        return

    tau = json.loads((attr_dir / "tau_circuit.json").read_text())["neurons"]
    input_attr = np.load(attr_dir / "per_pair_input_attr.npy")          # (P, F, T)
    output_contrib = np.load(attr_dir / "per_pair_output_contrib.npy")  # (P, F, K)
    top_k_ids = np.load(attr_dir / "top_k_logit_ids.npy")               # (P, K)
    lengths = np.load(attr_dir / "per_pair_lengths.npy")                # (P,)

    pair_ids_path = attr_dir / "per_pair_token_ids.json"
    pair_strs_path = attr_dir / "pair_token_strs.json"
    if pair_strs_path.exists():
        pair_token_strs: list[list[str]] = json.loads(pair_strs_path.read_text())
    elif pair_ids_path.exists():
        ids = json.loads(pair_ids_path.read_text())
        pair_token_strs = [[tokenizer.decode([t]) for t in row] for row in ids]
    else:
        return

    P, F, T_max = input_attr.shape
    F_circuit = min(F, len(tau))
    feature_idx = [(int(e["layer"]), int(e["neuron"])) for e in tau[:F_circuit]]

    payload: dict[str, dict] = {}
    for fi, (li, ni) in enumerate(feature_idx):
        # Pool across prompts: pick the prompt where this feature is most active
        # in magnitude. With 1 prompt (boundaries demo) this is just p=0.
        feat = input_attr[:, fi, :]                          # (P, T)
        abs_per_pair = np.nanmax(np.abs(feat), axis=1)       # (P,)
        if not np.isfinite(abs_per_pair).any():
            continue
        p = int(np.nanargmax(abs_per_pair))
        T = int(lengths[p])
        tokens = pair_token_strs[p][:T]
        if len(tokens) != T:
            tokens = (tokens + [""] * T)[:T]
        attr = feat[p, :T].astype(np.float64)

        for sign, side_acts in [("+", np.maximum(attr, 0.0)),
                                 ("-", np.maximum(-attr, 0.0))]:
            act_max = float(side_acts.max()) if side_acts.size else 0.0
            train_idx = int(np.argmax(side_acts)) if side_acts.size else 0

            contrib = output_contrib[p, fi]                  # (K,)
            top_ids_p = top_k_ids[p]
            if sign == "+":
                order = np.argsort(-contrib)
                pos_contrib = [int(top_ids_p[i]) for i in order if contrib[i] > 0]
                neg_contrib = [int(top_ids_p[i]) for i in order[::-1] if contrib[i] < 0]
            else:
                order = np.argsort(contrib)
                pos_contrib = [int(top_ids_p[i]) for i in order if contrib[i] < 0]
                neg_contrib = [int(top_ids_p[i]) for i in order[::-1] if contrib[i] > 0]
            top_logits = [tokenizer.decode([tid]) for tid in pos_contrib[:8]]
            bottom_logits = [tokenizer.decode([tid]) for tid in neg_contrib[:8]]

            payload[f"{li}_{ni}_{sign}"] = {
                "act_min": 0.0,
                "act_max": act_max if act_max > 0 else 1.0,
                "examples_quantiles": [
                    {
                        "quantile_name": f"On demo prompt (peak act {act_max:.2f})",
                        "examples": [
                            {
                                "tokens": tokens,
                                "tokens_acts_list": side_acts.tolist(),
                                "train_token_ind": train_idx,
                                "is_repeated_datapoint": False,
                            },
                        ],
                    },
                ],
                "top_logits": top_logits,
                "bottom_logits": bottom_logits,
            }

    (out_dir / "data" / "neuron_exemplars.json").write_text(json.dumps(payload))


def _build_one_graph(
    p: int,
    *,
    slug: str,
    scan: str,
    pair_token_ids: list[int],
    pair_token_strs: list[str],
    target_pos: int,
    feature_idx: list[tuple[int, int]],
    input_attr_p: np.ndarray,        # (F, T)
    output_contrib_p: np.ndarray,    # (F, K)
    top_k_logit_ids_p: np.ndarray,   # (K,)
    top_k_logit_vals_p: np.ndarray,  # (K,)
    edges_p: np.ndarray,             # (F, F) src -> tgt
    cluster_ids: np.ndarray | None,  # (F,) integer cluster id per feature
    cluster_descriptions: dict[int, str] | None,
    num_layers: int,
    decode_token_fn,
    cfg: ExportConfig,
) -> dict:
    F = len(feature_idx)
    T = len(pair_token_ids)

    # ── Nodes ──
    nodes: list[dict] = []
    node_id_set: set[str] = set()

    # Embedding nodes (one per token).
    for t, vid in enumerate(pair_token_ids):
        nid = f"E_{vid}_{t}"
        if nid in node_id_set:
            continue
        nodes.append({
            "node_id": nid,
            "jsNodeId": f"E_{vid}-{t}",
            "feature": t,
            "layer": "E",
            "ctx_idx": t,
            "feature_type": "embedding",
            "clerp": pair_token_strs[t],
            "influence": float(np.abs(input_attr_p[:, t]).sum()),
            "activation": 1.0,
        })
        node_id_set.add(nid)

    # Feature nodes (cross layer transcoder), one per τ-feature at the tgt pos.
    feature_node_id: dict[int, str] = {}
    feature_max_score = float(np.abs(input_attr_p).sum() + np.abs(output_contrib_p).sum() + 1e-12)
    for fi, (li, ni) in enumerate(feature_idx):
        nid = f"{li}_{ni}_{target_pos}"
        if nid in node_id_set:
            feature_node_id[fi] = nid
            continue
        # Influence = total absolute mass through this feature.
        infl = float(np.abs(input_attr_p[fi]).sum() + np.abs(output_contrib_p[fi]).sum())
        if infl < cfg.feature_threshold:
            continue
        clerp = ""
        if cluster_ids is not None and cluster_descriptions is not None:
            cid = int(cluster_ids[fi])
            desc = cluster_descriptions.get(cid, "")
            if desc:
                clerp = f"C{cid}: {desc[:120]}"
            else:
                clerp = f"L{li} N{ni}"
        else:
            clerp = f"L{li} N{ni}"
        attr_map = input_attr_p[fi].tolist()
        contrib_map = output_contrib_p[fi].tolist()
        nodes.append({
            "node_id": nid,
            "jsNodeId": f"{li}_{ni}-0",
            "feature": _cantor(li, ni),
            "layer": str(li),
            "ctx_idx": int(target_pos),
            "feature_type": "cross layer transcoder",
            "clerp": clerp,
            "influence": infl,
            "activation": float(np.abs(output_contrib_p[fi]).sum()),
            "attr_map": attr_map,
            "contrib_map": contrib_map,
        })
        node_id_set.add(nid)
        feature_node_id[fi] = nid

    # Logit nodes (top-K target logits).
    # The Transluce frontend parses the token probability out of the clerp via
    # `d.clerp.split('(p=')[1].split(')')[0]`. If the clerp doesn't contain
    # `(p=<num>)` exactly, formatData crashes with `Cannot read properties of
    # undefined (reading 'split')`. We compute softmax probabilities locally
    # over the top-K logit values (this is the conditional probability among
    # the top-K, not the full-vocab probability, which is fine for ranking).
    top_logits = np.asarray(top_k_logit_vals_p, dtype=np.float64)
    shifted = top_logits - top_logits.max()
    exps = np.exp(shifted)
    top_probs = (exps / exps.sum()).astype(np.float64)
    logit_node_ids: list[str] = []
    target_logit_tokens: list[str] = []
    for k, vid in enumerate(top_k_logit_ids_p):
        vid_int = int(vid)
        nid = f"{num_layers + 1}_{vid_int}_{target_pos}"
        tok = decode_token_fn(vid_int)
        target_logit_tokens.append(tok)
        if nid in node_id_set:
            logit_node_ids.append(nid)
            continue
        prob_k = float(top_probs[k])
        infl = float(np.abs(output_contrib_p[:, k]).sum())
        nodes.append({
            "node_id": nid,
            "jsNodeId": f"L_{vid_int}-{target_pos}",
            "feature": vid_int,
            "layer": str(num_layers + 1),
            "ctx_idx": int(target_pos),
            "feature_type": "logit",
            "clerp": f'Output "{tok}" (p={prob_k:.3f})',
            "influence": infl,
            "activation": float(top_k_logit_vals_p[k]),
            "is_target_logit": (k == 0),
            "token_prob": prob_k,
        })
        node_id_set.add(nid)
        logit_node_ids.append(nid)

    # ── Links ──
    links: list[dict] = []

    # Embed → feature: top-K embed sources per feature.
    for fi in range(F):
        if fi not in feature_node_id:
            continue
        attr = input_attr_p[fi]                                      # (T,)
        if attr.size == 0:
            continue
        order = np.argsort(-np.abs(attr))[: cfg.embed_top_k_per_feature]
        for t in order:
            t = int(t)
            w = float(attr[t])
            if abs(w) < cfg.edge_threshold:
                continue
            vid = pair_token_ids[t]
            src = f"E_{vid}_{t}"
            if src not in node_id_set:
                continue
            links.append({"source": src, "target": feature_node_id[fi], "weight": w})

    # Feature → feature: CLJA edges, top-K incoming per target.
    for tgt_fi in range(F):
        if tgt_fi not in feature_node_id:
            continue
        col = edges_p[:, tgt_fi]                                      # (F,)
        if not np.any(np.abs(col) > cfg.edge_threshold):
            continue
        order = np.argsort(-np.abs(col))[: cfg.edge_top_k_per_target]
        for src_fi in order:
            src_fi = int(src_fi)
            w = float(col[src_fi])
            if abs(w) < cfg.edge_threshold or src_fi == tgt_fi:
                continue
            if src_fi not in feature_node_id:
                continue
            links.append({
                "source": feature_node_id[src_fi],
                "target": feature_node_id[tgt_fi],
                "weight": w,
            })

    # Feature → logit: each top-K logit gets its incoming contribs.
    if cfg.logit_edges:
        for k, lid in enumerate(logit_node_ids):
            col = output_contrib_p[:, k]                              # (F,)
            if not np.any(np.abs(col) > cfg.edge_threshold):
                continue
            order = np.argsort(-np.abs(col))[: cfg.edge_top_k_per_target]
            for src_fi in order:
                src_fi = int(src_fi)
                if src_fi not in feature_node_id:
                    continue
                w = float(col[src_fi])
                if abs(w) < cfg.edge_threshold:
                    continue
                links.append({
                    "source": feature_node_id[src_fi],
                    "target": lid,
                    "weight": w,
                })

    # ── QParams (cluster supernodes, pinned ids, etc.) ──
    pinned_ids: list[str] = []
    supernodes: list[list[str]] = []
    if cluster_ids is not None and cluster_descriptions is not None:
        cluster_to_node_ids: dict[int, list[str]] = {}
        for fi in range(F):
            if fi not in feature_node_id:
                continue
            cid = int(cluster_ids[fi])
            cluster_to_node_ids.setdefault(cid, []).append(feature_node_id[fi])
        for cid, nids in sorted(cluster_to_node_ids.items()):
            label = cluster_descriptions.get(cid, "")
            head = f"C{cid}"
            if label:
                head = f"{head}: {label[:80]}"
            supernodes.append([head] + nids)
            pinned_ids.extend(nids)

    # Always pin the first logit node so the UI has a starting "clicked".
    clicked_id = logit_node_ids[0] if logit_node_ids else ""
    if clicked_id and clicked_id not in pinned_ids:
        pinned_ids.append(clicked_id)

    # ── Metadata ──
    influences = [n["influence"] for n in nodes if n.get("influence")]
    node_threshold = (max(influences) * 0.5) if influences else None

    metadata = {
        "slug": slug,
        "scan": scan,
        "transcoder_list": [],
        "prompt_tokens": pair_token_strs,
        "prompt": "".join(pair_token_strs),
        "node_threshold": node_threshold,
        "schema_version": 1,
        "target_logit_tokens": target_logit_tokens,
    }

    qparams = {
        "pinnedIds": pinned_ids,
        "supernodes": supernodes,
        "linkType": "both",
        "clickedId": clicked_id,
        "sg_pos": "",
    }

    return {
        "metadata": metadata,
        "qParams": qparams,
        "nodes": nodes,
        "links": links,
    }


def export_circuit_tracer(
    *,
    attr_dir: Path,
    out_dir: Path,
    tokenizer,
    num_layers: int,
    cfg: ExportConfig | None = None,
    pair_indices: list[int] | None = None,
    log: bool = True,
) -> list[Path]:
    """Read attr_dir/* and emit graph_data/*.json + graph-metadata.json.

    `attr_dir` must contain (from previous pipeline steps):
      tau_circuit.json
      per_pair_input_attr.npy        (P, F, T_max)
      per_pair_output_contrib.npy    (P, F, K)
      top_k_logit_ids.npy            (P, K)
      top_k_logit_vals.npy           (P, K)
      per_pair_token_ids.json        list of P lists of int
      per_pair_lengths.npy           (P,)
      edges.npz                      with key "edges" (P, F, F)
      clusters.json                  optional
      cluster_descriptions.json      optional
    """
    cfg = cfg or ExportConfig(out_dir=out_dir)
    cfg.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_data").mkdir(exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)

    _synth_neuron_exemplars_into(attr_dir, out_dir, tokenizer)

    # Load artifacts.
    tau_circuit = json.loads((attr_dir / "tau_circuit.json").read_text())["neurons"]

    input_attr = np.load(attr_dir / "per_pair_input_attr.npy")
    output_contrib = np.load(attr_dir / "per_pair_output_contrib.npy")
    top_k_logit_ids = np.load(attr_dir / "top_k_logit_ids.npy")
    top_k_logit_vals = np.load(attr_dir / "top_k_logit_vals.npy")
    per_pair_token_ids = json.loads((attr_dir / "per_pair_token_ids.json").read_text())
    per_pair_lengths = np.load(attr_dir / "per_pair_lengths.npy")

    # CLJA edges are optional. If `edges.npz` is missing (e.g. the user skipped
    # run_clja_edges for a quick demo), we substitute a zero edge matrix so the
    # graph still renders with embed→feature and feature→logit links. The
    # feature ordering then comes straight from the τ-circuit.
    edges_path = attr_dir / "edges.npz"
    if edges_path.exists():
        edges_npz = np.load(edges_path)
        edges = edges_npz["edges"]                                 # (P, F_full, F_full)
        feature_idx_arr = edges_npz["feature_idx"]                 # (F, 2)
        feature_idx = [(int(li), int(ni)) for li, ni in feature_idx_arr]

        F_edges = edges.shape[1]
        feature_keys_in_edges = set(feature_idx)
        keep = [i for i, e in enumerate(tau_circuit)
                if (int(e["layer"]), int(e["neuron"])) in feature_keys_in_edges]
        if len(keep) != F_edges:
            order_in_circuit = {(int(e["layer"]), int(e["neuron"])): i for i, e in enumerate(tau_circuit)}
            keep = [order_in_circuit[k] for k in feature_idx if k in order_in_circuit]

        F_profile = input_attr.shape[1]
        if F_profile == len(tau_circuit):
            keep_mask = np.array(keep, dtype=np.int64) if keep else np.arange(F_edges)
            input_attr = input_attr[:, keep_mask]
            output_contrib = output_contrib[:, keep_mask]
        F = input_attr.shape[1]
        if F != F_edges:
            raise RuntimeError(
                f"feature count mismatch between profiles ({F}) and edges ({F_edges})")
    else:
        if log:
            print(f"[export] {edges_path.name} missing — rendering without feature→feature edges",
                  flush=True)
        # Feature index = τ-circuit order, capped at the profile width.
        F = input_attr.shape[1]
        feature_idx = [(int(e["layer"]), int(e["neuron"])) for e in tau_circuit[:F]]
        if len(feature_idx) < F:
            # Profiles wider than the persisted circuit (shouldn't happen); pad.
            feature_idx = feature_idx + [(0, 0)] * (F - len(feature_idx))
        P_arr = input_attr.shape[0]
        edges = np.zeros((P_arr, F, F), dtype=np.float32)

    P = input_attr.shape[0]
    if pair_indices is None:
        pair_indices = list(range(P))

    # Optional: clusters + descriptions.
    cluster_ids = None
    cluster_descriptions: dict[int, str] = {}
    cl_path = attr_dir / "clusters.json"
    if cl_path.exists():
        cl = json.loads(cl_path.read_text())
        cluster_ids_full = np.array(cl["cluster_ids"], dtype=np.int32)
        # Subset to the same feature_idx (clusters were computed on the same
        # capped circuit, so their feature_keys should match feature_idx_arr).
        cl_keys = [tuple(k) for k in cl["feature_keys"]]
        if cl_keys == feature_idx:
            cluster_ids = cluster_ids_full
        else:
            # Fallback: align by (layer, neuron) lookup.
            cl_lookup = {tuple(k): cluster_ids_full[i] for i, k in enumerate(cl_keys)}
            cluster_ids = np.array(
                [int(cl_lookup.get((li, ni), -1)) for li, ni in feature_idx],
                dtype=np.int32,
            )

    desc_path = attr_dir / "cluster_descriptions.json"
    if desc_path.exists():
        for d in json.loads(desc_path.read_text()):
            cid = int(d["cluster_id"])
            # New ADAG schema (post-paper-rewrite): two-explainer outputs + Pearson r.
            # Old combined-describer schema kept the single "description" key.
            if "input_description" in d or "output_description" in d:
                in_desc = d.get("input_description", "").strip()
                out_desc = d.get("output_description", "").strip()
                r_in = d.get("input_simulator_pearson_r")
                r_out = d.get("output_simulator_pearson_r")
                pieces: list[str] = []
                if in_desc:
                    suf = f" (r={r_in:+.2f})" if isinstance(r_in, (int, float)) else ""
                    pieces.append(f"IN: {in_desc}{suf}")
                if out_desc:
                    suf = f" (r={r_out:+.2f})" if isinstance(r_out, (int, float)) else ""
                    pieces.append(f"OUT: {out_desc}{suf}")
                cluster_descriptions[cid] = " — ".join(pieces)
            else:
                cluster_descriptions[cid] = d.get("description", "")

    # Per-pair target_pos: re-derive as length-1 (matches our PairedPrompt).
    target_positions = (per_pair_lengths.astype(np.int64) - 1).tolist()

    written: list[Path] = []
    graph_metadata_entries: list[dict] = []

    def decode(tid: int) -> str:
        return tokenizer.decode([tid])

    for p in pair_indices:
        T = int(per_pair_lengths[p])
        pair_ids = per_pair_token_ids[p][:T]
        pair_strs = [decode(t) for t in pair_ids]
        target_pos = target_positions[p]

        # Truncate input_attr from T_max to T (saves bytes).
        attr_p = input_attr[p, :, :T]                              # (F, T)
        contrib_p = output_contrib[p]                              # (F, K)
        slug = f"{cfg.slug_prefix}__{p:03d}"

        graph_data = _build_one_graph(
            p,
            slug=slug,
            scan=cfg.scan,
            pair_token_ids=pair_ids,
            pair_token_strs=pair_strs,
            target_pos=target_pos,
            feature_idx=feature_idx,
            input_attr_p=attr_p,
            output_contrib_p=contrib_p,
            top_k_logit_ids_p=top_k_logit_ids[p],
            top_k_logit_vals_p=top_k_logit_vals[p],
            edges_p=edges[p],
            cluster_ids=cluster_ids,
            cluster_descriptions=cluster_descriptions,
            num_layers=num_layers,
            decode_token_fn=decode,
            cfg=cfg,
        )

        out_path = out_dir / "graph_data" / f"{slug}.json"
        out_path.write_text(json.dumps(graph_data))
        written.append(out_path)

        # Title line for the dropdown.
        first_word = next((w for w in pair_strs[-30:][::-1] if w.strip()), pair_strs[-1])
        graph_metadata_entries.append({
            "slug": slug,
            "scan": cfg.scan,
            "prompt": (graph_data["metadata"]["prompt"][:120].replace("\n", "↵")
                       + (" …" if len(graph_data["metadata"]["prompt"]) > 120 else "")),
            "node_threshold": graph_data["metadata"]["node_threshold"],
        })
        if log and (p + 1) % 10 == 0:
            print(f"[export] {p+1}/{len(pair_indices)} graphs written", flush=True)

    meta_path = out_dir / "data" / "graph-metadata.json"
    meta_path.write_text(json.dumps({"graphs": graph_metadata_entries}))
    written.append(meta_path)
    return written
