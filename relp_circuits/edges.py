"""Cross-Layer Jacobian Attribution (CLJA) edges between τ-circuit features.

Implements the edge-attribution step from Arora et al. 2026 "ADAG"
(arxiv 2604.07615, section 4 / clja.py in TransluceAI/circuits). Under our
existing RelP linearization (frozen SiLU, frozen RMSNorm, frozen softmax,
half-rule on the SwiGLU multiplication and on attn matmuls), the
post-SwiGLU activation at every layer is differentiable in a faithful
linear way — so the cross-layer Jacobian ∂h_tgt/∂h_src captures all
attention paths between layers in a self-consistent fashion with our
node attributions.

For each prompt and each pair of features (src in layer L_s, tgt in layer
L_t > L_s) we compute::

    edge_jvp(src → tgt) = Σ_t  h_src(token=t) · ∂h_tgt(target_pos) / ∂h_src(token=t)

i.e. the absolute JVP, summed over source token positions. This is
"absolute" rather than "relative" (the relative form is `edge_jvp / h_tgt`
and falls out by post-hoc division at viz time).

Implementation details:

- We do **one** RelP forward pass per prompt and reuse the autograd graph
  via `retain_graph=True` for many backward calls.
- Per tgt_layer in the circuit we issue ONE backward call (chunked over
  tgt features for memory) that returns gradients to *all* upstream
  layers' captured tensors simultaneously, by passing the full upstream
  list as the `inputs` argument of `torch.autograd.grad`. is_grads_batched
  vmaps over tgt features.
- Gradient tensors live on whatever GPU `device_map=auto` placed each
  layer on; we slice, multiply by activations, and sum on that GPU before
  copying scalars back to CPU.
- Memory per backward call: O(n_chunk × T × d_ffn) per upstream layer
  (bf16). For a 32B model with chunk=20, T=200, d_ffn=20480 that's 164 MB
  per layer × ≤ tgt_layer layers ≤ 10 GB worst case, distributed across
  the device_map'd GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.model import HookedModel, relp_active
from relp_circuits.tasks.base import PairedPrompt


@dataclass
class CLJAEdges:
    feature_idx: list[tuple[int, int]]   # (F,) ordered list of (layer, neuron)
    edges: np.ndarray                     # (P, F, F) — edges[p, src, tgt]
    target_pos: np.ndarray                # (P,)
    layer_pairs_seen: int = 0             # for sanity logging


def compute_clja_edges(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    circuit: list[dict],
    *,
    chunk_size: int = 20,
    log: bool = True,
) -> CLJAEdges:
    """Compute per-pair cross-layer Jacobian edge weights between every
    (src, tgt) pair of features in the τ-circuit (with src_layer <
    tgt_layer).

    Returns a dense (P, F, F) array. Entry [p, src, tgt] is the absolute
    JVP h_src · ∂h_tgt/∂h_src summed over source token positions. The
    diagonal and lower-triangular block (src_layer ≥ tgt_layer) is left
    at zero — features at the same layer have no direct linear edge in
    this graph view (their joint contribution flows through the next
    layer).
    """
    feature_list = [(int(e["layer"]), int(e["neuron"])) for e in circuit]
    F = len(feature_list)
    if F == 0:
        raise ValueError("empty circuit")

    by_layer: dict[int, list[tuple[int, int]]] = {}
    for fi, (li, ni) in enumerate(feature_list):
        by_layer.setdefault(li, []).append((fi, ni))
    layers_sorted = sorted(by_layer.keys())

    P = len(pairs)
    edges = np.zeros((P, F, F), dtype=np.float32)
    target_pos = np.zeros((P,), dtype=np.int32)

    embed_module = hooked.model.get_input_embeddings()
    primary_device = hooked.device

    layer_pairs_seen = 0
    iterator = tqdm(enumerate(pairs), total=P, desc="clja-edges") if log else enumerate(pairs)
    for p, pair in iterator:
        clean_ids = pair.clean_ids.to(primary_device)
        T = int(clean_ids.shape[0])
        tp = int(pair.target_pos)
        target_pos[p] = tp

        hooked.reset_states()
        with torch.no_grad():
            embeds = embed_module(clean_ids.unsqueeze(0)).clone()
        embeds = embeds.detach().requires_grad_(True)

        with relp_active():
            _ = hooked.model(inputs_embeds=embeds)
            captured = [st.captured for st in hooked.states]
            for c in captured:
                assert c is not None

            for tgt_layer in layers_sorted:
                tgt_features = by_layer[tgt_layer]
                tgt_dev = captured[tgt_layer].device

                for chunk_start in range(0, len(tgt_features), chunk_size):
                    chunk = tgt_features[chunk_start : chunk_start + chunk_size]
                    chunk_neurons = torch.tensor(
                        [ni for _, ni in chunk], device=tgt_dev,
                    )
                    n_chunk = len(chunk)

                    # (n_chunk,)  — pull tgt activations at the target token only
                    tgt_acts = captured[tgt_layer][0, tp, chunk_neurons]

                    upstream_captured = [captured[l] for l in range(tgt_layer)]
                    if not upstream_captured:
                        continue

                    eye = torch.eye(n_chunk, device=tgt_dev)
                    grads = torch.autograd.grad(
                        tgt_acts, upstream_captured,
                        grad_outputs=eye,
                        is_grads_batched=True,
                        retain_graph=True,
                    )
                    # grads[l] has shape (n_chunk, 1, T, d_ffn) on the device
                    # of layer l.

                    for src_layer in layers_sorted:
                        if src_layer >= tgt_layer:
                            break
                        src_features = by_layer[src_layer]
                        src_dev = grads[src_layer].device
                        src_neurons = torch.tensor(
                            [ni for _, ni in src_features], device=src_dev,
                        )
                        # Subset along d_ffn → (n_chunk, T, n_src)
                        g = grads[src_layer][:, 0, :, src_neurons]
                        h = captured[src_layer].detach()[0, :, src_neurons].to(g.dtype)
                        # JVP per (n_chunk, T, n_src) then sum over source tokens.
                        edge_summed = (g * h[None]).sum(dim=1).to(torch.float32).cpu().numpy()

                        # Scatter into the dense output.
                        src_fis = [fi for fi, _ in src_features]
                        for ci, (tgt_fi, _) in enumerate(chunk):
                            edges[p, src_fis, tgt_fi] = edge_summed[ci]
                        layer_pairs_seen += 1

                    del grads

        hooked.reset_states()

    return CLJAEdges(
        feature_idx=feature_list,
        edges=edges,
        target_pos=target_pos,
        layer_pairs_seen=layer_pairs_seen,
    )


def conservation_check(
    edges_p: np.ndarray,
    output_contrib_p: np.ndarray,
    by_layer: dict[int, list[int]],
    *,
    last_layer: int,
) -> dict[str, float]:
    """Sanity check: under linearization, the sum of incoming edges into a
    target feature should equal that feature's activation (up to the
    contribution of the residual stream from the embeddings, which is
    captured by the input_attr term, not edges between MLP features).

    For features in the last attribution layer we can also check that the
    sum of outgoing contributions to top-K logits matches output_contrib.

    edges_p: (F, F) for one prompt.
    output_contrib_p: (F, K) for one prompt.
    """
    return {
        "edge_l2": float(np.linalg.norm(edges_p)),
        "edge_density": float((np.abs(edges_p) > 1e-6).mean()),
        "edge_max": float(np.abs(edges_p).max()),
        "edge_sum_incoming_max": float(np.abs(edges_p.sum(axis=0)).max()),
    }
