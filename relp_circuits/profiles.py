"""ADAG attribution profiles (Arora & Wu et al. 2026, https://arxiv.org/abs/2604.07615).

For each circuit feature f and each prompt x we compute two views:

  Input attribution:
      Attr(f, x)_i = sum_{c} embed(x_i)_c · ∂f(x) / ∂embed(x_i)_c
    A scalar per input-token i. Captures which input tokens contribute to firing
    the feature (gradient-times-input on the token-embedding axis).

  Output contribution:
      Contrib(f, x)_j = f(x) · ∂logit_j(x) / ∂f(x)
    A scalar per top-K target logit j. Captures how the feature pushes the
    output logits at the answer position.

These two views feed the multi-view spectral clustering in clustering.py.

Implementation notes
--------------------
- Both views need the linearised RelP backward (silu frozen, RMSNorm frozen,
  attention softmax frozen, half-rule on multiplications) so attribution is
  consistent with our existing τ-circuit. We wrap forward + backward in
  ``relp_active()`` exactly like ``relp_attribution`` does.
- Input attribution uses ``inputs_embeds`` rather than ``input_ids`` so we have
  a leaf tensor to take gradients with respect to. The patched embed_tokens is
  module.embed_tokens; under HF device_map="auto" it lives on cuda:0.
- We compute the Jacobian of all τ-features w.r.t. embeds in **one** call via
  ``torch.autograd.grad(..., is_grads_batched=True)``. Memory: features ×
  batch × T × d_model × 4 bytes. For our 220-feature eval-awareness on 32B
  with T~16 and d_model=5120, that's ~70 MB. Fine.
- For output contribution we use the same trick: grads of top-K target logits
  w.r.t. all τ-features simultaneously. K × n_features × T × 4 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.model import HookedModel, relp_active
from relp_circuits.tasks.base import PairedPrompt


@dataclass
class AttributionProfiles:
    feature_idx: list[tuple[int, int]]      # (n_features,) ordered list of (layer, neuron)
    input_attr: np.ndarray                  # (n_pairs, n_features, T_max) — Attr(f, x)_i
    output_contrib: np.ndarray              # (n_pairs, n_features, K) — Contrib(f, x)_j
    top_k_logit_ids: np.ndarray             # (n_pairs, K) — token ids of top-K logits per pair
    top_k_logit_vals: np.ndarray            # (n_pairs, K) — logit values
    per_pair_lengths: np.ndarray            # (n_pairs,) — true T per pair


def _features_per_layer(circuit: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Group `(layer, neuron)` pairs by layer for vectorised grad collection."""
    grouped: dict[int, list[int]] = {}
    for fi, entry in enumerate(circuit):
        li = entry["layer"]
        ni = entry["neuron"]
        grouped.setdefault(li, []).append(fi)
    return grouped


def compute_attribution_profiles(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    circuit: list[dict],          # entries with "layer" and "neuron" keys
    top_k: int = 5,
    log: bool = True,
) -> AttributionProfiles:
    """Compute (input_attr, output_contrib) for every (prompt, feature).

    `circuit` is the τ-filter list from `tau_circuit.json` — each entry must
    have at least `layer` and `neuron` keys. Order is preserved as the
    feature axis.
    """
    feature_list = [(int(e["layer"]), int(e["neuron"])) for e in circuit]
    feature_idx = {(li, ni): i for i, (li, ni) in enumerate(feature_list)}
    n_features = len(feature_list)
    if n_features == 0:
        raise ValueError("empty circuit")
    by_layer = _features_per_layer(circuit)

    n_pairs = len(pairs)
    T_max = max(int(p.clean_ids.shape[0]) for p in pairs)
    primary_device = hooked.device

    input_attr = np.zeros((n_pairs, n_features, T_max), dtype=np.float32)
    output_contrib = np.zeros((n_pairs, n_features, top_k), dtype=np.float32)
    top_k_ids = np.zeros((n_pairs, top_k), dtype=np.int64)
    top_k_vals = np.zeros((n_pairs, top_k), dtype=np.float32)
    lengths = np.zeros((n_pairs,), dtype=np.int32)

    embed_module = hooked.model.get_input_embeddings()

    iterator = tqdm(enumerate(pairs), total=len(pairs), desc="profiles") if log else enumerate(pairs)
    for p, pair in iterator:
        clean_ids = pair.clean_ids.to(primary_device)
        T = int(clean_ids.shape[0])
        lengths[p] = T

        hooked.reset_states()
        # Build embeddings as a *leaf* so we can take gradients w.r.t. them.
        with torch.no_grad():
            embeds = embed_module(clean_ids.unsqueeze(0)).clone()
        embeds = embeds.detach().requires_grad_(True)
        # The embed module's output dtype is bf16; we keep it bf16 for the
        # forward pass and cast the jacobian-times-embed contraction to fp32
        # later.

        with relp_active():
            out = hooked.model(inputs_embeds=embeds)
            logits = out.logits[0, pair.target_pos]  # (V,)
            captured = [st.captured for st in hooked.states]    # list of (1, T, d_ffn)
            for c in captured:
                assert c is not None

            # ─── Output contribution ───
            # Differentiate each of the top-K logits w.r.t. ALL captured tensors
            # (the actual graph leaves we care about — torch.stack of slices
            # creates a new tensor not in the original graph, hence the earlier
            # "not used in the graph" error). Per (l, n) the contribution is
            # Σ_t  g(t)·h(t)  for that neuron's row.
            top_vals, top_ids = logits.topk(top_k)
            top_k_ids[p] = top_ids.detach().cpu().numpy()
            top_k_vals[p] = top_vals.detach().to(torch.float32).cpu().numpy()
            for k in range(top_k):
                grads_per_layer = torch.autograd.grad(
                    top_vals[k], captured, retain_graph=True,
                )
                for fi, (li, ni) in enumerate(feature_list):
                    g = grads_per_layer[li][0, :, ni].to(torch.float32)
                    h = captured[li].detach()[0, :, ni].to(torch.float32)
                    output_contrib[p, fi, k] = float((g * h).sum().item())

        # ─── Input attribution ───
        # Re-do forward with retain_graph=True so we can take more gradients.
        # We process features layer-by-layer so we can request grads of ALL
        # features at that layer in one batched-Jacobian call.
        hooked.reset_states()
        embeds2 = embeds.detach().clone().requires_grad_(True)
        with relp_active():
            _ = hooked.model(inputs_embeds=embeds2)
            captured2 = [st.captured for st in hooked.states]

            for li, fi_list in by_layer.items():
                h = captured2[li]                         # (1, T, d_ffn)
                # Under HF device_map=auto, h lives on the GPU hosting layer li;
                # the index tensor must match that device.
                ni_tensor = torch.tensor([feature_list[fi][1] for fi in fi_list],
                                          device=h.device)
                feat_per_token = h[0, :, ni_tensor]       # (T, n_li)
                feat_sums = feat_per_token.sum(dim=0)     # (n_li,)
                # Identity vector-Jacobian to get the full Jacobian. Must be on
                # the same device as feat_sums (grad source) — autograd routes
                # the resulting grad back to embeds2's device on its own.
                eye = torch.eye(feat_sums.shape[0], device=feat_sums.device)
                grad_embeds = torch.autograd.grad(
                    feat_sums, embeds2,
                    grad_outputs=eye,
                    is_grads_batched=True,
                    retain_graph=True,
                )[0]                                      # (n_li, 1, T, d_model)
                # Attribute = embed · grad_embed, summed over d_model.
                contrib_per_tok = (
                    grad_embeds.to(torch.float32) * embeds2.detach().to(torch.float32)
                ).sum(dim=-1).squeeze(1)                  # (n_li, T)

                for j, fi in enumerate(fi_list):
                    input_attr[p, fi, :T] = contrib_per_tok[j].cpu().numpy()

        hooked.reset_states()

    return AttributionProfiles(
        feature_idx=feature_list,
        input_attr=input_attr,
        output_contrib=output_contrib,
        top_k_logit_ids=top_k_ids,
        top_k_logit_vals=top_k_vals,
        per_pair_lengths=lengths,
    )
