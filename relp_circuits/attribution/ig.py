"""Integrated-gradients attribution over MLP post-activation neurons.

For each layer l we treat the d_ffn neuron vector h_l as the variable being attributed.
The integration path goes from h_l(cf) (counterfactual) to h_l(clean) for every layer
simultaneously, in K straight-line steps. At each step we run a forward pass with all
layers' h overridden to the interpolated value, compute the metric, backward, and
accumulate grad_h_l * (h_l(clean) - h_l(cf)) / K.

This matches the IG-activations definition in Arora et al. 2026 (Eq. for IG_act_v) and
its 10-step numerical-integration choice. K=10 is the default.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.attribution.common import AttributionResult, logit_diff_metric
from relp_circuits.model import HookedModel
from relp_circuits.tasks.base import PairedPrompt


@torch.no_grad()
def _capture_acts(hooked: HookedModel, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Forward pass with no grad, returns the captured h tensor for each layer."""
    hooked.reset_states()
    hooked.model(input_ids.to(hooked.device).unsqueeze(0))
    out: list[torch.Tensor] = []
    for st in hooked.states:
        assert st.captured is not None
        out.append(st.captured.detach().clone())
    return out


def ig_attribution(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    n_steps: int = 10,
) -> AttributionResult:
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device

    accum = torch.zeros((n_layers, d_ffn), dtype=torch.float32, device=device)
    n_used = 0

    for pair in tqdm(pairs, desc="ig"):
        assert pair.clean_ids is not None and pair.cf_ids is not None
        assert pair.target_pos is not None
        clean_ids = pair.clean_ids.to(device)
        cf_ids = pair.cf_ids.to(device)

        h_clean = _capture_acts(hooked, clean_ids)   # list of (1, T, d_ffn)
        h_cf = _capture_acts(hooked, cf_ids)         # list of (1, T, d_ffn)
        delta = [(c - r).to(torch.float32) for c, r in zip(h_clean, h_cf, strict=True)]

        # Sum of grad * delta across steps; divided by K at the end.
        per_layer_attr = [torch.zeros_like(d) for d in delta]

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            # Build leaf override tensors at the interpolated point.
            overrides: list[torch.Tensor] = []
            for c, r in zip(h_clean, h_cf, strict=True):
                interp = (r + alpha * (c - r)).detach().to(hooked.dtype)
                interp.requires_grad_(True)
                overrides.append(interp)
            hooked.reset_states()
            hooked.set_overrides(overrides)
            # Use the clean input ids during the integration: only h is interpolated.
            # (Standard IG-activations: input is fixed, neuron activations vary.)
            logits = hooked.model(clean_ids.unsqueeze(0)).logits
            metric = logit_diff_metric(
                logits,
                target_pos=pair.target_pos,
                correct_id=pair.metric_correct_id(),
                incorrect_id=pair.metric_incorrect_id(),
            )
            grads = torch.autograd.grad(metric, overrides, retain_graph=False)
            for li, g in enumerate(grads):
                per_layer_attr[li] += g.to(torch.float32) * delta[li]

        for li in range(n_layers):
            # Average over steps and over sequence/batch (sum over those axes is what IG
            # gives; sum -> per-token attribution. We sum over tokens to get a per-neuron
            # score, since we care about neurons, not token-positions, for circuit ID.)
            score = per_layer_attr[li].sum(dim=(0, 1)) / n_steps  # (d_ffn,)
            accum[li] += score

        hooked.reset_states()
        n_used += 1

    scores = (accum / max(n_used, 1)).cpu().numpy().astype(np.float32)
    return AttributionResult(
        method="ig",
        scores=scores,
        n_pairs=n_used,
        n_layers=n_layers,
        d_ffn=d_ffn,
        extras={"n_steps": n_steps},
    )
