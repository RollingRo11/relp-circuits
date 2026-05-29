"""Validate attribution by zeroing the top-k attributed neurons and measuring metric drop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from relp_circuits.attribution.common import AttributionResult, logit_diff_metric
from relp_circuits.model import HookedModel
from relp_circuits.tasks.base import PairedPrompt


@dataclass
class AblationReport:
    baseline_metric: float
    ablated_metric: float
    random_baseline_metric: float
    metric_drop: float            # baseline - ablated
    random_drop: float            # baseline - random_baseline
    k: int
    n_pairs: int


@torch.no_grad()
def _metric_with_ablation(
    hooked: HookedModel,
    pair: PairedPrompt,
    ablate_layer_idx: np.ndarray | None,
    ablate_neuron_idx: np.ndarray | None,
    metric_mode: str = "logit_diff",
    top_k_ids: list[int] | None = None,
) -> float:
    """Run forward; if ablation indices provided, zero those neurons in the override.

    metric_mode:
      "logit_diff"      : logit(correct) - logit(incorrect) at target position. Default
                          for paired RelP / SVA / multihop where a contrast exists.
      "logit_value"     : logit(correct) at target position. Used by unpaired RelP
                          where there's no contrast token — keeps the ablation in
                          the same metric space as the attribution.
      "top_k_logit_sum" : Σ logits at `top_k_ids` (which must be passed in from the
                          attribution pass). Used by relp_paper to keep the ablation
                          metric identical to the paper's attribution target.
    """
    assert pair.clean_ids is not None and pair.target_pos is not None
    device = hooked.device
    clean_ids = pair.clean_ids.to(device).unsqueeze(0)

    hooked.reset_states()
    hooked.model(clean_ids)
    overrides: list[torch.Tensor] = []
    for li, st in enumerate(hooked.states):
        assert st.captured is not None
        h = st.captured.detach().clone()
        if ablate_layer_idx is not None and ablate_neuron_idx is not None:
            mask = ablate_layer_idx == li
            if mask.any():
                neurons = ablate_neuron_idx[mask]
                h[..., neurons] = 0.0
        overrides.append(h)

    hooked.reset_states()
    hooked.set_overrides(overrides)
    logits = hooked.model(clean_ids).logits
    if metric_mode == "top_k_logit_sum":
        if not top_k_ids:
            raise ValueError("top_k_logit_sum metric requires top_k_ids")
        idx = torch.as_tensor(top_k_ids, device=logits.device, dtype=torch.long)
        m = logits[0, pair.target_pos].index_select(0, idx).sum()
    elif metric_mode == "logit_value":
        m = logits[0, pair.target_pos, pair.metric_correct_id()]
    else:
        m = logit_diff_metric(
            logits,
            target_pos=pair.target_pos,
            correct_id=pair.metric_correct_id(),
            incorrect_id=pair.metric_incorrect_id(),
        )
    hooked.reset_states()
    return float(m.item())


def ablate_topk(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    attribution: AttributionResult,
    k: int,
    rng: np.random.Generator | None = None,
    metric_mode: str = "logit_diff",
    per_pair_top_k_ids: list[list[int]] | None = None,
) -> AblationReport:
    rng = rng or np.random.default_rng(0)
    layer_idx, neuron_idx = attribution.topk_global(k)
    return _ablate_indexed(
        hooked, pairs, attribution.n_layers, attribution.d_ffn,
        layer_idx, neuron_idx, k, rng, metric_mode=metric_mode,
        per_pair_top_k_ids=per_pair_top_k_ids,
    )


def ablate_specific(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    n_layers: int,
    d_ffn: int,
    layer_neurons: list[tuple[int, int]],
    rng: np.random.Generator | None = None,
    metric_mode: str = "logit_diff",
    per_pair_top_k_ids: list[list[int]] | None = None,
) -> AblationReport:
    rng = rng or np.random.default_rng(0)
    if not layer_neurons:
        base = 0.0
        for i, pair in enumerate(pairs):
            ids = per_pair_top_k_ids[i] if per_pair_top_k_ids else None
            base += _metric_with_ablation(hooked, pair, None, None, metric_mode, ids)
        n = max(len(pairs), 1)
        base /= n
        return AblationReport(
            baseline_metric=base, ablated_metric=base, random_baseline_metric=base,
            metric_drop=0.0, random_drop=0.0, k=0, n_pairs=len(pairs),
        )
    layer_idx = np.array([l for l, _ in layer_neurons], dtype=np.int64)
    neuron_idx = np.array([n for _, n in layer_neurons], dtype=np.int64)
    return _ablate_indexed(hooked, pairs, n_layers, d_ffn, layer_idx, neuron_idx,
                            len(layer_neurons), rng, metric_mode=metric_mode,
                            per_pair_top_k_ids=per_pair_top_k_ids)


def _ablate_indexed(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    n_layers: int,
    d_ffn: int,
    layer_idx: np.ndarray,
    neuron_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
    metric_mode: str = "logit_diff",
    per_pair_top_k_ids: list[list[int]] | None = None,
) -> AblationReport:
    flat_size = n_layers * d_ffn
    k_eff = min(k, flat_size)
    rand_flat = rng.choice(flat_size, size=k_eff, replace=False)
    rand_layer = rand_flat // d_ffn
    rand_neuron = rand_flat % d_ffn

    base, abl, rnd = 0.0, 0.0, 0.0
    for i, pair in enumerate(pairs):
        ids = per_pair_top_k_ids[i] if per_pair_top_k_ids else None
        base += _metric_with_ablation(hooked, pair, None, None, metric_mode, ids)
        abl += _metric_with_ablation(hooked, pair, layer_idx, neuron_idx, metric_mode, ids)
        rnd += _metric_with_ablation(hooked, pair, rand_layer, rand_neuron, metric_mode, ids)
    n = max(len(pairs), 1)
    base /= n
    abl /= n
    rnd /= n
    return AblationReport(
        baseline_metric=base,
        ablated_metric=abl,
        random_baseline_metric=rnd,
        metric_drop=base - abl,
        random_drop=base - rnd,
        k=k,
        n_pairs=len(pairs),
    )
