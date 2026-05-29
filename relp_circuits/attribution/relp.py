"""RelP attribution variants over MLP post-activation neurons.

Three flavours are provided:

  * `paper_relp_attribution` — the ADAG paper's (arXiv 2604.07615) canonical
    definition: bare h · ∂target/∂h with target = Σ top-K logits at the answer
    position. No counterfactual; no baseline subtraction. Preserves the RelP
    conservation property Σ h·grad = target. This is the default.

  * `relp_attribution` — paired Relevance-Patching variant of RelP: uses
    (h_clean − h_cf) · grad against `logit_diff(correct, incorrect)`. Matches
    the original RelP paper's patching formulation (Jafari et al. 2025).

  * `unpaired_relp_attribution` — replaces the per-example cf with a fixed
    corpus-mean baseline. Kept for comparison against the patching variants.

  * `atp_attribution` — Attribution Patching baseline (no linearization), using
    (h_clean − h_cf) · grad with the standard backward pass.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.attribution.common import (
    AttributionResult,
    logit_diff_metric,
    top_k_logit_sum_metric,
)
from relp_circuits.model import HookedModel, relp_active
from relp_circuits.tasks.base import PairedPrompt


@torch.no_grad()
def _capture_acts(hooked: HookedModel, input_ids: torch.Tensor) -> list[torch.Tensor]:
    hooked.reset_states()
    hooked.model(input_ids.to(hooked.device).unsqueeze(0))
    return [st.captured.detach().clone() for st in hooked.states if st.captured is not None]


def atp_attribution(hooked: HookedModel, pairs: list[PairedPrompt]) -> AttributionResult:
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device

    accum = torch.zeros((n_layers, d_ffn), dtype=torch.float32, device=device)
    n_used = 0

    for pair in tqdm(pairs, desc="atp"):
        assert pair.clean_ids is not None and pair.cf_ids is not None
        assert pair.target_pos is not None
        clean_ids = pair.clean_ids.to(device)
        cf_ids = pair.cf_ids.to(device)

        # Capture cf activations under no_grad.
        h_cf = _capture_acts(hooked, cf_ids)

        # Clean forward with grad enabled. Captured tensors are non-leaf graph nodes;
        # autograd.grad against them gives ∂m/∂h_l without materializing weight grads.
        hooked.reset_states()
        logits = hooked.model(clean_ids.unsqueeze(0)).logits
        metric = logit_diff_metric(
            logits,
            target_pos=pair.target_pos,
            correct_id=pair.metric_correct_id(),
            incorrect_id=pair.metric_incorrect_id(),
        )
        captured = [st.captured for st in hooked.states]
        for c in captured:
            assert c is not None
        grads = torch.autograd.grad(metric, captured, retain_graph=False)

        for li, (g, h_clean) in enumerate(zip(grads, captured, strict=True)):
            delta = (h_clean.detach() - h_cf[li]).to(torch.float32)
            accum[li] += (g.to(torch.float32) * delta).sum(dim=(0, 1))

        hooked.reset_states()
        n_used += 1

    scores = (accum / max(n_used, 1)).cpu().numpy().astype(np.float32)
    return AttributionResult(
        method="atp",
        scores=scores,
        n_pairs=n_used,
        n_layers=n_layers,
        d_ffn=d_ffn,
    )


def unpaired_relp_attribution(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    baseline_mean: np.ndarray,        # (n_layers, d_ffn) per-neuron corpus mean
    store_per_pair: bool = True,
) -> AttributionResult:
    """Unpaired RelP (Arora et al. Section 6).

    Replaces the per-example counterfactual h_l(x') with a fixed corpus-mean
    baseline E[h_l(x') for x' ~ corpus]. This lets attribution run on any prompt
    without constructing a minimal pair — the input arriving here as `pairs` is
    treated as a list of clean prompts; the cf side is ignored. Use the same
    PairedPrompt dataclass for code reuse.

    `baseline_mean` should be precomputed once via scripts/compute_baseline.py
    and shared across many unpaired attribution runs for the same model.
    """
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device

    if baseline_mean.shape != (n_layers, d_ffn):
        raise ValueError(
            f"baseline_mean shape {baseline_mean.shape} doesn't match "
            f"(n_layers={n_layers}, d_ffn={d_ffn}) of model"
        )
    baseline_t = torch.from_numpy(baseline_mean).to(device=device, dtype=torch.float32)

    accum = torch.zeros((n_layers, d_ffn), dtype=torch.float32, device=device)
    per_pair_scores: list[np.ndarray] = []
    per_pair_token_scores: list[np.ndarray] = []
    per_pair_metric: list[float] = []
    per_pair_token_ids: list[list[int]] = []
    n_used = 0

    for pair in tqdm(pairs, desc="relp-unpaired"):
        assert pair.clean_ids is not None
        clean_ids = pair.clean_ids.to(device)

        hooked.reset_states()
        with relp_active():
            logits = hooked.model(clean_ids.unsqueeze(0)).logits
            # Metric: logit at the target position for the correct token. Same
            # form as paired RelP, but with no cf to define a contrast — we
            # attribute the logit *value* itself (paper's "total logit").
            target_id = pair.metric_correct_id() if pair.clean_correct_id is not None else \
                int(logits[0, pair.target_pos].argmax().item())
            metric = logits[0, pair.target_pos, target_id]
            captured = [st.captured for st in hooked.states]
            for c in captured:
                assert c is not None
            grads = torch.autograd.grad(metric, captured, retain_graph=False)

        T_pair = captured[0].shape[1]
        pair_token_attr = torch.zeros(
            (n_layers, T_pair, d_ffn), dtype=torch.float32, device=device,
        )
        for li, (g, h_clean) in enumerate(zip(grads, captured, strict=True)):
            # Δh = h(x) - baseline_mean. Under HF device_map="auto", h_clean lives on
            # whichever GPU layer li sits on, so move the baseline row to match.
            base_li = baseline_t[li].view(1, 1, d_ffn).to(h_clean.device)
            delta = h_clean.detach().to(torch.float32) - base_li
            pair_token_attr[li] = (g.to(torch.float32) * delta).sum(dim=0).to(device)

        pair_attr = pair_token_attr.sum(dim=1)
        accum += pair_attr
        if store_per_pair:
            per_pair_scores.append(pair_attr.cpu().numpy())
            per_pair_token_scores.append(pair_token_attr.cpu().numpy())
            per_pair_metric.append(float(metric.detach().item()))
            per_pair_token_ids.append(pair.clean_ids.cpu().tolist())

        hooked.reset_states()
        n_used += 1

    scores = (accum / max(n_used, 1)).cpu().numpy().astype(np.float32)

    pps = ppm = ppts = ppl = pptid = None
    if store_per_pair and per_pair_scores:
        pps = np.stack(per_pair_scores, axis=0).astype(np.float32)
        ppm = np.asarray(per_pair_metric, dtype=np.float32)
        T_max = max(s.shape[1] for s in per_pair_token_scores)
        ppts = np.full(
            (len(per_pair_token_scores), n_layers, T_max, d_ffn),
            np.nan, dtype=np.float32,
        )
        for p, s in enumerate(per_pair_token_scores):
            ppts[p, :, : s.shape[1], :] = s
        ppl = np.asarray([s.shape[1] for s in per_pair_token_scores], dtype=np.int32)
        pptid = per_pair_token_ids
    return AttributionResult(
        method="relp_unpaired",
        scores=scores,
        n_pairs=n_used,
        n_layers=n_layers,
        d_ffn=d_ffn,
        per_pair_scores=pps,
        per_pair_metric=ppm,
        per_pair_token_scores=ppts,
        per_pair_lengths=ppl,
        per_pair_token_ids=pptid,
    )


def relp_attribution(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    store_per_pair: bool = True,
) -> AttributionResult:
    """Paper-faithful RelP attribution.

    Forward+backward run inside `relp_active()`, which switches every Olmo3MLP,
    Olmo3RMSNorm, and the eager attention path to their linearized replacements:

      * Olmo3MLP        — silu frozen, half-rule on the SwiGLU multiplication.
      * Olmo3RMSNorm    — rms and weight treated as constants in backward.
      * Attention       — softmax frozen (zero grad), half-rule on Q·K^T and softmax·V.

    The forward output is identical to the standard model; only the backward gradient
    is replaced. Per-neuron score = Σ_(B,T) ∂m/∂h · (h_clean - h_cf), averaged over pairs.

    When `store_per_pair=True`, returns the per-pair score tensors and metric values so
    the caller can apply the paper's τ-filter (Sec. 6: keep (l,n) iff
    |RelP_v(x)| ≥ τ · |m(M,x)| on a per-example basis).
    """
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device

    accum = torch.zeros((n_layers, d_ffn), dtype=torch.float32, device=device)
    per_pair_scores: list[np.ndarray] = []
    per_pair_token_scores: list[np.ndarray] = []  # each (n_layers, T_pair, d_ffn) float32
    per_pair_metric: list[float] = []
    per_pair_token_ids: list[list[int]] = []
    n_used = 0

    for pair in tqdm(pairs, desc="relp"):
        assert pair.clean_ids is not None and pair.cf_ids is not None
        clean_ids = pair.clean_ids.to(device)
        cf_ids = pair.cf_ids.to(device)

        h_cf = _capture_acts(hooked, cf_ids)

        hooked.reset_states()
        with relp_active():
            logits = hooked.model(clean_ids.unsqueeze(0)).logits
            metric = logit_diff_metric(
                logits,
                target_pos=pair.target_pos,
                correct_id=pair.metric_correct_id(),
                incorrect_id=pair.metric_incorrect_id(),
            )
            captured = [st.captured for st in hooked.states]
            for c in captured:
                assert c is not None
            grads = torch.autograd.grad(metric, captured, retain_graph=False)

        T_pair = captured[0].shape[1]
        pair_token_attr = torch.zeros(
            (n_layers, T_pair, d_ffn), dtype=torch.float32, device=device
        )
        for li, (g, h_clean) in enumerate(zip(grads, captured, strict=True)):
            delta = (h_clean.detach() - h_cf[li]).to(torch.float32)
            # Sum over batch dim (size 1), keep token axis.
            pair_token_attr[li] = (g.to(torch.float32) * delta).sum(dim=0)

        pair_attr = pair_token_attr.sum(dim=1)  # (n_layers, d_ffn)
        accum += pair_attr
        if store_per_pair:
            per_pair_scores.append(pair_attr.cpu().numpy())
            per_pair_token_scores.append(pair_token_attr.cpu().numpy())
            per_pair_metric.append(float(metric.detach().item()))
            per_pair_token_ids.append(pair.clean_ids.cpu().tolist())

        hooked.reset_states()
        n_used += 1

    scores = (accum / max(n_used, 1)).cpu().numpy().astype(np.float32)

    pps = ppm = ppts = ppl = pptid = None
    if store_per_pair and per_pair_scores:
        pps = np.stack(per_pair_scores, axis=0).astype(np.float32)
        ppm = np.asarray(per_pair_metric, dtype=np.float32)
        # Pad token dim to T_max and stack.
        T_max = max(s.shape[1] for s in per_pair_token_scores)
        ppts = np.full(
            (len(per_pair_token_scores), n_layers, T_max, d_ffn),
            np.nan, dtype=np.float32,
        )
        for p, s in enumerate(per_pair_token_scores):
            ppts[p, :, : s.shape[1], :] = s
        ppl = np.asarray([s.shape[1] for s in per_pair_token_scores], dtype=np.int32)
        pptid = per_pair_token_ids
    return AttributionResult(
        method="relp",
        scores=scores,
        n_pairs=n_used,
        n_layers=n_layers,
        d_ffn=d_ffn,
        per_pair_scores=pps,
        per_pair_metric=ppm,
        per_pair_token_scores=ppts,
        per_pair_lengths=ppl,
        per_pair_token_ids=pptid,
    )


def paper_relp_attribution(
    hooked: HookedModel,
    pairs: list[PairedPrompt],
    top_k: int = 5,
    store_per_pair: bool = True,
) -> AttributionResult:
    """ADAG paper's canonical RelP attribution (arXiv 2604.07615).

    Score per (layer, token, neuron):

        α(m_u^{(l,t)}) = m_u^{(l,t)}(x) · ∂target/∂m_u^{(l,t)}(x)

    where

        target = Σ_{k=0}^{K-1} M(x)_{k}^{(s)}

    is the sum of the top-K logits at the answer position s. With the RelP
    backward replacement model installed (silu-as-constant-multiplier, frozen
    softmax, frozen RMSNorm, half-rule on bilinear matmuls), this satisfies the
    conservation property Σ h·grad = target — i.e. the attribution explains the
    entire target value.

    Input is a list of `PairedPrompt` for code compatibility with the rest of
    the pipeline, but only `pair.clean_ids` and `pair.target_pos` are used —
    no counterfactual, no baseline subtraction.
    """
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device

    accum = torch.zeros((n_layers, d_ffn), dtype=torch.float32, device=device)
    per_pair_scores: list[np.ndarray] = []
    per_pair_token_scores: list[np.ndarray] = []
    per_pair_metric: list[float] = []
    per_pair_token_ids: list[list[int]] = []
    per_pair_top_logit_ids: list[list[int]] = []
    n_used = 0

    for pair in tqdm(pairs, desc="relp-paper"):
        assert pair.clean_ids is not None
        assert pair.target_pos is not None
        clean_ids = pair.clean_ids.to(device)

        hooked.reset_states()
        with relp_active():
            logits = hooked.model(clean_ids.unsqueeze(0)).logits
            metric, top_ids, _top_vals = top_k_logit_sum_metric(
                logits, target_pos=pair.target_pos, top_k=top_k,
            )
            captured = [st.captured for st in hooked.states]
            for c in captured:
                assert c is not None
            grads = torch.autograd.grad(metric, captured, retain_graph=False)

        T_pair = captured[0].shape[1]
        pair_token_attr = torch.zeros(
            (n_layers, T_pair, d_ffn), dtype=torch.float32, device=device,
        )
        for li, (g, h_clean) in enumerate(zip(grads, captured, strict=True)):
            # Bare h * grad — no subtraction. Sum over batch dim (size 1), keep token axis.
            score_lt = (g.to(torch.float32) * h_clean.detach().to(torch.float32)).sum(dim=0)
            pair_token_attr[li] = score_lt.to(device)

        pair_attr = pair_token_attr.sum(dim=1)
        accum += pair_attr
        if store_per_pair:
            per_pair_scores.append(pair_attr.cpu().numpy())
            per_pair_token_scores.append(pair_token_attr.cpu().numpy())
            per_pair_metric.append(float(metric.detach().item()))
            per_pair_token_ids.append(pair.clean_ids.cpu().tolist())
            per_pair_top_logit_ids.append(top_ids.cpu().tolist())

        hooked.reset_states()
        n_used += 1

    scores = (accum / max(n_used, 1)).cpu().numpy().astype(np.float32)

    pps = ppm = ppts = ppl = pptid = None
    extras: dict | None = {"top_k": top_k}
    if store_per_pair and per_pair_scores:
        pps = np.stack(per_pair_scores, axis=0).astype(np.float32)
        ppm = np.asarray(per_pair_metric, dtype=np.float32)
        T_max = max(s.shape[1] for s in per_pair_token_scores)
        ppts = np.full(
            (len(per_pair_token_scores), n_layers, T_max, d_ffn),
            np.nan, dtype=np.float32,
        )
        for p, s in enumerate(per_pair_token_scores):
            ppts[p, :, : s.shape[1], :] = s
        ppl = np.asarray([s.shape[1] for s in per_pair_token_scores], dtype=np.int32)
        pptid = per_pair_token_ids
        extras["per_pair_top_logit_ids"] = per_pair_top_logit_ids

    return AttributionResult(
        method="relp_paper",
        scores=scores,
        n_pairs=n_used,
        n_layers=n_layers,
        d_ffn=d_ffn,
        per_pair_scores=pps,
        per_pair_metric=ppm,
        per_pair_token_scores=ppts,
        per_pair_lengths=ppl,
        per_pair_token_ids=pptid,
        extras=extras,
    )
