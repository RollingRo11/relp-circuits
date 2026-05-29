"""Shared types and the logit-difference metric used by all attribution methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AttributionResult:
    """Per-(layer, neuron) attribution scores, summed (or averaged) over a set of pairs."""
    method: str                   # "atp" | "ig" | "relp"
    scores: np.ndarray            # shape (n_layers, d_ffn), float32 — averaged over pairs
    n_pairs: int
    n_layers: int
    d_ffn: int
    # Per-pair attribution scores and metric values (only populated when the caller asks
    # for them — needed for the paper's τ-filter, where neurons are kept iff they pass
    # |score| ≥ τ · |metric| on the per-example level).
    per_pair_scores: np.ndarray | None = None   # (n_pairs, n_layers, d_ffn) float32 — summed over token
    per_pair_metric: np.ndarray | None = None   # (n_pairs,) float32
    # Per-pair, per-token attribution. Padded with NaN to T_max.
    per_pair_token_scores: np.ndarray | None = None   # (n_pairs, n_layers, T_max, d_ffn) float32
    per_pair_lengths: np.ndarray | None = None        # (n_pairs,) int — true T per pair
    per_pair_token_ids: list[list[int]] | None = None # (n_pairs,) list of clean token-id lists
    extras: dict | None = None

    def topk_global(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (layer_idx, neuron_idx) of top-k by absolute score across all layers."""
        flat = self.scores.reshape(-1)
        idx = np.argsort(-np.abs(flat))[:k]
        layer_idx = idx // self.d_ffn
        neuron_idx = idx % self.d_ffn
        return layer_idx, neuron_idx

    def tau_circuit(self, tau: float) -> list[dict]:
        """Per-example τ-filter (Arora et al. eq. for unpaired RelP):
            keep (l, n) iff |score_per_pair[l, n]| ≥ τ · |metric_per_pair| on at least one pair.
        Returns a list of dicts sorted by descending firing frequency, with
        {layer, neuron, frequency, max_norm_score, agg_score}.
        """
        if self.per_pair_scores is None or self.per_pair_metric is None:
            raise ValueError("per_pair_scores/per_pair_metric not stored on this result")
        from collections import Counter
        scores_pp = np.abs(self.per_pair_scores)         # (P, L, D)
        thresholds = tau * np.abs(self.per_pair_metric)  # (P,)
        thresholds = np.maximum(thresholds, 1e-12)
        passes = scores_pp >= thresholds[:, None, None]  # (P, L, D) bool
        freq = passes.sum(axis=0)                         # (L, D) int
        if not freq.any():
            return []

        # Build the per-(l, n) max normalized score across pairs that passed.
        norm = self.per_pair_scores / np.where(
            np.abs(self.per_pair_metric)[:, None, None] > 0,
            self.per_pair_metric[:, None, None], 1.0
        )
        # zero out pairs that didn't pass so max picks from passing ones only
        masked = np.where(passes, np.abs(norm), 0.0)
        max_norm = masked.max(axis=0)                     # (L, D)

        ls, ns = np.nonzero(freq)
        out = []
        for l, n in zip(ls.tolist(), ns.tolist(), strict=True):
            out.append({
                "layer": int(l),
                "neuron": int(n),
                "frequency": int(freq[l, n]),
                "max_norm_score": float(max_norm[l, n]),
                "agg_score": float(self.scores[l, n]),
            })
        out.sort(key=lambda r: (-r["frequency"], -r["max_norm_score"]))
        return out


def logit_diff_metric(
    logits: torch.Tensor,
    target_pos: int,
    correct_id: int,
    incorrect_id: int,
) -> torch.Tensor:
    """Logits at the answer position: correct minus incorrect. Higher = more correct."""
    # logits shape: (batch=1, seq, vocab)
    return logits[0, target_pos, correct_id] - logits[0, target_pos, incorrect_id]


def top_k_logit_sum_metric(
    logits: torch.Tensor,
    target_pos: int,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ADAG paper target (Eq. 8): sum of the top-K logits at the answer position.

    Returns (metric_scalar, top_ids, top_vals). The scalar is differentiable; the
    top-K ids/vals are detached for bookkeeping.

    Why top-K instead of a single logit: RelP's conservation property says
        Σ_{l,t,d} h_d^{(l,t)} · ∂M/∂h_d^{(l,t)} = M
    where M is the metric being attributed. Selecting the *single* argmax logit
    risks reducing attribution to one prediction; the paper sums the top-K
    (K=5) so the attribution captures the model's full distributional
    intent at the answer position.
    """
    pos_logits = logits[0, target_pos]
    top_vals, top_ids = pos_logits.topk(top_k)
    metric = top_vals.sum()
    return metric, top_ids.detach(), top_vals.detach()
