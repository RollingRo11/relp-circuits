"""Compute the per-(layer, neuron) baseline mean activation needed by unpaired RelP.

Unpaired RelP (Arora et al. Section 6) replaces the per-example counterfactual
with `E[h_l(x') for x' ~ corpus]` so attribution can run on any prompt without
constructing a minimal pair. We compute that mean once per (model, corpus
sample, seq_len) and cache it on disk.

The mean is in-place numerically stable: we accumulate fp64 sums on GPU and
divide by token count at the end. Padding tokens are masked out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.model import HookedModel


@torch.no_grad()
def compute_baseline_mean(
    hooked: HookedModel,
    docs: Iterator[tuple[int, str]],
    seq_len: int = 512,
    batch_size: int = 8,
    max_tokens: int = 100_000,
) -> tuple[np.ndarray, int]:
    """Run forward over `docs` and return the per-(layer, neuron) mean activation.

    Returns: (mean: (n_layers, d_ffn) float32, n_tokens_seen: int)
    """
    n_layers = hooked.n_layers
    d_ffn = hooked.d_ffn
    device = hooked.device
    tokenizer = hooked.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # fp64 accumulator to keep numerical error small over millions of tokens.
    sum_acts = torch.zeros((n_layers, d_ffn), dtype=torch.float64, device=device)
    n_tokens_seen = 0

    batch_doc_ids: list[int] = []
    batch_token_arrays: list[np.ndarray] = []
    pbar = tqdm(total=max_tokens, desc="baseline", unit="tok")

    def flush() -> None:
        nonlocal n_tokens_seen
        if not batch_doc_ids:
            return
        T = seq_len
        B = len(batch_doc_ids)
        ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
        valid = torch.zeros((B, T), dtype=torch.bool, device=device)
        for bi, toks in enumerate(batch_token_arrays):
            n = min(len(toks), T)
            ids[bi, :n] = torch.from_numpy(toks[:n].astype(np.int64)).to(device)
            valid[bi, :n] = True

        hooked.reset_states()
        hooked.model(ids)
        valid_flat = valid.reshape(-1)
        n_valid = int(valid_flat.sum().item())
        for li, st in enumerate(hooked.states):
            assert st.captured is not None
            h = st.captured.reshape(-1, d_ffn)
            # Sum only valid positions; cast to fp64 for accumulation. Under
            # device_map="auto" each layer's h lives on a different GPU; move
            # the valid mask to match before indexing.
            mask = valid_flat.to(h.device)
            sum_acts[li] += h[mask].to(torch.float64).sum(dim=0).to(sum_acts.device)
        n_tokens_seen += n_valid
        pbar.update(n_valid)
        batch_doc_ids.clear()
        batch_token_arrays.clear()

    for doc_id, text in docs:
        toks = tokenizer.encode(text, add_special_tokens=True)[:seq_len]
        if not toks:
            continue
        toks_arr = np.asarray(toks, dtype=np.int32)
        batch_doc_ids.append(doc_id)
        batch_token_arrays.append(toks_arr)
        if len(batch_doc_ids) >= batch_size:
            flush()
        if n_tokens_seen >= max_tokens:
            break

    flush()
    pbar.close()

    if n_tokens_seen == 0:
        raise RuntimeError("compute_baseline_mean saw zero tokens")
    mean = (sum_acts / float(n_tokens_seen)).cpu().numpy().astype(np.float32)
    return mean, n_tokens_seen


def baseline_cache_path(
    artifacts_root: Path,
    model_id: str,
    seq_len: int,
    n_tokens: int,
) -> Path:
    """Stable path for the per-(model, seq_len, n_tokens) baseline cache."""
    safe = model_id.replace("/", "__")
    return artifacts_root / "baselines" / f"{safe}_seq{seq_len}_n{n_tokens}.npy"
