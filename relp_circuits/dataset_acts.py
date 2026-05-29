"""Streaming pass through a tokenized corpus to find, for each MLP neuron, the dataset
spans where it fires most strongly.

The output is a per-(layer, neuron) top-K index, stored as two arrays plus a doc manifest:

  topk_vals  shape (n_layers, K, d_ffn)        float32  activation values
  topk_meta  shape (n_layers, K, d_ffn, 2)     int64    (doc_id, token_pos)
  docs.parquet                                          doc_id -> text + tokens

This is the data layer behind a Neuronpedia-style view: pick a (layer, neuron), look up
the K highest-activating tokens across the corpus, slice the surrounding context out of
the doc manifest. The same index supports the inverse query: given a token in a doc,
find which neurons activated highly there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm

from relp_circuits.model import HookedModel


@dataclass
class StreamingTopK:
    """Vectorised running top-K over per-neuron activations.

    Per layer, holds (K, d_ffn) best values and (K, d_ffn, 2) — (doc_id, pos) — meta.
    """
    n_layers: int
    d_ffn: int
    K: int
    device: torch.device
    vals: torch.Tensor                 # (n_layers, K, d_ffn) float32, init -inf
    doc_ids: torch.Tensor              # (n_layers, K, d_ffn) int64
    positions: torch.Tensor            # (n_layers, K, d_ffn) int64

    @classmethod
    def empty(cls, n_layers: int, d_ffn: int, K: int, device: torch.device) -> "StreamingTopK":
        return cls(
            n_layers=n_layers,
            d_ffn=d_ffn,
            K=K,
            device=device,
            vals=torch.full((n_layers, K, d_ffn), float("-inf"), dtype=torch.float32, device=device),
            doc_ids=torch.full((n_layers, K, d_ffn), -1, dtype=torch.int64, device=device),
            positions=torch.full((n_layers, K, d_ffn), -1, dtype=torch.int64, device=device),
        )

    def update(
        self,
        batch_acts: list[torch.Tensor],          # n_layers entries of (B, T, d_ffn)
        batch_doc_ids: torch.Tensor,             # (B*T,) int64
        batch_positions: torch.Tensor,           # (B*T,) int64
        batch_valid_mask: torch.Tensor,          # (B*T,) bool — False at padding
    ) -> None:
        for li, h in enumerate(batch_acts):
            flat = h.reshape(-1, self.d_ffn).to(torch.float32)         # (N, d_ffn)
            # Mask out padding by replacing with -inf.
            invalid = ~batch_valid_mask
            if invalid.any():
                flat = flat.clone()
                flat[invalid] = float("-inf")
            N = flat.shape[0]
            K_use = min(self.K, N)
            batch_vals, batch_idx = flat.topk(K_use, dim=0)            # (K_use, d_ffn)
            batch_doc = batch_doc_ids[batch_idx]                       # (K_use, d_ffn)
            batch_pos = batch_positions[batch_idx]                     # (K_use, d_ffn)

            # Concatenate with current top-K and select top-K again.
            combined_vals = torch.cat([self.vals[li], batch_vals], dim=0)            # (K + K_use, d_ffn)
            combined_doc = torch.cat([self.doc_ids[li], batch_doc], dim=0)
            combined_pos = torch.cat([self.positions[li], batch_pos], dim=0)
            new_vals, new_idx = combined_vals.topk(self.K, dim=0)                    # (K, d_ffn)
            self.vals[li] = new_vals
            self.doc_ids[li] = torch.gather(combined_doc, 0, new_idx)
            self.positions[li] = torch.gather(combined_pos, 0, new_idx)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "topk_vals.npy", self.vals.cpu().numpy())
        np.save(out_dir / "topk_doc_ids.npy", self.doc_ids.cpu().numpy())
        np.save(out_dir / "topk_positions.npy", self.positions.cpu().numpy())


@torch.no_grad()
def collect_activations(
    hooked: HookedModel,
    docs: Iterator[tuple[int, str]],
    seq_len: int,
    batch_size: int,
    K: int,
    max_tokens: int,
    log_every: int = 50,
) -> tuple[StreamingTopK, list[tuple[int, np.ndarray]]]:
    """Run forward over `docs`, updating a streaming top-K per neuron.

    Returns the top-K state and the list of (doc_id, token_ids_int32) that were
    processed (used to render context windows at query time). Tokens are stored as
    numpy int32 arrays rather than python lists — for billion-token runs the python-int
    overhead would otherwise blow past 100 GB RAM.
    """
    tokenizer = hooked.tokenizer
    device = hooked.device
    topk = StreamingTopK.empty(
        n_layers=hooked.n_layers, d_ffn=hooked.d_ffn, K=K, device=device
    )
    seen_tokens = 0
    seen_docs: list[tuple[int, np.ndarray]] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_doc_ids: list[int] = []
    batch_token_arrays: list[np.ndarray] = []
    pbar = tqdm(total=max_tokens, desc="acts", unit="tok")

    def flush() -> None:
        nonlocal seen_tokens
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
        per_layer = [st.captured for st in hooked.states]
        for c in per_layer:
            assert c is not None

        doc_ids_flat = torch.tensor(batch_doc_ids, dtype=torch.int64, device=device).repeat_interleave(T)
        positions_flat = torch.arange(T, dtype=torch.int64, device=device).repeat(B)
        valid_flat = valid.reshape(-1)

        topk.update(per_layer, doc_ids_flat, positions_flat, valid_flat)
        added = int(valid_flat.sum().item())
        seen_tokens += added
        pbar.update(added)
        batch_doc_ids.clear()
        batch_token_arrays.clear()

    for doc_id, text in docs:
        toks = tokenizer.encode(text, add_special_tokens=True)
        toks = toks[:seq_len]
        if not toks:
            continue
        toks_arr = np.asarray(toks, dtype=np.int32)
        seen_docs.append((doc_id, toks_arr))
        batch_doc_ids.append(doc_id)
        batch_token_arrays.append(toks_arr)
        if len(batch_doc_ids) >= batch_size:
            flush()
        if seen_tokens >= max_tokens:
            break

    flush()
    pbar.close()
    return topk, seen_docs
