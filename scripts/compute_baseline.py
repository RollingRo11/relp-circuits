"""Compute and cache the unpaired-RelP baseline mean for a given model.

This is a one-shot per (model, seq_len, n_tokens) — the baseline is shared across
any number of unpaired RelP attribution runs.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from relp_circuits.baseline import baseline_cache_path, compute_baseline_mean
from relp_circuits.model import HookedModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--local-shards", type=Path, required=True,
                   help="directory of *.jsonl.zst dolma shards")
    p.add_argument("--text-field", default="text")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument("--device", default="cuda",
                   help='"cuda" picks the freest single GPU; "auto" spreads across all visible GPUs.')
    p.add_argument("--artifacts-root", type=Path,
                   default=Path("/data/artifacts/rohan/relp-circuits"))
    return p.parse_args()


def _iter_shards(shard_dir: Path, text_field: str):
    import zstandard as zstd
    files = sorted(shard_dir.rglob("*.jsonl.zst"))
    if not files:
        raise SystemExit(f"no *.jsonl.zst under {shard_dir}")
    dctx = zstd.ZstdDecompressor()
    i = 0
    for path in files:
        with open(path, "rb") as f, dctx.stream_reader(f) as zf:
            text_stream = io.TextIOWrapper(zf, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = rec.get(text_field)
                if not txt:
                    continue
                yield i, txt
                i += 1


def main() -> int:
    args = parse_args()
    print(f"[baseline] model={args.model} seq_len={args.seq_len} "
          f"max_tokens={args.max_tokens:,}", flush=True)
    # SDPA forward is fine for baseline (no backward needed). For 32B+ use device="auto".
    hooked = HookedModel.load(args.model, device=args.device, dtype=torch.bfloat16,
                              attn_implementation="sdpa")
    print(f"[baseline] forward device(s) configured", flush=True)
    print(f"[baseline] model loaded; layers={hooked.n_layers} d_ffn={hooked.d_ffn}",
          flush=True)

    mean, n_seen = compute_baseline_mean(
        hooked,
        _iter_shards(args.local_shards, args.text_field),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    print(f"[baseline] saw {n_seen:,} tokens; mean shape {mean.shape}", flush=True)

    out = baseline_cache_path(args.artifacts_root, args.model, args.seq_len, n_seen)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mean)
    print(f"[baseline] saved {out}", flush=True)
    print(f"[baseline] stats per-layer (mean of |neuron means|): "
          f"{np.abs(mean).mean(axis=-1)[:6].tolist()}…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
