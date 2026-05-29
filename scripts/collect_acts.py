"""Submit-via-slurm entry point for the streaming max-activating-examples pass.

Default corpus is allenai/dolma3_mix-6T-1025-7B (the OLMo-3 pretraining mix). Streaming
mode means we never download the full 5T tokens; the iterator just walks shards from HF
on the fly and we stop at --max-tokens.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from relp_circuits.dataset_acts import collect_activations
from relp_circuits.model import HookedModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="allenai/dolma3_mix-6T-1025-7B")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--local-shards", type=Path, default=None,
                   help="Path to a directory of pre-downloaded *.jsonl.zst shards. If "
                        "set, iterate them directly instead of HF streaming (kills the "
                        "network bottleneck that dropped 14k → 5-9k tok/s mid-run).")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--max-docs", type=int, default=None,
                   help="Cap on docs streamed before max-tokens fires. Defensive bound.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _iter_local_shards(shard_dir: Path, text_field: str):
    """Yield (i, text) from local zstd-compressed jsonl shards.

    **Round-robin across source directories** so a 100M-token cap doesn't burn
    out one corpus subset (e.g. all the science PDFs) before touching others.
    Within each shard we read its docs in file order; across shards we
    interleave one-doc-per-shard-per-step.
    """
    import io
    import json as _json
    import zstandard as zstd

    files = sorted(shard_dir.rglob("*.jsonl.zst"))
    if not files:
        raise SystemExit(f"no *.jsonl.zst found under {shard_dir}")
    # Group by parent dir (which encodes the Dolma source).
    by_parent: dict[Path, list[Path]] = {}
    for f in files:
        by_parent.setdefault(f.parent, []).append(f)
    print(f"[acts] iterating {len(files)} local shards across {len(by_parent)} sources, "
          f"round-robin", flush=True)

    # Open one doc-yielding generator per shard, then interleave them.
    # IMPORTANT: each generator needs its OWN ZstdDecompressor — sharing one
    # across concurrent stream_readers corrupts the underlying context state.

    def shard_docs(path: Path):
        local_dctx = zstd.ZstdDecompressor()
        try:
            with open(path, "rb") as fh, local_dctx.stream_reader(fh) as zf:
                for line in io.TextIOWrapper(zf, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    txt = rec.get(text_field)
                    if txt:
                        yield txt
        except (zstd.ZstdError, OSError) as e:
            print(f"[acts] WARN skipping corrupt shard {path.name}: {e}", flush=True)

    # One generator per shard, ordered so adjacent shards come from different sources.
    interleaved_files: list[Path] = []
    cursors = {p: 0 for p in by_parent}
    while True:
        progressed = False
        for parent in sorted(by_parent):
            if cursors[parent] < len(by_parent[parent]):
                interleaved_files.append(by_parent[parent][cursors[parent]])
                cursors[parent] += 1
                progressed = True
        if not progressed:
            break

    gens = [shard_docs(f) for f in interleaved_files]
    active = list(range(len(gens)))
    i = 0
    while active:
        next_active = []
        for gi in active:
            try:
                txt = next(gens[gi])
            except StopIteration:
                continue
            yield i, txt
            i += 1
            next_active.append(gi)
        active = next_active


def docs_iterator(args, max_docs: int | None):
    if args.local_shards is not None:
        it = _iter_local_shards(args.local_shards, args.text_field)
        for i, text in it:
            if max_docs is not None and i >= max_docs:
                return
            yield i, text
        return
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = ex.get(args.text_field)
        if not text:
            continue
        yield i, text


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[acts] loading {args.model}", flush=True)
    t0 = time.time()
    # collect_acts is forward-only: no RelP backward, no need for the eager attention
    # path. SDPA gets us 1.5-2x on 7B without changing the captured neurons.
    hooked = HookedModel.load(args.model, device=args.device, dtype=torch.bfloat16,
                              attn_implementation="sdpa")
    print(f"[acts] model loaded in {time.time()-t0:.1f}s; "
          f"layers={hooked.n_layers} d_ffn={hooked.d_ffn}", flush=True)

    # max_docs bound: if not specified, allow ~max_tokens/seq_len docs plus 50% headroom.
    max_docs = args.max_docs or int((args.max_tokens / max(args.seq_len, 1)) * 1.5)
    print(f"[acts] streaming up to {args.max_tokens:,} tokens "
          f"({max_docs:,} doc cap) seq_len={args.seq_len} batch={args.batch_size} K={args.topk}",
          flush=True)

    t0 = time.time()
    topk, seen_docs = collect_activations(
        hooked,
        docs_iterator(args, max_docs=max_docs),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        K=args.topk,
        max_tokens=args.max_tokens,
    )
    print(f"[acts] streaming pass done in {time.time()-t0:.1f}s; "
          f"docs_processed={len(seen_docs)}", flush=True)

    topk.save(args.output)

    # Save the doc tokens so query_acts can render context. For 10M tokens at int32
    # this is ~40MB; for larger runs we'd switch to per-doc parquet files.
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "doc_id": [d for d, _ in seen_docs],
        # tokens come in as int32 numpy arrays; pyarrow handles that as a list column.
        "tokens": [t.tolist() for _, t in seen_docs],
    })
    pq.write_table(table, args.output / "docs.parquet")

    (args.output / "config.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "split": args.dataset_split,
        "seq_len": args.seq_len, "batch_size": args.batch_size, "topk": args.topk,
        "max_tokens": args.max_tokens, "n_layers": hooked.n_layers, "d_ffn": hooked.d_ffn,
        "n_docs_processed": len(seen_docs),
    }, indent=2))

    print(f"[acts] OK. Artifacts: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
