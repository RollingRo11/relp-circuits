"""SGLang-driven activation collection for OLMo-3.

This is the fast path: forward via SGLang's FA3 kernels, capture
`silu(gate(x))·up(x)` per MLP layer, update an in-process streaming top-K
heap. Atexit hook in the SGLang subprocess flushes the heap to disk; the
driver (this process) writes the doc manifest separately.

Run via slurm/collect_acts_sglang.sbatch — `SGLANG_EXTERNAL_MODEL_PACKAGE`
must point at our shim package, which means `relp_circuits` must be on
PYTHONPATH inside the SGLang scheduler subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="allenai/dolma3_mix-6T-1025-7B")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--local-shards", type=Path, default=None,
                   help="Path to a directory of *.jsonl.zst shards to iterate locally. "
                        "Bypasses HF streaming.")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64,
                   help="how many docs to push per engine.generate() call")
    p.add_argument("--max-tokens", type=int, default=10_000_000)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--chunked-prefill-size", type=int, default=32768)
    p.add_argument("--max-running-requests", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", default="",
                   help="comma-separated layer indices to capture, blank = all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Read the model config first so we can size the heap state correctly. (The
    # shim auto-detection from inside DecoderLayer __init__ races with capture
    # in some setups; passing it explicitly via env is robust.)
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(args.model)
    n_layers = getattr(model_cfg, "num_hidden_layers", None)
    print(f"[acts-sglang] model has {n_layers} layers, "
          f"intermediate_size={getattr(model_cfg, 'intermediate_size', '?')}", flush=True)

    # Wire env vars BEFORE importing sglang. The shim reads these at import time.
    # SGLang's registry iterates `package.__path__` to discover modules with
    # EntryClass, so the env var must name a PACKAGE (not a single module).
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "relp_circuits.sglang_harvester"
    os.environ["HARVEST_OUT_DIR"] = str(args.output)
    os.environ["HARVEST_K"] = str(args.topk)
    os.environ["HARVEST_TP_SIZE"] = str(args.tp_size)
    if n_layers:
        os.environ["HARVEST_N_LAYERS"] = str(n_layers)
    if args.layers:
        os.environ["HARVEST_LAYERS"] = args.layers

    # PYTHONPATH so the SGLang scheduler subprocess can import our shim.
    repo_root = Path(__file__).resolve().parents[1]
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{repo_root}:{existing_pp}" if existing_pp else str(repo_root)
    )

    print(f"[acts-sglang] model={args.model}", flush=True)
    print(f"[acts-sglang] dataset={args.dataset} max_tokens={args.max_tokens:,}", flush=True)
    print(f"[acts-sglang] batch={args.batch_size} seq_len={args.seq_len} K={args.topk}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[acts-sglang] tokenizer loaded", flush=True)

    def _doc_iter():
        if args.local_shards is not None:
            import io
            import json as _json
            import zstandard as zstd
            files = sorted(args.local_shards.rglob("*.jsonl.zst"))
            if not files:
                raise SystemExit(f"no *.jsonl.zst under {args.local_shards}")
            # Round-robin across source dirs so a small token budget doesn't
            # burn out one corpus subset before touching the others.
            by_parent: dict = {}
            for f in files:
                by_parent.setdefault(f.parent, []).append(f)
            print(f"[acts-sglang] iterating {len(files)} local shards across "
                  f"{len(by_parent)} sources, round-robin", flush=True)

            def shard_docs(path):
                # IMPORTANT: each generator needs its OWN ZstdDecompressor.
                # A single dctx is not safe for concurrent stream_reader use —
                # they share the underlying ZSTD_DCtx state and corrupt each
                # other's reads (we saw this as <100k tokens/run instead of 100M).
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
                            txt = rec.get(args.text_field)
                            if txt:
                                yield txt
                except (zstd.ZstdError, OSError) as e:
                    print(f"[acts-sglang] WARN skipping corrupt shard {path.name}: {e}",
                          flush=True)

            interleaved = []
            cursors = {p: 0 for p in by_parent}
            while True:
                stepped = False
                for parent in sorted(by_parent):
                    if cursors[parent] < len(by_parent[parent]):
                        interleaved.append(by_parent[parent][cursors[parent]])
                        cursors[parent] += 1
                        stepped = True
                if not stepped:
                    break
            gens = [shard_docs(f) for f in interleaved]
            active = list(range(len(gens)))
            i = 0
            while active:
                nxt = []
                for gi in active:
                    try:
                        txt = next(gens[gi])
                    except StopIteration:
                        continue
                    yield i, txt
                    i += 1
                    nxt.append(gi)
                active = nxt
        else:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
            for i, ex in enumerate(ds):
                txt = ex.get(args.text_field) or ""
                if not txt:
                    continue
                yield i, txt

    import sglang as sgl
    print(f"[acts-sglang] starting engine (tp={args.tp_size})", flush=True)
    t_engine = time.time()
    engine = sgl.Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        chunked_prefill_size=args.chunked_prefill_size,
        max_running_requests=args.max_running_requests,
        random_seed=0,
        log_level="warning",
    )
    print(f"[acts-sglang] engine ready in {time.time()-t_engine:.1f}s", flush=True)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}

    docs_index: list[tuple[int, list[int]]] = []
    doc_id = 0
    tokens_so_far = 0
    batch_input_ids: list[list[int]] = []
    batch_rids: list[str] = []
    t_run = time.time()
    last_log = t_run

    def flush_batch() -> int:
        nonlocal tokens_so_far
        if not batch_input_ids:
            return 0
        n_tok = sum(len(x) for x in batch_input_ids)
        engine.generate(
            input_ids=batch_input_ids,
            sampling_params=sampling_params,
            rid=batch_rids,
        )
        batch_input_ids.clear()
        batch_rids.clear()
        tokens_so_far += n_tok
        return n_tok

    for _, text in _doc_iter():
        if tokens_so_far >= args.max_tokens:
            break
        ids = tok.encode(text, add_special_tokens=True)[: args.seq_len]
        if not ids:
            continue
        docs_index.append((doc_id, ids))
        batch_input_ids.append(ids)
        batch_rids.append(f"{doc_id}|0")
        doc_id += 1
        if len(batch_input_ids) >= args.batch_size:
            flush_batch()
            now = time.time()
            if now - last_log > 10:
                rate = tokens_so_far / max(now - t_run, 1e-3)
                pct = 100 * tokens_so_far / max(args.max_tokens, 1)
                print(f"[acts-sglang] {tokens_so_far:>10,} / {args.max_tokens:,}  "
                      f"({pct:5.1f}%)  {rate:.0f} tok/s", flush=True)
                last_log = now

    flush_batch()
    elapsed = time.time() - t_run
    rate = tokens_so_far / max(elapsed, 1e-3)
    print(f"[acts-sglang] streaming done — {tokens_so_far:,} tokens in "
          f"{elapsed:.1f}s ({rate:.0f} tok/s)", flush=True)

    # Driver writes docs.parquet (so query_acts can render context spans).
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "doc_id": [d for d, _ in docs_index],
        "tokens": [t for _, t in docs_index],
    })
    pq.write_table(table, args.output / "docs.parquet")

    (args.output / "config.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "split": args.dataset_split,
        "seq_len": args.seq_len, "batch_size": args.batch_size, "topk": args.topk,
        "max_tokens": args.max_tokens, "n_docs_processed": len(docs_index),
        "elapsed_s": elapsed, "tok_per_s": rate, "engine": "sglang",
        "tp_size": args.tp_size,
    }, indent=2))

    # engine.shutdown() triggers our SIGTERM handler in each scheduler subprocess
    # (one per TP rank), each writes its column-sharded heap to shard_rank{N}/.
    print("[acts-sglang] shutting down engine (each rank flushes its shard)…", flush=True)
    engine.shutdown()

    # Driver stitches the per-rank shards into a single (n_layers, K, d_ffn) view
    # so query_acts.py / render_circuit.py can use it the same as a TP=1 run.
    print(f"[acts-sglang] stitching per-rank shards", flush=True)
    _stitch_rank_shards(args.output, args.tp_size)

    print(f"[acts-sglang] OK. Artifacts: {args.output}", flush=True)
    return 0


def _stitch_rank_shards(out_dir: Path, tp_size: int) -> None:
    """Concatenate the per-rank heap shards along the d_ffn axis to produce the
    final topk_vals.npy / topk_doc_ids.npy / topk_positions.npy at the run
    root. doc_id and position metadata are per-(rank, neuron-shard); after
    concat the indices line up with global neuron columns automatically."""
    import numpy as np
    shard_dirs = sorted([out_dir / f"shard_rank{r:02d}" for r in range(tp_size)])
    missing = [d for d in shard_dirs if not (d / "topk_vals.npy").exists()]
    if missing:
        print(f"[stitch] WARNING: {len(missing)} of {tp_size} rank shards missing — "
              f"is incremental flush working in every rank?", flush=True)
        shard_dirs = [d for d in shard_dirs if d not in missing]

    if not shard_dirs:
        print(f"[stitch] no rank shards found under {out_dir}; skipping", flush=True)
        return

    print(f"[stitch] reading {len(shard_dirs)} rank shards", flush=True)
    vals = np.concatenate([np.load(d / "topk_vals.npy") for d in shard_dirs], axis=-1)
    docs = np.concatenate([np.load(d / "topk_doc_ids.npy") for d in shard_dirs], axis=-1)
    poss = np.concatenate([np.load(d / "topk_positions.npy") for d in shard_dirs], axis=-1)
    np.save(out_dir / "topk_vals.npy", vals)
    np.save(out_dir / "topk_doc_ids.npy", docs)
    np.save(out_dir / "topk_positions.npy", poss)
    print(f"[stitch] wrote {out_dir}/topk_*.npy  shape={vals.shape}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
