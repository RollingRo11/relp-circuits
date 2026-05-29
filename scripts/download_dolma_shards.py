"""Pre-download a fixed number of Dolma 3 mix shards to /data so collect_acts can
iterate from local files instead of streaming over the HF hub (which dropped us
from ~14k to ~5-9k tok/s mid-run on the 1B job).

The mix is shipped as zstd-compressed jsonl shards (one row per doc, "text" field).
We snapshot the first N shards into the configured cache and let HF datasets find
them via its normal cache resolver.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="allenai/dolma3_mix-6T-1025-7B")
    p.add_argument("--out", type=Path,
                   default=Path("/data/artifacts/rohan/relp-circuits/dolma3"))
    p.add_argument("--shards-per-source", type=int, default=2,
                   help="how many shards to pull from each non-skipped source — round-robin "
                        "for diverse coverage (replaces the old --num-shards alphabetic pick)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-source-substr", action="append",
                   default=["adult_content"],
                   help="substrings; any shard whose path contains one is dropped (default: "
                        "adult_content). Repeatable.")
    return p.parse_args()


def _source_of(path: str) -> str:
    """Group by directory under data/, dropping the trailing -NNNN id (e.g.
    'common_crawl-finance_and_business-0034' → 'common_crawl-finance_and_business')."""
    parts = path.split("/")
    if len(parts) < 2:
        return path
    sub = parts[1]
    return sub.rsplit("-", 1)[0]


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    print(f"[dl] listing files in {args.repo}", flush=True)
    files = api.list_repo_files(args.repo, repo_type="dataset")
    matching = sorted([f for f in files if f.endswith(".jsonl.zst") or f.endswith(".jsonl.gz")])

    # Drop any source matching the skip list.
    filtered = []
    for f in matching:
        if any(s in f for s in args.skip_source_substr):
            continue
        filtered.append(f)

    # Group by source and pick the first N shards per source.
    by_source: dict[str, list[str]] = defaultdict(list)
    for f in filtered:
        by_source[_source_of(f)].append(f)

    chosen: list[str] = []
    for src in sorted(by_source.keys()):
        chosen.extend(by_source[src][: args.shards_per_source])
    print(f"[dl] {len(matching)} total shards, {len(filtered)} after skip — "
          f"selecting {args.shards_per_source} per source across {len(by_source)} sources "
          f"= {len(chosen)} shards", flush=True)
    for src in sorted(by_source.keys()):
        n_picked = min(args.shards_per_source, len(by_source[src]))
        print(f"      [{n_picked:>2}]  {src}")

    t0 = time.time()
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        cache_dir=str(args.out),
        allow_patterns=chosen + ["*.json", "*.md", "README*"],
        max_workers=args.workers,
    )
    elapsed = time.time() - t0
    print(f"[dl] done in {elapsed:.0f}s. cache root: {args.out}", flush=True)
    # Print a one-liner summary of what's local now
    total_bytes = 0
    n_files = 0
    for root, _dirs, fs in os.walk(args.out):
        for fn in fs:
            n_files += 1
            try:
                total_bytes += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    print(f"[dl] local cache: {n_files} files, {total_bytes/1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
