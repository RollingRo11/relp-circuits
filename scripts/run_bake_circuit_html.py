"""Bake a self-contained interactive circuit-tracer HTML from one of our
attribution runs.

Reads `--attr-dir` (must contain RelP attribution + ADAG profiles +
clusters + descriptions + edges.npz), runs the circuit-tracer JSON
export, then bundles everything into a single HTML using Transluce's
vendored frontend (under third_party/transluce/frontend_assets).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from transformers import AutoTokenizer

from relp_circuits.bake_html import bake_folder, bake_html
from relp_circuits.circuit_tracer_export import ExportConfig, export_circuit_tracer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET_ROOT = REPO_ROOT / "third_party/transluce/frontend_assets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr-dir", type=Path, required=True)
    p.add_argument("--model", required=True,
                   help="HF model id (used for tokenizer + scan label)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output HTML path")
    p.add_argument("--num-graphs", type=int, default=20,
                   help="How many of the prompt graphs to bake into the HTML")
    p.add_argument("--prompt-stride", type=int, default=1,
                   help="Bake every N-th prompt (useful for eval/deploy alternation)")
    p.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    p.add_argument("--slug-prefix", default="circuit")
    p.add_argument("--scan", default=None,
                   help="`scan` label for the frontend dropdown (defaults to model name)")
    p.add_argument("--edge-top-k-per-target", type=int, default=20)
    p.add_argument("--embed-top-k-per-feature", type=int, default=8)
    p.add_argument("--feature-threshold", type=float, default=1e-3)
    p.add_argument("--edge-threshold", type=float, default=1e-3)
    p.add_argument("--title", default=None)
    p.add_argument("--folder-out", type=Path, default=None,
                   help="If set, also emit a folder containing index.html + data.js "
                        "(side-by-side, loaded via <script src>) to this path. "
                        "Useful for file:// opens where the inlined-HTML approach is "
                        "fragile.")
    p.add_argument("--zip-out", type=Path, default=None,
                   help="If set together with --folder-out, also write a .zip of the "
                        "folder to this path so the artifact can be downloaded as a "
                        "single file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[bake] loading tokenizer for {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Look up num_layers from config without loading the full model.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    num_layers = int(cfg.num_hidden_layers)

    scan = args.scan or args.model.split("/")[-1]
    cfg_export = ExportConfig(
        out_dir=Path("placeholder"),    # filled in below
        slug_prefix=args.slug_prefix,
        scan=scan,
        edge_top_k_per_target=args.edge_top_k_per_target,
        embed_top_k_per_feature=args.embed_top_k_per_feature,
        feature_threshold=args.feature_threshold,
        edge_threshold=args.edge_threshold,
    )

    # Decide which prompts to bake.
    import numpy as np
    P_total = int(np.load(args.attr_dir / "per_pair_lengths.npy").shape[0])
    indices = list(range(0, P_total, args.prompt_stride))[: args.num_graphs]
    print(f"[bake] baking {len(indices)} of {P_total} prompts: {indices[:6]}{' ...' if len(indices) > 6 else ''}",
          flush=True)

    with tempfile.TemporaryDirectory() as td:
        bundle_root = Path(td)
        cfg_export.out_dir = bundle_root
        t0 = time.time()
        export_circuit_tracer(
            attr_dir=args.attr_dir,
            out_dir=bundle_root,
            tokenizer=tokenizer,
            num_layers=num_layers,
            cfg=cfg_export,
            pair_indices=indices,
        )
        n_files = len(list((bundle_root / "graph_data").glob("*.json")))
        sizes = sum((bundle_root / "graph_data" / f.name).stat().st_size
                    for f in (bundle_root / "graph_data").iterdir())
        print(f"[bake] exporter wrote {n_files} per-prompt graphs "
              f"({sizes / 1e6:.1f} MB) in {time.time()-t0:.1f}s",
              flush=True)

        title = args.title or f"RelP circuit — {scan}"
        # Pre-set the URL so the frontend opens on the first prompt.
        first_slug = f"{args.slug_prefix}__{indices[0]:03d}"

        size = bake_html(
            asset_root=args.asset_root,
            bundle_root=bundle_root,
            out_html=args.out,
            title=title,
            initial_slug=first_slug,
        )
        print(f"[bake] wrote {args.out} ({size / 1e6:.1f} MB)", flush=True)

        if args.folder_out is not None:
            sizes = bake_folder(
                asset_root=args.asset_root,
                bundle_root=bundle_root,
                out_dir=args.folder_out,
                title=title,
                initial_slug=first_slug,
            )
            print(
                f"[bake] folder {args.folder_out}/  "
                f"index.html={sizes['html_bytes'] / 1e3:.0f} KB, "
                f"data.js={sizes['data_bytes'] / 1e6:.1f} MB",
                flush=True,
            )

            if args.zip_out is not None:
                import zipfile
                args.zip_out.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(args.zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in args.folder_out.iterdir():
                        zf.write(p, arcname=f"{args.folder_out.name}/{p.name}")
                print(f"[bake] zip   {args.zip_out} ({args.zip_out.stat().st_size / 1e6:.1f} MB)",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
