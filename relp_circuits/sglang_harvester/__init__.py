"""SGLang external-package shim that captures OLMo-3 MLP post-`silu(gate)*up`
intermediate (d_ffn=11008) at the speed of FA3 inference.

Layout mirrors the standalone activation-harvester repo's pattern but at a
different capture point: we patch `Olmo2MLP.forward` (since SGLang routes
Olmo3 through Olmo2 via `sglang/srt/configs/olmo3.py`), exposing the d_ffn
vector that feeds `down_proj`. State lives entirely in the SGLang scheduler
subprocess; on shutdown the streaming top-K-per-neuron heap is flushed to
the configured output dir as `topk_vals.npy / topk_doc_ids.npy /
topk_positions.npy`. The caller (driver process) is responsible for writing
`docs.parquet` with the doc_id → token_ids mapping.

Entry point is `relp_circuits.sglang_harvester.olmo2`, registered via
`SGLANG_EXTERNAL_MODEL_PACKAGE=relp_circuits.sglang_harvester.olmo2`.
"""
