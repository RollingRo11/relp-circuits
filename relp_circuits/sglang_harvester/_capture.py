"""Activation-capture machinery for the SGLang external-package shim.

NB: SGLang shuts down the scheduler subprocess via SIGTERM, which doesn't run
atexit reliably. We therefore flush the streaming top-K heap to disk
incrementally (every `_FLUSH_EVERY_N_BATCHES` batches and at-most every
`_FLUSH_MIN_INTERVAL_S` seconds), and additionally on SIGTERM/SIGINT. The atexit
hook is kept as a final-line-of-defence but we no longer rely on it.


State-of-the-world

  install_universal_patches() — patches `ForwardBatch.init_new` so each batch
                                   carries `_harvest_rids` we set on the user
                                   side via `engine.generate(rid=[...])`.

  patch_olmo3_capture()       — patches `Olmo2DecoderLayer.__init__` to record
                                   layer_id on its `mlp` member, patches
                                   `Olmo2DecoderLayer.forward` to stash the
                                   forward_batch in a thread-local before
                                   running, and patches `Olmo2MLP.forward` to
                                   recompute the SwiGLU intermediate in named
                                   submodules so we can pick off the d_ffn
                                   vector (the input to `down_proj`).

  flush_state_to_disk(out)    — atexit hook. Serialises the in-process top-K
                                   tensors to {topk_vals, topk_doc_ids,
                                   topk_positions}.npy in `out`.

Configuration is via env vars set BEFORE `import sglang` in the driver:

  HARVEST_OUT_DIR   directory to write top-K state into on shutdown
  HARVEST_LAYERS    comma-separated layer indices to capture (default: all)
  HARVEST_K         top-K per (layer, neuron); default 32
  HARVEST_TP_SIZE   tensor-parallel size (informational; only tp_rank=0 captures)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- env config

_HARVEST_OUT_DIR = os.environ.get("HARVEST_OUT_DIR")
_HARVEST_LAYERS_RAW = os.environ.get("HARVEST_LAYERS", "")
_HARVEST_LAYERS: set[int] | None = (
    {int(x) for x in _HARVEST_LAYERS_RAW.split(",") if x.strip()} if _HARVEST_LAYERS_RAW else None
)
_HARVEST_K = int(os.environ.get("HARVEST_K", "32"))
_HARVEST_TP_SIZE = int(os.environ.get("HARVEST_TP_SIZE", "1"))
_HARVEST_N_LAYERS = int(os.environ.get("HARVEST_N_LAYERS", "0")) or None
_FLUSH_EVERY_N_BATCHES = int(os.environ.get("HARVEST_FLUSH_EVERY", "20"))
_FLUSH_MIN_INTERVAL_S = float(os.environ.get("HARVEST_FLUSH_MIN_INTERVAL_S", "10"))


# ---------------------------------------------------------------- per-process state

_LOCAL = threading.local()
_state_lock = threading.Lock()
_state: dict | None = None     # {'vals', 'doc_ids', 'positions'} tensors on GPU
_state_n_layers: int | None = None
_state_d_ffn: int | None = None

_universal_patched = False
_module_patched = False
_flushed = False
_n_capture_calls = 0   # batch counter, drives periodic flush
_last_flush_t = 0.0    # wall-clock of last flush
_signal_installed = False


def _tp_rank() -> int:
    """tp_rank for this worker. 0 if distributed not initialised."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() % max(1, _HARVEST_TP_SIZE)
    return 0


def _ensure_state(device: torch.device, n_layers: int, d_ffn: int) -> None:
    global _state, _state_n_layers, _state_d_ffn
    if _state is not None:
        return
    with _state_lock:
        if _state is not None:
            return
        _state = {
            "vals": torch.full((n_layers, _HARVEST_K, d_ffn), float("-inf"),
                                dtype=torch.float32, device=device),
            "doc_ids": torch.full((n_layers, _HARVEST_K, d_ffn), -1,
                                   dtype=torch.int64, device=device),
            "positions": torch.full((n_layers, _HARVEST_K, d_ffn), -1,
                                     dtype=torch.int64, device=device),
        }
        _state_n_layers = n_layers
        _state_d_ffn = d_ffn


def _update_topk(layer_id: int, h: torch.Tensor,
                 doc_ids: torch.Tensor, positions: torch.Tensor) -> None:
    """h: (N, d_ffn) flat over tokens in this batch. doc_ids, positions: (N,) int64."""
    K = _HARVEST_K
    N = h.shape[0]
    if N == 0:
        return
    K_use = min(K, N)
    h_f32 = h.to(torch.float32)
    batch_vals, batch_idx = h_f32.topk(K_use, dim=0)        # (K_use, d_ffn)
    batch_doc = doc_ids[batch_idx]                          # (K_use, d_ffn)
    batch_pos = positions[batch_idx]                        # (K_use, d_ffn)

    s = _state
    combined_vals = torch.cat([s["vals"][layer_id], batch_vals], dim=0)         # (K + K_use, d_ffn)
    combined_doc = torch.cat([s["doc_ids"][layer_id], batch_doc], dim=0)
    combined_pos = torch.cat([s["positions"][layer_id], batch_pos], dim=0)
    new_vals, new_idx = combined_vals.topk(K, dim=0)
    s["vals"][layer_id] = new_vals
    s["doc_ids"][layer_id] = torch.gather(combined_doc, 0, new_idx)
    s["positions"][layer_id] = torch.gather(combined_pos, 0, new_idx)


def _capture_intermediate(layer_id: int | None, intermediate: torch.Tensor) -> None:
    """Called from the patched Olmo2MLP.forward with `silu(gate(x)) * up(x)`,
    shape (N_total, d_ffn // tp_size) on each TP rank — the d_ffn dim is sharded
    along the columns of the row-parallel `down_proj`.

    **Each rank captures its own shard independently** (no NCCL comm). Ranks
    write to per-rank subdirectories `shard_rank{R}/topk_*.npy`; the driver
    stitches them by concatenation along the d_ffn axis after engine shutdown.
    This skips the all_gather we previously did on every MLP layer × every
    batch — major speedup for many-layer captures (e.g. 64-layer 32B at TP=4
    where the all_gather was dominating wall time).

    Demuxes per rid using the forward_batch stashed in thread-local.
    """
    if layer_id is None:
        return
    if _HARVEST_LAYERS is not None and layer_id not in _HARVEST_LAYERS:
        return

    fb = getattr(_LOCAL, "forward_batch", None)
    if fb is None:
        return
    extend_seq_lens = getattr(fb, "extend_seq_lens_cpu", None)
    if extend_seq_lens is None:
        return  # decode batch — capture only on prefill
    rids = getattr(fb, "_harvest_rids", None)
    extend_prefix_lens = getattr(fb, "extend_prefix_lens_cpu", None)
    if rids is None or extend_prefix_lens is None or len(rids) != len(extend_seq_lens):
        return

    device = intermediate.device
    n_total, d_ffn = intermediate.shape
    # Heap size: prefer HARVEST_N_LAYERS env (set by driver from model config — robust);
    # fall back to the value picked up in patched_layer_init; final fallback 32.
    n_layers = _HARVEST_N_LAYERS or _STATE_N_LAYERS_CONFIGURED or 32
    _ensure_state(device, n_layers, d_ffn)

    # Build (N_total,) doc_id and position vectors.
    doc_ids_flat = torch.empty(n_total, dtype=torch.int64, device=device)
    positions_flat = torch.empty(n_total, dtype=torch.int64, device=device)
    cursor = 0
    for rid, seq_len, prefix_len in zip(rids, extend_seq_lens, extend_prefix_lens, strict=True):
        seq_len = int(seq_len)
        prefix_len = int(prefix_len)
        if seq_len == 0:
            continue
        try:
            pid_str, _ = rid.split("|", 1)
            pid = int(pid_str)
        except (ValueError, AttributeError):
            pid = -1
        doc_ids_flat[cursor : cursor + seq_len] = pid
        positions_flat[cursor : cursor + seq_len] = torch.arange(
            prefix_len, prefix_len + seq_len, dtype=torch.int64, device=device,
        )
        cursor += seq_len

    if cursor < n_total:
        # Mask the leftover (decode rows or padding) by setting their values to -inf,
        # so they never displace a real prefill token in the heap.
        h = intermediate.clone()
        h[cursor:] = float("-inf")
    else:
        h = intermediate

    _update_topk(layer_id, h, doc_ids_flat, positions_flat)
    _maybe_flush()


def _maybe_flush() -> None:
    """Periodic incremental flush so we don't depend on atexit (SGLang subprocess
    is SIGTERM'd at shutdown). Called once per capture; fast-path skips most calls."""
    global _n_capture_calls, _last_flush_t
    _n_capture_calls += 1
    if _n_capture_calls % _FLUSH_EVERY_N_BATCHES != 0:
        return
    import time as _time
    now = _time.monotonic()
    if now - _last_flush_t < _FLUSH_MIN_INTERVAL_S:
        return
    _last_flush_t = now
    _do_flush(final=False)


# ---------------------------------------------------------------- patches

_STATE_N_LAYERS_CONFIGURED: int | None = None


def install_universal_patches() -> None:
    """Patch ForwardBatch.init_new so per-batch rids are reachable from any
    layer's forward. Idempotent."""
    global _universal_patched
    if _universal_patched:
        return
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    _orig_init_new = ForwardBatch.init_new.__func__

    def patched_init_new(cls, batch, model_runner):
        ret = _orig_init_new(cls, batch, model_runner)
        reqs = getattr(batch, "reqs", None)
        ret._harvest_rids = [r.rid for r in reqs] if reqs else None
        return ret

    ForwardBatch.init_new = classmethod(patched_init_new)
    _universal_patched = True


def patch_olmo3_capture() -> None:
    """Install per-instance patches on Olmo2DecoderLayer + Olmo2MLP. Olmo2 is
    SGLang's stand-in for Olmo3 (config rewriter routes Olmo3 → Olmo2). The
    patched MLP forward uncomposes SiluAndMul into named submodules so we can
    pick off the d_ffn intermediate that feeds down_proj."""
    global _module_patched, _STATE_N_LAYERS_CONFIGURED
    if _module_patched:
        return

    install_universal_patches()

    from sglang.srt.models.olmo2 import Olmo2DecoderLayer, Olmo2MLP

    _orig_layer_init = Olmo2DecoderLayer.__init__
    _orig_layer_forward = Olmo2DecoderLayer.forward
    _orig_mlp_forward = Olmo2MLP.forward

    def patched_layer_init(self, *args, **kwargs):
        _orig_layer_init(self, *args, **kwargs)
        global _STATE_N_LAYERS_CONFIGURED
        cfg = getattr(self, "config", None) or (args[0] if args else None)
        if cfg is not None:
            n_layers = getattr(cfg, "num_hidden_layers", None)
            if n_layers and _STATE_N_LAYERS_CONFIGURED is None:
                _STATE_N_LAYERS_CONFIGURED = int(n_layers)
        # Recover layer_id (matches harvester's recovery pattern).
        lid = kwargs.get("layer_id")
        if lid is None:
            attn = getattr(self, "self_attn", None)
            inner = getattr(attn, "attn", None) if attn is not None else None
            lid = getattr(inner, "layer_id", None)
        # Stash on the MLP child so its patched forward can read it.
        if hasattr(self, "mlp"):
            self.mlp._harvest_layer_id = lid

    def patched_layer_forward(self, *args, **kwargs):
        forward_batch = args[2] if len(args) > 2 else kwargs.get("forward_batch")
        prev = getattr(_LOCAL, "forward_batch", None)
        _LOCAL.forward_batch = forward_batch
        try:
            return _orig_layer_forward(self, *args, **kwargs)
        finally:
            _LOCAL.forward_batch = prev

    def patched_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original is `down_proj(act_fn(gate_up_proj(x)))`. We re-thread it so
        # the post-act-fn tensor (the d_ffn intermediate that is the input to
        # down_proj) is exposed for capture.
        gate_up, _ = self.gate_up_proj(x)
        intermediate = self.act_fn(gate_up)             # SiluAndMul: silu(gate)·up
        layer_id = getattr(self, "_harvest_layer_id", None)
        _capture_intermediate(layer_id, intermediate)
        out, _ = self.down_proj(intermediate)
        return out

    Olmo2DecoderLayer.__init__ = patched_layer_init
    Olmo2DecoderLayer.forward = patched_layer_forward
    Olmo2MLP.forward = patched_mlp_forward
    _module_patched = True


# ---------------------------------------------------------------- shutdown / flush

def _do_flush(final: bool) -> None:
    """Atomic write of this rank's heap shard to shard_rank{R}/. Per-rank means
    each TP worker writes its own column-shard; the driver stitches after
    engine shutdown. SIGTERM-safe via temp-file + rename."""
    global _flushed
    if _state is None or _HARVEST_OUT_DIR is None:
        return
    rank = _tp_rank()
    out = Path(_HARVEST_OUT_DIR) / f"shard_rank{rank:02d}"
    out.mkdir(parents=True, exist_ok=True)

    def _atomic_save(arr: np.ndarray, dst: Path) -> None:
        tmp = dst.parent / (dst.stem + ".tmp" + dst.suffix)
        np.save(tmp, arr)
        tmp.replace(dst)

    _atomic_save(_state["vals"].cpu().numpy(), out / "topk_vals.npy")
    _atomic_save(_state["doc_ids"].cpu().numpy(), out / "topk_doc_ids.npy")
    _atomic_save(_state["positions"].cpu().numpy(), out / "topk_positions.npy")
    (out / "harvest_meta.json").write_text(json.dumps({
        "rank": rank,
        "n_layers": _state_n_layers,
        "d_ffn_shard": _state_d_ffn,
        "K": _HARVEST_K,
        "tp_size": _HARVEST_TP_SIZE,
        "layers_captured": sorted(_HARVEST_LAYERS) if _HARVEST_LAYERS else None,
        "n_capture_calls": _n_capture_calls,
        "final": final,
    }, indent=2))
    if final:
        _flushed = True


def flush_state_to_disk() -> None:
    """atexit / signal hook. Final-line-of-defence flush; we already write
    incrementally via _maybe_flush, so by the time we get here the on-disk state
    is at most 10s + N-batches stale. Idempotent."""
    if _flushed:
        return
    _do_flush(final=True)


def _install_signal_handlers() -> None:
    """SGLang sends SIGTERM to the scheduler subprocess at shutdown. We catch it,
    flush, and re-raise the default handler so SGLang's normal cleanup proceeds."""
    global _signal_installed
    if _signal_installed:
        return
    import signal

    def _handler(signum, frame):
        try:
            flush_state_to_disk()
        finally:
            # restore default and re-raise so SGLang's shutdown path runs.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not a main thread / can't install
    _signal_installed = True
