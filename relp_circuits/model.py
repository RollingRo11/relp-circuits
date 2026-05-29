"""Model wrapper that exposes per-layer MLP post-activation neurons for attribution.

The "neuron" basis used by Arora & Wu et al. 2026 (ADAG, arXiv 2604.07615) is the
d_ffn-vector that feeds each MLP's down-projection — i.e. `silu(gate_proj(x)) * up_proj(x)`
for both LlamaMLP (the paper's primary model) and Olmo3MLP. We monkey-patch each MLP's
forward to (a) stash this tensor for later read, and (b) optionally substitute an
externally-supplied tensor in its place — needed by IG and patching workflows.

This module also installs the **full RelP linearization** (Jafari et al. 2025) when
`relp_active()` is the active context:

  * SiLU is treated as a constant multiplier. Since SiLU(g) = g · sigmoid(g), the
    multiplier is sigmoid(g) and the bilinear approximation of the SwiGLU
    is `[sigmoid(gate_pre) · gate_pre] · up`. The half-rule then splits relevance
    equally between the gate branch (carrying sigmoid(gate_pre) · 0.5 · up) and
    the up branch (carrying silu(gate_pre) · 0.5). This matches Eq. 8's
    conservation property.
  * Softmax behaves as a constant — zero gradient flows back through it.
    Combined with the half-rule on Q·K^T, the gradient flowing through the
    query and key paths is effectively killed (paper's "stop-grad through Q/K").
  * RMSNorm freezes (rms_scale, weight) at clean values: ∂RMSNorm/∂x = weight·rsqrt.
  * Bilinear matmuls in attention (Q·K^T and softmax·V) use the half-rule.

Outside the `relp_active()` context the model is bit-identical to the upstream
transformers' implementation, so it's safe to install these patches at load time and
leave them in place for non-RelP forwards.

Supported architectures: `llama` (paper's primary) and `olmo3` (project-specific
extension for OLMo-3-7B-Think and OLMo-3.1-32B-Think).
"""

from __future__ import annotations

import contextlib
import importlib
import types
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# When this is True, all patched forwards switch to the linearized RelP variants.
_RELP_ACTIVE: bool = False


@contextlib.contextmanager
def relp_active() -> Iterator[None]:
    """Within this context, the model's forward+backward uses the RelP replacement model."""
    global _RELP_ACTIVE
    prev = _RELP_ACTIVE
    _RELP_ACTIVE = True
    try:
        yield
    finally:
        _RELP_ACTIVE = prev


def pick_freest_gpu() -> int:
    """Return the index of the visible GPU with the most free memory.

    Workaround for shared cluster nodes where another user's non-Slurm process is
    occupying one of the GPUs Slurm gave us. Requires more than one visible GPU; if
    only one is visible, returns 0.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible")
    n = torch.cuda.device_count()
    if n == 1:
        return 0
    best_idx, best_free = 0, -1
    for i in range(n):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_idx, best_free = i, free
    return best_idx


@dataclass
class LayerState:
    captured: torch.Tensor | None = None
    override: torch.Tensor | None = None
    retain_grad: bool = False


# === RelP custom autograd Functions =========================================

class _RelPSwiGLU(torch.autograd.Function):
    """SwiGLU forward = silu(gate_pre) * up. Linearized backward implements RelP:

    SiLU(g) = g · sigmoid(g), so the "constant multiplier" interpretation of the
    nonlinearity replaces silu(g) with sigmoid(g) · g, with sigmoid(g) frozen at
    its clean value. The post-activation then becomes the bilinear

        h ≈ [sigmoid(g) · g] · up = sigmoid(g) · g · up

    with sigmoid(g) treated as a constant. The half-rule then splits relevance
    equally between the two remaining factors:

        ∂h/∂g  = sigmoid(g) · up · 0.5
        ∂h/∂up = sigmoid(g) · g · 0.5 = silu(g) · 0.5
    """

    @staticmethod
    def forward(ctx, gate_pre: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(gate_pre)
        silu_gate = gate_pre * sig
        ctx.save_for_backward(sig, silu_gate, up)
        return silu_gate * up

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor):
        sig, silu_gate, up = ctx.saved_tensors
        grad_gate_pre = grad_h * sig * up * 0.5
        grad_up = grad_h * silu_gate * 0.5
        return grad_gate_pre, grad_up


class _LinearizedRMSNorm(torch.autograd.Function):
    """RMSNorm forward identical to standard. Backward treats `rms(x)` and `weight` as
    constants — the gradient through the normalisation reduces to a per-channel scale.
    Equivalent to replacing RMSNorm with `x · (weight / rms_clean)` in the replacement
    model.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(var + eps)
        normalized = x_f32 * rsqrt
        y = (weight * normalized).to(input_dtype)
        ctx.save_for_backward(rsqrt.to(input_dtype), weight)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        rsqrt, weight = ctx.saved_tensors
        grad_x = grad_y * (weight * rsqrt)
        return grad_x, None, None  # frozen weight & eps


class _FrozenSoftmax(torch.autograd.Function):
    """Forward: standard softmax. Backward: zero gradient — softmax is treated as a
    constant in the replacement model. Combined with the half-rule on Q·K^T this
    realises the paper's stop-gradient through the query and key paths."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.softmax(x, dim=dim)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        return torch.zeros_like(grad_y), None


class _HalfRuleMatmul(torch.autograd.Function):
    """Matmul a @ b. Backward applies the half-rule: gradient w.r.t. each factor is
    halved relative to the standard chain rule. Used for the bilinear ops in attention
    (Q·K^T and softmax·V) — both factors are functions of the same residual stream, so
    splitting the relevance equally is the LRP-style prescription used in the paper."""

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_a = torch.matmul(grad_y, b.transpose(-2, -1)) * 0.5
        grad_b = torch.matmul(a.transpose(-2, -1), grad_y) * 0.5
        return grad_a, grad_b


# === RelP-flavoured replacements for transformers ops ========================

def _linearized_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Per-instance replacement for {Olmo3,Llama}RMSNorm.forward. Defers to the
    standard form when RelP is not active."""
    if not _RELP_ACTIVE:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)
    return _LinearizedRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)


def _make_relp_eager_attention(orig_fn, repeat_kv_fn):
    """Build a drop-in replacement for `eager_attention_forward` against a specific
    transformers modeling module. Falls back to `orig_fn` when RelP is inactive."""

    def _relp_eager(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        if not _RELP_ACTIVE:
            return orig_fn(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs
            )

        key_states = repeat_kv_fn(key, module.num_key_value_groups)
        value_states = repeat_kv_fn(value, module.num_key_value_groups)

        attn_weights = _HalfRuleMatmul.apply(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = _FrozenSoftmax.apply(attn_weights.to(torch.float32), -1).to(query.dtype)
        attn_output = _HalfRuleMatmul.apply(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return _relp_eager


# Architecture registry. Each entry maps a HuggingFace model_type to the transformers
# modeling module path, the RMSNorm class name, and a flag tracking install state.
_ARCH_SPECS: dict[str, dict] = {
    "llama": {
        "module": "transformers.models.llama.modeling_llama",
        "rmsnorm_cls": "LlamaRMSNorm",
    },
    "olmo3": {
        "module": "transformers.models.olmo3.modeling_olmo3",
        "rmsnorm_cls": "Olmo3RMSNorm",
    },
}
_INSTALLED_ARCHS: set[str] = set()


def _install_relp_patches_for(arch: str) -> None:
    """Install the eager-attention monkey patch for `arch`. Idempotent."""
    if arch in _INSTALLED_ARCHS:
        return
    if arch not in _ARCH_SPECS:
        raise RuntimeError(f"unsupported model architecture for RelP: {arch}")
    spec = _ARCH_SPECS[arch]
    mod = importlib.import_module(spec["module"])
    orig = mod.eager_attention_forward
    repeat_kv = mod.repeat_kv
    mod.eager_attention_forward = _make_relp_eager_attention(orig, repeat_kv)
    _INSTALLED_ARCHS.add(arch)


def _rmsnorm_classes() -> tuple[type, ...]:
    """Return the tuple of RMSNorm classes for all installed architectures.

    Used by `isinstance(...)` when sweeping modules during patching.
    """
    classes: list[type] = []
    for arch in _INSTALLED_ARCHS:
        spec = _ARCH_SPECS[arch]
        mod = importlib.import_module(spec["module"])
        cls = getattr(mod, spec["rmsnorm_cls"])
        classes.append(cls)
    return tuple(classes)


@dataclass
class HookedModel:
    model: nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    d_ffn: int
    states: list[LayerState] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
    ) -> "HookedModel":
        if isinstance(device, str) and device == "auto":
            device_map: str | torch.device | dict = "auto"
            primary_device = torch.device("cuda:0")
            print(f"[model.load] device_map=auto across {torch.cuda.device_count()} GPUs",
                  flush=True)
        elif isinstance(device, str) and device == "cuda":
            idx = pick_freest_gpu()
            device = f"cuda:{idx}"
            free, total = torch.cuda.mem_get_info(idx)
            print(f"[model.load] picked {device} (free={free/1e9:.1f}GB / total={total/1e9:.1f}GB)",
                  flush=True)
            device_map = device
            primary_device = torch.device(device)
        else:
            device_map = device if isinstance(device, str) else str(device)
            primary_device = torch.device(device) if isinstance(device, str) else device

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        model.eval()
        cfg = model.config
        arch = cfg.model_type
        _install_relp_patches_for(arch)

        n_layers = cfg.num_hidden_layers
        states = [LayerState() for _ in range(n_layers)]
        wrapped = cls(
            model=model,
            tokenizer=tokenizer,
            device=primary_device,
            dtype=dtype,
            n_layers=n_layers,
            d_ffn=cfg.intermediate_size,
            states=states,
        )
        wrapped._patch_mlps()
        wrapped._patch_rmsnorms()
        return wrapped

    def _patch_rmsnorms(self) -> None:
        """Replace every {Llama,Olmo3}RMSNorm forward with the RelP-aware version.
        Outside the `relp_active()` context this falls back to the standard
        computation, so the patch is safe even when RelP is not in use."""
        norm_classes = _rmsnorm_classes()
        for mod in self.model.modules():
            if isinstance(mod, norm_classes):
                mod.forward = types.MethodType(_linearized_rmsnorm_forward, mod)

    def _patch_mlps(self) -> None:
        layers = self.model.model.layers
        if len(layers) != self.n_layers:
            raise RuntimeError(
                f"layer count mismatch: model has {len(layers)} but config says {self.n_layers}"
            )
        for idx, layer in enumerate(layers):
            self._patch_one(idx, layer.mlp)

    def _patch_one(self, idx: int, mlp: nn.Module) -> None:
        gate = mlp.gate_proj
        up = mlp.up_proj
        down = mlp.down_proj
        act = mlp.act_fn
        states = self.states

        def forward(x: torch.Tensor) -> torch.Tensor:
            st = states[idx]
            if st.override is not None:
                h = st.override
            elif _RELP_ACTIVE:
                # Linearized SwiGLU: SiLU as constant multiplier (sigmoid(g))
                # + half-rule splitting relevance equally across the gate and up branches.
                h = _RelPSwiGLU.apply(gate(x), up(x))
            else:
                h = act(gate(x)) * up(x)
            if st.retain_grad and h.requires_grad and h.grad_fn is not None:
                h.retain_grad()
            st.captured = h
            return down(h)

        mlp.forward = forward  # type: ignore[method-assign]

    def reset_states(self) -> None:
        for st in self.states:
            st.captured = None
            st.override = None
            st.retain_grad = False

    def set_retain_grad(self, value: bool = True) -> None:
        for st in self.states:
            st.retain_grad = value

    def set_overrides(self, overrides: list[torch.Tensor | None]) -> None:
        if len(overrides) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} overrides, got {len(overrides)}")
        for st, ov in zip(self.states, overrides, strict=True):
            st.override = ov

    def captured_acts(self) -> list[torch.Tensor | None]:
        return [st.captured for st in self.states]

    def captured_grads(self) -> list[torch.Tensor | None]:
        out: list[torch.Tensor | None] = []
        for st in self.states:
            out.append(st.captured.grad if st.captured is not None else None)
        return out
