"""ADAG explainer + simulator pipeline (arXiv 2604.07615, Sec. 4).

For each cluster of features we follow the paper's two-explainer setup:

  * `describe_input_attribution(cluster)` — produces a 1-2 sentence description
    of *which input tokens / concept* fires the cluster, given top-mass input
    tokens (across the prompt set). Paper uses `Transluce/llama_8b_explainer`
    (a fine-tuned Llama 3.1 8B). We default to that model when the HF or vLLM
    backend is selected.

  * `describe_output_contribution(cluster)` — produces a 1-2 sentence
    description of *which output tokens* the cluster pushes the model towards,
    given top contribution-weighted top-K logits. Paper uses Anthropic's
    Claude Haiku via API.

Then the **simulator** asks an LLM to predict per-example scores from the
description alone, on held-out examples, and we compute the Pearson correlation
between predicted and true scores. This measures the description's faithfulness
(higher r ⇒ description explains more of the cluster's variance).

Outputs per cluster:

  {
    "cluster_id": int,
    "input_description": str,
    "output_description": str,
    "input_simulator_pearson_r": float,
    "output_simulator_pearson_r": float,
  }

Backends:
  hf    - HuggingFace transformers (default: Transluce/llama_8b_explainer for
          input; falls back to same model for output if no API key set).
  vllm  - serve a local HF model via vLLM AsyncLLM for higher throughput.
  api   - Anthropic API (paper's choice for output-contribution descriptions).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ───────────────────────────── prompt templates ─────────────────────────────

_INPUT_ATTR_PROMPT = """\
You are interpreting a *cluster* of MLP neurons inside a large language model.
The cluster co-activates across many examples in a coherent way. Below are the
input tokens that contribute most to firing this cluster, aggregated across the
prompt set:

INPUT TOKENS (sorted by total input-attribution mass; higher = more important):
{input_token_block}

NEURONS in this cluster (top {n_neurons} by attribution magnitude, with
max-activating Dolma example spans where available):
{neuron_block}

Describe in 1-2 plain-English sentences what *concept in the input* fires this
cluster. Start with the concept name (e.g. "Mentions of US state capitals" or
"Negation polarity in factual statements"). Avoid hedging ("appears to",
"possibly"). If the input tokens are heterogeneous and you cannot identify a
single concept, say "Mixed; no single input concept" and stop.
"""


_OUTPUT_CONTRIB_PROMPT = """\
You are interpreting a *cluster* of MLP neurons inside a large language model.
The cluster pushes the model toward specific output tokens at the answer
position. Below are the output tokens that the cluster contributes to most,
aggregated across the prompt set:

OUTPUT TOKENS (sorted by total output-contribution magnitude):
{output_token_block}

NEURONS in this cluster (top {n_neurons} by attribution magnitude, with
max-activating Dolma example spans where available):
{neuron_block}

Describe in 1-2 plain-English sentences what *output behaviour* this cluster
promotes. Start with the action (e.g. "Pushes the model to refuse harmful
requests" or "Promotes capital-city completions"). Avoid hedging. If the
output tokens are heterogeneous and you cannot identify a single behaviour,
say "Mixed; no single output behaviour" and stop.
"""


_SIM_INPUT_PROMPT = """\
You are simulating an MLP neuron cluster's input attribution from a natural
language description. Given the description and a list of input tokens, output
a score from 0 to 10 for each token indicating how strongly the cluster would
respond to that token (10 = strongly fires the cluster, 0 = no response).

DESCRIPTION: {description}

TOKENS (one per line, numbered):
{numbered_tokens}

Return exactly {n_tokens} integer scores in [0, 10], one per line, in the same
order as the tokens above. Output only the integers, nothing else.
"""


_SIM_OUTPUT_PROMPT = """\
You are simulating an MLP neuron cluster's output contribution from a natural
language description. Given the description and a list of candidate output
tokens, output a score from 0 to 10 for each token indicating how strongly the
cluster would push the model to produce that token (10 = strongly pushed, 0 =
no contribution).

DESCRIPTION: {description}

CANDIDATE OUTPUT TOKENS (one per line, numbered):
{numbered_tokens}

Return exactly {n_tokens} integer scores in [0, 10], one per line, in the same
order as the tokens above. Output only the integers, nothing else.
"""


# ───────────────────────────── data structures ──────────────────────────────

@dataclass
class ClusterContext:
    """Inputs to the describer/simulator pipeline for a single cluster."""

    cluster_id: int
    feature_idxs: list[int]                       # indices into the τ-circuit
    feature_keys: list[tuple[int, int]]           # [(layer, neuron), ...]
    aggregate_score: float                        # sum of |agg_score| for ranking clusters

    # Per-cluster, per-(prompt, token) input attribution.
    # Shape: (n_prompts, T_max). Padded with NaN past per_pair_lengths[p].
    input_attr_per_token: list[list[float]] = field(default_factory=list)
    # The decoded token strings per prompt (same shape).
    pair_token_strs: list[list[str]] = field(default_factory=list)

    # Per-cluster output contribution to top-K logits (n_prompts, K).
    output_contrib_top_k: list[list[float]] = field(default_factory=list)
    top_k_logit_ids: list[list[int]] = field(default_factory=list)

    # Pre-rendered summaries, used by the explainer prompts.
    input_token_block: str = ""
    output_token_block: str = ""
    neuron_block: str = ""
    differential_summary: str = ""


@dataclass
class ClusterExplanation:
    """Output of the full ADAG explainer + simulator pipeline for one cluster."""

    cluster_id: int
    n_features: int
    aggregate_score: float
    input_description: str
    output_description: str
    input_simulator_pearson_r: float
    output_simulator_pearson_r: float


# ─────────────────────── building cluster contexts ──────────────────────────

def build_cluster_contexts(
    circuit: list[dict],
    cluster_ids,                                  # (F,) int array
    input_attr,                                   # (P, F, T_max) np
    output_contrib,                               # (P, F, K) np
    top_k_logit_ids,                              # (P, K) np
    per_pair_lengths,                             # (P,) np
    pair_token_strs: list[list[str]],             # P lists of T-token strings
    tokenizer,                                    # rust tokenizer for vocab decode
    neuron_examples: dict[str, list[dict]],       # {"L:N": [{act, html, ...}], ...}
    top_neurons_per_cluster: int = 8,
    top_input_tokens: int = 12,
    top_output_tokens: int = 8,
) -> list[ClusterContext]:
    """Build per-cluster contexts containing both the aggregated text blocks
    (for the explainer prompts) and the raw per-prompt input/output profiles
    (for the simulator)."""
    import numpy as np
    F = len(circuit)
    feature_keys = [(int(e["layer"]), int(e["neuron"])) for e in circuit]
    agg_scores = np.array([float(e.get("max_norm_score", e.get("agg_score", 0))) for e in circuit])

    contexts: list[ClusterContext] = []
    for c in sorted(set(int(x) for x in cluster_ids)):
        members = [i for i in range(F) if int(cluster_ids[i]) == c]
        if not members:
            continue
        members.sort(key=lambda i: -abs(agg_scores[i]))

        # Per-prompt cluster-level input attribution (sum |members| over feature axis).
        # Shape: (P, T_max). Mask past the per-prompt length with NaN.
        per_token_mag = np.abs(input_attr[:, members, :]).sum(axis=1)   # (P, T_max)
        masked_per_token = per_token_mag.copy()
        for p in range(per_token_mag.shape[0]):
            T = int(per_pair_lengths[p])
            if T < per_token_mag.shape[1]:
                masked_per_token[p, T:] = np.nan

        # Top input tokens summary (paper-style: rank by total mass across prompts).
        input_token_freq: dict[str, float] = {}
        for p in range(per_token_mag.shape[0]):
            T = int(per_pair_lengths[p])
            for t in range(T):
                tok = pair_token_strs[p][t] if t < len(pair_token_strs[p]) else ""
                input_token_freq[tok] = input_token_freq.get(tok, 0.0) + float(per_token_mag[p, t])
        ranked = sorted(input_token_freq.items(), key=lambda x: -x[1])[:top_input_tokens]
        input_block = "\n".join(
            f"  {i + 1}. {repr(tok)}  (mass {mass:.2f})"
            for i, (tok, mass) in enumerate(ranked)
            if tok.strip()
        ) or "  (no clear input tokens)"

        # Per-prompt cluster-level output contribution (sum members, keep K axis).
        per_pair_output = np.abs(output_contrib[:, members, :]).sum(axis=1)   # (P, K)

        # Top output tokens summary (rank by total |contribution| across prompts).
        output_token_score: dict[int, float] = {}
        for p in range(top_k_logit_ids.shape[0]):
            for k in range(top_k_logit_ids.shape[1]):
                tid = int(top_k_logit_ids[p, k])
                output_token_score[tid] = output_token_score.get(tid, 0.0) + float(
                    per_pair_output[p, k]
                )
        ranked_out = sorted(output_token_score.items(), key=lambda x: -x[1])[:top_output_tokens]
        output_block = "\n".join(
            f"  {i + 1}. {repr(tokenizer.decode([tid]))}  (contrib {sc:.2f})"
            for i, (tid, sc) in enumerate(ranked_out)
        ) or "  (no clear output tokens)"

        # Per-neuron block (max-activating Dolma examples).
        lines: list[str] = []
        for i in members[:top_neurons_per_cluster]:
            li, ni = feature_keys[i]
            key = f"{li}:{ni}"
            lines.append(f"\n  L{li}/N{ni}  agg|score|={abs(float(agg_scores[i])):.3f}")
            ex = (neuron_examples.get(key, []) or [])[:3]
            if not ex:
                lines.append("    (no Dolma examples)")
            else:
                for r in ex:
                    snippet = r["html"]
                    snippet = snippet.replace("<mark>", "<<").replace("</mark>", ">>")
                    snippet = snippet.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                    snippet = snippet.replace("\n", "↵")
                    if len(snippet) > 220:
                        snippet = snippet[:217] + "…"
                    lines.append(f"    act={r['act']:.0f}  {snippet}")
        neuron_block = "".join(lines).lstrip("\n")

        contexts.append(ClusterContext(
            cluster_id=c,
            feature_idxs=members,
            feature_keys=[feature_keys[i] for i in members],
            aggregate_score=float(sum(abs(agg_scores[i]) for i in members)),
            input_attr_per_token=masked_per_token.tolist(),
            pair_token_strs=pair_token_strs,
            output_contrib_top_k=per_pair_output.tolist(),
            top_k_logit_ids=top_k_logit_ids.tolist(),
            input_token_block=input_block,
            output_token_block=output_block,
            neuron_block=neuron_block,
        ))
    return contexts


# ───────────────────── explainer prompt rendering ───────────────────────────

def render_input_explainer_prompt(ctx: ClusterContext, *, n_neurons: int = 8) -> str:
    return _INPUT_ATTR_PROMPT.format(
        input_token_block=ctx.input_token_block,
        neuron_block=ctx.neuron_block,
        n_neurons=n_neurons,
    )


def render_output_explainer_prompt(ctx: ClusterContext, *, n_neurons: int = 8) -> str:
    return _OUTPUT_CONTRIB_PROMPT.format(
        output_token_block=ctx.output_token_block,
        neuron_block=ctx.neuron_block,
        n_neurons=n_neurons,
    )


# ─────────────────────────── simulator helpers ──────────────────────────────

# Some reasoning models (e.g. OLMo-3-Think, Qwen3-Think) emit a chain-of-thought
# delimited by `<think>...</think>` before the final answer. The describer just
# wants the final answer, so we strip these blocks post-decode.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove `<think>...</think>` blocks. If the model ran out of tokens before
    closing the think block, drop everything from `<think>` onwards."""
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()


_INT_LINE = re.compile(r"^\s*(-?\d+)")


def _parse_scores(text: str, n_expected: int) -> list[float]:
    """Parse `n_expected` integers from the simulator's output. Tolerates
    blank lines and trailing prose. Pads with 0 if the LLM under-produces."""
    scores: list[float] = []
    for line in text.splitlines():
        m = _INT_LINE.match(line)
        if m:
            try:
                scores.append(float(int(m.group(1))))
            except ValueError:
                continue
        if len(scores) >= n_expected:
            break
    while len(scores) < n_expected:
        scores.append(0.0)
    return scores[:n_expected]


def render_input_simulator_prompts(
    description: str,
    pair_token_strs: list[list[str]],
    per_pair_lengths: list[int],
    prompt_indices: list[int],
    max_tokens_per_prompt: int = 64,
) -> tuple[list[str], list[tuple[int, list[int]]]]:
    """Build simulator prompts for the held-out prompts at `prompt_indices`.

    Returns (prompts, sites). Each `sites[i]` is `(prompt_idx, token_indices)`
    listing which prompt and which token positions were included (capped at
    `max_tokens_per_prompt` to fit context).
    """
    prompts: list[str] = []
    sites: list[tuple[int, list[int]]] = []
    for p in prompt_indices:
        T = min(int(per_pair_lengths[p]), max_tokens_per_prompt)
        token_idxs = list(range(T))
        tokens = [pair_token_strs[p][t] for t in token_idxs]
        numbered = "\n".join(f"{i + 1}. {repr(tok)}" for i, tok in enumerate(tokens))
        prompts.append(_SIM_INPUT_PROMPT.format(
            description=description,
            numbered_tokens=numbered,
            n_tokens=len(tokens),
        ))
        sites.append((p, token_idxs))
    return prompts, sites


def render_output_simulator_prompts(
    description: str,
    top_k_logit_ids: list[list[int]],
    prompt_indices: list[int],
    tokenizer,
) -> tuple[list[str], list[tuple[int, list[int]]]]:
    """Build simulator prompts for output contribution on held-out prompts."""
    prompts: list[str] = []
    sites: list[tuple[int, list[int]]] = []
    for p in prompt_indices:
        ids = top_k_logit_ids[p]
        tokens = [tokenizer.decode([tid]) for tid in ids]
        numbered = "\n".join(f"{i + 1}. {repr(tok)}" for i, tok in enumerate(tokens))
        prompts.append(_SIM_OUTPUT_PROMPT.format(
            description=description,
            numbered_tokens=numbered,
            n_tokens=len(tokens),
        ))
        sites.append((p, list(range(len(ids)))))
    return prompts, sites


def pearson_score(predicted: list[float], actual: list[float]) -> float:
    """Pearson r between predicted and actual scores. Returns 0.0 when one side
    has zero variance (constant prediction or constant ground truth)."""
    import numpy as np
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() < 2:
        return 0.0
    p = p[mask]
    a = a[mask]
    if p.std() == 0 or a.std() == 0:
        return 0.0
    r = float(np.corrcoef(p, a)[0, 1])
    return r if np.isfinite(r) else 0.0


# ──────────────────────────── LLM backends ──────────────────────────────────

def describe_via_hf(
    prompts: list[str],
    *,
    model_id: str = "Transluce/llama_8b_explainer",
    max_tokens: int = 220,
    temperature: float = 0.0,
    device: str = "auto",
    _model_cache: dict = {},
) -> list[str]:
    """HuggingFace backend. Default model is the paper's input-attribution
    explainer (`Transluce/llama_8b_explainer`, a fine-tuned Llama 3.1 8B).

    Caches the (model_id, device) pair across calls so repeated cluster batches
    don't reload weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache_key = (model_id, device)
    if cache_key in _model_cache:
        tok, model = _model_cache[cache_key]
    else:
        print(f"[describe-hf] loading {model_id}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
        )
        model.eval()
        _model_cache[cache_key] = (tok, model)

    out: list[str] = []
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-6),
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = gen[0, inputs.input_ids.shape[1] :]
        text_out = tok.decode(new_tokens, skip_special_tokens=True).strip()
        text_out = _strip_thinking(text_out)
        out.append(text_out)
        print(f"[describe-hf] {i+1}/{len(prompts)} → {text_out[:80]!r}", flush=True)
    return out


def describe_via_vllm(
    prompts: list[str],
    *,
    model_id: str = "Transluce/llama_8b_explainer",
    max_tokens: int = 220,
    temperature: float = 0.0,
) -> list[str]:
    """vLLM-served version. Faster but pulls vllm into the venv."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, dtype="bfloat16", trust_remote_code=True)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outs]


def describe_via_api(
    prompts: list[str],
    *,
    model_id: str = "claude-3-5-haiku-latest",
    max_tokens: int = 220,
) -> list[str]:
    """Anthropic API. Used by the paper for the output-contribution explainer
    (Claude Haiku) and as the default backend when ANTHROPIC_API_KEY is set."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("ANTHROPIC_API_KEY not set; pick --backend hf/vllm or set the key")
    import anthropic
    client = anthropic.Anthropic()
    out: list[str] = []
    for prompt in prompts:
        msg = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        out.append("".join(b.text for b in msg.content if b.type == "text").strip())
    return out


BackendCallable = Callable[[list[str]], list[str]]


def make_backend(name: str, model_id: str) -> BackendCallable:
    """Resolve a backend name + model id into a callable that takes a list of
    prompts and returns a list of completions."""
    if name == "hf":
        return lambda prompts: describe_via_hf(prompts, model_id=model_id)
    if name == "vllm":
        return lambda prompts: describe_via_vllm(prompts, model_id=model_id)
    if name == "api":
        return lambda prompts: describe_via_api(prompts, model_id=model_id)
    raise ValueError(f"unknown backend: {name}")


# ────────────────── end-to-end pipeline per cluster ─────────────────────────

def explain_and_score_clusters(
    contexts: list[ClusterContext],
    *,
    input_backend: BackendCallable,
    output_backend: BackendCallable,
    simulator_backend: BackendCallable,
    tokenizer,
    per_pair_lengths: list[int],
    holdout_frac: float = 0.5,
    max_input_tokens_per_prompt: int = 64,
    seed: int = 0,
) -> list[ClusterExplanation]:
    """Full ADAG describer pipeline:

    1. For each cluster, call the input-attribution explainer.
    2. For each cluster, call the output-contribution explainer.
    3. Split the prompt set into train/held-out (uses the train half implicitly
       in step 1-2 via the aggregated context; step 3 scores on the held-out).
    4. Run the simulator on held-out prompts for both views; compute Pearson r.
    """
    import numpy as np
    if not contexts:
        return []

    # Step 1: input-attribution descriptions, batched across clusters.
    input_prompts = [render_input_explainer_prompt(ctx) for ctx in contexts]
    input_descs = input_backend(input_prompts)

    # Step 2: output-contribution descriptions.
    output_prompts = [render_output_explainer_prompt(ctx) for ctx in contexts]
    output_descs = output_backend(output_prompts)

    # Step 3: train/held-out split. Use a deterministic shuffle so the same
    # split is used across clusters in a single run.
    n_prompts = len(per_pair_lengths)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_prompts)
    n_holdout = max(1, int(round(holdout_frac * n_prompts)))
    holdout = perm[:n_holdout].tolist()

    # Step 4: simulate + Pearson r for each cluster.
    out: list[ClusterExplanation] = []
    for ctx, in_desc, out_desc in zip(contexts, input_descs, output_descs, strict=True):
        # --- input simulator ---
        in_prompts, in_sites = render_input_simulator_prompts(
            in_desc, ctx.pair_token_strs, per_pair_lengths, holdout,
            max_tokens_per_prompt=max_input_tokens_per_prompt,
        )
        in_completions = simulator_backend(in_prompts)
        pred_in: list[float] = []
        true_in: list[float] = []
        for (p, token_idxs), completion in zip(in_sites, in_completions, strict=True):
            scores = _parse_scores(completion, n_expected=len(token_idxs))
            actual = [ctx.input_attr_per_token[p][t] for t in token_idxs]
            pred_in.extend(scores)
            true_in.extend(actual)
        r_in = pearson_score(pred_in, true_in)

        # --- output simulator ---
        out_prompts_sim, out_sites = render_output_simulator_prompts(
            out_desc, ctx.top_k_logit_ids, holdout, tokenizer,
        )
        out_completions = simulator_backend(out_prompts_sim)
        pred_out: list[float] = []
        true_out: list[float] = []
        for (p, k_idxs), completion in zip(out_sites, out_completions, strict=True):
            scores = _parse_scores(completion, n_expected=len(k_idxs))
            actual = [ctx.output_contrib_top_k[p][k] for k in k_idxs]
            pred_out.extend(scores)
            true_out.extend(actual)
        r_out = pearson_score(pred_out, true_out)

        out.append(ClusterExplanation(
            cluster_id=ctx.cluster_id,
            n_features=len(ctx.feature_idxs),
            aggregate_score=ctx.aggregate_score,
            input_description=in_desc,
            output_description=out_desc,
            input_simulator_pearson_r=r_in,
            output_simulator_pearson_r=r_out,
        ))
        print(
            f"[describe] cluster {ctx.cluster_id}: "
            f"r_in={r_in:+.3f} r_out={r_out:+.3f}  "
            f"in='{in_desc[:60]}…' out='{out_desc[:60]}…'",
            flush=True,
        )
    return out


def dump_explanations(explanations: list[ClusterExplanation], path: Path) -> None:
    payload = [
        {
            "cluster_id": e.cluster_id,
            "n_features": e.n_features,
            "aggregate_score": e.aggregate_score,
            "input_description": e.input_description,
            "output_description": e.output_description,
            "input_simulator_pearson_r": e.input_simulator_pearson_r,
            "output_simulator_pearson_r": e.output_simulator_pearson_r,
        }
        for e in explanations
    ]
    Path(path).write_text(json.dumps(payload, indent=2))
