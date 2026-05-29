"""Eval-awareness as a *flat* dataset of 100 unpaired prompts (the ADAG-style setup).

We take the 50 benign task bodies from `eval_awareness_paired._TASKS` and emit
each of them in *both* framings, so the resulting list is::

    [prompt 0:  eval-framed task 0,
     prompt 1:  deploy-framed task 0,
     prompt 2:  eval-framed task 1,
     prompt 3:  deploy-framed task 1,
     ...
     prompt 99: deploy-framed task 49]

Each entry is a single PairedPrompt with the cf side stubbed out (unpaired
RelP ignores it). The `meta` dict carries an `is_eval` flag so the post-hoc
differential analysis can compute `mean(|attr|_eval) − mean(|attr|_deploy)`
per cluster — the eval-aware clusters fall out as the ones with high Δ
(rather than being baked in via a contrast token, which is what the paired
approach was effectively doing).

Target metric for unpaired RelP: absolute logit of " testing" at the
assistant lead-in position. We use the same target for both framings, and
let the differential post-hoc analysis surface what changes between them.
"""

from __future__ import annotations

from relp_circuits.tasks.base import PairedPrompt
from relp_circuits.tasks.eval_awareness_paired import (
    _DEPLOY_FRAMES,
    _EVAL_FRAMES,
    _LEAD_INS,
    _TASKS,
    _resolve_single_token,
)


def build_eval_awareness_flat(
    tokenizer,
    num_pairs: int | None = None,
) -> list[PairedPrompt]:
    """Return up to `num_pairs` unpaired prompts (default: all 100). Order
    interleaves eval/deploy framings so a smaller `num_pairs` still gives a
    balanced split."""
    eval_word = _resolve_single_token(tokenizer, [" testing"])
    if eval_word is None:
        raise RuntimeError("tokenizer doesn't have ' testing' as a single token")
    target_id = eval_word[1]

    out: list[PairedPrompt] = []
    for i, task in enumerate(_TASKS):
        ef = _EVAL_FRAMES[i % len(_EVAL_FRAMES)]
        df = _DEPLOY_FRAMES[i % len(_DEPLOY_FRAMES)]
        lead = _LEAD_INS[i % len(_LEAD_INS)]
        for is_eval, frame in [(True, ef), (False, df)]:
            text = (
                f"System: {frame}\n\nUser: {task}\n\n"
                f"Assistant (private reasoning): {lead}"
            )
            ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")[0]
            out.append(
                PairedPrompt(
                    clean_text=text,
                    cf_text="",
                    clean_ids=ids,
                    cf_ids=ids,                  # unused under relp_unpaired
                    target_pos=int(ids.shape[0] - 1),
                    clean_correct_id=target_id,
                    cf_correct_id=target_id,     # unused
                    meta={
                        "task": "eval_awareness_flat",
                        "task_body": task,
                        "frame": frame,
                        "is_eval": is_eval,
                        "lead_in": lead,
                        "task_idx": i,
                    },
                )
            )
            if num_pairs is not None and len(out) >= num_pairs:
                return out
    return out
