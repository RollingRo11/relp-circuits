"""Generic paired-prompt task type used by all attribution methods.

A PairedPrompt is two tokenizations (clean, counterfactual) and the two single-token
answer ids being scored against each other at a target position. The metric is always
`logits[target_pos, correct_id] - logits[target_pos, incorrect_id]` evaluated on the
clean prompt; cf is used to define the contrast (Δh used by AtP/IG).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PairedPrompt:
    clean_text: str
    cf_text: str
    clean_ids: torch.Tensor          # (T,)
    cf_ids: torch.Tensor             # (T,) — must match clean_ids in length
    target_pos: int                  # position whose next-token logits we score
    clean_correct_id: int            # the correct token id under the clean prompt
    cf_correct_id: int               # the correct token id under the cf prompt (= incorrect under clean)
    meta: dict | None = None         # task-specific bookkeeping (e.g. subject number)

    def metric_correct_id(self) -> int:
        return self.clean_correct_id

    def metric_incorrect_id(self) -> int:
        return self.cf_correct_id


def first_token_id(tokenizer, text: str) -> int | None:
    """Return the first token id of `text` (a leading-space-prefixed answer string).

    None if it can't tokenize as expected (e.g. empty)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return None
    return ids[0]
