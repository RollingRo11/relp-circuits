"""Load BLiMP subject-verb-agreement suites and convert (good, bad) sentence pairs into
the PairedPrompt format used by attribution.

BLiMP gives us minimal pairs that differ in exactly one token (typically the verb form).
We:
  1. Find the diverging position by walking token-by-token.
  2. Use the prefix up to and including the first matching token as the prompt.
  3. Score logit-diff between good[diverge] and bad[diverge] at the position one before.

Note: BLiMP's "good" sentence has the *grammatical* form. For SVA suites, that means the
correct verb. So `clean = good`, `cf = bad`. Attribution then targets the metric
`logit(good_token) - logit(bad_token)` at the prediction position — same setup as our
template SVA, just from a real benchmark.
"""

from __future__ import annotations

from relp_circuits.tasks.base import PairedPrompt

# The SVA-relevant suites in BLiMP. There are more, but these are the cleanest for our use.
SVA_SUITES = [
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
]


def _diverge_position(good_ids: list[int], bad_ids: list[int]) -> int | None:
    n = min(len(good_ids), len(bad_ids))
    for i in range(n):
        if good_ids[i] != bad_ids[i]:
            return i
    return None


def build_blimp_sva(
    tokenizer,
    suites: list[str] | None = None,
    num_pairs: int | None = 200,
    cache_dir: str | None = None,
) -> list[PairedPrompt]:
    from datasets import load_dataset

    suites = suites or SVA_SUITES
    pairs: list[PairedPrompt] = []
    import torch

    for suite in suites:
        try:
            ds = load_dataset("nyu-mll/blimp", suite, split="train", cache_dir=cache_dir)
        except Exception as e:
            print(f"[blimp] skipping suite {suite}: {e}")
            continue

        for ex in ds:
            good = ex["sentence_good"]
            bad = ex["sentence_bad"]
            good_ids = tokenizer.encode(good, add_special_tokens=True)
            bad_ids = tokenizer.encode(bad, add_special_tokens=True)
            if len(good_ids) != len(bad_ids):
                # We require length parity for a single-position logit-diff metric.
                continue
            div = _diverge_position(good_ids, bad_ids)
            if div is None or div < 1:
                continue
            # Prompt is everything up through the token *before* the diverging one. The
            # divergence is the next-token prediction we score.
            prompt_ids_good = good_ids[:div]
            prompt_ids_bad = bad_ids[:div]
            if len(prompt_ids_good) != len(prompt_ids_bad):
                continue
            # Both prompts should be identical in token content up through `div`.
            if prompt_ids_good != prompt_ids_bad:
                continue
            pairs.append(
                PairedPrompt(
                    clean_text=good,
                    cf_text=bad,
                    clean_ids=torch.tensor(prompt_ids_good, dtype=torch.long),
                    cf_ids=torch.tensor(prompt_ids_bad, dtype=torch.long),
                    target_pos=int(div - 1),
                    clean_correct_id=good_ids[div],
                    cf_correct_id=bad_ids[div],
                    meta={"task": "blimp_sva", "suite": suite},
                )
            )
            if num_pairs is not None and len(pairs) >= num_pairs:
                return pairs
    return pairs
