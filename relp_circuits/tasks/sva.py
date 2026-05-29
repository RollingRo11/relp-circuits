"""Subject-verb agreement paired prompts.

Two flavors:
- `build_sva_pairs(tokenizer)`: ~30 hand-written templates with prepositional-phrase distractors
  (subject and verb separated by an intervening NP that flips number to test agreement).
- `build_blimp_sva(tokenizer)`: BLiMP suites — see `tasks.blimp`.

For attribution we want clean (plural subject) → predicts " are" and cf (singular subject) →
predicts " is", so the metric `logit(' are') - logit(' is')` is positive on clean and
negative on cf. Skip pairs whose tokenizations don't share length (rare with these short
templates but cheap to defend against).
"""

from __future__ import annotations

from relp_circuits.tasks.base import PairedPrompt


def _resolve_single_token(tokenizer, candidates: list[str]) -> tuple[str, int]:
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return cand, ids[0]
    raise ValueError(f"no single-token form among {candidates} for this tokenizer")


_TEMPLATES: list[tuple[str, str, str]] = [
    ("The key", "The keys", "to the cabinet"),
    ("The author", "The authors", "near the editors"),
    ("The book", "The books", "on the shelves"),
    ("The pilot", "The pilots", "by the planes"),
    ("The senator", "The senators", "in the chamber"),
    ("The dog", "The dogs", "behind the fences"),
    ("The chef", "The chefs", "in the kitchens"),
    ("The painting", "The paintings", "in the galleries"),
    ("The student", "The students", "with the laptops"),
    ("The nurse", "The nurses", "near the patients"),
    ("The musician", "The musicians", "with the violins"),
    ("The farmer", "The farmers", "by the tractors"),
    ("The captain", "The captains", "of the ships"),
    ("The athlete", "The athletes", "from the schools"),
    ("The translator", "The translators", "with the manuscripts"),
    ("The architect", "The architects", "behind the projects"),
    ("The detective", "The detectives", "near the suspects"),
    ("The carpenter", "The carpenters", "with the hammers"),
    ("The reporter", "The reporters", "around the buildings"),
    ("The cyclist", "The cyclists", "near the cars"),
    ("The waiter", "The waiters", "at the tables"),
    ("The dancer", "The dancers", "behind the curtains"),
    ("The judge", "The judges", "in the courtrooms"),
    ("The driver", "The drivers", "near the trucks"),
    ("The professor", "The professors", "in the offices"),
    ("The engineer", "The engineers", "by the bridges"),
    ("The actress", "The actresses", "near the cameras"),
    ("The technician", "The technicians", "with the cables"),
    ("The poet", "The poets", "behind the desks"),
    ("The cashier", "The cashiers", "at the registers"),
]


def build_sva_pairs(tokenizer, num_pairs: int | None = None) -> list[PairedPrompt]:
    _, plural_id = _resolve_single_token(tokenizer, [" are", " were"])
    _, singular_id = _resolve_single_token(tokenizer, [" is", " was"])

    pairs: list[PairedPrompt] = []
    chosen = _TEMPLATES if num_pairs is None else _TEMPLATES[:num_pairs]
    for sg, pl, distractor in chosen:
        clean_text = f"{pl} {distractor}"
        cf_text = f"{sg} {distractor}"
        clean_ids = tokenizer.encode(clean_text, add_special_tokens=True, return_tensors="pt")[0]
        cf_ids = tokenizer.encode(cf_text, add_special_tokens=True, return_tensors="pt")[0]
        if cf_ids.shape[0] != clean_ids.shape[0]:
            continue
        pairs.append(
            PairedPrompt(
                clean_text=clean_text,
                cf_text=cf_text,
                clean_ids=clean_ids,
                cf_ids=cf_ids,
                target_pos=int(clean_ids.shape[0] - 1),
                clean_correct_id=plural_id,
                cf_correct_id=singular_id,
                meta={"task": "sva", "kind": "templates"},
            )
        )
    return pairs


# Back-compat alias for any old callers.
SVAPair = PairedPrompt
