"""Multi-hop city→state→capital paired prompts (the paper's secondary benchmark).

A pair is two (city, state, capital) tuples sharing the same prompt template. The
metric is the logit difference between the two correct capitals at the answer position.
The model has to chain city→state then state→capital — a two-hop fact lookup.

Tokenization filter: we keep a tuple only if (a) clean and cf tokenize to the same
length, and (b) the leading-space-prefixed first capital token is a single token under
the tokenizer (so logit-diff at one position is well-defined).
"""

from __future__ import annotations

from relp_circuits.tasks.base import PairedPrompt, first_token_id

_TEMPLATE = "{city} is in {state}. The capital of {state} is"

# (city, state, capital). State must be the same the model would output given the city.
_TUPLES: list[tuple[str, str, str]] = [
    ("Dallas", "Texas", " Austin"),
    ("Houston", "Texas", " Austin"),
    ("San Diego", "California", " Sacramento"),
    ("Los Angeles", "California", " Sacramento"),
    ("Miami", "Florida", " Tallahassee"),
    ("Orlando", "Florida", " Tallahassee"),
    ("Chicago", "Illinois", " Springfield"),
    ("Detroit", "Michigan", " Lansing"),
    ("Buffalo", "New York", " Albany"),
    ("Pittsburgh", "Pennsylvania", " Harrisburg"),
    ("Philadelphia", "Pennsylvania", " Harrisburg"),
    ("Cleveland", "Ohio", " Columbus"),
    ("Cincinnati", "Ohio", " Columbus"),
    ("Memphis", "Tennessee", " Nashville"),
    ("Charlotte", "North Carolina", " Raleigh"),
    ("Atlanta", "Georgia", " Atlanta"),
    ("New Orleans", "Louisiana", " Baton"),
    ("Baltimore", "Maryland", " Annapolis"),
    ("Seattle", "Washington", " Olympia"),
    ("Portland", "Oregon", " Salem"),
    ("Las Vegas", "Nevada", " Carson"),
    ("Tucson", "Arizona", " Phoenix"),
    ("Boulder", "Colorado", " Denver"),
    ("Albuquerque", "New Mexico", " Santa"),
    ("Salt Lake City", "Utah", " Salt"),
    ("Spokane", "Washington", " Olympia"),
    ("Tampa", "Florida", " Tallahassee"),
]


def build_multihop_pairs(tokenizer, num_pairs: int | None = None) -> list[PairedPrompt]:
    """Pair each tuple with another tuple from a *different* state, so the answers differ."""
    # Resolve first-token capital ids; drop tuples whose capital first-token isn't single.
    resolved: list[tuple[str, str, str, int]] = []
    for city, state, capital in _TUPLES:
        tid = first_token_id(tokenizer, capital)
        if tid is None:
            continue
        # Verify capital starts with a single token (we score on first token only).
        ids_full = tokenizer.encode(capital, add_special_tokens=False)
        if len(ids_full) < 1 or ids_full[0] != tid:
            continue
        resolved.append((city, state, capital, tid))

    pairs: list[PairedPrompt] = []
    for i, (city_a, state_a, _, tid_a) in enumerate(resolved):
        for city_b, state_b, _, tid_b in resolved[i + 1 :]:
            if state_a == state_b or tid_a == tid_b:
                continue
            clean_text = _TEMPLATE.format(city=city_a, state=state_a)
            cf_text = _TEMPLATE.format(city=city_b, state=state_b)
            clean_ids = tokenizer.encode(clean_text, add_special_tokens=True, return_tensors="pt")[0]
            cf_ids = tokenizer.encode(cf_text, add_special_tokens=True, return_tensors="pt")[0]
            if clean_ids.shape[0] != cf_ids.shape[0]:
                continue
            pairs.append(
                PairedPrompt(
                    clean_text=clean_text,
                    cf_text=cf_text,
                    clean_ids=clean_ids,
                    cf_ids=cf_ids,
                    target_pos=int(clean_ids.shape[0] - 1),
                    clean_correct_id=tid_a,
                    cf_correct_id=tid_b,
                    meta={"task": "multihop", "states": (state_a, state_b)},
                )
            )
            if num_pairs is not None and len(pairs) >= num_pairs:
                return pairs
    return pairs
