"""The random baseline of the mutation analysis: request sets of the suite's size drawn from the pool of all requests, repeated.

For each repetition, a mutant counts as killed if the sample holds a request the mutant answers differently from the oracle.
"""

from __future__ import annotations

import random


def repetitions(pool: list, size: int, differing: dict[str, set], seed: int, reps: int = 30) -> list[dict[str, bool]]:
    """`differing[mutant]` is the set of pool requests that mutant answers wrongly; returns, per repetition, mutant -> killed by the sample."""
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        sample = set(rng.sample(pool, min(size, len(pool))))
        out.append({m: bool(d & sample) for m, d in differing.items()})
    return out
