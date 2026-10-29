"""E6 table: the runs in which a tool returned a document or version that the independent oracle does not let the asker read, per model and condition.

    python -m evalkit.e6.analysis [eval/results/runs/e6/context_leaks.json.gz]

The runs and the leaks are those of `python -m evalkit.e6.context_leaks`. A rate carries its 95% Wilson interval; a zero is reported with its upper bound.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

Z = 1.959964
CONDITIONS = ("no_access_control", "filtering", "filtering_guidance")
ROOT = Path(__file__).resolve().parents[3]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    r = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def load(path: Path) -> list[dict]:
    """The runs of a results file (`.jsonl`, or `.jsonl.gz`)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def table(runs: list[dict]) -> list[list]:
    rows = []
    for model in sorted({r["model"] for r in runs}):
        for cond in CONDITIONS:
            rs = [r for r in runs if r["model"] == model and r["condition"] == cond]
            if not rs:
                continue
            k = sum(bool(r["context_leaks"]) for r in rs)
            lo, hi = wilson(k, len(rs))
            rows.append([model, cond, len(rs), k, f"{k / len(rs):.1%}", f"{lo:.1%} to {hi:.1%}"])
    return rows


def main(argv: list[str]) -> None:
    path = Path(argv[0]) if argv else ROOT / "eval" / "results" / "runs" / "e6" / "context_leaks.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    head = ["model", "condition", "runs", "runs with a context leak", "rate", "95% Wilson interval"]
    rows = [head] + [[str(c) for c in r] for r in table(runs)]
    width = [max(len(r[i]) for r in rows) for i in range(len(head))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, width)))


if __name__ == "__main__":
    main(sys.argv[1:])
