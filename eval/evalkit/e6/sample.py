"""The 200 stratified questions (88 content, 25 version, 33 status, 54 regulation), drawn with a fixed seed.

Used for the human check file and for the second model of E6. The 54 regulation questions are shared over their kinds in the proportion of the 291 regulation questions
(content 158, version 75, status 58); inside a kind the questions are shared over the workflows (regulation: over the states) in proportion, by largest remainder; inside a
cell they are taken in the order of a checksum of (seed, id), so the draw does not depend on the order of the file.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

NON_REGULATION = {"content": 88, "version": 25, "status": 33}
REGULATION = 54
SEED = 1


def largest_remainder(weights: dict, total: int) -> dict:
    s = sum(weights.values())
    quota = {k: w * total / s for k, w in weights.items()}
    take = {k: int(v) for k, v in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - take[k]), str(k)))[: total - sum(take.values())]:
        take[k] += 1
    return take


def draw(questions: list[dict], seed: int = SEED) -> list[dict]:
    order = lambda q: zlib.crc32(f"{seed}\0{q['id']}".encode())
    reg = [q for q in questions if q["workflow"] == "regulation_single"]
    non = [q for q in questions if q["workflow"] != "regulation_single"]
    plan = [(non, kind, n, "workflow") for kind, n in NON_REGULATION.items()]
    kinds = largest_remainder(dict(Counter(q["kind"] for q in reg)), REGULATION)
    plan += [(reg, kind, n, "state") for kind, n in kinds.items()]
    out = []
    for pool, kind, n, by in plan:
        group = [q for q in pool if q["kind"] == kind]
        cells = defaultdict(list)
        for q in group:
            cells[q[by]].append(q)
        take = largest_remainder({c: len(v) for c, v in cells.items()}, n)
        for c, v in cells.items():
            out += sorted(v, key=order)[: take[c]]
    return sorted(out, key=lambda q: q["id"])


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    qs = json.loads((root / "eval/data/questions" / "questions.json").read_text(encoding="utf-8"))
    picked = draw(qs)
    (root / "eval/data/questions" / "stratified200.json").write_text(json.dumps([q["id"] for q in picked]), encoding="utf-8")
    print(len(picked), dict(Counter((q["workflow"] == "regulation_single", q["kind"]) for q in picked)))


if __name__ == "__main__":
    main()
