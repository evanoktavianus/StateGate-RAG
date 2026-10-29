"""Human check of the guidance judges: a blind sample for a labeller, a check of the labels, and the agreement of each judge with them.

    python -m evalkit.e6.guidance_human build --answers answers.jsonl.gz --verdicts judge_a.jsonl.gz --verdicts2 judge_b.jsonl.gz --out DIR [--seed 20261029] [--per-cell 10]
    python -m evalkit.e6.guidance_human check LABELS.csv
    python -m evalkit.e6.guidance_human score LABELS.csv --sample sample.json [--b LABELS_B.csv] [--out report.txt]

`build` draws, from the answers the two judges rated, one item per (run, document) that code says the answer names, for the four status kinds in the three conditions: `--per-cell` items
in each cell by simple random sampling with a fixed seed (no selection by verdict), all shuffled together. It writes `human_check.csv` for the labeller (question, answer, document, state,
guidance, rule, empty `label` and `note`; the condition and the judges' verdicts are not in it) and `sample.json` (the seed, the sizes of the cells, and for every item its condition, kind and
both judges' verdicts).
A label is one of BREAKS (the answer misstates the document's status under the rule), KEEPS (it draws on the document and does not misstate its status) or NOT USED (it does not draw on
the document). `check` refuses a file in which any row has another label and warns when one label covers more than 90% of the rows. `score` compares the labels with each judge, binary
(BREAKS against not breaking): agreement and Cohen's kappa with a 95% bootstrap interval (2,000 resamples, seed 7), per condition and per kind, the confusion matrix, every disagreement
with the judge's reason, and the break rates per condition of the labeller and of each judge on the same items; with `--b`, the kappa between two labellers.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from . import guidance as G

KINDS = G.RESTRICTIONS
CONDITIONS = G.CONDITIONS
ALLOWED = ("BREAKS", "KEEPS", "NOT USED")
JUDGES = ("glm", "muse")
FIELDS = ["id", "question", "answer", "document", "state", "guidance", "rule", "label", "note"]


def build(answers: list[dict], vg: list[dict], vm: list[dict], questions: dict[str, str], seed: int = 20261029, per_cell: int = 10) -> tuple[list[dict], dict]:
    """(rows for the labeller, sample record)."""
    cells: dict[tuple, list] = {(k, c): [] for k in KINDS for c in CONDITIONS}
    for idx, (a, g, m) in enumerate(zip(answers, vg, vm)):
        assert (a["pair"], a["condition"]) == (g["pair"], g["condition"]) == (m["pair"], m["condition"])
        if g["error"] or m["error"]:
            continue
        for it in a["items"]:
            if it["kind"] in KINDS and G.names_regulation(it, a["answer"]):
                cells[(it["kind"], a["condition"])].append((idx, it))
    rng = random.Random(seed)
    picked, report = [], []
    for cell, pool in cells.items():
        pool.sort(key=lambda x: (answers[x[0]]["pair"], answers[x[0]]["condition"], x[1]["doc"]))
        take = rng.sample(pool, min(per_cell, len(pool)))
        picked += [(cell, x) for x in take]
        report.append({"kind": cell[0], "condition": cell[1], "eligible": len(pool), "taken": len(take), "shortfall": max(0, per_cell - len(pool))})
    rng.shuffle(picked)
    rows, items = [], []
    for n, ((kind, cond), (idx, it)) in enumerate(picked, 1):
        a = answers[idx]
        verdict = lambda vs: next({k: v[k] for k in ("uses_document", "follows_guidance", "reason")} for v in vs[idx]["items"] if v["doc"] == it["doc"])
        rows.append({"id": n, "question": questions[a["qid"]], "answer": a["answer"], "document": f'{it["doc"]} = {G.head(it["doc"], it["version"])}', "state": it["state"],
                     "guidance": it["guidance"], "rule": G.RUBRIC[kind], "label": "", "note": ""})
        items.append({"id": n, "pair": a["pair"], "condition": cond, "kind": kind, "doc": it["doc"], "version": it["version"], "answer_index": idx,
                      "glm": verdict(vg), "muse": verdict(vm)})
    return rows, {"seed": seed, "per_cell": per_cell, "rubric_digest": G.rubric_digest(), "cells": report, "items": items}


def read_labels(path: str | Path) -> dict[int, str]:
    """id -> label for a file that passes the check; ValueError otherwise."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [(r["id"], r["label"]) for r in rows if r["label"].strip().upper() not in ALLOWED]
    if bad:
        raise ValueError(f"{len(bad)} rows have no allowed label (allowed: {', '.join(ALLOWED)}); first: {bad[:5]}")
    ids = [int(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("an id appears twice")
    return {int(r["id"]): r["label"].strip().upper() for r in rows}


def check(labels: dict[int, str]) -> list[str]:
    counts = Counter(labels.values())
    out = [f"{len(labels)} rows; " + ", ".join(f"{k} {counts[k]}" for k in ALLOWED)]
    top, n = counts.most_common(1)[0]
    if n / len(labels) > 0.9:
        out.append(f"WARNING: {top} covers {n / len(labels):.0%} of the rows; check for fill-down or replace-all")
    return out


def kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return None if pe == 1 else (po - pe) / (1 - pe)


def fmt(k: float | None) -> str:
    return "n/a" if k is None else f"{k:.3f}"


def bootstrap(a: list, b: list, reps: int = 2000, seed: int = 7) -> tuple[float | None, float | None]:
    rng, n, ks = random.Random(seed), len(a), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            ks.append(k)
    ks.sort()
    return (ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks)) - 1]) if ks else (None, None)


def verdict(j: dict) -> str:
    return "NOT USED" if not j["uses_document"] else ("KEEPS" if j["follows_guidance"] else "BREAKS")


def table(rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] + ["| " + " | ".join(r) + " |" for r in rows[1:]]


def judge_section(name: str, items: list[dict], human: dict[int, str]) -> list[str]:
    h = [human[i["id"]] == "BREAKS" for i in items]
    j = [verdict(i[name]) == "BREAKS" for i in items]
    lo, hi = bootstrap(h, j)
    agree = sum(x == y for x, y in zip(h, j))
    out = [f"Judge {name}", "", f"Overall ({len(items)} items): agreement {agree}/{len(items)} ({100 * agree / len(items):.1f}%), kappa {fmt(kappa(h, j))} (95% bootstrap interval {fmt(lo)} to {fmt(hi)}).", ""]
    rows = [["group", "items", "agreement", "kappa", "human break rate", "judge break rate", "note"]]
    groups = [(f"condition {c}", [n for n, i in enumerate(items) if i["condition"] == c]) for c in CONDITIONS] + [(f"kind {k}", [n for n, i in enumerate(items) if i["kind"] == k]) for k in KINDS]
    for label, sel in groups:
        a, b = [h[n] for n in sel], [j[n] for n in sel]
        rows.append([label, str(len(sel)), f"{sum(x == y for x, y in zip(a, b))}/{len(sel)}", fmt(kappa(a, b)), f"{sum(a)}/{len(sel)}", f"{sum(b)}/{len(sel)}", "small cell (under 30)" if len(sel) < 30 else ""])
    c = Counter(zip(h, j))
    out += table(rows) + ["", "Confusion matrix (rows: human, columns: judge), binary:", ""]
    out += table([["human \\ judge", "breaks", "not breaking"], ["breaks", str(c[(True, True)]), str(c[(True, False)])], ["not breaking", str(c[(False, True)]), str(c[(False, False)])]])
    c3 = Counter((human[i["id"]], verdict(i[name])) for i in items)
    out += ["", "Three-way table (rows: human, columns: judge):", ""]
    out += table([["human \\ judge", *ALLOWED]] + [[a, *[str(c3[(a, b)]) for b in ALLOWED]] for a in ALLOWED])
    out += ["", f"Disagreements on the binary label ({sum(x != y for x, y in zip(h, j))}):", ""]
    dis = [["id", "kind", "condition", "human", "judge", "judge's reason"]]
    for n, i in enumerate(items):
        if h[n] != j[n]:
            dis.append([str(i["id"]), i["kind"], i["condition"], human[i["id"]], verdict(i[name]), i[name]["reason"].replace("|", "/")])
    return out + (table(dis) if len(dis) > 1 else ["none"]) + [""]


def score(sample: dict, human: dict[int, str], human_b: dict[int, str] | None = None) -> str:
    items = sample["items"]
    assert sorted(human) == sorted(i["id"] for i in items), "the labels do not cover the items of the sample"
    out = ["Human check of the guidance judges", "", f"Rubric digest {sample['rubric_digest']}; sample seed {sample['seed']}, {sample['per_cell']} items per cell, {len(items)} items; binary label BREAKS against not breaking.", "",
           "Per-cell sample sizes:", ""]
    out += table([["kind", "condition", "eligible", "taken", "shortfall"]] + [[c["kind"], c["condition"], str(c["eligible"]), str(c["taken"]), str(c["shortfall"])] for c in sample["cells"]]) + [""]
    counts = Counter(human.values())
    out += ["Labels: " + ", ".join(f"{k} {counts[k]}" for k in ALLOWED), ""]
    rows = [["condition", "items", "human break rate", *[f"{n} break rate" for n in JUDGES]]]
    for c in CONDITIONS:
        sel = [i for i in items if i["condition"] == c]
        rows.append([c, str(len(sel)), f'{sum(human[i["id"]] == "BREAKS" for i in sel)}/{len(sel)}', *[f'{sum(verdict(i[n]) == "BREAKS" for i in sel)}/{len(sel)}' for n in JUDGES]])
    out += ["Break rate per condition on the same items (10 items per kind and condition, so these are not the rates of the full run):", ""] + table(rows) + [""]
    for name in JUDGES:
        out += judge_section(name, items, human)
    if human_b is not None:
        ids = sorted(human)
        a, b = [human[i] == "BREAKS" for i in ids], [human_b[i] == "BREAKS" for i in ids]
        lo, hi = bootstrap(a, b)
        out += ["Two labellers", "", f"Agreement {sum(x == y for x, y in zip(a, b))}/{len(ids)}, binary kappa {fmt(kappa(a, b))} (95% bootstrap interval {fmt(lo)} to {fmt(hi)}).", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    cmd, rest = argv[0], argv[1:]
    positional = [a for a in rest[:1] if not a.startswith("--")]
    opts = dict(zip(rest[len(positional)::2], rest[len(positional) + 1::2]))
    if cmd == "build":
        answers, vg, vm = G.load(Path(opts["--answers"])), G.load(Path(opts["--verdicts"])), G.load(Path(opts["--verdicts2"]))
        questions = {q["id"]: q["text"] for q in json.loads(G.QUESTIONS.read_text(encoding="utf-8"))}
        rows, sample = build(answers, vg, vm, questions, int(opts.get("--seed", 20261029)), int(opts.get("--per-cell", 10)))
        out = Path(opts["--out"])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "human_check.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, FIELDS, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)
        (out / "sample.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{len(rows)} items, seed {sample['seed']}; " + ", ".join(f"{c['kind']}/{c['condition']} {c['taken']} of {c['eligible']}" for c in sample["cells"][:2]) + " ...")
    elif cmd == "check":
        try:
            print("\n".join(check(read_labels(positional[0]))))
        except ValueError as e:
            print("REFUSED:", e)
            return 1
    elif cmd == "score":
        sample = json.loads(Path(opts["--sample"]).read_text(encoding="utf-8"))
        text = score(sample, read_labels(positional[0]), read_labels(opts["--b"]) if "--b" in opts else None)
        if "--out" in opts:
            Path(opts["--out"]).write_text(text + "\n", encoding="utf-8")
        print(text)
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
