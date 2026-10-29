"""A static access-control baseline, scored on the stored request grids. No engine run, no model call.

A static policy gives each person class of a workflow file one fixed outcome for every state: allow (the newest version, as the unrestricted run serves it) or deny. It cannot
depend on the state, so it cannot serve the version as of a state. Only read requests count (list, open, open version, the three searches, and the agent tools); actions are
guarded transitions, not reads.

    python -m evalkit.static_baseline [--out eval/results/runs/static_baseline/static.json] [--md table.md]

For every read request the stored grid has the oracle's answer (`want`), and the stored run with the access check switched off has what an allow-everything policy returns
(`got`). A static policy that allows a class returns that second answer for the class's requests; one that denies returns nothing. Each outcome is scored exactly as the grid
scores a request (`Run._record`): a wrongful allow (something returned that the oracle does not allow), a wrongful denial (something the oracle allows is missing) and a wrong
version (the right document at another version; it counts in the other two as well, as in the grid).

Policies, per (workflow file, person class):
  permissive   allow if the oracle lets the class open the document in some situation
  restrictive  allow only if the oracle lets the class open the document in every situation
  best static  allow or deny, whichever gives the fewer failed requests over the workflow's reads: the lower bound of any state-blind policy
The class opens the document, in the oracle's terms, when the `get` request of its situation wants it.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from .paths import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "eval" / "results" / "runs"
READ = ("list", "get", "get_version", "keyword", "semantic", "hybrid", "tool_list", "tool_get", "tool_get_version", "tool_search")
POLICIES = ("permissive", "restrictive", "best static")


def load(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.open(path.with_name(path.name + ".gz"), "rt", encoding="utf-8").read())


def score(want, got) -> tuple[bool, bool, bool]:
    """(wrongful allow, wrongful denial, wrong version), as `Run._record` computes them."""
    w, g = {tuple(x) for x in want}, {tuple(x) for x in got}
    wrong_version = any(d == d2 and v != v2 for d, v in g - w for d2, v2 in w - g)
    return bool(g - w), bool(w - g), wrong_version


def grid(workflow: str) -> tuple[list[dict], list[dict]]:
    d = RUNS / ("e1" if workflow == WORKFLOWS[0] else "e2")
    own, off = load(d / f"{workflow}.json")["records"], load(d / f"{workflow}.baseline.json")["records"]
    key = lambda r: (r["path"], r["index"], r["class"], r["person"], r["channel"], str(r["target"]))
    assert [key(r) for r in own] == [key(r) for r in off] and all(a["want"] == b["want"] for a, b in zip(own, off)), f"{workflow}: the two runs do not line up"
    pairs = [(a, b) for a, b in zip(own, off) if a["channel"] in READ]
    return [a for a, _ in pairs], [b for _, b in pairs]


def tally(rows: list[tuple[bool, bool, bool]]) -> dict:
    return {"requests": len(rows), "wrongful_allow": sum(r[0] for r in rows), "wrongful_denial": sum(r[1] for r in rows), "wrong_version": sum(r[2] for r in rows),
            "failed_requests": sum(r[0] or r[1] for r in rows)}


def one_workflow(workflow: str) -> dict:
    own, off = grid(workflow)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(own):
        by_class[r["class"]].append(i)
    opens = {c: [bool(own[i]["want"]) for i in idx if own[i]["channel"] == "get"] for c, idx in by_class.items()}
    versions = {c: {tuple(map(tuple, own[i]["want"])) for i in idx if own[i]["channel"] == "get"} for c, idx in by_class.items()}
    decision = {"permissive": {}, "restrictive": {}, "best static": {}}
    for c, idx in by_class.items():
        decision["permissive"][c] = any(opens[c])
        decision["restrictive"][c] = all(opens[c])
        fails = {}
        for allow in (True, False):
            fails[allow] = sum(score(own[i]["want"], off[i]["got"] if allow else [])[:2] != (False, False) for i in idx)
        decision["best static"][c] = fails[True] <= fails[False]  # a tie allows: the newest version at least serves a reader
    out = {"read_requests": len(own), "classes": len(by_class)}
    out["stategate"] = tally([score(r["want"], r["got"]) for r in own])
    for p in POLICIES:
        out[p] = tally([score(own[i]["want"], off[i]["got"] if decision[p][own[i]["class"]] else []) for i in range(len(own))])
        out[p]["classes_allowed"] = sum(decision[p].values())
    changing = [c for c in by_class if len(set(opens[c])) > 1]  # opens in some situations and not in others
    changing_any = [c for c in by_class if len(versions[c]) > 1]  # the answer to `get` differs between situations (opened or not, or which version)
    out["classes_whose_access_changes"] = len(changing)
    out["classes_whose_access_or_version_changes"] = len(changing_any)
    out["share_changes"] = round(len(changing) / len(by_class), 3)
    out["share_changes_access_or_version"] = round(len(changing_any) / len(by_class), 3)
    return out


def main(argv: list[str]) -> int:
    args = dict(zip(argv[::2], argv[1::2]))
    res = {"workflows": {w: one_workflow(w) for w in WORKFLOWS}}
    total = {"read_requests": sum(v["read_requests"] for v in res["workflows"].values()), "classes": sum(v["classes"] for v in res["workflows"].values())}
    for p in ("stategate", *POLICIES):
        total[p] = {k: sum(v[p][k] for v in res["workflows"].values()) for k in res["workflows"][WORKFLOWS[0]][p]}
    for k in ("classes_whose_access_changes", "classes_whose_access_or_version_changes"):
        total[k] = sum(v[k] for v in res["workflows"].values())
    total["share_changes"] = round(total["classes_whose_access_changes"] / total["classes"], 3)
    total["share_changes_access_or_version"] = round(total["classes_whose_access_or_version_changes"] / total["classes"], 3)
    res["all"] = total
    bad = []
    for name, v in [*res["workflows"].items(), ("all", total)]:
        if v["stategate"]["failed_requests"] or v["stategate"]["wrong_version"]:
            bad.append(f"{name}: StateGate has errors {v['stategate']}")
        for p in ("permissive", "restrictive"):
            if v["best static"]["failed_requests"] > v[p]["failed_requests"]:
                bad.append(f"{name}: best static ({v['best static']['failed_requests']}) has more failed requests than {p} ({v[p]['failed_requests']})")
    res["sanity_checks"] = {"passed": not bad, "problems": bad}
    out = Path(args.get("--out", RUNS / "static_baseline" / "static.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    lines = ["| workflow | read requests | classes | classes whose access changes | " + " | ".join(f"{p}: allow / denial / version" for p in POLICIES) + " | StateGate |", "|---|---|---|---|---|---|---|---|"]
    for name, v in [*res["workflows"].items(), ("All", total)]:
        cells = [f"{v[p]['wrongful_allow']} / {v[p]['wrongful_denial']} / {v[p]['wrong_version']}" for p in (*POLICIES, "stategate")]
        lines.append(f"| {name} | {v['read_requests']:,} | {v['classes']} | {v['classes_whose_access_changes']} ({v['share_changes']:.0%}) | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    if "--md" in args:
        Path(args["--md"]).write_text(text + "\n")
    print(text)
    print("sanity checks:", "passed" if not bad else "FAILED\n  " + "\n  ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
