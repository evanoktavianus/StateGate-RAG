"""Action sequences that reach every state, take every transition, and every consecutive pair of transitions.

For one workflow: node coverage (every state), edge coverage (every transition: all-transitions) and edge-pair coverage (every pair of
consecutive transitions). Each path is a list of real steps (who did what, with which input) that the system accepted; a transition
that cannot be fired from the fixtures is reported as uncoverable with the reason. Extra paths cover every option of every choice
and every value of every attribute set at creation, each carried on to the deepest state it reaches.

    python -m evalkit.paths [workflow ...] [--json eval/results/paths.json] [--md eval/results/paths.md]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import model as model_mod
from .model import Edge, Model
from .replay import Firing, Sim, Step, Unfireable, replay
from .worlds import ROOT, Spec, load_spec

PAIR_ROUTES = 3


@dataclass
class Path_:
    id: str
    attrs: dict
    steps: list[Step]
    firings: list[Firing]
    why: str = ""  # what this path was made for

    def to_json(self) -> dict:
        return {"id": self.id, "attrs": self.attrs, "why": self.why, "steps": [s.to_json() for s in self.steps],
                "firings": [{"edge": list(f.key), "before": f.before, "after": f.after, "option": f.option} for f in self.firings]}


@dataclass
class PathSet:
    workflow: str
    model: Model
    spec: Spec
    paths: list[Path_] = field(default_factory=list)
    uncoverable_edges: dict[tuple, str] = field(default_factory=dict)
    uncoverable_pairs: dict[tuple, str] = field(default_factory=dict)
    limits: dict[str, str] = field(default_factory=dict)  # an attribute value whose document cannot be carried to the deepest state, and why
    seconds: float = 0.0

    # ── coverage ──
    def states_covered(self) -> set[str]:
        out = {self.model.initial}
        for p in self.paths:
            out |= {f.after for f in p.firings} | {f.before for f in p.firings}
        return out

    def edges_covered(self) -> set[tuple]:
        return {f.key for p in self.paths for f in p.firings}

    def pairs_covered(self) -> set[tuple]:
        out = set()
        for p in self.paths:
            for a, b in zip(p.firings, p.firings[1:]):
                if a.after == b.before:
                    out.add((a.key, b.key))
        return out

    def options_covered(self) -> set[tuple]:
        return {(f.key, f.option) for p in self.paths for f in p.firings if f.option is not None}

    def options_total(self) -> set[tuple]:
        return {(e.key, o) for e in self.model.edges for _, opts in e.choose for o in opts}

    def table_row(self) -> dict:
        m = self.model
        pairs = {(a.key, b.key) for a, b in m.pairs()}
        return {
            "workflow": self.workflow,
            "states": [len(self.states_covered() & set(m.states)), len(m.states)],
            "edges": [len(self.edges_covered()), len(m.edges)],
            "edge_pairs": [len(self.pairs_covered() & pairs), len(pairs)],
            "options": [len(self.options_covered()), len(self.options_total())],
            "paths": len(self.paths), "steps": sum(len(p.steps) for p in self.paths),
            "situations": sum(len(p.steps) + 1 for p in self.paths),
            "uncoverable_edges": [{"edge": list(k), "reason": v} for k, v in self.uncoverable_edges.items()],
            "limits": [{"attributes": k, "what": v} for k, v in self.limits.items()],
            "uncoverable_pairs": [{"pair": [list(a), list(b)], "reason": v} for (a, b), v in self.uncoverable_pairs.items()],
            "unexplained": {
                "edges": [list(e.key) for e in m.edges if e.key not in self.edges_covered() and e.key not in self.uncoverable_edges],
                "pairs": [[list(k[0]), list(k[1])] for k in sorted(pairs - self.pairs_covered() - set(self.uncoverable_pairs))],
                "options": [[list(k), o] for k, o in sorted(self.options_total() - self.options_covered())
                            if (k, o) not in self.uncoverable_edges],
            },
            "seconds": round(self.seconds, 2),
        }


def _run(spec: Spec, model: Model, attrs: dict, ops: list[tuple[Edge, str | None]]) -> Sim:
    sim = Sim(spec, model, attrs)
    for e, option in ops:
        sim.fire(e, option)
    return sim


def _first_success(spec, model, attrs, candidates: list[list[tuple[Edge, str | None]]], wanted=None) -> tuple[Sim | None, str]:
    """The first candidate sequence the system accepts (and, when `wanted` is given, that does what it was made for)."""
    reason = "no route from the initial state"
    for ops in candidates:
        try:
            sim = _run(spec, model, attrs, ops)
        except Unfireable as u:
            reason = u.reason
            continue
        if wanted is None or wanted(sim):
            return sim, ""
        reason = "accepted, but the actions needed in between (a choice, more people) came between the two"
    return None, reason


def _has_pair(a: Edge, b: Edge):
    def check(sim: Sim) -> bool:
        return any(x.after == y.before
                   for x, y in zip(sim.firings, sim.firings[1:]) if x.key == a.key and y.key == b.key)
    return check


def _routes_to(model: Model, state: str, k: int) -> list[list[Edge]]:
    return model.routes(model.initial, state, k)


def generate(name: str, spec: Spec | None = None, upload_rule: bool = True) -> PathSet:
    """`upload_rule=False` gives the path set without step 6; the corpus replay uses it, so the states it gives to the documents (and so the question set) are the ones recorded."""
    t0 = time.time()
    spec = spec or load_spec(name)
    model = model_mod.from_machine(name, spec.machine())
    ps = PathSet(name, model, spec)
    base = spec.variants()[0]
    counter = [0]

    def add(sim: Sim, attrs: dict, why: str) -> Path_:
        counter[0] += 1
        p = Path_(f"{name}#{counter[0]:03d}", attrs, sim.steps, sim.firings, why)
        ps.paths.append(p)
        return p

    # 1. every transition: route to its state, then fire it
    for e in model.edges:
        if e.key in ps.edges_covered():
            continue
        routes = _routes_to(model, e.state, 4)
        cands = [[(x, None) for x in r] + [(e, None)] for r in routes] or []
        sim, reason = _first_success(spec, model, base, cands)
        if sim is None:
            ps.uncoverable_edges[e.key] = reason
        else:
            add(sim, base, f"edge {e.label}")
    # 2. every state not reached by an edge (the initial one is reached by creating the document)
    # 3. every consecutive pair of transitions
    for a, b in model.pairs():
        pk = (a.key, b.key)
        if pk in ps.pairs_covered() or a.key in ps.uncoverable_edges or b.key in ps.uncoverable_edges:
            if pk not in ps.pairs_covered():
                ps.uncoverable_pairs[pk] = "one of its edges is uncoverable"
            continue
        cands = []
        prefixes = _routes_to(model, a.state, PAIR_ROUTES) + [w for w in model.walks(model.initial, a.state) if w not in _routes_to(model, a.state, PAIR_ROUTES)]
        for r in prefixes:
            cands.append([(x, None) for x in r] + [(a, None), (b, None)])
        sim, reason = _first_success(spec, model, base, cands, _has_pair(a, b))
        if sim is None:
            ps.uncoverable_pairs[pk] = reason
        else:
            add(sim, base, f"pair {a.label} then {b.label}")
    # 4. every option of every choice, carried to the deepest state it reaches
    levels = model.levels()
    for e in model.edges:
        for _, options in e.choose:
            for option in options:
                if (e.key, option) in ps.options_covered() or e.key in ps.uncoverable_edges:
                    continue
                after = [s for s in model.moves(e)][0]
                deepest = max((s for s in levels if s != after), key=lambda s: (_depth_from(model, after, s), s), default=None)
                tails = [r for r in model.routes(after, deepest, 2)] if deepest and _depth_from(model, after, deepest) > 0 else [[]]
                cands = [[(x, None) for x in r] + [(e, option)] + [(x, None) for x in t]
                         for r in _routes_to(model, e.state, 2) for t in tails]
                sim, reason = _first_success(spec, model, base, cands)
                if sim is None:
                    ps.uncoverable_edges.setdefault((e.key, option), reason)  # type: ignore[arg-type]
                else:
                    add(sim, base, f"option {option} of {e.label}")
    # 5. every other value of every attribute set at creation, carried as far as it can go: the deepest state first, then the next
    for attrs in spec.variants()[1:]:
        reason, done = "", False
        for target in sorted(levels, key=lambda s: (-levels[s], s)):
            sim, reason_here = _first_success(spec, model, attrs, [[(x, None) for x in r] for r in _routes_to(model, target, 2)])
            if sim is not None:
                add(sim, attrs, f"attributes {attrs}")
                if target != max(levels, key=lambda s: (levels[s], s)) or reason:
                    ps.limits[json.dumps(attrs, sort_keys=True)] = f"reaches {sim.state}; {reason}"
                done = True
                break
            reason = reason or reason_here  # why the deepest state could not be reached
        if not done:
            ps.uncoverable_edges[("create", json.dumps(attrs, sort_keys=True))] = reason  # type: ignore[assignment]
    # 6. the upload rule: in every state that allows an upload, the document is uploaded to once after each way of entering the state (the creation, and every
    #    other state's action that leads here, returns included), then carried on to the deepest state. So a version made after a state was last entered exists
    #    in the situations after it, and "the version as of a state" differs from "the latest" wherever the policy makes them differ.
    deepest = max(levels, key=lambda s: (levels[s], s))
    for up in [e for e in model.edges if e.action == "upload"] if upload_rule else []:
        entries = ([None] if up.state == model.initial else []) + [e for e in model.edges if e.to == up.state and e.state != up.state and e.action != "upload"]
        for entry in entries:
            heads = [[]] if entry is None else [[(x, None) for x in r] + [(entry, None)] for r in _routes_to(model, entry.state, 2)]
            tails = [[(x, None) for x in t] for t in model.routes(up.state, deepest, 2)] if deepest != up.state else []
            cands = [h + [(up, None)] + t for h in heads for t in tails] + [h + [(up, None)] for h in heads]
            sim, reason = _first_success(spec, model, base, cands)
            if sim is None:
                ps.uncoverable_edges.setdefault((up.key, "upload after " + (entry.label if entry else "creation")), reason)  # type: ignore[arg-type]
            else:
                add(sim, base, f"upload in {up.state} after " + (entry.label if entry else "creation"))
    ps.paths = _drop_prefixes(ps.paths)
    # a state the document never reached is reported with the edges that lead to it
    ps.seconds = time.time() - t0
    return ps


def _depth_from(model: Model, src: str, dst: str) -> int:
    r = model.routes(src, dst, 1)
    return len(r[0]) if r else -1


def _drop_prefixes(paths: list[Path_]) -> list[Path_]:
    """A path whose steps are a prefix of another path's, with the same attributes, adds nothing; of two equal paths the later one goes (by position, not by memory address, so the result does not change from run to run)."""
    def prefix(i: int, a: Path_, j: int, b: Path_) -> bool:
        return i != j and a.attrs == b.attrs and len(a.steps) <= len(b.steps) and a.steps == b.steps[: len(a.steps)] and (len(a.steps) < len(b.steps) or i > j)
    return [p for i, p in enumerate(paths) if not any(prefix(i, p, j, q) for j, q in enumerate(paths))]


# ── reading a path back ──
def situations(ps: PathSet, path: Path_, store=None):
    """Replays a path in a fresh world; yields (situation, live world) after the creation and after each step."""
    yield from replay(ps.spec, ps.model, path.attrs, path.steps, path.id, store)


def replay_all(ps: PathSet) -> int:
    """Replays every path; returns the number of situations. A refusal raises `Unfireable`."""
    n = 0
    for p in ps.paths:
        for _ in situations(ps, p):
            n += 1
    return n


# ── report ──
WORKFLOWS = [
    "card1_policy_analysis_simple", "card2_policy_analysis_individual", "policy_analysis_team_single",
    "policy_analysis_other_unit_single", "card5_policy_analysis_sensitive", "knowledge_asset", "card7_official_document",
    "regulation_single",
]


def report(sets: list[PathSet]) -> str:
    rows = [ps.table_row() for ps in sets]
    cell = lambda x: f"{x[0]}/{x[1]}"
    out = ["| workflow | states | edges | edge-pairs | choice options | paths | steps | situations |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['workflow']} | {cell(r['states'])} | {cell(r['edges'])} | {cell(r['edge_pairs'])} | {cell(r['options'])} | {r['paths']} | {r['steps']} | {r['situations']} |")
    out += ["", "Uncoverable (never dropped):", ""]
    any_ = False
    for r in rows:
        for u in r["uncoverable_edges"]:
            any_ = True
            out.append(f"- {r['workflow']}: edge {tuple(u['edge'])}: {u['reason']}")
        for u in r["uncoverable_pairs"]:
            any_ = True
            out.append(f"- {r['workflow']}: pair {u['pair']}: {u['reason']}")
    if not any_:
        out.append("- none")
    out += ["", "Attribute values whose document cannot be carried to the deepest state (not a failure: nobody of the cast can take the next action):", ""]
    lim = [f"- {r['workflow']}: {u['attributes']}: {u['what']}" for r in rows for u in r["limits"]]
    out += lim or ["- none"]
    for r in rows:
        u = r["unexplained"]
        if u["edges"] or u["pairs"] or u["options"]:
            out.append(f"- NOT COVERED AND NOT EXPLAINED, {r['workflow']}: {u}")
    return "\n".join(out)


def main(argv: list[str]) -> None:
    args = list(argv)
    opts = {}
    for flag in ("--json", "--md"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i : i + 2]
    names = args or WORKFLOWS
    sets = [generate(n) for n in names]
    for ps in sets:
        replay_all(ps)
    text = report(sets)
    print(text)
    if "--json" in opts:
        Path(opts["--json"]).write_text(json.dumps({"workflows": [ps.table_row() for ps in sets],
                                                    "paths": {ps.workflow: [p.to_json() for p in ps.paths] for ps in sets}}, indent=1))
    if "--md" in opts:
        import datetime, os, platform
        head = [
            "# Paths: coverage of the workflows", "",
            f"Commit `{os.environ.get('EVAL_COMMIT', 'unknown')}`; {datetime.date.today()}; machine {platform.node()}; command `python -m evalkit.paths"
            + (" " + " ".join(argv) if argv else "") + "`; run time " + f"{sum(ps.seconds for ps in sets):.1f} s (generation) ; no seed (the generator is deterministic).", "",
            "Every path is a list of real steps the system accepted, replayed again from a fresh world without a refusal. States are the states of the "
            "workflow; edges are its transitions (an action of a state); "
            "edge-pairs are consecutive transitions. Choice options are the options of every `choose`; each is made and carried to the deepest state. "
            "Each value of each attribute set at creation has a path of its own. This is the coverage of the test suite's situations, not an evaluation result.", "",
        ]
        Path(opts["--md"]).write_text("\n".join(head) + "\n" + text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
