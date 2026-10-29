"""A check of equivalence for the version-selector mutants that survive, from the graph of the workflow (not from the cast or the generated paths).

A rule `serve`/`find` names a version in one of two ways: `latest`, or `from: A`, "the version the document had when it last entered state A". A mutant that changes
the selector of a rule in state Y gives another version only if, in some history that ends in Y, the two selectors name different versions. Versions are made only by
`upload`, so two selectors differ in a history only if an upload falls between the moments they refer to:

    A against latest   an upload after the last entry into A and before the end (in Y): a route A -> U -> Y, U a state that allows upload (U may be A or Y),
                       that does not enter A again after the start;
    A against B        an upload between the last entry into A and the last entry into B (or the reverse): a route A -> U -> B (not entering A again), or B -> U -> A;
    a state that is not on every route to Y   a history can end in Y without having entered it, and the selector then names nothing.

If none of these routes exists in the graph, the two selectors name the same version in every history that ends in Y, whoever asks: the mutant is equivalent, with
no reference to the cast. If one exists, the mutant is not equivalent by this check and its survival is a gap of the suite (the paths do not reach that history).
Only the `change_version` mutants are decided here; the other operators are read one by one.
"""

from __future__ import annotations

import json
import re

from .model import Model

DESC = re.compile(r"^(?P<state>[\w.]+)\.access\[(?P<i>\d+)\]: (?P<kind>serve|find) (?P<a>.+) -> (?P<b>.+)$")


def _sel(text: str):
    try:
        x = json.loads(text)
    except ValueError:
        return "unread"
    if isinstance(x, str):
        return x  # "latest", or "nothing" (not read here)
    return x["from"]


def parse(desc: str):
    m = DESC.match(desc)
    return (m["state"], m["kind"], _sel(m["a"]), _sel(m["b"])) if m else None


class Graph:
    def __init__(self, model: Model) -> None:
        self.initial = model.initial
        self.succ: dict[str, set[str]] = {s: set() for s in model.states}
        for e in model.edges:
            if e.action == "upload":
                continue
            for t in model.moves(e):
                if t != e.state:
                    self.succ.setdefault(e.state, set()).add(t)
        self.uploads = {e.state for e in model.edges if e.action == "upload"}

    def reach(self, src: str, avoid: frozenset = frozenset()) -> set[str]:
        """States reachable from `src` (itself included) without entering a state of `avoid` after the start."""
        seen, todo = {src}, [src]
        while todo:
            for t in self.succ.get(todo.pop(), ()):
                if t not in seen and t not in avoid:
                    seen.add(t)
                    todo.append(t)
        return seen

    def differs_from_latest(self, a: str, y: str) -> str | None:
        for u in sorted(self.uploads & self.reach(a, frozenset({a}))):
            if y in self.reach(u, frozenset({a})):
                return f"a version can be uploaded in {u} after {a} was entered and before the document is in {y}"
        return None

    def differs(self, a: str, b: str, y: str) -> str | None:
        for first, second in ((a, b), (b, a)):
            for u in sorted(self.uploads & self.reach(first, frozenset({first}))):
                if second in self.reach(u, frozenset({first})) and y in self.reach(second, frozenset({first})):
                    return f"a version can be uploaded in {u} between the last entry into {first} and the last entry into {second}"
        return None

    def missing(self, x: str, y: str) -> bool:
        """True if a document can be in `y` without ever having entered `x`."""
        return x != y and x != self.initial and y in self.reach(self.initial, frozenset({x}))


def decide(model: Model, desc: str) -> dict | None:
    """{"verdict": "equivalent" | "gap", "reason": ...} for a change_version mutant, None for a description this check does not read."""
    p = parse(desc)
    if p is None:
        return None
    y, kind, a, b = p
    if {"nothing", "unread"} & {a, b}:
        return None
    g = Graph(model)
    if y not in g.succ:  # a state named with a track prefix or the like: not read here
        return None
    ups = ", ".join(sorted(g.uploads)) or "no state"
    for x in (a, b):
        if x != "latest" and x != y and g.missing(x, y):
            return {"verdict": "gap", "reason": f"a document can be in {y} without having entered {x}, so `from {x}` names nothing there; the suite's paths do not show it"}
    if "latest" in (a, b):
        s = b if a == "latest" else a
        why = g.differs_from_latest(s, y)
        if why:
            return {"verdict": "gap", "reason": why + "; the suite's paths do not show it"}
        return {"verdict": "equivalent", "reason": f"uploads are made only in {ups}; none can be reached from {s} on the way to {y} without entering {s} again, so the version when {s} was last entered is the latest version in {y}"}
    why = g.differs(a, b, y)
    if why:
        return {"verdict": "gap", "reason": why + "; the suite's paths do not show it"}
    return {"verdict": "equivalent", "reason": f"uploads are made only in {ups}; none falls between the last entry into {a} and the last entry into {b} on any route to {y}, so `from {a}` and `from {b}` name the same version there"}
