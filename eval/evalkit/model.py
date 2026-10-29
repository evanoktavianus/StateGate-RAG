"""The graph of a workflow: its states, and the transitions between them, read from a loaded workflow.

A transition is an `Edge`: an action of a state. Edge identity is `(state, action)`, which is what coverage counts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from docflow import Machine


@dataclass(frozen=True)
class Edge:
    state: str  # the state of the workflow the action is taken in
    action: str  # the action, `upload` included
    to: str  # the state it leads to
    by: str | None = None  # the `by` of the file, as text
    event: str | None = None
    requires: str | None = None  # the requirement type, if any
    assign: tuple[str, ...] = ()
    choose: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.state, self.action)

    @property
    def label(self) -> str:
        return f"{self.state} --{self.action}--> {self.to}"


@dataclass
class Model:
    name: str
    initial: str
    states: tuple[str, ...]
    edges: tuple[Edge, ...]
    create_assign: tuple[str, ...] = ()
    create_set: tuple[str, ...] = ()
    attributes: dict[str, tuple[str, ...] | None] = field(default_factory=dict)
    machine: Machine | None = None

    # ── reading ──
    def edges_from(self, state: str) -> list[Edge]:
        return [e for e in self.edges if e.state == state]

    def edge(self, state: str, action: str) -> Edge:
        return next(e for e in self.edges if e.key == (state, action))

    def moves(self, e: Edge) -> list[str]:
        """The states the document may be in right after `e` fires."""
        return [e.to]

    def next_edges(self, e: Edge) -> list[Edge]:
        """The edges that can fire right after `e`: those of the state the document is then in."""
        return [x for s in self.moves(e) for x in self.edges_from(s)]

    def pairs(self) -> list[tuple[Edge, Edge]]:
        return [(a, b) for a in self.edges for b in self.next_edges(a)]

    def levels(self) -> dict[str, int]:
        """Distance from the initial state, in moves of the document."""
        dist = {self.initial: 0}
        queue = deque([self.initial])
        while queue:
            s = queue.popleft()
            for e in self.edges_from(s):
                if e.to not in dist:
                    dist[e.to] = dist[s] + 1
                    queue.append(e.to)
        return dist

    def walks(self, src: str, dst: str, extra: int = 3, k: int = 60) -> list[list[Edge]]:
        return _walks(self, src, dst, extra, k)

    def routes(self, src: str, dst: str, k: int = 4) -> list[list[Edge]]:
        """Up to `k` simple routes, shortest first, from state `src` to state `dst`, as lists of edges that move the document."""
        if src == dst:
            return [[]]
        found: list[list[Edge]] = []
        queue = deque([(src, [], {src})])
        while queue and len(found) < k:
            s, path, seen = queue.popleft()
            for e in self.edges_from(s):
                if e.to in seen:
                    continue
                if e.to == dst:
                    found.append([*path, e])
                    if len(found) >= k:
                        break
                else:
                    queue.append((e.to, [*path, e], seen | {e.to}))
        return found


def _walks(model: "Model", src: str, dst: str, extra: int = 3, k: int = 60) -> list[list[Edge]]:
    """Walks from `src` to `dst` that may pass a state twice (a detour, a self-loop), shortest first, each edge used at most twice."""
    base = model.routes(src, dst, 1)
    if not base:
        return []
    limit = len(base[0]) + extra
    found: list[list[Edge]] = []
    queue = deque([(src, [])])
    while queue and len(found) < k:
        s, path = queue.popleft()
        for e in model.edges_from(s):
            if len(path) + 1 > limit or path.count(e) >= 2:
                continue
            nxt = [*path, e]
            if e.to == dst:
                found.append(nxt)
            queue.append((e.to, nxt))
    return found


def from_machine(name: str, machine: Machine) -> Model:
    edges: list[Edge] = []
    for state, actions in machine.states.items():
        for action, t in actions.items():
            edges.append(_edge(state, action, t))
    create = machine.create
    return Model(
        name=name, initial=machine.initial, states=tuple(machine.states), edges=tuple(edges),
        create_assign=tuple(create.assign) if create else (), create_set=tuple(create.set) if create else (),
        attributes=dict(machine.attributes), machine=machine,
    )


def _edge(state: str, action: str, t) -> Edge:
    return Edge(state, action, t.to, by=_text(t.by), event=t.event, requires=(t.requires or {}).get("type"),
                assign=tuple(sorted(t.assign)), choose=tuple((n, tuple(o)) for n, o in sorted(t.choose.items())))


def _text(who) -> str | None:
    if who is None:
        return None
    return " | ".join(f"{t.kind}:{','.join(map(str, t.values)) if t.values else ''}" for t in who)


def load(path: str | Path) -> Model:
    path = Path(path)
    return from_machine(path.stem, Machine.load(path))
