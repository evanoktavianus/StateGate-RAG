"""Driving a document with real actions, and snapshotting a situation after every step.

`Sim` builds a fresh world and fires edges of the model: it looks for a person of the cast the system accepts, fills in the people an
action asks for, and records each accepted step. `replay` runs a recorded path again with no search and fails on any refusal.
Nothing writes state or history directly: every step is `World.act` or, for an outside event, `World.fire_event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from docflow.access_control.access_policy_engine import machine as machine_mod
from docflow.stores.world import World
from docflow.support.model import Refused

from .model import Edge, Model
from .worlds import Spec, build_world


class Unfireable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Step:
    actor: str | None  # None: the host fires an outside event
    action: str
    input: dict = field(default_factory=dict)
    event: str | None = None

    def to_json(self) -> dict:
        return {"actor": self.actor, "action": self.action, "input": self.input, "event": self.event}


@dataclass(frozen=True)
class Firing:
    key: tuple[str, str]  # (state, action) of the edge that fired
    before: str
    after: str
    option: str | None = None  # the option of a choice the action made, if any


@dataclass(frozen=True)
class Situation:
    """A document in a known state with a known history."""

    path: str
    index: int  # steps taken so far
    state: str
    version: int
    entries: tuple
    assigned: tuple
    choices: tuple
    actors: tuple  # who acted, in order


def snapshot(world: World, spec: Spec, path_id: str, index: int, steps: list[Step]) -> Situation:
    tl = world.documents[spec.doc_id].timeline
    return Situation(path_id, index, tl.state, tl.version, tl.entries,
                     tuple(sorted(tl.assigned().items())), tuple(sorted(tl.choices().items())),
                     tuple(s.actor for s in steps[:index] if s.actor))


def input_for(model: Model, world: World, spec: Spec, e: Edge, option: str | None = None) -> dict:
    """The input a valid request for the edge carries: the people it asks for, an option of each choice, a signature of the current version."""
    inp: dict = {}
    m = model.machine
    t = m.states[e.state].get(e.action)
    doc = world.documents[spec.doc_id]
    for name in e.assign:
        if name in spec.assign:
            inp[name] = list(spec.assign[name])
            continue
        rule = t.assign[name]
        ctx = replace(world.context(doc), assigned=doc.timeline.assigned())
        cands = [p.id for p in machine_mod.candidates(rule, world.members(spec.workspace), ctx)]
        n = rule.min + (1 if rule.odd and rule.min % 2 == 0 else 0)
        if rule.max is not None:
            n = min(n, rule.max)
        inp[name] = cands[:n]
    for name, options in e.choose:
        inp[name] = option if option in options else options[0]
    if e.requires == "signature":
        inp["signature"] = {"ref": "sig-evalkit", "digest": world.digest(spec.doc_id, doc.timeline.version)}
    return inp


class Sim:
    """A fresh world and the steps taken in it."""

    def __init__(self, spec: Spec, model: Model, attrs: dict[str, str], store=None) -> None:
        self.spec, self.model, self.attrs = spec, model, attrs
        self.world = build_world(spec, attrs, model.machine, store)
        self.steps: list[Step] = []
        self.firings: list[Firing] = []
        self.people = list(spec.people)

    # ── reading ──
    @property
    def tl(self):
        return self.world.documents[self.spec.doc_id].timeline

    @property
    def state(self) -> str:
        return self.tl.state

    # ── one accepted step ──
    def _try(self, actor: str | None, action: str, inp: dict, event: str | None):
        w, doc = self.world, self.spec.doc_id
        if event is not None:
            if w.fire_event(doc, action, event) is None:
                raise Refused(w.log[-1]["reason"])
        else:
            w.act(doc, actor, action, input=inp)
        self.steps.append(Step(actor, action, dict(inp), event))

    def _assign_input(self, e: Edge) -> dict:
        return input_for(self.model, self.world, self.spec, e)

    # ── firing an edge ──
    def _take(self, actor: str | None, e: Edge, inp: dict) -> bool:
        """One accepted step. True when the action took effect, False when only this person's part was recorded (it needs more people)."""
        size = len(self.tl.entries)
        self._try(actor, e.action, inp, e.event)
        return not any(x[0] == "given" for x in self.tl.entries[size:])

    def fire(self, e: Edge, option: str | None = None, depth: int = 0) -> None:
        """Fires the edge or raises `Unfireable` with the reason. Records the steps and the firing."""
        before = self.state
        inp = input_for(self.model, self.world, self.spec, e, option)
        reasons: list[str] = []
        fired = False
        for actor in ([None] if e.event else self.people):
            try:
                fired = self._take(actor, e, inp)
            except Refused as r:
                if r.reason == "choice_needed" and depth < 3:
                    name = r.detail.get("name")
                    pre = next((x for x in self.model.edges_from(self.state) if any(n == name for n, _ in x.choose)), None)
                    if pre is None:
                        raise Unfireable(f"needs the choice {name!r}, which no action of {self.state} makes")
                    self.fire(pre, None, depth + 1)
                    return self.fire(e, option, depth + 1)
                reasons.append(f"{actor}: {r.reason}")
                continue
            if not fired:  # their part is recorded; other people take theirs until the action takes effect
                for other in self.people:
                    try:
                        fired = self._take(other, e, inp)
                    except Refused as r:
                        reasons.append(f"{other}: {r.reason}")
                        continue
                    if fired:
                        break
            break
        if not fired:
            raise Unfireable("; ".join(sorted(set(reasons))) or "not enough people could take part")
        self.firings.append(Firing(e.key, before, self.state, inp[e.choose[0][0]] if e.choose else None))


def replay(spec: Spec, model: Model, attrs: dict[str, str], steps: list[Step], path_id: str = "path", store=None):
    """Yields (situation, world) after the creation and after each step; the world is live and moves on at the next iteration.
    Any refusal is an error: a recorded path was accepted when it was made."""
    world = build_world(spec, attrs, model.machine, store)
    yield snapshot(world, spec, path_id, 0, steps), world
    for i, s in enumerate(steps, 1):
        if s.event is not None:
            if world.fire_event(spec.doc_id, s.action, s.event) is None:
                raise Unfireable(f"step {i} {s.action} by the host was rejected: {world.log[-1]['reason']}")
        else:
            try:
                world.act(spec.doc_id, s.actor, s.action, input=s.input)
            except Refused as r:
                raise Unfireable(f"step {i} {s.action} by {s.actor} was refused: {r.reason}") from r
        yield snapshot(world, spec, path_id, i, steps), world
