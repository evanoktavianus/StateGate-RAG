"""The append-only history of one document under a workflow.

Nothing is ever changed or removed; every action that is accepted adds
entries (tuples, oldest first):

    ("version", n)            a version was uploaded
    ("state", name)           the document moved to a state
    ("given", action, who)    who did their part of an action whose requirement needs more people
    ("assigned", name, ids)   people were chosen under a name; they stay chosen until chosen again
    ("chose", name, value)    an action made a choice (an audience); it stays until chosen again

Where the document stands is always read off the history: its state is the
last `state` entry and its version the highest `version`. The version it had
when it last arrived in a state is `version_at(state)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..access_control.access_policy_engine import requirements, who as whomod
from ..access_control.access_policy_engine.machine import Machine, Transition
from ..support.model import Context, Refused

Entry = tuple

UPLOAD = "upload"


@dataclass(frozen=True)
class Timeline:
    machine: Machine
    entries: tuple[Entry, ...]

    @classmethod
    def new(cls, machine: Machine, version: int = 1, assigned: dict[str, list] | None = None, attributes: dict[str, str] | None = None) -> Timeline:
        """A document entering the workflow at `version`, in the initial state; `assigned` are the people named
        when it was created and `attributes` the values given then (a secrecy level, a requesting unit)."""
        named = tuple(("assigned", n, tuple(ids)) for n, ids in sorted((assigned or {}).items()))
        given = tuple(("chose", n, v) for n, v in sorted((attributes or {}).items()))
        return cls(machine, (("version", version), *named, *given, ("state", machine.initial)))

    # ── Acting ─────────────────────────────────────────────────────────

    def act(self, action: str, ctx: Context | None = None) -> Timeline:
        """Applies an action and returns the timeline with what it appended.

        Raises `Refused`, adding nothing. An action whose requirement isn't
        met yet is not refused: the actor's part is recorded and the
        document stays where it is. Doing it a second time is refused with
        `already_given`.
        """
        ctx = ctx or Context()
        earlier = self.given(action)
        who = ctx.actor.id if ctx.actor else None
        ctx = replace(
            ctx,
            given=len(({who} | set(earlier)) - {None}),
            assigned=self.assigned(),
            choices=self.choices(),
        )
        transition = self.transition(action)
        if transition is not None and transition.requires is not None:
            ctx = replace(ctx, asked=sum(1 for m in ctx.members if whomod.covers(transition.by, m, ctx)))

        try:
            to = self.machine.step(self.state, action, ctx)
        except Refused as refusal:
            if refusal.reason == "requirement_unmet" and requirements.counts_people(transition.requires):
                return self._give(action, who, earlier, refusal, transition, ctx)
            raise
        return (
            self._uploaded(action, ctx)
            ._evidenced(action, who, transition, ctx)
            ._chosen(transition, ctx)
            ._moved(self.state, to)
        )

    def _give(self, action: str, who: str | None, earlier: list, refusal: Refused, transition: Transition, ctx: Context) -> Timeline:
        if who is None:
            raise refusal
        if who in earlier:
            raise Refused("already_given", action=action)
        return self._evidenced(action, who, transition, ctx)._append(("given", action, who))

    def _evidenced(self, action: str, who: str | None, transition: Transition | None, ctx: Context) -> Timeline:
        """Keeps what a requirement keeps about this person's part (a signature's reference and digest), tied to the version."""
        ev = requirements.evidence(transition.requires, ctx) if transition and transition.requires else None
        return self if ev is None else self._append(("evidence", action, who, {**ev, "version": self.version}))

    def _uploaded(self, action: str, ctx: Context) -> Timeline:
        if action != UPLOAD:
            return self
        return self._append(("version", ctx.version if ctx.version is not None else self.version + 1))

    def _chosen(self, transition: Transition, ctx: Context) -> Timeline:
        """What the action was given is recorded before the move it causes."""
        out = self
        for name in sorted(transition.assign):
            out = out._append(("assigned", name, tuple(dict.fromkeys(ctx.input.get(name, [])))))
        for name in sorted(transition.choose):
            out = out._append(("chose", name, ctx.input[name]))
        return out

    def _moved(self, frm: str, to: str) -> Timeline:
        return self if frm == to else self._append(("state", to))

    def _append(self, entry: Entry) -> Timeline:
        return replace(self, entries=(*self.entries, entry))

    # ── Reading ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """The state the document is in."""
        return self._last("state")

    @property
    def version(self) -> int:
        """The newest version (the highest, not the last recorded)."""
        return max(e[1] for e in self.entries if e[0] == "version")

    def assigned(self) -> dict[str, tuple]:
        """Who is chosen under each name right now: the latest choice made for it."""
        return {e[1]: tuple(e[2]) for e in self.entries if e[0] == "assigned"}

    def assigned_everyone(self) -> set:
        """Everyone ever chosen under any name on this document."""
        return {p for ids in self.assigned().values() for p in ids}

    def choices(self) -> dict[str, str]:
        """What was chosen under each name: the latest choice made for it."""
        return {e[1]: e[2] for e in self.entries if e[0] == "chose"}

    def given(self, action: str) -> list:
        """Who has done their part of `action` on the current version, in the current state.

        The run of `given` entries at the end of the history; anything else starts it again.
        """
        run = []
        for entry in reversed(self.entries):
            if entry[0] not in ("given", "evidence"):
                break
            run.append(entry)
        return [e[2] for e in run if e[0] == "given" and e[1] == action]

    def evidence(self, action: str) -> list[dict]:
        """What was kept about each person's part of `action` on the current version: who, and what they gave."""
        run = []
        for entry in reversed(self.entries):
            if entry[0] not in ("given", "evidence"):
                break
            run.append(entry)
        return [{"who": e[2], **e[3]} for e in reversed(run) if e[0] == "evidence" and e[1] == action]

    def transition(self, action: str) -> Transition | None:
        """What `action` does where the document stands, or `None`."""
        return self.machine.transition(self.state, action)

    def version_at(self, state: str) -> int | None:
        """The version the document had when it last arrived in `state`, or `None` if it never has."""
        newest = at = None
        for e in self.entries:
            if e[0] == "version":
                newest = e[1] if newest is None else max(newest, e[1])
            elif e == ("state", state):
                at = newest
        return at

    def _last(self, kind: str):
        for e in reversed(self.entries):
            if e[0] == kind:
                return e[1]
        return None
