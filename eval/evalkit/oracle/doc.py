"""One document under a policy: its history, and the answers the policy gives for it. Written from the language reference, not from the engine.

The history is kept as entries, oldest first, as the reference describes them: ("version", n), ("state", name), ("assigned", name, ids),
("chose", name, value), ("given", action, who), ("evidence", action, who). Where the document stands is read off them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import who as W
from .policy import Policy

ORDER = ("admin", "editor", "contributor", "viewer")


@dataclass(frozen=True)
class Person:
    id: str
    org: str | None = None
    unit: str | None = None
    groups: tuple[str, ...] = ()
    positions: tuple[str, ...] = ()
    attrs: dict = field(default_factory=dict, hash=False, compare=False)
    org_admin: bool = False


class Cast(dict):
    """id -> Person, in the order of the fixture."""

    @classmethod
    def from_people(cls, people: dict[str, dict]) -> "Cast":
        return cls({pid: Person(pid, p.get("org"), p.get("unit"), tuple(g.lower() for g in p.get("groups", ())),
                                tuple(x.lower() for x in p.get("positions", ())), dict(p.get("attrs") or {}), bool(p.get("org_admin")))
                    for pid, p in people.items()})


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None


YES = Decision(True)


def no(reason: str) -> Decision:
    return Decision(False, reason)


class Doc:
    def __init__(self, policy: Policy, cast: Cast, workspace: dict, workspace_org: str | None, doc_id: str, owner: str, invited=()) -> None:
        self.policy, self.cast, self.workspace, self.workspace_org = policy, cast, workspace, workspace_org
        self.id, self.owner, self.invited = doc_id, owner, set(invited)
        self.unit = cast[owner].unit
        self.entries: list[tuple] = []

    # ── creating ──
    @classmethod
    def create(cls, policy, cast, workspace, workspace_org, doc_id, owner, creator=None, assign=None, attrs=None, invited=()) -> "Doc | Decision":
        doc = cls(policy, cast, workspace, workspace_org, doc_id, owner, invited)
        rule = policy.create
        assign, attrs = dict(assign or {}), dict(attrs or {})
        if rule is None:
            if assign or attrs:
                return no("not_accepted")
        else:
            who = cast[creator or owner]
            if not W.covers(rule.get("by"), who, doc, "act", policy.default_access):
                return no("not_permitted")
            bad = doc.check_assign(rule.get("assign") or {}, assign)
            if bad:
                return bad
            for name in rule.get("set") or []:
                allowed = (policy.attributes.get(name) or {}).get("values")
                if attrs.get(name) is None or (allowed is not None and attrs[name] not in allowed):
                    return no("attribute_needed")
        doc.entries = [("version", 1), *(("assigned", n, tuple(ids)) for n, ids in sorted(assign.items())),
                       *(("chose", n, v) for n, v in sorted(attrs.items())), ("state", policy.initial)]
        return doc

    # ── reading the history ──
    @property
    def state(self) -> str:
        return next(e[1] for e in reversed(self.entries) if e[0] == "state")

    @property
    def version(self) -> int:
        return max(e[1] for e in self.entries if e[0] == "version")

    @property
    def assigned(self) -> dict[str, tuple]:
        return {e[1]: tuple(e[2]) for e in self.entries if e[0] == "assigned"}

    @property
    def choices(self) -> dict[str, str]:
        return {e[1]: e[2] for e in self.entries if e[0] == "chose"}

    def arrived(self, state: str) -> int | None:
        """The version the document had when it last arrived in `state`."""
        newest = at = None
        for e in self.entries:
            if e[0] == "version":
                newest = e[1] if newest is None else max(newest, e[1])
            elif e == ("state", state):
                at = newest
        return at

    def given(self, action: str) -> list[str]:
        """Who has done their part of `action` since anything else was last recorded."""
        out = []
        for e in reversed(self.entries):
            if e[0] not in ("given", "evidence"):
                break
            if e[0] == "given" and e[1] == action:
                out.append(e[2])
        return out

    # ── what a person is given ──
    def access(self, person_id: str, find: bool = False) -> int | None:
        """The version the person opens (or, with `find`, is shown by a search); None for none."""
        rule = self._rule(person_id)
        if rule is not None:
            if rule.get("role") == "none":
                return None
            sel = rule.get("find", rule.get("serve", "latest")) if find else rule.get("serve", "latest")
            return self._version(sel)
        person = self.cast[person_id]
        if self.policy.default_access == "deny":
            return None
        return self.version if W.person_role(person, self) is not None else None

    def _rule(self, person_id: str) -> dict | None:
        """The first rule of the state that applies now and covers the person, or None."""
        person = self.cast[person_id]
        for rule in self.policy.access_rules(self.state):
            if any(str(self.choices.get(n)) not in [str(x) for x in (v if isinstance(v, list) else [v])] for n, v in (rule.get("when") or {}).items()):
                continue
            if W.covers(rule["who"], person, self, "access", self.policy.default_access):
                return rule
        return None

    def _version(self, sel: Any) -> int | None:
        if sel == "latest":
            return self.version
        if sel == "nothing":
            return None
        return self.arrived(sel["from"])

    def guidance(self, person_id: str | None = None) -> str | None:
        """The text of the rule that covers the person if that rule has one, else the state's; `{name}` filled in from the choices."""
        rule = self._rule(person_id) if person_id is not None else None
        text = rule["guidance"] if rule and isinstance(rule.get("guidance"), str) else self.policy.guidance(self.state)
        for name, value in self.choices.items():
            text = text and text.replace("{" + name + "}", str(value))
        return text

    # ── actions ──
    def available(self) -> list[str]:
        """The actions the state accepts now."""
        return list(self.policy.actions(self.state))

    def _transition(self, action: str) -> dict | None:
        return self.policy.actions(self.state).get(action)

    def check_assign(self, rules: dict, given: dict) -> Decision | None:
        for name, rule in sorted(rules.items()):
            ids = list(dict.fromkeys(given.get(name, [])))
            lo, hi = rule.get("min", 1), rule.get("max")
            if len(ids) < lo:
                return no("assignment_needed")
            pool = {p.id for p in self.cast.values()
                    if W.covers(rule["from"], p, self, "act", self.policy.default_access)
                    and not (rule.get("except") is not None and W.covers(rule["except"], p, self, "act", self.policy.default_access))}
            if (hi is not None and len(ids) > hi) or not set(ids) <= pool:
                return no("assignment_invalid")
            if rule.get("odd") and len(ids) % 2 == 0:
                return no("assignment_invalid")
        return None

    def decide(self, person_id: str, action: str, inp: dict | None = None, event: str | None = None) -> Decision:
        """Whether the system must accept `action` by this person now (or, with `event`, fired by the host). Changes nothing."""
        inp = inp or {}
        person, p = self.cast[person_id] if person_id else None, self.policy
        if event is None and p.default_access != "deny":
            standing = (W.person_role(person, self) is not None or person.id in self.invited or person.id == self.owner
                        or any(person.id in ids for ids in self.assigned.values()))
            if not standing:
                return no("not_permitted")
        t = self._transition(action)
        if t is None:
            return no("not_accepted")
        if t.get("event") is not None or event is not None:
            if t.get("event") != event:
                return no("event_only")
        else:
            by = t.get("by")
            if by is None and p.default_access == "deny":
                return no("not_permitted")
            if not W.covers(by, person, self, "act", p.default_access):
                return no("not_permitted")
        bad = self.check_assign(t.get("assign") or {}, inp)
        if bad:
            return bad
        for name, options in (t.get("choose") or {}).items():
            if str(inp.get(name)) not in [str(o) for o in options]:
                return no("choice_needed")
        req = t.get("requires")
        if req:
            if req["type"] == "signature":
                sig = inp.get("signature")
                if not (isinstance(sig, dict) and sig.get("ref") and isinstance(sig.get("digest"), str)):
                    return no("signature_needed")
                if sig["digest"] != f"{self.id}@{self.version}":  # the host's digest of a version is "<document>@<version>"
                    return no("signature_invalid")
            for name in req.get("choices") or []:
                if name not in self.choices and name not in inp:
                    return no("choice_needed")
            earlier = self.given(action)
            count = len(set(earlier) | ({person_id} if person_id else set()))
            if count < self._needed(req, t):
                return no("already_given") if person_id in earlier else YES
        return YES

    def _needed(self, req: dict, t: dict) -> int:
        n = req.get("pass", "any")
        if n == "any":
            return 1
        if n == "all":
            return sum(1 for p in self.cast.values() if W.covers(t.get("by"), p, self, "act", self.policy.default_access))
        return int(n)

    def apply(self, person_id: str | None, action: str, inp: dict | None = None, event: str | None = None) -> Decision:
        """Decides, and when the action is accepted, records what it does."""
        inp = inp or {}
        d = self.decide(person_id, action, inp, event)
        if not d.allowed:
            return d
        t = self._transition(action)
        req = t.get("requires")
        if req:
            earlier = self.given(action)
            if len(set(earlier) | {person_id}) < self._needed(req, t):  # their part is recorded; the action waits for more people
                if req["type"] == "signature":
                    self.entries.append(("evidence", action, person_id))
                self.entries.append(("given", action, person_id))
                return d
            if req["type"] == "signature":
                self.entries.append(("evidence", action, person_id))
        if action == "upload":
            self.entries.append(("version", self.version + 1))
        for name in sorted(t.get("assign") or {}):
            self.entries.append(("assigned", name, tuple(dict.fromkeys(inp.get(name, [])))))
        for name in sorted(t.get("choose") or {}):
            self.entries.append(("chose", name, inp[name]))
        if t["to"] != self.state:
            self.entries.append(("state", t["to"]))
        return d
