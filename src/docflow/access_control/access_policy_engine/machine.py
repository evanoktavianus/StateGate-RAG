"""A workflow: the states a document can be in and, for each state, the actions it accepts.

Pure data and pure functions, no database. `Machine.step` answers the one
question everything else is built on: in this state, when this action
happens, where does the document go? States and actions are strings: they
come from workflow files, which are data.

The file format is the workflow language of the paper; `workflow.schema.json` is its schema.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from . import dsl, requirements, who as whomod
from ...stores.workflow_definitions import _Loader
from ...support.model import Context, InvalidWorkflow, Person, Refused
from .who import Who

# `start` and `undo` are what the host records itself in a document's history.
RESERVED = ("start", "undo")
ACCESS_ROLES = ("editor", "contributor", "viewer", "none")

# A version an access rule gives: ("latest",), ("nothing",) or ("from", state).
Serve = tuple[str, ...]


@dataclass(frozen=True)
class AssignRule:
    from_: Who
    min: int = 1
    max: int | None = None
    except_: Who = None
    odd: bool = False


@dataclass(frozen=True)
class Transition:
    to: str
    by: Who = None
    requires: dict[str, Any] | None = None
    assign: dict[str, AssignRule] = field(default_factory=dict)
    choose: dict[str, tuple[str, ...]] = field(default_factory=dict)
    event: str | None = None  # fired only by this outside event (a recorded signature, a regulation coming into force), never by a person


@dataclass(frozen=True)
class AccessRule:
    who: Who
    role: str | None = None
    serve: Serve = ("latest",)
    find: Serve = ("latest",)
    when: dict[str, tuple[str, ...]] = field(default_factory=dict)  # choice or attribute -> the values that make the rule apply
    guidance: str | None = None  # what an agent is told for a person this rule covers (left out: the state's)


@dataclass(frozen=True)
class CreateRule:
    """How a document comes into being: who may create it and the people they must name (its roles)."""

    by: Who
    assign: dict[str, AssignRule] = field(default_factory=dict)
    set: tuple[str, ...] = ()  # attributes the creator must give, declared under `attributes`


@dataclass(frozen=True)
class Machine:
    name: str
    initial: str
    states: dict[str, dict[str, Transition]]
    access: dict[str, tuple[AccessRule, ...]] = field(default_factory=dict)
    guidance: dict[str, str] = field(default_factory=dict)  # state -> text
    default_access: str = "own"  # "deny": a person no rule covers gets nothing
    create: CreateRule | None = None
    attributes: dict[str, tuple[str, ...] | None] = field(default_factory=dict)  # declared attribute -> allowed values (None: any)

    # ── Loading ────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path) -> Machine:
        """Reads and checks a workflow file; raises `InvalidWorkflow` listing every problem."""
        try:
            raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_Loader)
        except (OSError, yaml.YAMLError) as error:
            raise InvalidWorkflow([f"{path}: could not be read ({error})"]) from error
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Any) -> Machine:
        states = raw.get("states") if isinstance(raw, dict) else None
        if not isinstance(states, dict) or not states:
            raise InvalidWorkflow(["a workflow needs at least one state under `states`"])

        problems: list[str] = []
        unbuildable: list[str] = []  # what stops the rest being read at all
        dsl.check_keys("the workflow", raw, "workflow", problems)
        for name, state in states.items():
            _check_name("a state's name", name, unbuildable)
            if state is not None and not isinstance(state, dict):
                unbuildable.append(f"{name}: a state holds settings (`on`, `access`), got {state!r}")
            elif state:
                dsl.check_keys(str(name), state, "state", problems)
        if unbuildable:
            raise InvalidWorkflow(problems + unbuildable)

        machine = cls(
            name=raw.get("name"),
            initial=raw.get("initial"),
            states={n: _parse_on(n, (s or {}).get("on"), RESERVED, problems) for n, s in states.items()},
            access={
                n: _parse_access(n, s["access"], problems)
                for n, s in states.items()
                if s and "access" in s
            },
            guidance={n: s["guidance"] for n, s in states.items() if s and s.get("guidance") is not None and _guidance_ok(s["guidance"])},
            default_access=raw.get("default_access", "own"),
            create=_parse_create(raw.get("create"), problems),
            attributes=_parse_attributes(raw.get("attributes"), problems),
        )
        if machine.default_access not in ("own", "deny"):
            problems.append(f"`default_access` is `own` or `deny`, got {machine.default_access!r}")
        for n, s in states.items():
            if s and "guidance" in s and not _guidance_ok(s["guidance"]):
                problems.append(f"{n}: `guidance` is text, got {s['guidance']!r}")
        problems += machine._problems()
        if problems:
            raise InvalidWorkflow(problems)
        return machine

    # ── Reading ────────────────────────────────────────────────────────

    def actions(self, state: str) -> list[str]:
        """The actions a state accepts."""
        return list(self.states.get(state, {}))

    def transition(self, state: str, action: str) -> Transition | None:
        """What `action` does in `state`, or `None` when the state doesn't accept it."""
        return self.states.get(state, {}).get(action)

    def access_rules(self, state: str) -> tuple[AccessRule, ...]:
        """The rules of who reaches the document while it is in `state`, in order."""
        return self.access.get(state, ())

    def guidance_for(self, state: str, choices: dict[str, str] | None = None, text: str | None = None) -> str | None:
        """What the host tells an agent about a document in `state` (or `text`, an access rule's own guidance, instead);
        `{name}` is replaced by that choice."""
        text = self.guidance.get(state) if text is None else text
        for name, value in (choices or {}).items():
            text = text and text.replace("{" + name + "}", value)
        return text

    def choices(self) -> dict[str, list[str]]:
        """Every choice an action of the workflow makes, with its options."""
        out: dict[str, list[str]] = {}
        for transition in self._transitions():
            for name, options in transition.choose.items():
                merged = out.setdefault(name, [])
                merged += [o for o in options if o not in merged]
        return out

    def next_states(self, state: str) -> list[str]:
        """Where a document in `state` can go by an action."""
        found = [t.to for _, t in sorted(self.states.get(state, {}).items())]
        return list(dict.fromkeys(s for s in found if s in self.states))

    # ── Running ────────────────────────────────────────────────────────

    def check(self, transition: Transition, ctx: Context) -> None:
        """Raises `Refused` unless `ctx.actor` may do `transition` now (or, for an event transition, the event matches)."""
        if transition.event is not None or ctx.event is not None:
            if transition.event != ctx.event:
                raise Refused("event_only", needs=transition.event)
        else:
            if transition.by is None and self.default_access == "deny":
                raise Refused("not_permitted", why="default deny: an action says who may do it")
            _permitted(transition.by, ctx)
        _assignable(transition.assign, ctx)
        _chosen(transition.choose, ctx)
        requirements.validate(transition.requires, ctx)
        for name in (transition.requires or {}).get("choices", ()):
            if name not in ctx.choices:
                raise Refused("choice_needed", name=name, options=self.choices().get(name, []))
        if not requirements.satisfied(transition.requires, ctx):
            raise Refused("requirement_unmet", type=transition.requires["type"])

    def check_create(self, ctx: Context) -> None:
        """Raises `Refused` unless the actor may create a document and has named the people its roles need."""
        if self.create is None:
            raise Refused("not_accepted", action="create", state=None)
        _permitted(self.create.by, ctx)
        _assignable(self.create.assign, ctx)
        for name in self.create.set:
            allowed = self.attributes.get(name)
            if ctx.input.get(name) is None or (allowed is not None and ctx.input[name] not in allowed):
                raise Refused("attribute_needed", name=name, options=list(allowed or ()))

    def step(self, state: str, action: str, ctx: Context | None = None) -> str:
        """Where `action` takes a document in `state`; raises `Refused` when it can't."""
        if state not in self.states:
            raise Refused("unknown_state", state=state)
        transition = self.states[state].get(action)
        if transition is None:
            raise Refused("not_accepted", action=action, state=state)
        self.check(transition, ctx or Context())
        return transition.to

    # ── Checks ─────────────────────────────────────────────────────────

    def _transitions(self) -> list[Transition]:
        return [t for actions in self.states.values() for t in actions.values()]

    def _problems(self) -> list[str]:
        out: list[str] = []
        if not isinstance(self.name, str):
            out.append("a workflow needs a `name`")
        if self.initial not in self.states:
            out.append(f"`initial` must name one of the states, got {self.initial!r}")

        out += [f"{state}: `upload` must lead to {state!r}, not {t.to!r} (an upload adds a version and never moves the document)"
                for state, actions in self.states.items() for action, t in actions.items() if action == "upload" and t.to != state]

        assigned = {name for t in self._transitions() for name in t.assign} | set(self.create.assign if self.create else ())

        known = set(self.attributes) | set(self.choices()) | {"unit"}

        def check_who(where: str, who: Who) -> None:
            for term in who or ():
                if term.kind == "all":
                    check_who(where, term.parts)
                if term.kind == "same":
                    for _, (subject,) in term.pairs:
                        if subject not in known:
                            out.append(f"{where}: `same` names {subject!r}, which is not a declared attribute or choice")
                if term.kind == "assigned" and term.values[0] not in assigned:
                    out.append(f"{where}: no action assigns {term.values[0]!r}")

        def check_transition(where: str, t: Transition) -> None:
            if t.to not in self.states:
                out.append(f"{where}: `to` must name one of the states, got {t.to!r}")
            check_who(f"{where}: `by`", t.by)
            for name, rule in t.assign.items():
                check_who(f"{where}: `assign` {name}: `from`", rule.from_)
                check_who(f"{where}: `assign` {name}: `except`", rule.except_)

        for state, actions in self.states.items():
            for action, t in actions.items():
                check_transition(f"{state} → {action}", t)

        if self.create:
            for name in self.create.set:
                if name not in self.attributes:
                    out.append(f"`create`: `set` names {name!r}, which `attributes` does not declare")
            check_who("`create`: `by`", self.create.by)
            for name, rule in self.create.assign.items():
                check_who(f"`create`: `assign` {name}: `from`", rule.from_)

        choices = self.choices()
        for state, rules in self.access.items():
            for rule in rules:
                where = f"{state}: `access`"
                check_who(f"{where}: `who`", rule.who)
                for key, serve in (("serve", rule.serve), ("find", rule.find)):
                    if serve[0] == "from" and serve[1] not in self.states:
                        out.append(f"{where}: `{key}` names a state that isn't there, {serve[1]!r}")
                for name, values in rule.when.items():
                    allowed = choices.get(name, self.attributes.get(name))
                    if name not in choices and name not in self.attributes:
                        out.append(f"{where}: `when` names {name!r}, which no action chooses (and `attributes` does not declare it)")
                    elif allowed is not None:
                        for value in values:
                            if value not in allowed:
                                out.append(
                                    f"{where}: `when` gives {name} the value {value!r}, "
                                    f"which no action offers ({', '.join(allowed)})"
                                )

        if self.initial in self.states:
            reachable = set(_reach(self.initial, self.next_states))
            out += [f"state {s} can never be reached from {self.initial}" for s in self.states if s not in reachable]
        return out


# ── Parsing ────────────────────────────────────────────────────────────



def _check_name(what: str, name: Any, problems: list[str]) -> None:
    if not isinstance(name, str):
        problems.append(f"{what} must be text, got {name!r} (put it in quotes)")
    elif not name.strip():
        problems.append(f"{what} can't be empty")
    elif "/" in name:
        problems.append(f"{what} can't contain `/`, got {name!r}")


def _parse_on(where: str, on: Any, reserved: tuple[str, ...], problems: list[str]) -> dict[str, Transition]:
    if on is None:
        return {}
    if not isinstance(on, dict):
        problems.append(f"{where}: `on` lists actions by name, got {on!r}")
        return {}
    out: dict[str, Transition] = {}
    for action, raw in on.items():
        _check_name(f"{where}: an action's name", action, problems)
        if action in reserved:
            problems.append(f"{where}: `{action}` is a name the host uses itself; call the action something else")
        if not isinstance(raw, dict):
            problems.append(f"{where} → {action}: an action needs `to: …`, got {raw!r}")
            continue
        out[str(action)] = _parse_transition(f"{where} → {action}", raw, problems)
    return out


def _parse_transition(where: str, raw: dict, problems: list[str]) -> Transition:
    dsl.check_keys(where, raw, "transition", problems)
    to = raw.get("to")
    if not isinstance(to, str):
        problems.append(f"{where}: `to` must name one of the states, got {to!r}")
        to = ""

    requires = raw.get("requires")
    if requires is not None:
        if not isinstance(requires, dict) or "type" not in requires:
            problems.append(f"{where}: `requires` needs a `type`, got {requires!r}")
            requires = None
        elif requires["type"] not in requirements.types():
            problems.append(f"{where}: unknown requirement type {requires['type']!r}")
            requires = None
        else:
            allowed = getattr(requirements.types()[requires["type"]], "settings", None)
            for key in requires if allowed else ():
                if key != "type" and key not in allowed:
                    problems.append(f"{where}: `requires` {requires['type']} has no {key!r} (the settings are {', '.join(allowed)})")
            problems += [f"{where}: {p}" for p in requirements.problems(requires)]

    return Transition(
        to=to,
        by=whomod.parse(raw.get("by"), f"{where}: `by`", problems),
        requires=requires,
        assign=_parse_assign(where, raw.get("assign"), problems),
        choose=_parse_choose(where, raw.get("choose"), problems),
        event=raw.get("event") if isinstance(raw.get("event"), str) else None,
    )


def _guidance_ok(g: Any) -> bool:
    return g is None or isinstance(g, str)


def _parse_create(raw: Any, problems: list[str]) -> CreateRule | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        problems.append(f"`create` is `by:` and `assign:`, got {raw!r}")
        return None
    dsl.check_keys("`create`", raw, "create", problems)
    names = raw.get("set", [])
    if not (isinstance(names, list) and all(isinstance(n, str) for n in names)):
        problems.append(f"`create`: `set` lists attribute names, got {names!r}")
        names = []
    return CreateRule(whomod.parse(raw.get("by"), "`create`: `by`", problems), _parse_assign("`create`", raw.get("assign"), problems), tuple(names))


def _parse_attributes(raw: Any, problems: list[str]) -> dict[str, tuple[str, ...] | None]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        problems.append(f"`attributes` names the attributes a document carries, got {raw!r}")
        return {}
    out: dict[str, tuple[str, ...] | None] = {}
    for name, spec in raw.items():
        if spec is None or spec == {}:
            out[str(name)] = None
        elif isinstance(spec, dict) and isinstance(spec.get("values"), list) and spec["values"]:
            dsl.check_keys(f"`attributes` {name}", spec, "attribute", problems)
            out[str(name)] = tuple(str(v) for v in spec["values"])
        else:
            problems.append(f"`attributes` {name}: leave it empty (any value) or give `values: [..]`, got {spec!r}")
    return out


def _parse_assign(where: str, raw: Any, problems: list[str]) -> dict[str, AssignRule]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        problems.append(f"{where}: `assign` needs names with `from: …`, got {raw!r}")
        return {}
    out: dict[str, AssignRule] = {}
    for name, rule in raw.items():
        at = f"{where}: `assign` {name}"
        if not isinstance(rule, dict) or rule.get("from") is None:
            problems.append(f"{at} needs `from: …`")
            continue
        dsl.check_keys(at, rule, "assign", problems)
        low, high = rule.get("min", 1), rule.get("max")
        if not isinstance(low, int) or isinstance(low, bool) or low < 0:
            problems.append(f"{at}: `min` must be a number")
            low = 1
        if high is not None:
            if not isinstance(high, int) or isinstance(high, bool):
                problems.append(f"{at}: `max` must be a number")
                high = None
            elif high < low:
                problems.append(f"{at}: `max` is less than `min`")
        out[str(name)] = AssignRule(
            from_=whomod.parse(rule["from"], f"{at}: `from`", problems),
            min=low,
            max=high,
            except_=whomod.parse(rule.get("except"), f"{at}: `except`", problems),
            odd=rule.get("odd") is True,
        )
    return out


def _parse_choose(where: str, raw: Any, problems: list[str]) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        problems.append(f"{where}: `choose` is names with their options (`audience: [unit, public]`), got {raw!r}")
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name, options in raw.items():
        if isinstance(options, list) and options and all(isinstance(o, (str, int)) and not isinstance(o, bool) for o in options):
            out[str(name)] = tuple(str(o) for o in options)
        else:
            problems.append(
                f"{where}: `choose` gives each name a list of options (`audience: [unit, public]`), got {name!r}: {options!r}"
            )
    return out


def _parse_access(state: str, raw: Any, problems: list[str]) -> tuple[AccessRule, ...]:
    where = f"{state}: `access`"
    if not isinstance(raw, list):
        problems.append(f"{where} is a list of rules, each with `who` (and `role`, `serve`, `find`), got {raw!r}")
        return ()
    rules: list[AccessRule] = []
    for rule in raw:
        if not isinstance(rule, dict) or "who" not in rule:
            problems.append(f"{where} is a list of rules, each with `who` (and `role`, `serve`, `find`), got {rule!r}")
            continue
        for key in rule.keys() - {"who", "role", "serve", "find", "when", "guidance"}:
            problems.append(f"{where}: there is no {key!r} (`who`, `role`, `serve`, `find`, `when`, `guidance`)")
        if "guidance" in rule and not _guidance_ok(rule["guidance"]):
            problems.append(f"{where}: `guidance` is text, got {rule['guidance']!r}")
        role = rule.get("role")
        if role is not None and role not in ACCESS_ROLES:
            problems.append(f"{where}: `role` is `editor`, `contributor`, `viewer` or `none`, got {role!r}")
        serve = _parse_serve("serve", rule.get("serve", "latest"), where, problems)
        find = _parse_serve("find", rule["find"], where, problems) if "find" in rule else serve
        when = rule.get("when")
        if when is not None and not isinstance(when, dict):
            problems.append(f"{where}: `when` is a choice or attribute and its value (`when: {{ audience: unit }}`), got {when!r}")
            when = None
        rules.append(
            AccessRule(
                who=whomod.parse(rule["who"], f"{where}: `who`", problems),
                role=role,
                serve=serve,
                find=find,
                when={str(k): tuple(str(x) for x in (v if isinstance(v, list) else [v])) for k, v in (when or {}).items()},
                guidance=rule["guidance"] if isinstance(rule.get("guidance"), str) else None,
            )
        )
    return tuple(rules)


def _parse_serve(key: str, raw: Any, where: str, problems: list[str]) -> Serve:
    if raw in ("latest", "nothing"):
        return (raw,)
    if isinstance(raw, dict) and isinstance(raw.get("from"), str):
        return ("from", raw["from"])
    problems.append(f"{where}: `{key}` is `latest`, `nothing` or `from: <state>`, got {raw!r}")
    return ("latest",)


# ── Running ────────────────────────────────────────────────────────────


def _permitted(by: Who, ctx: Context) -> None:
    if by is None:
        return
    if ctx.actor is None or not whomod.covers(by, ctx.actor, ctx):
        raise Refused("not_permitted")


def candidates(rule: AssignRule, members: tuple[Person, ...] | list[Person], ctx: Context) -> list[Person]:
    """The people an assign rule lets the actor choose from."""
    return [
        p
        for p in members
        if whomod.covers(rule.from_, p, ctx) and not (rule.except_ is not None and whomod.covers(rule.except_, p, ctx))
    ]


def _assignable(assign: dict[str, AssignRule], ctx: Context) -> None:
    for name, rule in sorted(assign.items()):
        given = ctx.input.get(name, [])
        if not isinstance(given, (list, tuple)):
            raise Refused("assignment_invalid", name=name)
        chosen = list(dict.fromkeys(given))
        allowed = {p.id for p in candidates(rule, ctx.members, ctx)}
        if len(chosen) < rule.min:
            raise Refused("assignment_needed", name=name, min=rule.min)
        if (rule.max is not None and len(chosen) > rule.max) or not set(chosen) <= allowed:
            raise Refused("assignment_invalid", name=name)
        if rule.odd and len(chosen) % 2 == 0:
            raise Refused("assignment_invalid", name=name, needs="an odd number of people")


def _chosen(choose: dict[str, tuple[str, ...]], ctx: Context) -> None:
    for name, options in sorted(choose.items()):
        if ctx.input.get(name) not in options:
            raise Refused("choice_needed", name=name, options=list(options))


def _reach(start: str, nxt: Callable[[str], list[str]]) -> list[str]:
    seen, queue = {start}, deque([start])
    while queue:
        for to in nxt(queue.popleft()):
            if to not in seen:
                seen.add(to)
                queue.append(to)
    return list(seen)


def _shortest(start: str, goal: Callable[[str], bool], nxt: Callable[[str], list[str]]) -> list[str]:
    seen, queue = {start}, deque([[start]])
    while queue:
        path = queue.popleft()
        if goal(path[-1]):
            return path
        for to in nxt(path[-1]):
            if to not in seen:
                seen.add(to)
                queue.append([*path, to])
    return []


