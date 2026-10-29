"""What a state says about who reaches the document, as an answer for one person.

Under a workflow a person keeps their own role on the document and opens
and finds the latest version, unless the state the document is in says
otherwise. A state's `access` is a list of rules; the first one that covers
the person is theirs. Opening and finding are two answers: `serve` is the
version they open, `find` the version a search finds for them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import who as whomod
from .machine import Serve
from ...support.model import Context, Person
from ...stores.state_history import Timeline


@dataclass(frozen=True)
class Access:
    """What the workflow gives one person here. `role` is `None` for their own, else
    `editor` / `contributor` / `viewer` / `none`. `serve` and `find` are version numbers,
    `None` for nothing."""

    role: str | None
    serve: int | None
    find: int | None
    guidance: str | None = None  # the covering rule's own guidance, if it has one


def resolve(timeline: Timeline, person: Person, ctx: Context | None = None) -> Access | None:
    """The first rule of the current state that speaks of `person`, or `None` when none does
    (then their own role and the latest version stand)."""
    # Who was assigned is the timeline's to say, never the caller's.
    ctx = replace(ctx or Context(), assigned=timeline.assigned(), choices=timeline.choices())
    choices = timeline.choices()
    for rule in timeline.machine.access_rules(timeline.state):
        if any(choices.get(name) not in values for name, values in rule.when.items()):
            continue
        if whomod.covers(rule.who, person, ctx, "access"):
            return Access(rule.role, _version(rule.serve, timeline), _version(rule.find, timeline), rule.guidance)
    return None


def readable_version(timeline: Timeline, person: Person, ctx: Context | None = None, *, for_search: bool = False) -> int | None:
    """The one version `person` reads (or, with `for_search`, is shown by a search), or `None` for none.

    Someone no rule speaks of reads the latest version if they have a role at all.
    """
    access = resolve(timeline, person, ctx)
    if access is None:
        if timeline.machine.default_access == "deny":
            return None  # nothing says they may: default deny
        return timeline.version if person.role is not None else None
    if access.role == "none":
        return None
    return access.find if for_search else access.serve


def _version(serve: Serve, timeline: Timeline) -> int | None:
    match serve:
        case ("latest",):
            return timeline.version
        case ("from", state):
            return timeline.version_at(state)
    return None
