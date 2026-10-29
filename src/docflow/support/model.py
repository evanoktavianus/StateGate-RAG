"""The small shared vocabulary: who someone is, the moment an action happens, and why one is refused."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLES = ("admin", "editor", "contributor", "viewer")


@dataclass(frozen=True)
class Person:
    """Someone a `who` is matched against.

    `role` is the role they have on the document's scope (workspace, project…),
    their own or through a group; `None` means none at all. `groups` are the
    names, lower case, of the groups they are in.
    """

    id: str
    email: str | None = None
    role: str | None = None
    groups: tuple[str, ...] = ()
    org_admin: bool = False
    in_org: bool = False
    unit: str | None = None
    positions: tuple[str, ...] = ()  # a place in the organization (leadership, staff …), not a workspace role
    attrs: dict = field(default_factory=dict)  # any other attribute of the person, by name (clearance, region …)


@dataclass(frozen=True)
class Context:
    """The moment an action happens.

    `assigned` (who was chosen so far, by name) is filled in by the timeline.
    `input` is what the actor gives: people chosen (`{"reviewers": [id, …]}`)
    and choices made (`{"audience": "public"}`). `members` are the people who
    could take part, for `assign … from` and for `pass: all`.
    """

    actor: Person | None = None
    author_id: str | None = None
    assigned: dict[str, tuple[str, ...]] = field(default_factory=dict)
    input: dict[str, Any] = field(default_factory=dict)
    members: tuple[Person, ...] = ()
    invited: frozenset[str] = frozenset()
    given: int = 0
    asked: int = 1
    version: int | None = None
    unit: str | None = None  # the unit the document belongs to (its author's)
    choices: dict[str, str] = field(default_factory=dict)  # choices made so far, by name
    digest: str | None = None  # the host's digest of the version being acted on (what a signature is bound to)
    event: str | None = None  # set when an outside event, not a person, fires the action


class Refused(Exception):
    """An action that isn't allowed. `reason` says why; `detail` has the specifics.

    Reasons: unknown_state, not_accepted, not_permitted, assignment_needed,
    assignment_invalid, choice_needed, requirement_unmet, already_given.
    """

    def __init__(self, reason: str, **detail: Any) -> None:
        super().__init__(reason, detail)
        self.reason = reason
        self.detail = detail

    def __str__(self) -> str:
        extra = ", ".join(f"{k}={v!r}" for k, v in self.detail.items())
        return f"{self.reason}({extra})" if extra else self.reason


class InvalidWorkflow(Exception):
    """A workflow file that doesn't make sense; every problem is listed."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
