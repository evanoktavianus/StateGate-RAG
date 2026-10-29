"""People, roles, and units: the users, their organization, unit, positions and attributes, and the roles they hold in workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    id: str
    org: str | None = None  # None: signed in, belongs to no organization here
    groups: tuple[str, ...] = ()  # lower case
    org_admin: bool = False
    unit: str | None = None
    positions: tuple[str, ...] = ()
    attrs: dict = field(default_factory=dict)  # any other attribute (clearance, region …): rules reach it with `attr` and `same`


@dataclass
class Workspace:
    id: str
    org: str | None
    members: dict[str, str] = field(default_factory=dict)  # user id -> role
    group_roles: dict[str, str] = field(default_factory=dict)  # group name (lower) -> role
