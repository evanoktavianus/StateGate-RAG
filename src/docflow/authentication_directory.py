"""Authentication and directory: who is asking, and who the host knows.

A session token maps to exactly one person. The host issues it and hands it to that person; the person is fixed by the
token for the whole run and is passed to the tool controller by the host. A person is never a tool argument and
nothing in a question or a model's reply can change who is asking.

Only a hash of a token is kept, never the token. An unknown, malformed or revoked token is refused the same way.

The directory half answers the lookups the policy needs about people, units, groups and roles. It reads the people store
and hands out copies, so a caller cannot change a person.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import replace

from .stores.people_roles_units import User
from .stores.world import World

MAX_TOKEN_CHARS = 512


class Unauthenticated(Exception):
    """The token does not belong to a person. The same for an unknown, malformed or revoked token."""

    def __init__(self) -> None:
        super().__init__("unauthenticated")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthenticationDirectory:
    def __init__(self, world: World) -> None:
        self._world = world
        self._sessions: dict[str, str] = {}  # sha256(token) -> person

    def __repr__(self) -> str:
        return f"AuthenticationDirectory(sessions={len(self._sessions)})"

    # ── Authentication ─────────────────────────────────────────────────

    def issue(self, user: str) -> str:
        """A new token for `user`; shown once, here. Raises KeyError for someone the host does not know."""
        if user not in self._world.users:
            raise KeyError(user)
        token = "dfl_" + secrets.token_urlsafe(32)
        self._sessions[_hash(token)] = user
        return token

    def revoke(self, token: str) -> None:
        if isinstance(token, str):
            self._sessions.pop(_hash(token), None)

    def authenticate(self, token) -> str:
        """The person the token belongs to, or `Unauthenticated`."""
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_CHARS:
            raise Unauthenticated()
        user = self._sessions.get(_hash(token))
        if user is None or user not in self._world.users:  # a person the host has since removed has no session
            raise Unauthenticated()
        return user

    # ── Directory ──────────────────────────────────────────────────────

    def people(self) -> list[str]:
        return list(self._world.users)

    def person(self, user: str) -> User:
        return replace(self._world.users[user], attrs=dict(self._world.users[user].attrs))

    def unit_of(self, user: str) -> str | None:
        return self._world.users[user].unit

    def users_in_unit(self, unit: str) -> list[str]:
        return [u.id for u in self._world.users.values() if u.unit == unit]

    def groups_of(self, user: str) -> tuple[str, ...]:
        return self._world.users[user].groups

    def role_in(self, user: str, workspace: str) -> str | None:
        return self._world.role_in(user, workspace)
