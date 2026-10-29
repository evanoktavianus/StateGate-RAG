"""A small world to test workflows in: organizations, people, workspaces, documents under workflows.

Everything the engine leaves to the host lives here, in memory: who has which role in which
workspace, who a document was opened to, and which workflow each document is under. Documents
can be added at any time, each under any workflow in the registry.

    world = World(workflows={"policy": Machine.load(...), "report": Machine.load(...)})
    world.add_org("O1"); world.add_user("u1", org="O1"); world.add_workspace("W1", org="O1", members={"u1": "contributor"})
    world.add_document("d1", workspace="W1", owner="u1", workflow="policy")
    world.act("d1", "u1", "submit", input={"reviewer": ["qa1"]})
    world.readable_version("d1", "someone")
"""

from __future__ import annotations

import functools
from dataclasses import replace

from ..access_control.access_policy_engine.access import readable_version, resolve
from ..access_control.access_policy_engine.machine import Machine
from ..support.model import Context, Person, Refused
from .documents import Document
from .people_roles_units import User, Workspace
from .state_history import Timeline


def _writes(method):
    """After the outermost call of an action that changes the world, give the store the chance to keep it (a `WriteThrough`; see
    world_store.py). A call inside another (an outside event, which acts on the document) is part of
    the outer one. If the call raised and left the world half changed, the store puts it back as the repository holds it."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self._depth += 1
        try:
            result = method(self, *args, **kwargs)
        except BaseException:
            self._depth -= 1
            if self._depth == 0 and self.store is not None:
                self.store.recover(self)
            raise
        self._depth -= 1
        if self._depth == 0 and self.store is not None:
            self.store.flush(self)
        return result

    return wrapper


class World:
    def __init__(self, workflows: dict[str, Machine], store=None) -> None:
        self.store, self._depth = store, 0  # a store (world_store.WriteThrough) keeps the repository up to date after every accepted action
        self.workflows = workflows
        self.orgs: set[str] = set()
        self.users: dict[str, User] = {}
        self.workspaces: dict[str, Workspace] = {}
        self.documents: dict[str, Document] = {}
        self.digest = lambda doc_id, version: f"{doc_id}@{version}"  # the host's digest of a version: what a signature binds to
        self.log: list[dict] = []  # every state change an event causes and every rejected event: document, old and new state, trigger, actor

    # ── Building the world ─────────────────────────────────────────────

    @_writes
    def add_org(self, org: str) -> None:
        self.orgs.add(org)

    @_writes
    def add_user(self, id: str, org: str | None = None, groups=(), org_admin: bool = False, unit: str | None = None, positions=(), attrs: dict | None = None) -> User:
        user = User(id, org, tuple(g.lower() for g in groups), org_admin, unit, tuple(p.lower() for p in positions), dict(attrs or {}))
        self.users[id] = user
        return user

    @_writes
    def add_workspace(self, id: str, org: str | None, members: dict[str, str] | None = None, group_roles: dict[str, str] | None = None) -> Workspace:
        ws = Workspace(id, org, dict(members or {}), {k.lower(): v for k, v in (group_roles or {}).items()})
        self.workspaces[id] = ws
        return ws

    @_writes
    def add_document(self, id: str, workspace: str, owner: str, workflow: str, version: int = 1, invited=(),
                     creator: str | None = None, assign: dict | None = None, attrs: dict | None = None) -> Document:
        """Puts a new document under a workflow, in its initial state at `version`. A workflow with `create`
        says who may create one and which people (`assign`) and attributes (`attrs`) they must give."""
        machine = self.workflows[workflow]
        if machine.create is not None:
            ctx = Context(actor=self.person(creator or owner, workspace), author_id=owner, unit=self.users[owner].unit,
                          input={**(assign or {}), **(attrs or {})}, members=self.members(workspace))
            machine.check_create(ctx)
        elif assign or attrs:
            raise Refused("not_accepted", why=f"{workflow} has no `create`: it takes no named people or attributes")
        doc = Document(id, workspace, owner, workflow, Timeline.new(machine, version, assign, attrs), set(invited))
        self.documents[id] = doc
        return doc

    # ── People as the engine sees them ─────────────────────────────────

    def role_in(self, user: str, workspace: str) -> str | None:
        """The strongest role a user has in a workspace: their own, or through a group."""
        ws, u = self.workspaces[workspace], self.users[user]
        order = ("admin", "editor", "contributor", "viewer")
        roles = [r for r in (ws.members.get(user), *(ws.group_roles.get(g) for g in u.groups)) if r]
        return min(roles, key=order.index) if roles else None

    def person(self, user: str, workspace: str) -> Person:
        u, ws = self.users[user], self.workspaces[workspace]
        return Person(
            id=u.id, email=f"{u.id}@example.test", role=self.role_in(user, workspace),
            groups=u.groups, org_admin=u.org_admin, in_org=u.org is not None and u.org == ws.org, unit=u.unit, positions=u.positions, attrs=u.attrs,
        )

    def members(self, workspace: str) -> tuple[Person, ...]:
        """Everyone the host knows, as the workspace sees them: the pool `assign … from` and `pass: all` draw on.
        Who may be chosen is the workflow's to say (`from`, `except`), never the host's."""
        return tuple(self.person(u, workspace) for u in self.users)

    def context(self, doc: Document, **kw) -> Context:
        return Context(
            author_id=doc.owner, members=self.members(doc.workspace), invited=frozenset(doc.invited),
            unit=self.users[doc.owner].unit, digest=self.digest(doc.id, doc.timeline.version), **kw,
        )

    # ── Acting and reading ─────────────────────────────────────────────

    @_writes
    def act(self, doc_id: str, user: str, action: str, **kw) -> Timeline:
        """Performs `action` as `user`; raises `Refused`, changing nothing."""
        doc = self.documents[doc_id]
        actor = self.person(user, doc.workspace)
        if (doc.timeline.machine.default_access != "deny" and actor.role is None and user not in doc.invited
                and user != doc.owner and user not in doc.timeline.assigned_everyone()):
            raise Refused("not_permitted")  # no standing on this document (under `deny`, standing is the workflow's to give)
        ctx = replace(self.context(doc, **kw), actor=actor)
        doc.timeline = doc.timeline.act(action, ctx)
        return doc.timeline

    def readable_version(self, doc_id: str, user: str, *, for_search: bool = False) -> int | None:
        doc = self.documents[doc_id]
        return readable_version(doc.timeline, self.person(user, doc.workspace), self.context(doc), for_search=for_search)

    def state_of(self, doc_id: str) -> str:
        return self.documents[doc_id].timeline.state

    def guidance(self, doc_id: str, user: str | None = None) -> str | None:
        """What an agent is told about the document where it stands, with `{name}` filled in by the last choice made: the text of the
        access rule that covers `user` if that rule has one, else the state's. Without a user (the host's own view), the state's."""
        doc = self.documents[doc_id]
        tl = doc.timeline
        rule = resolve(tl, self.person(user, doc.workspace), self.context(doc)) if user is not None else None
        return tl.machine.guidance_for(tl.state, tl.choices(), rule.guidance if rule else None)

    @_writes
    def fire_event(self, doc_id: str, action: str, event: str, source: str | None = None) -> Timeline | None:
        """An outside event (a recorded signature, a regulation coming into force), fired by the host, moves a document
        (no person takes it). One its state doesn't leave through is rejected and logged."""
        doc = self.documents[doc_id]
        before = doc.timeline.state
        try:
            self.act(doc_id, doc.owner, action, event=event)
        except Refused as r:
            self.log.append({"kind": "rejected", "doc": doc_id, "state": before, "event": event, "reason": r.reason})
            return None
        if doc.timeline.state != before:
            self.log.append({"kind": "state_change", "doc": doc_id, "from": before, "to": doc.timeline.state, "trigger": f"event:{event}", "actor": source})
        return doc.timeline

    def visible(self, user: str, *, for_search: bool = True) -> dict[str, int]:
        """Every document the user can read, with the version they read: the filter a search applies."""
        out = {}
        for doc_id in self.documents:
            version = self.readable_version(doc_id, user, for_search=for_search)
            if version is not None:
                out[doc_id] = version
        return out
