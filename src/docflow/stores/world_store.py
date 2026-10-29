"""Saving a world to a repository, keeping it there as it changes, and loading one back.

`save_world` writes what the stores hold: people and workspaces, the workflow definitions (the YAML text and its hash),
each document's metadata, their history as append-only events, and every version's content with its
passages. Nothing about where a document stands is written, because it is read off the history: `load_world` rebuilds
the timelines from the events and gets the same states and versions.

Saving twice appends only the events that are new. A world whose history is not an extension of what is stored (an
earlier event differs) is refused, not overwritten.

`WriteThrough` keeps a repository up to date as a world changes: give it to `World(workflows, store=...)`, and after every
accepted action (the outermost call, with whatever it set going) everything that is new is written in one transaction: the new
history entries, the changed metadata, and every new version's row, hash and passages (the file of a new version is written to disk just
before, and removed if the transaction does not commit). An action that is refused
changes nothing and writes nothing. If writing fails, nothing of the action is kept in the repository and the world is rebuilt
from the repository, so the two agree. The content of a new version comes from a function the host gives (`content`); an action
that makes a version the host cannot supply content for is undone (`MissingContent`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml

from ..access_control.access_policy_engine.machine import Machine
from .documents import Document
from .people_roles_units import User, Workspace
from .repository import UNASSIGNED, DocumentRecord, PassageIn, RepositoryError
from .state_history import Timeline
from .workflow_definitions import _Loader
from .world import World


class HistoryDiverged(RepositoryError):
    pass


class MissingContent(RepositoryError):
    """A version was made that the host gave no content for."""


def machine_from_text(text: str) -> Machine:
    return Machine.from_dict(yaml.load(text, Loader=_Loader))


def partition_of(world: World, doc: Document) -> str:
    """The unit whose tables and folder hold a document: its owner's, else the unassigned one."""
    return world.users[doc.owner].unit or UNASSIGNED


def _encode(entry: tuple) -> list:
    return json.loads(json.dumps(entry))


def _decode(entry: list) -> tuple:
    if entry[0] == "assigned":
        return ("assigned", entry[1], tuple(entry[2]))
    return tuple(entry)


def versions_of(doc: Document) -> set[int]:
    """The version numbers a document has, read off its history."""
    return {e[1] for e in doc.timeline.entries if e[0] == "version"}


class Writer:
    """Brings a repository up to date with a world: only what is new or changed is written, in one transaction.

    It remembers what it has written (counts of events, the last record of each document, person and workspace), so a
    flush does not read the repository again; those memories are updated only after the transaction has committed."""

    def __init__(self, repo, workflow_files: dict[str, str | Path], passages_for) -> None:
        self.repo, self.files, self.passages_for = repo, workflow_files, passages_for
        self.events: dict[str, int] = {}
        self.records: dict[tuple, object] = {}
        self.versions: dict[str, set[int]] = {}
        self.units: set[str] | None = None  # what the repository holds, read the first time it is needed
        self.orgs: set[str] = set()
        self.workflows: set[str] = set()

    def _ready(self) -> None:
        """What the repository already holds of the things that are not per subject (so attaching to a loaded world writes only what is new)."""
        if self.units is None:
            self.units = set(self.repo.units())
            self.orgs = set(self.repo.orgs())
            self.workflows = set(self.repo.workflows())

    def _stored_events(self, unit: str, subject: str, timeline: Timeline) -> int:
        """How many events of this subject the repository holds, read the first time it is seen, with a check that they are a prefix."""
        if subject not in self.events:
            if unit not in self.units:  # a unit the repository has not seen yet holds nothing
                return 0
            stored = [e["entry"] for e in self.repo.events(unit, subject)]
            if [_encode(e) for e in timeline.entries[: len(stored)]] != stored:
                raise HistoryDiverged(f"the history of {subject} differs from the one stored")
            self.events[subject] = len(stored)
        return self.events[subject]

    def _stored_versions(self, unit: str, doc: str) -> set[int]:
        if doc not in self.versions:
            if unit not in self.units:
                return set()
            self.versions[doc] = {v.version for v in self.repo.versions(unit, doc)}
        return self.versions[doc]

    def plan(self, world: World, also=()) -> dict:
        """What is not in the repository yet. Empty when the repository is up to date."""
        self._ready()
        plan: dict = {"workflows": [], "orgs": [], "units": [], "users": [], "workspaces": [], "docs": [], "events": [], "versions": []}
        plan["workflows"] = [n for n in world.workflows if n not in self.workflows]
        plan["orgs"] = [o for o in sorted(world.orgs) if o not in self.orgs]
        units = [u.unit for u in world.users.values() if u.unit] + [partition_of(world, d) for d in world.documents.values()]
        plan["units"] = [u for u in dict.fromkeys([UNASSIGNED, *units]) if u not in self.units]
        plan["users"] = [u for u in world.users.values() if self.records.get(("user", u.id)) != u]
        plan["workspaces"] = [w for w in world.workspaces.values() if self.records.get(("workspace", w.id)) != w]
        extra: dict[str, set[int]] = {}
        for doc_id, version in also:
            extra.setdefault(doc_id, set()).add(version)
        for doc in world.documents.values():
            unit = partition_of(world, doc)
            record = DocumentRecord(doc.id, unit, doc.workflow, doc.owner, doc.workspace, tuple(sorted(doc.invited)))
            if self.records.get(("doc", doc.id)) != record:
                plan["docs"].append(record)
            stored = self._stored_events(unit, doc.id, doc.timeline)
            if len(doc.timeline.entries) > stored:
                plan["events"].append((unit, doc.id, stored, [_encode(e) for e in doc.timeline.entries[stored:]], doc.timeline))
            wanted = versions_of(doc) | extra.get(doc.id, set())
            plan["versions"] += [(unit, doc.id, v) for v in sorted(wanted - self._stored_versions(unit, doc.id))]
        return plan

    def in_sync(self, world: World, also=()) -> bool:
        return not any(self.plan(world, also).values())

    def sync(self, world: World, also=()) -> None:
        plan = self.plan(world, also)
        if not any(plan.values()):
            return
        # the content of every new version is asked for before anything is written, so a version the host cannot supply refuses the action
        content = {}
        for unit, doc_id, version in plan["versions"]:
            passages = self.passages_for(doc_id, version)
            if passages is None:
                raise MissingContent(f"no content for {doc_id} version {version}")
            content[(doc_id, version)] = passages
        repo = self.repo
        with repo.transaction():
            for name in plan["workflows"]:
                repo.put_workflow(name, Path(self.files[name]).read_text(encoding="utf-8"))
            for org in plan["orgs"]:
                repo.put_org(org)
            for unit in plan["units"]:
                repo.put_unit(unit)
            for user in plan["users"]:
                repo.put_person(user)
            for workspace in plan["workspaces"]:
                repo.put_workspace(workspace)
            for record in plan["docs"]:
                repo.put_document(record)
            for unit, subject, start, entries, _ in plan["events"]:
                for offset, entry in enumerate(entries):
                    repo.append_event(unit, subject, start + offset, {"entry": entry})
            for unit, doc_id, version in plan["versions"]:
                passages = content[(doc_id, version)]
                repo.ingest_version(unit, doc_id, version, "\n\n".join(p.content for p in passages).encode("utf-8"), passages)
        # committed: now it is true that the repository holds these
        self.workflows.update(plan["workflows"]), self.orgs.update(plan["orgs"]), self.units.update(plan["units"])
        for user in plan["users"]:
            self.records[("user", user.id)] = replace(user, attrs=dict(user.attrs))
        for workspace in plan["workspaces"]:
            self.records[("workspace", workspace.id)] = replace(workspace, members=dict(workspace.members), group_roles=dict(workspace.group_roles))
        for record in plan["docs"]:
            self.records[("doc", record.doc_id)] = record
        for unit, subject, start, entries, _ in plan["events"]:
            self.events[subject] = start + len(entries)
        for unit, doc_id, version in plan["versions"]:
            self.versions.setdefault(doc_id, set()).add(version)


def save_world(world: World, repo, corpus, workflow_files: dict[str, str | Path]) -> None:
    """`workflow_files` names the YAML file each of the world's workflows was loaded from."""
    def passages_for(doc_id: str, version: int):
        found = corpus.passages_of(doc_id, version)
        if not found:
            return None
        vectors = [tuple(corpus.vector_of(p)) for p in found] if getattr(corpus, "_embeddings", None) is not None else [None] * len(found)
        return [PassageIn(p.seq, tuple(p.pages), p.breadcrumb, p.content, v) for p, v in zip(found, vectors)]

    Writer(repo, workflow_files, passages_for).sync(world, also=list(corpus.by_version))


class WriteThrough:
    """Keeps a repository up to date as a world changes; see the top of this file."""

    def __init__(self, repo, workflow_files: dict[str, str | Path], content=None) -> None:
        self.repo = repo
        self.writer = Writer(repo, workflow_files, content or (lambda doc_id, version: None))

    def flush(self, world: World) -> None:
        """After an accepted action. If it cannot be written, nothing of it stays in the repository and the world is rebuilt from it."""
        try:
            self.writer.sync(world)
        except BaseException:
            self.restore(world)
            raise

    def recover(self, world: World) -> None:
        """After an action that raised. A refusal changed nothing, and then there is nothing to do; if the action got partway before it
        raised, the world is rebuilt from the repository."""
        if not self.writer.in_sync(world):
            self.restore(world)

    def restore(self, world: World) -> None:
        fresh = load_world(self.repo)
        world.orgs, world.users, world.workspaces = fresh.orgs, fresh.users, fresh.workspaces
        world.documents = fresh.documents

    def in_sync(self, world: World) -> bool:
        return self.writer.in_sync(world)


def load_world(repo) -> World:
    workflows = {name: machine_from_text(text) for name, (text, _) in repo.workflows().items()}
    world = World(workflows)
    world.orgs = set(repo.orgs())
    for user in repo.people():
        world.users[user.id] = user
    for workspace in repo.workspaces():
        world.workspaces[workspace.id] = workspace

    def timeline(unit: str, subject: str, workflow: str) -> Timeline:
        return Timeline(workflows[workflow], tuple(_decode(e["entry"]) for e in repo.events(unit, subject)))

    for r in repo.documents():
        world.documents[r.doc_id] = Document(r.doc_id, r.workspace, r.owner, r.workflow, timeline(r.unit, r.doc_id, r.workflow), set(r.invited))
    return world
