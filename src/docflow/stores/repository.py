"""The repository service: the one interface to the stores, and its in-memory implementation.

    people, roles and units   fed by the host's directory; the engine only reads them
    workflow definitions      the YAML text and its hash (the files in `workflows/` stay the source)
    documents                 metadata, and the original file of every version (files on disk, metadata in the database)
    state history             append-only events per document; the state is read off them, never stored
    search index              the passages of every version, with breadcrumb and vector, one index per unit

Every call that names a unit is checked against the units the people store knows; a unit that is not registered is
refused before anything is built from it. A caller never gives a file path: it gives the content and the repository
decides where it goes (`<unit>/<document>/<version>/` under its root) and stores the path relative to that root.

`InMemoryRepository` keeps everything in Python objects, with the same behaviour as the Postgres one
(`postgres_repository.py`); one suite of tests runs against both.
"""

from __future__ import annotations

import copy
import hashlib
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, replace

from .people_roles_units import User, Workspace
from .search_index import DIMS, Corpus, Passage

SUFFIX = re.compile(r"^\.[a-z0-9]{1,8}$")
UNASSIGNED = "-"  # the unit of a document whose owner belongs to none


class RepositoryError(Exception):
    pass


class UnknownUnit(RepositoryError):
    """The unit is not registered in the people store."""


class NoSuchRecord(RepositoryError):
    pass


class VersionExists(RepositoryError):
    pass


class HistoryConflict(RepositoryError):
    """An event was appended out of order: its number is not the next one."""


class BadRecord(RepositoryError):
    pass


class ContentIntegrityError(RepositoryError):
    """The stored file no longer has the hash it was stored with."""


@dataclass(frozen=True)
class PassageIn:
    seq: int
    pages: tuple[int, int]
    breadcrumb: str
    content: str
    vector: tuple[float, ...] | None = None


@dataclass(frozen=True)
class VersionRecord:
    doc_id: str
    version: int
    sha256: str
    size: int
    rel_path: str  # relative to the repository's root, decided by the repository


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    unit: str  # the partition: the unit whose tables and folder hold it
    workflow: str
    owner: str
    workspace: str = ""
    invited: tuple[str, ...] = ()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def folder_name(text: str) -> str:
    """A name safe on a file system for any id: its readable part and a short hash, so two ids never share a folder."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", text).strip(".")[:40] + "-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def relative_path(unit: str, doc_id: str, version: int, suffix: str) -> str:
    return f"{folder_name(unit)}/{folder_name(doc_id)}/{version}/original{suffix}"


def check_version(version, suffix: str, passages: list[PassageIn], content) -> None:
    """What every repository refuses before it writes: a bad version number, suffix, content, or vectors that do not fit."""
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise BadRecord("a version is a whole number from 1")
    if not isinstance(suffix, str) or not SUFFIX.match(suffix):
        raise BadRecord("the file suffix must be a dot and up to eight lower-case letters or digits")
    if not isinstance(content, (bytes, bytearray)):
        raise BadRecord("the content is bytes")
    with_vectors = [p.vector is not None for p in passages]
    if any(with_vectors) and not all(with_vectors):
        raise BadRecord("a version has vectors for all its passages or for none")


class UnitIndex(ABC):
    """The search index of one unit."""

    @abstractmethod
    def passages_of(self, doc: str, version: int) -> list[Passage]: ...

    @abstractmethod
    def search(self, query: str, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        """Keyword search (BM25) over the passages of the versions in `visible`, with statistics from those passages only."""

    @abstractmethod
    def similar(self, vector, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        """Search by meaning (cosine), ranked within the versions in `visible`."""


class Repository(ABC):
    # ── Orgs, units, people, roles (the host feeds them; the engine reads) ──
    @abstractmethod
    def put_org(self, org: str) -> None: ...
    @abstractmethod
    def orgs(self) -> list[str]: ...
    @abstractmethod
    def put_unit(self, unit: str, org: str | None = None) -> None: ...
    @abstractmethod
    def units(self) -> list[str]: ...
    @abstractmethod
    def put_person(self, user: User) -> None: ...
    @abstractmethod
    def people(self) -> list[User]: ...
    @abstractmethod
    def put_workspace(self, workspace: Workspace) -> None: ...
    @abstractmethod
    def workspaces(self) -> list[Workspace]: ...

    # ── Workflow definitions ───────────────────────────────────────────
    @abstractmethod
    def put_workflow(self, name: str, text: str) -> str:
        """Keeps the YAML text; returns its sha256."""
    @abstractmethod
    def workflows(self) -> dict[str, tuple[str, str]]:
        """name -> (YAML text, sha256)"""

    # ── Documents ──────────────────────────────────────────────────────
    @abstractmethod
    def put_document(self, record: DocumentRecord) -> None: ...
    @abstractmethod
    def documents(self) -> list[DocumentRecord]:
        """Every document, in the order they were first stored."""
    @abstractmethod
    def unit_of_document(self, doc_id: str) -> str: ...

    @abstractmethod
    def ingest_version(self, unit: str, doc_id: str, version: int, content: bytes, passages: list[PassageIn], *,
                       suffix: str = ".txt") -> VersionRecord:
        """The file, its metadata row, its passages and their vectors, all written together or not at all."""
    @abstractmethod
    def versions(self, unit: str, doc_id: str) -> list[VersionRecord]: ...
    @abstractmethod
    def read_version(self, unit: str, doc_id: str, version: int) -> bytes:
        """The stored file; its hash is checked against the one stored with it."""

    # ── State history (append-only) ────────────────────────────────────
    @abstractmethod
    def append_event(self, unit: str, subject_id: str, seq: int, event: dict) -> None:
        """Adds event number `seq` (from 0) of a document; it must be the next one."""
    @abstractmethod
    def events(self, unit: str, subject_id: str) -> list[dict]: ...

    # ── Search ─────────────────────────────────────────────────────────
    @abstractmethod
    def index(self, unit: str) -> UnitIndex: ...

    @abstractmethod
    def transaction(self):
        """A context manager: everything written inside it is kept together, or none of it is. They can be nested; the outermost decides."""

    def close(self) -> None:
        pass


class _MemoryIndex(UnitIndex):
    def __init__(self, repo: InMemoryRepository, unit: str) -> None:
        self.repo, self.unit = repo, unit

    @property
    def corpus(self) -> Corpus:
        return self.repo._corpora[self.unit]

    def passages_of(self, doc, version):
        return self.corpus.passages_of(doc, version)

    def search(self, query, visible, limit=5):
        return self.corpus.search(query, visible, limit)

    def similar(self, vector, visible, limit=5):
        vectored = self.repo._vectored[self.unit]
        return self.corpus.similar(vector, {d: v for d, v in visible.items() if (d, v) in vectored}, limit)


class InMemoryRepository(Repository):
    def __init__(self) -> None:
        self._orgs: list[str] = []
        self._units: dict[str, str | None] = {}
        self._people: dict[str, User] = {}
        self._workspaces: dict[str, Workspace] = {}
        self._workflows: dict[str, tuple[str, str]] = {}
        self._documents: dict[str, DocumentRecord] = {}
        self._versions: dict[tuple[str, str], dict[int, VersionRecord]] = {}
        self._files: dict[tuple[str, str, int], bytes] = {}
        self._events: dict[tuple[str, str], list[dict]] = {}
        self._corpora: dict[str, Corpus] = {}
        self._vectored: dict[str, set[tuple[str, int]]] = {}

    _depth = 0

    def _unit(self, unit) -> str:
        if not isinstance(unit, str) or unit not in self._units:
            raise UnknownUnit("that unit is not registered")
        return unit

    @contextmanager
    def transaction(self):
        """Kept by copying everything first and putting the copy back if the block fails. This repository is the test double of the
        PostgreSQL one: it is meant for small worlds, and copying is not."""
        outer = self._depth == 0
        snapshot = copy.deepcopy(self.__dict__) if outer else None
        self._depth += 1
        try:
            yield
        except BaseException:
            if outer:
                self.__dict__.clear()
                self.__dict__.update(snapshot)
            raise
        finally:
            if outer:
                self._depth = 0
            else:
                self._depth -= 1

    # people
    def put_org(self, org):
        if org not in self._orgs:
            self._orgs.append(org)

    def orgs(self):
        return list(self._orgs)

    def put_unit(self, unit, org=None):
        if not isinstance(unit, str) or not unit or len(unit) > 64 or any(ord(c) < 32 for c in unit):
            raise BadRecord("a unit id is 1 to 64 printable characters")
        if unit not in self._units:
            self._units[unit] = org
            self._corpora[unit] = Corpus([], _array())
            self._vectored[unit] = set()

    def units(self):
        return list(self._units)

    def put_person(self, user):
        if user.unit is not None:
            self._unit(user.unit)
        self._people[user.id] = replace(user, attrs=dict(user.attrs))

    def people(self):
        return [replace(u, attrs=dict(u.attrs)) for u in self._people.values()]

    def put_workspace(self, workspace):
        self._workspaces[workspace.id] = replace(workspace, members=dict(workspace.members), group_roles=dict(workspace.group_roles))

    def workspaces(self):
        return [replace(w, members=dict(w.members), group_roles=dict(w.group_roles)) for w in self._workspaces.values()]

    # workflows
    def put_workflow(self, name, text):
        sha = sha256_hex(text.encode("utf-8"))
        self._workflows[name] = (text, sha)
        return sha

    def workflows(self):
        return dict(self._workflows)

    # documents
    def put_document(self, record):
        self._unit(record.unit)
        known = self._documents.get(record.doc_id)
        if known is not None and known.unit != record.unit:
            raise BadRecord("a document stays in its unit")
        self._documents[record.doc_id] = record

    def documents(self):
        return list(self._documents.values())

    def unit_of_document(self, doc_id):
        if doc_id not in self._documents:
            raise NoSuchRecord("no such document")
        return self._documents[doc_id].unit

    def ingest_version(self, unit, doc_id, version, content, passages, *, suffix=".txt"):
        self._unit(unit)
        check_version(version, suffix, passages, content)
        record = self._documents.get(doc_id)
        if record is None or record.unit != unit:
            raise NoSuchRecord("no such document in that unit")
        if version in self._versions.get((unit, doc_id), {}):
            raise VersionExists("that version is already stored")
        if len({p.seq for p in passages}) != len(passages):
            raise BadRecord("two passages have the same number")
        if any(p.vector is not None and len(p.vector) != DIMS for p in passages):
            raise BadRecord("a passage vector has the wrong size")
        content = bytes(content)
        rec = VersionRecord(doc_id, version, sha256_hex(content), len(content), relative_path(unit, doc_id, version, suffix))
        # nothing above wrote anything; from here it all goes in
        corpus = self._corpora[unit]
        vectored = bool(passages) and passages[0].vector is not None
        if vectored:
            corpus.add_passages(doc_id, version, [(p.seq, p.pages, p.breadcrumb, p.content) for p in passages], [list(p.vector) for p in passages])
            self._vectored[unit].add((doc_id, version))
        else:
            corpus.add_passages(doc_id, version, [(p.seq, p.pages, p.breadcrumb, p.content) for p in passages], [[0.0] * DIMS for _ in passages])
        self._files[(unit, doc_id, version)] = content
        self._versions.setdefault((unit, doc_id), {})[version] = rec
        return rec

    def versions(self, unit, doc_id):
        self._unit(unit)
        return [r for _, r in sorted(self._versions.get((unit, doc_id), {}).items())]

    def read_version(self, unit, doc_id, version):
        self._unit(unit)
        rec = self._versions.get((unit, doc_id), {}).get(version)
        if rec is None:
            raise NoSuchRecord("no such version")
        data = self._files.get((unit, doc_id, version))
        if data is None:
            raise ContentIntegrityError(f"{doc_id} version {version}: the stored file is missing")
        if sha256_hex(data) != rec.sha256:
            raise ContentIntegrityError(f"{doc_id} version {version}: the stored content does not match its hash")
        return data

    # history
    def append_event(self, unit, subject_id, seq, event):
        self._unit(unit)
        history = self._events.setdefault((unit, subject_id), [])
        if seq != len(history):
            raise HistoryConflict(f"event {seq} is not the next one")
        history.append(_copy(event))

    def events(self, unit, subject_id):
        self._unit(unit)
        return [_copy(e) for e in self._events.get((unit, subject_id), [])]

    def index(self, unit):
        self._unit(unit)
        return _MemoryIndex(self, unit)


def _copy(value):
    import json
    return json.loads(json.dumps(value))


def _array():
    import array
    return array.array("f")
