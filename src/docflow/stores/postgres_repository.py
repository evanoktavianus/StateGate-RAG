"""The repository on PostgreSQL with pgvector, and files on disk.

    <root>/<unit>/<document>/<version>/original.<suffix>     the original file of every version
    <namespace>                  schema: people, units, workspaces, workflow definitions, the map from a document to its unit
    <namespace>_u_<unit>_<hash>  one schema per unit: its documents, versions, passages with vectors, and history

Needs the optional `psycopg` (version 3) driver; the engine and the in-memory repository do not.

How it keeps its promises:
  * a unit named in a call is looked up among the registered units, and only the schema name stored for it is used, quoted;
    nothing a caller types is ever part of an SQL name or a path;
  * the file of a version is written first; then its row, its passages and their vectors are written in one transaction, and the
    file is removed if the transaction does not commit (a file left by a run that died before it committed has no row, nothing refers
    to it, and the next write of that version replaces it); a version whose row exists always has its file;
  * the sha256 of a version is stored with it and checked on every read;
  * history, versions and passages cannot be updated, deleted or truncated (a trigger refuses, for every role);
  * no table holds where a document stands: that is read off the history.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from pathlib import Path

try:
    import psycopg
    from psycopg import errors, sql
except ImportError as error:  # the in-memory repository works without it
    raise ImportError("docflow.stores.postgres_repository needs the optional 'psycopg' driver (version 3)") from error

from . import migrate
from .people_roles_units import User, Workspace
from .repository import (DIMS, BadRecord, ContentIntegrityError, DocumentRecord, HistoryConflict, NoSuchRecord, PassageIn,
                         Repository, UnitIndex, UnknownUnit, VersionExists, VersionRecord, check_version, folder_name, relative_path,
                         sha256_hex)
from .search_index import Corpus, Passage

NAMESPACE = re.compile(r"^[a-z][a-z0-9_]{0,19}$")


def vector_literal(vector) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


class PostgresRepository(Repository):
    def __init__(self, dsn: str, root: str | Path, namespace: str = "docflow") -> None:
        if not NAMESPACE.match(namespace):
            raise ValueError("the namespace is a lower-case word of up to 20 letters, digits and underscores")
        self.ns, self.root = namespace, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth, self._tx_files = 0, []
        self._conn = psycopg.connect(dsn, autocommit=True)
        migrate.apply_global(self._conn, self.ns)
        self._schemas: dict[str, tuple[str, str]] = {}  # unit -> (schema name, folder name), as read from the units table

    def __repr__(self) -> str:
        return f"PostgresRepository(namespace={self.ns!r}, root={str(self.root)!r})"

    def close(self) -> None:
        self._conn.close()

    # ── transactions ───────────────────────────────────────────────────

    @contextmanager
    def transaction(self):
        """Everything written inside is committed together or rolled back together, and the files of a version written inside a
        transaction that does not commit are removed. One connection is used, under a lock: calls from several threads wait for each other."""
        with self._lock:
            outer = self._tx_depth == 0
            if outer:
                self._tx_files = []
            self._tx_depth += 1
            try:
                with self._conn.transaction():
                    yield
            except BaseException:
                if outer:
                    for path in self._tx_files:
                        path.unlink(missing_ok=True)
                raise
            finally:
                self._tx_depth -= 1
                if outer:
                    self._tx_files = []

    # ── helpers ────────────────────────────────────────────────────────

    def _t(self, table: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(self.ns), sql.Identifier(table))

    def _x(self, unit: str, table: str) -> sql.Composed:
        return sql.SQL("{}.{}").format(sql.Identifier(self._unit(unit)[0]), sql.Identifier(table))

    def _unit(self, unit) -> tuple[str, str]:
        """The schema and folder of a registered unit. Anything else is refused before an SQL name is built."""
        if not isinstance(unit, str):
            raise UnknownUnit("that unit is not registered")
        if unit not in self._schemas:
            row = self._conn.execute(sql.SQL("SELECT schema_name, dir_name FROM {} WHERE unit_id = %s").format(self._t("units")), (unit,)).fetchone()
            if row is None:
                raise UnknownUnit("that unit is not registered")
            self._schemas[unit] = (row[0], row[1])
        return self._schemas[unit]

    # ── orgs, units, people ────────────────────────────────────────────

    def put_org(self, org):
        with self._lock:
            self._conn.execute(sql.SQL("INSERT INTO {} (org_id) VALUES (%s) ON CONFLICT DO NOTHING").format(self._t("orgs")), (org,))

    def orgs(self):
        with self._lock:
            return [r[0] for r in self._conn.execute(sql.SQL("SELECT org_id FROM {} ORDER BY ordinal").format(self._t("orgs")))]

    def put_unit(self, unit, org=None):
        if not isinstance(unit, str) or not unit or len(unit) > 64 or any(ord(c) < 32 for c in unit):
            raise BadRecord("a unit id is 1 to 64 printable characters")
        schema = f"{self.ns}_u_{re.sub(r'[^a-z0-9]', '', unit.lower())[:16]}_{hashlib.sha256(unit.encode()).hexdigest()[:8]}"
        with self._lock, self._conn.transaction():
            self._conn.execute(sql.SQL("INSERT INTO {} (unit_id, schema_name, dir_name, org) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING").format(self._t("units")),
                               (unit, schema, folder_name(unit), org))
            self._conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        migrate.apply_unit(self._conn, self.ns, schema)

    def units(self):
        with self._lock:
            return [r[0] for r in self._conn.execute(sql.SQL("SELECT unit_id FROM {} ORDER BY ordinal").format(self._t("units")))]

    def put_person(self, user):
        if user.unit is not None:
            self._unit(user.unit)
        with self._lock:
            self._conn.execute(sql.SQL(
                "INSERT INTO {} (user_id, org, unit, groups, org_admin, positions, attrs) VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb) "
                "ON CONFLICT (user_id) DO UPDATE SET org = EXCLUDED.org, unit = EXCLUDED.unit, groups = EXCLUDED.groups, "
                "org_admin = EXCLUDED.org_admin, positions = EXCLUDED.positions, attrs = EXCLUDED.attrs").format(self._t("people")),
                (user.id, user.org, user.unit, json.dumps(list(user.groups)), user.org_admin, json.dumps(list(user.positions)), json.dumps(user.attrs)))

    def people(self):
        with self._lock:
            rows = self._conn.execute(sql.SQL("SELECT user_id, org, unit, groups, org_admin, positions, attrs FROM {} ORDER BY ordinal").format(self._t("people"))).fetchall()
        return [User(r[0], r[1], tuple(r[3]), r[4], r[2], tuple(r[5]), dict(r[6])) for r in rows]

    def put_workspace(self, workspace):
        with self._lock:
            self._conn.execute(sql.SQL(
                "INSERT INTO {} (ws_id, org, members, group_roles) VALUES (%s, %s, %s::jsonb, %s::jsonb) "
                "ON CONFLICT (ws_id) DO UPDATE SET org = EXCLUDED.org, members = EXCLUDED.members, group_roles = EXCLUDED.group_roles").format(self._t("workspaces")),
                (workspace.id, workspace.org, json.dumps(workspace.members), json.dumps(workspace.group_roles)))

    def workspaces(self):
        with self._lock:
            rows = self._conn.execute(sql.SQL("SELECT ws_id, org, members, group_roles FROM {} ORDER BY ws_id").format(self._t("workspaces"))).fetchall()
        return [Workspace(r[0], r[1], dict(r[2]), dict(r[3])) for r in rows]

    # ── workflow definitions ───────────────────────────────────────────

    def put_workflow(self, name, text):
        sha = sha256_hex(text.encode("utf-8"))
        with self._lock:
            self._conn.execute(sql.SQL(
                "INSERT INTO {} (name, yaml, sha256) VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET yaml = EXCLUDED.yaml, sha256 = EXCLUDED.sha256, loaded_at = now()")
                .format(self._t("workflows")), (name, text, sha))
        return sha

    def workflows(self):
        with self._lock:
            return {r[0]: (r[1], r[2]) for r in self._conn.execute(sql.SQL("SELECT name, yaml, sha256 FROM {}").format(self._t("workflows")))}

    # ── documents ───────────────────────────────────────────

    def put_document(self, record):
        self._unit(record.unit)
        with self._lock, self._conn.transaction():
            known = self._conn.execute(sql.SQL("SELECT unit FROM {} WHERE doc_id = %s").format(self._t("document_units")), (record.doc_id,)).fetchone()
            if known is not None and known[0] != record.unit:
                raise BadRecord("a document stays in its unit")
            if known is None:
                self._conn.execute(sql.SQL("INSERT INTO {} (doc_id, unit) VALUES (%s, %s)").format(self._t("document_units")), (record.doc_id, record.unit))
            self._conn.execute(sql.SQL(
                "INSERT INTO {} (doc_id, workflow, owner, workspace, invited) VALUES (%s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (doc_id) DO UPDATE SET workflow = EXCLUDED.workflow, owner = EXCLUDED.owner, workspace = EXCLUDED.workspace, "
                "invited = EXCLUDED.invited").format(self._x(record.unit, "documents")),
                (record.doc_id, record.workflow, record.owner, record.workspace, json.dumps(list(record.invited))))

    def documents(self):
        with self._lock:
            order = self._conn.execute(sql.SQL("SELECT doc_id, unit FROM {} ORDER BY ordinal").format(self._t("document_units"))).fetchall()
            by_unit: dict[str, dict[str, tuple]] = {}
            for unit in {u for _, u in order}:
                rows = self._conn.execute(sql.SQL("SELECT doc_id, workflow, owner, workspace, invited FROM {}")
                                          .format(self._x(unit, "documents"))).fetchall()
                by_unit[unit] = {r[0]: r for r in rows}
        out = []
        for doc_id, unit in order:
            r = by_unit[unit].get(doc_id)
            if r is not None:
                out.append(DocumentRecord(r[0], unit, r[1], r[2], r[3], tuple(r[4])))
        return out

    def unit_of_document(self, doc_id):
        with self._lock:
            row = self._conn.execute(sql.SQL("SELECT unit FROM {} WHERE doc_id = %s").format(self._t("document_units")), (doc_id,)).fetchone()
        if row is None:
            raise NoSuchRecord("no such document")
        return row[0]

    # ── versions: the file, the row, the passages, the vectors ─────────

    def ingest_version(self, unit, doc_id, version, content, passages, *, suffix=".txt"):
        _, unit_dir = self._unit(unit)
        check_version(version, suffix, passages, content)
        content = bytes(content)
        rel = relative_path(unit, doc_id, version, suffix)
        assert rel.startswith(unit_dir + "/")
        path = (self.root / rel)
        if self.root.resolve() not in path.resolve().parents:
            raise BadRecord("the file would fall outside the root")
        record = VersionRecord(doc_id, version, sha256_hex(content), len(content), rel)
        with self._lock:
            known = self._conn.execute(sql.SQL("SELECT 1 FROM {} WHERE doc_id = %s").format(self._x(unit, "documents")), (doc_id,)).fetchone()
            if known is None:
                raise NoSuchRecord("no such document in that unit")
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._conn.execute(sql.SQL("SELECT 1 FROM {} WHERE doc_id = %s AND version = %s").format(self._x(unit, "versions")), (doc_id, version)).fetchone():
                raise VersionExists("that version is already stored")
            path.unlink(missing_ok=True)  # a file with no row is left over from a run that died before it committed: nothing refers to it
            with open(path, "xb") as f:
                f.write(content)
            if self._tx_depth:
                self._tx_files.append(path)
            try:
                with self._conn.transaction():
                    self._conn.execute(sql.SQL("INSERT INTO {} (doc_id, version, sha256, size, rel_path) VALUES (%s, %s, %s, %s, %s)")
                                       .format(self._x(unit, "versions")), (doc_id, version, record.sha256, record.size, rel))
                    with self._conn.cursor() as cur:
                        cur.executemany(
                            sql.SQL("INSERT INTO {} (doc_id, version, seq, pages, breadcrumb, content, embedding) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s::vector)")
                            .format(self._x(unit, "chunks")),
                            [(doc_id, version, p.seq, json.dumps(list(p.pages)), p.breadcrumb, p.content, None if p.vector is None else vector_literal(p.vector))
                             for p in passages])
            except errors.UniqueViolation as e:
                path.unlink(missing_ok=True)
                raise (VersionExists("that version is already stored") if (e.diag.constraint_name or "").startswith("versions") else BadRecord("two passages have the same number")) from None
            except errors.DataException:
                path.unlink(missing_ok=True)
                raise BadRecord("a passage vector has the wrong size") from None
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return record

    def versions(self, unit, doc_id):
        with self._lock:
            rows = self._conn.execute(sql.SQL("SELECT doc_id, version, sha256, size, rel_path FROM {} WHERE doc_id = %s ORDER BY version")
                                      .format(self._x(unit, "versions")), (doc_id,)).fetchall()
        return [VersionRecord(*r) for r in rows]

    def read_version(self, unit, doc_id, version):
        with self._lock:
            row = self._conn.execute(sql.SQL("SELECT sha256, rel_path FROM {} WHERE doc_id = %s AND version = %s").format(self._x(unit, "versions")), (doc_id, version)).fetchone()
        if row is None:
            raise NoSuchRecord("no such version")
        try:
            data = (self.root / row[1]).read_bytes()
        except OSError:
            raise ContentIntegrityError(f"{doc_id} version {version}: the stored file is missing") from None
        if sha256_hex(data) != row[0]:
            raise ContentIntegrityError(f"{doc_id} version {version}: the stored content does not match its hash")
        return data

    # ── history ────────────────────────────────────────────────────────

    def append_event(self, unit, subject_id, seq, event):
        table = self._x(unit, "history")
        try:
            with self._lock:
                cur = self._conn.execute(
                    sql.SQL("INSERT INTO {t} (subject_id, seq, event) SELECT %s, %s, %s::jsonb "
                            "WHERE %s = (SELECT coalesce(max(seq) + 1, 0) FROM {t} WHERE subject_id = %s)").format(t=table),
                    (subject_id, seq, json.dumps(event), seq, subject_id))
                inserted = cur.rowcount
        except (errors.UniqueViolation, errors.CheckViolation):
            raise HistoryConflict(f"event {seq} is not the next one") from None
        if inserted != 1:
            raise HistoryConflict(f"event {seq} is not the next one")

    def events(self, unit, subject_id):
        with self._lock:
            return [r[0] for r in self._conn.execute(sql.SQL("SELECT event FROM {} WHERE subject_id = %s ORDER BY seq").format(self._x(unit, "history")), (subject_id,))]

    def index(self, unit):
        self._unit(unit)
        return _PostgresIndex(self, unit)

    # ── for tests: remove everything this installation made ────────────

    def drop_all(self) -> None:
        """Drops the namespace and every unit schema under it. Used by the tests on a namespace of their own."""
        with self._lock:
            names = [r[0] for r in self._conn.execute("SELECT nspname FROM pg_namespace WHERE nspname = %s OR starts_with(nspname, %s)", (self.ns, self.ns + "_u_"))]
            for name in names:
                self._conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


def _passage(r) -> Passage:
    return Passage(r[0], r[1], r[2], tuple(r[3]), r[4], r[5], r[6])


class _PostgresIndex(UnitIndex):
    def __init__(self, repo: PostgresRepository, unit: str) -> None:
        self.repo, self.unit = repo, unit

    def _visible(self, visible: dict[str, int]) -> tuple[list[str], list[int]]:
        return list(visible), [visible[d] for d in visible]

    COLUMNS = "doc_id, version, seq, pages, breadcrumb, content, row_no"

    def passages_of(self, doc, version):
        with self.repo._lock:
            rows = self.repo._conn.execute(sql.SQL("SELECT " + self.COLUMNS + " FROM {} WHERE doc_id = %s AND version = %s ORDER BY seq")
                                           .format(self.repo._x(self.unit, "chunks")), (doc, version)).fetchall()
        return [_passage(r) for r in rows]

    def search(self, query, visible, limit=5):
        """BM25 is computed here, over the passages of the visible versions and no others: the database ranks nothing, because
        its own text ranking would take its statistics from the whole index, and a hidden document could move a score."""
        if not visible:
            return []
        docs, versions = self._visible(visible)
        with self.repo._lock:
            rows = self.repo._conn.execute(
                sql.SQL("SELECT " + self.COLUMNS + " FROM {} WHERE (doc_id, version) IN (SELECT * FROM unnest(%s::text[], %s::int[])) ORDER BY row_no")
                .format(self.repo._x(self.unit, "chunks")), (docs, versions)).fetchall()
        return Corpus([_passage(r) for r in rows]).search(query, visible, limit)

    def similar(self, vector, visible, limit=5):
        """The visible versions are part of the query, so the nearest passages are chosen among them; never the nearest first, filtered after."""
        if not visible:
            return []
        if len(vector) != DIMS:
            raise BadRecord("the query vector has the wrong size")
        docs, versions = self._visible(visible)
        q = vector_literal(vector)
        with self.repo._lock:
            rows = self.repo._conn.execute(
                sql.SQL("SELECT " + self.COLUMNS + ", 1 - (embedding <=> %s::vector) AS similarity FROM {} "
                        "WHERE (doc_id, version) IN (SELECT * FROM unnest(%s::text[], %s::int[])) AND embedding IS NOT NULL "
                        "ORDER BY embedding <=> %s::vector, row_no LIMIT %s").format(self.repo._x(self.unit, "chunks")),
                (q, docs, versions, q, limit)).fetchall()
        return [(float(r[7]), _passage(r)) for r in rows]
