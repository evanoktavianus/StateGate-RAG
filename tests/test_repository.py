"""The repository service, on both implementations: the same tests run against the in-memory repository and, when DATABASE_URL
is set, against PostgreSQL with pgvector (a namespace of its own for each test, dropped afterwards, and files in a temp folder).

Who may read what comes from the table in test_access_control.py and the workflows."""

import hashlib
import os
import re
import secrets
from pathlib import Path

import pytest

from docflow import Machine
from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.document_service.retrieve import PartitionedIndex
from docflow.stores.documents import Document
from docflow.stores.people_roles_units import User, Workspace
from docflow.stores.repository import (BadRecord, ContentIntegrityError, DocumentRecord, HistoryConflict, InMemoryRepository, NoSuchRecord,
                                       PassageIn, Repository, UnknownUnit, VersionExists, relative_path)
from docflow.stores.search_index import DIMS, Corpus
from docflow.stores.state_history import Timeline
from docflow.stores.world_store import HistoryDiverged, load_world, save_world
from docflow.support import scenario
from conftest import WORKFLOWS
from test_access_control import ORACLE, PEOPLE, QUERY, expected_for, gw, vec, world  # noqa: F401  (the ministry fixture and its table)

ROOT = Path(__file__).parent.parent
DATABASE_URL = os.environ.get("DATABASE_URL")


class Harness:
    """A repository with what a test needs to reach behind it (to damage a file, to ask whether one exists)."""

    def __init__(self, repo: Repository, root: Path, kind: str) -> None:
        self.repo, self.root, self.kind = repo, root, kind

    def tamper(self, unit, doc, version, data=b"changed") -> None:
        if self.kind == "memory":
            self.repo._files[(unit, doc, version)] = data
        else:
            (self.root / relative_path(unit, doc, version, ".txt")).write_bytes(data)

    def remove_file(self, unit, doc, version) -> None:
        if self.kind == "memory":
            del self.repo._files[(unit, doc, version)]
        else:
            (self.root / relative_path(unit, doc, version, ".txt")).unlink()

    def has_file(self, unit, doc, version) -> bool:
        if self.kind == "memory":
            return (unit, doc, version) in self.repo._files
        return (self.root / relative_path(unit, doc, version, ".txt")).exists()


@pytest.fixture(params=["memory", "postgres"])
def h(request, tmp_path):
    if request.param == "memory":
        yield Harness(InMemoryRepository(), tmp_path, "memory")
        return
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL is not set: the PostgreSQL repository is not tested")
    pytest.importorskip("psycopg")
    from docflow.stores.postgres_repository import PostgresRepository

    repo = PostgresRepository(DATABASE_URL, tmp_path / "files", namespace="t" + secrets.token_hex(4))
    try:
        yield Harness(repo, tmp_path / "files", "postgres")
    finally:
        repo.drop_all()
        repo.close()


@pytest.fixture
def pg(h):
    if h.kind != "postgres":
        pytest.skip("this checks the database itself")
    return h


def doc(unit="U1", doc_id="d1", **kw):
    return DocumentRecord(doc_id, unit, kw.pop("workflow", "w"), kw.pop("owner", "ann"), **kw)


def passage(seq, text, a=None, crumb=""):
    return PassageIn(seq, (0, 0), crumb, text, None if a is None else tuple(vec(a)))


def setup_unit(repo, unit="U1", docs=("d1",)):
    repo.put_unit(unit)
    for d in docs:
        repo.put_document(doc(unit, d))


# ── Units ──────────────────────────────────────────────────────────────

def test_a_unit_that_is_not_registered_is_refused_before_anything_is_built(h):
    repo = h.repo
    setup_unit(repo)
    attempts = ["U9", "U1\"; DROP TABLE people; --", "../../etc", "", "u1; select 1", "U1 ", "u1", None, 5, ["U1"]]
    for unit in attempts:
        for call in (lambda: repo.put_document(doc(unit)), lambda: repo.versions(unit, "d1"), lambda: repo.read_version(unit, "d1", 1),
                     lambda: repo.ingest_version(unit, "d1", 1, b"x", []), lambda: repo.append_event(unit, "d1", 0, {}),
                     lambda: repo.events(unit, "d1"), lambda: repo.index(unit)):
            with pytest.raises(UnknownUnit):
                call()
    assert repo.units() == ["U1"]


def test_a_unit_id_must_be_printable_text(h):
    for bad in ("", "x" * 65, "a\nb", "a\x00b", None):
        with pytest.raises(BadRecord):
            h.repo.put_unit(bad)
    h.repo.put_unit("U-1 (2)")  # spaces and punctuation are fine as an id; they never become part of a name
    assert h.repo.units() == ["U-1 (2)"]


def test_a_unit_name_never_reaches_sql(pg):
    setup_unit(pg.repo)
    with pg.repo._conn.transaction():
        pg.repo._conn.execute(f"CREATE TABLE {pg.repo.ns}.canary (x int)")
    for evil in ('U1"; DROP TABLE canary; --', "U1'; DROP SCHEMA " + pg.repo.ns + " CASCADE; --"):
        with pytest.raises(UnknownUnit):
            pg.repo.versions(evil, "d1")
    assert pg.repo._conn.execute(f"SELECT count(*) FROM {pg.repo.ns}.canary").fetchone() == (0,)
    # a registered unit with a hostile id is stored safely: its schema name is built from letters, digits and a hash
    pg.repo.put_unit('X"; DROP SCHEMA ' + pg.repo.ns + " CASCADE; --")
    names = [r[0] for r in pg.repo._conn.execute("SELECT schema_name FROM " + pg.repo.ns + ".units")]
    assert all(re.fullmatch(r"[a-z0-9_]+", n) for n in names)


# ── Versions: the file, its hash, its passages ─────────────────────────

def test_a_version_is_stored_with_its_hash_and_read_back(h):
    setup_unit(h.repo)
    record = h.repo.ingest_version("U1", "d1", 1, b"hello", [passage(0, "hello")])
    assert record.sha256 == hashlib.sha256(b"hello").hexdigest() == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert record.size == 5
    assert re.fullmatch(r"[^/]+/[^/]+/1/original\.txt", record.rel_path) and ".." not in record.rel_path  # relative to the root, decided here
    assert h.repo.read_version("U1", "d1", 1) == b"hello"
    assert h.repo.versions("U1", "d1") == [record]
    assert h.has_file("U1", "d1", 1)


def test_the_hash_is_checked_on_every_read(h):
    setup_unit(h.repo)
    h.repo.ingest_version("U1", "d1", 1, b"hello", [passage(0, "hello")])
    assert h.repo.read_version("U1", "d1", 1) == b"hello"
    h.tamper("U1", "d1", 1)
    for _ in range(2):  # not only the first time
        with pytest.raises(ContentIntegrityError) as raised:
            h.repo.read_version("U1", "d1", 1)
        assert "d1 version 1" in str(raised.value) and str(h.root) not in str(raised.value)  # says which, not where


def test_a_missing_file_is_an_integrity_error(h):
    setup_unit(h.repo)
    h.repo.ingest_version("U1", "d1", 1, b"hello", [passage(0, "hello")])
    h.remove_file("U1", "d1", 1)
    with pytest.raises(ContentIntegrityError, match="missing"):
        h.repo.read_version("U1", "d1", 1)
    with pytest.raises(NoSuchRecord):
        h.repo.read_version("U1", "d1", 2)


@pytest.mark.parametrize("passages", [
    [passage(0, "first", 0.1), passage(0, "same number", 0.2)],               # two passages with one number
    [passage(0, "first", 0.1), PassageIn(1, (0, 0), "", "short vector", (1.0, 2.0, 3.0))],  # the second vector is the wrong size
    [passage(0, "with", 0.1), passage(1, "without")],                          # vectors for some passages only
], ids=["duplicate number", "wrong vector size", "mixed vectors"])
def test_ingest_writes_everything_or_nothing(h, passages):
    setup_unit(h.repo)
    with pytest.raises((BadRecord, VersionExists)):
        h.repo.ingest_version("U1", "d1", 1, b"content", passages)
    assert h.repo.versions("U1", "d1") == []  # no row
    assert h.repo.index("U1").passages_of("d1", 1) == []  # no passage
    assert not h.has_file("U1", "d1", 1)  # no file
    h.repo.ingest_version("U1", "d1", 1, b"content", [passage(0, "fine", 0.1)])  # and nothing is left in the way of a good one
    assert h.repo.read_version("U1", "d1", 1) == b"content"


def test_a_failed_ingest_leaves_no_row_chunk_or_file_in_the_database(pg):
    setup_unit(pg.repo)
    with pytest.raises(BadRecord):
        pg.repo.ingest_version("U1", "d1", 1, b"x", [passage(0, "a", 0.1), PassageIn(1, (0, 0), "", "b", (1.0,))])
    schema = pg.repo._unit("U1")[0]
    for table in ("versions", "chunks"):
        assert pg.repo._conn.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone() == (0,)
    assert list(pg.root.rglob("original*")) == []


def test_a_version_is_stored_once(h):
    setup_unit(h.repo)
    h.repo.ingest_version("U1", "d1", 1, b"first", [passage(0, "first")])
    with pytest.raises(VersionExists):
        h.repo.ingest_version("U1", "d1", 1, b"second", [passage(0, "second")])
    assert h.repo.read_version("U1", "d1", 1) == b"first" and h.repo.index("U1").passages_of("d1", 1)[0].content == "first"


def test_the_caller_never_chooses_a_path(h):
    import inspect
    assert "path" not in " ".join(inspect.signature(h.repo.ingest_version).parameters)
    setup_unit(h.repo)
    for suffix in ("/etc/passwd", ".TXT", "txt", ".a/b", "", "..", ".toolongsuffix", None, 5):
        with pytest.raises(BadRecord):
            h.repo.ingest_version("U1", "d1", 1, b"x", [], suffix=suffix)
    for version in (0, -1, "1", None, True):
        with pytest.raises(BadRecord):
            h.repo.ingest_version("U1", "d1", version, b"x", [])
    with pytest.raises(BadRecord):
        h.repo.ingest_version("U1", "d1", 1, "text, not bytes", [])
    assert h.repo.versions("U1", "d1") == []
    h.repo.ingest_version("U1", "d1", 1, b"x", [], suffix=".pdf")
    assert h.repo.versions("U1", "d1")[0].rel_path.endswith("/1/original.pdf")


def test_a_document_stays_in_its_unit(h):
    setup_unit(h.repo, "U1", ("d1",))
    h.repo.put_unit("U2")
    with pytest.raises(BadRecord):
        h.repo.put_document(doc("U2", "d1"))
    with pytest.raises(NoSuchRecord):
        h.repo.ingest_version("U2", "d1", 1, b"x", [])
    with pytest.raises(NoSuchRecord):
        h.repo.ingest_version("U1", "never-stored", 1, b"x", [])
    assert h.repo.unit_of_document("d1") == "U1"
    with pytest.raises(NoSuchRecord):
        h.repo.unit_of_document("never-stored")


def test_paths_are_relative_inside_the_root_and_checked_by_the_database(pg):
    setup_unit(pg.repo)
    record = pg.repo.ingest_version("U1", "d1", 1, b"x", [])
    schema = pg.repo._unit("U1")[0]
    assert pg.repo._conn.execute(f"SELECT rel_path FROM {schema}.versions").fetchone() == (record.rel_path,)
    assert (pg.root / record.rel_path).is_file() and pg.root.resolve() in (pg.root / record.rel_path).resolve().parents
    import psycopg
    for bad in ("/etc/passwd", "../outside"):
        with pytest.raises(psycopg.errors.CheckViolation):
            pg.repo._conn.execute(f"INSERT INTO {schema}.versions (doc_id, version, sha256, size, rel_path) VALUES ('d1', 9, %s, 1, %s)", ("0" * 64, bad))


# ── State history: append-only ─────────────────────────────────────────

def test_history_only_grows_and_in_order(h):
    setup_unit(h.repo)
    for seq, entry in enumerate([["version", 1], ["state", "working"], ["state", "review"]]):
        h.repo.append_event("U1", "d1", seq, {"entry": entry})
    assert h.repo.events("U1", "d1") == [{"entry": ["version", 1]}, {"entry": ["state", "working"]}, {"entry": ["state", "review"]}]
    for seq in (0, 2, 5, -1):  # a repeat, and a gap, and a negative number
        with pytest.raises(HistoryConflict):
            h.repo.append_event("U1", "d1", seq, {"entry": ["state", "other"]})
    assert len(h.repo.events("U1", "d1")) == 3 and h.repo.events("U1", "other") == []
    got = h.repo.events("U1", "d1")
    got[0]["entry"].append("changed")
    got.pop()
    assert h.repo.events("U1", "d1")[0] == {"entry": ["version", 1]} and len(h.repo.events("U1", "d1")) == 3  # what it returns is a copy


def test_the_repository_has_no_way_to_change_or_remove_history(h):
    names = [n for n in dir(h.repo) if not n.startswith("_")]
    assert not [n for n in names if re.match(r"(update|delete|remove|truncate|rewrite|edit)", n)]
    assert {"append_event", "events"} <= set(names)


def test_the_database_refuses_to_change_history_versions_or_passages(pg):
    setup_unit(pg.repo)
    pg.repo.ingest_version("U1", "d1", 1, b"x", [passage(0, "text", 0.1)])
    pg.repo.append_event("U1", "d1", 0, {"entry": ["version", 1]})
    schema = pg.repo._unit("U1")[0]
    import psycopg
    for table in ("history", "versions", "chunks"):
        for statement in (f"UPDATE {schema}.{table} SET doc_id = doc_id" if table != "history" else f"UPDATE {schema}.history SET seq = seq",
                          f"DELETE FROM {schema}.{table}", f"TRUNCATE {schema}.{table} CASCADE"):  # CASCADE: so it is the trigger that refuses
            with pytest.raises(psycopg.errors.IntegrityConstraintViolation, match="append-only"):
                pg.repo._conn.execute(statement)
        assert pg.repo._conn.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone() == (1,)


def test_no_table_stores_where_a_document_stands(pg):
    setup_unit(pg.repo)
    rows = pg.repo._conn.execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = ANY(%s)",
                                 ([pg.repo.ns, pg.repo._unit("U1")[0]],)).fetchall()
    assert rows  # we looked at something
    assert not [r for r in rows if re.search(r"state|status|stage|released|current", r[1])], [r for r in rows if re.search(r"state|status|stage|released|current", r[1])]


def test_migrations_are_applied_once(pg):
    from docflow.stores import migrate
    assert migrate.apply_global(pg.repo._conn, pg.repo.ns) == []  # the repository applied them when it was made
    setup_unit(pg.repo)
    assert migrate.apply_unit(pg.repo._conn, pg.repo.ns, pg.repo._unit("U1")[0]) == []


# ── The people store and the workflow definitions ──────────────────────

def test_people_roles_workspaces_and_workflows_round_trip(h):
    repo = h.repo
    repo.put_org("O1"), repo.put_org("O2"), repo.put_org("O1")
    repo.put_unit("U1")
    ann = User("ann", "O1", ("knowledge management",), True, "U1", ("leadership",), {"clearance": "high"})
    repo.put_person(ann)
    repo.put_person(User("ext", "O2"))
    repo.put_workspace(Workspace("ws", "O1", {"ann": "contributor"}, {"committee": "viewer"}))
    assert repo.orgs() == ["O1", "O2"]
    assert repo.people() == [ann, User("ext", "O2")]  # in the order they were given
    assert repo.workspaces() == [Workspace("ws", "O1", {"ann": "contributor"}, {"committee": "viewer"})]
    with pytest.raises(UnknownUnit):
        repo.put_person(User("bob", "O1", unit="U9"))  # a person's unit is one the people store knows
    sha = repo.put_workflow("w", "name: W\n")
    assert sha == hashlib.sha256(b"name: W\n").hexdigest() and repo.workflows() == {"w": ("name: W\n", sha)}
    sha2 = repo.put_workflow("w", "name: W2\n")
    assert sha2 != sha and repo.workflows()["w"] == ("name: W2\n", sha2)


def test_people_come_back_as_copies(h):
    h.repo.put_unit("U1")
    h.repo.put_person(User("ann", unit="U1", attrs={"a": 1}))
    got = h.repo.people()[0]
    got.attrs["a"] = 2
    got.unit = "U9"
    assert h.repo.people()[0].attrs == {"a": 1} and h.repo.people()[0].unit == "U1"


# ── Searching: the two behaviours that must survive a database ─────────

def reference(items):
    """The original in-memory search over exactly these passages: [(doc, version, text, vector tilt)]."""
    import array
    corpus = Corpus([], array.array("f"))
    for d, v, text, a in items:
        corpus.add(d, v, [text], [vec(a)])
    return corpus


def fill(repo, items, unit="U1"):
    repo.put_unit(unit)
    for d, v, text, a in items:
        if d not in {r.doc_id for r in repo.documents()}:
            repo.put_document(doc(unit, d))
        repo.ingest_version(unit, d, v, text.encode(), [passage(0, text, a)])


ITEMS = [("A", 1, "zebra zebra alpha", 0.1), ("B", 1, "zebra beta gamma delta", 0.2), ("C", 1, "zebra epsilon", 0.3),
         ("D", 1, "unrelated words", 0.4), ("E", 1, "zebra zebra zebra", 0.5), ("F", 1, "zeta eta", 0.6)]


def test_keyword_statistics_come_from_the_visible_passages_only(h):
    """The database's own text ranking would take its statistics from every passage. Here BM25 is computed over the passages of the
    visible versions only, so a hidden document stuffed with the query words moves no score, order or count."""
    visible = {"A": 1, "B": 1, "C": 1}
    fill(h.repo, ITEMS[:3])
    before = h.repo.index("U1").search("zebra alpha", visible, 10)
    h.repo.put_document(doc("U1", "HIDDEN"))
    h.repo.ingest_version("U1", "HIDDEN", 1, b"x", [passage(0, "zebra alpha " * 50, 0.0)])
    after = h.repo.index("U1").search("zebra alpha", visible, 10)
    assert [(p.doc, p.version, p.seq, s) for s, p in after] == [(p.doc, p.version, p.seq, s) for s, p in before]  # exactly the same, scores too
    # and it is the original search: the same documents, order and scores as the in-memory index of just these passages
    expected = reference(ITEMS[:3]).search("zebra alpha", visible, 10)
    assert [(p.doc, s) for s, p in after] == [(p.doc, s) for s, p in expected]
    # only A has `alpha`; A has `zebra` twice in three words, B and C once, in four and two
    assert [p.doc for _, p in after] == ["A", "C", "B"]
    assert h.repo.index("U1").search("zebra", {}, 10) == []


def test_vector_search_ranks_within_the_visible_versions(h):
    """The visible versions are part of the query: the nearest passages are chosen among them. Cosine to QUERY is 1 / sqrt(1 + a*a),
    so a smaller tilt is nearer: A (0.1) is nearest overall and F (0.6) is farthest."""
    fill(h.repo, ITEMS)
    index = h.repo.index("U1")
    visible = {"B": 1, "D": 1, "F": 1}  # the three nearest overall (A, B, C) are not all allowed
    top2 = index.similar(QUERY, visible, 2)
    assert [p.doc for _, p in top2] == ["B", "D"]  # the nearest two of the three, not the nearest two overall filtered afterwards
    assert [p.doc for _, p in index.similar(QUERY, visible, 10)] == ["B", "D", "F"]
    assert [s for s, _ in index.similar(QUERY, visible, 10)] == pytest.approx([1 / (1 + a * a) ** 0.5 for a in (0.2, 0.4, 0.6)], abs=1e-6)
    assert [p.doc for _, p in index.similar(QUERY, {"F": 1}, 1)] == ["F"]  # someone who may read only the farthest still gets it
    expected = reference(ITEMS).similar(QUERY, visible, 10)
    assert [p.doc for _, p in index.similar(QUERY, visible, 10)] == [p.doc for _, p in expected]
    assert index.similar(QUERY, {}, 5) == []


def test_a_version_without_vectors_is_not_found_by_meaning(h):
    h.repo.put_unit("U1")
    h.repo.put_document(doc("U1", "A"))
    h.repo.put_document(doc("U1", "B"))
    h.repo.ingest_version("U1", "A", 1, b"x", [passage(0, "has a vector", 0.1)])
    h.repo.ingest_version("U1", "B", 1, b"y", [passage(0, "has none")])
    visible = {"A": 1, "B": 1}
    assert [p.doc for _, p in h.repo.index("U1").similar(QUERY, visible, 10)] == ["A"]
    assert {p.doc for _, p in h.repo.index("U1").search("has", visible, 10)} == {"A", "B"}  # words still find both


def test_a_search_across_units_asks_each_unit_and_merges_by_rank(h):
    """U1 holds a1 (`zebra` three times) and a2 (`zebra alpha`); U2 holds b1 (`zebra`). Each unit ranks its own: U1 [a1, a2], U2 [b1].
    Merged with k=60: a1 1/61, b1 1/61, a2 1/62. a1 and b1 tie; the unit that sorts first (U1) goes first."""
    repo = h.repo
    for unit, d, text, a in (("U1", "a1", "zebra zebra zebra", 0.1), ("U1", "a2", "zebra alpha", 0.5), ("U2", "b1", "zebra", 0.3)):
        repo.put_unit(unit)
        repo.put_document(doc(unit, d))
        repo.ingest_version(unit, d, 1, text.encode(), [passage(0, text, a)])
    index = PartitionedIndex(repo)
    both = {"a1": 1, "a2": 1, "b1": 1}
    merged = index.search("zebra", both, 10)
    assert [p.doc for _, p in merged] == ["a1", "b1", "a2"]
    assert [s for s, _ in merged] == pytest.approx([1 / 61, 1 / 61, 1 / 62])
    assert [p.doc for _, p in index.search("zebra", both, 2)] == ["a1", "b1"]
    # by meaning: U1 [a1 (0.1), a2 (0.5)], U2 [b1 (0.3)]: the same shape, the same merge
    assert [p.doc for _, p in index.similar(QUERY, both, 10)] == ["a1", "b1", "a2"]
    # someone who reaches one unit only gets that unit's own ranking and scores, not rank scores
    only_u1 = index.search("zebra", {"a1": 1, "a2": 1}, 10)
    assert [p.doc for _, p in only_u1] == ["a1", "a2"] and only_u1[0][0] != pytest.approx(1 / 61)
    assert [(p.doc, s) for s, p in only_u1] == [(p.doc, s) for s, p in repo.index("U1").search("zebra", {"a1": 1, "a2": 1}, 10)]
    assert index.search("zebra", {}, 10) == []


def two_units(repo):
    """U1: a1 `zebra` three times, a2 `zebra alpha`. U2: b1 `zebra`, b2 `zebra beta`."""
    for unit, d, text, a in (("U1", "a1", "zebra zebra zebra", 0.1), ("U1", "a2", "zebra alpha", 0.5), ("U2", "b1", "zebra", 0.3), ("U2", "b2", "zebra beta", 0.6)):
        repo.put_unit(unit)
        repo.put_document(doc(unit, d))
        repo.ingest_version(unit, d, 1, text.encode(), [passage(0, text, a)])


def test_a_hidden_document_in_one_unit_cannot_change_visible_results_in_another(h):
    """Four visible documents in two units. Then a hidden document stuffed with the query words, and with the query's own vector, is
    added to each unit in turn. The merged keyword and meaning results (documents, order and scores) do not change."""
    two_units(h.repo)
    index = PartitionedIndex(h.repo)
    visible = {"a1": 1, "a2": 1, "b1": 1, "b2": 1}

    def results():
        return ([(p.doc, s) for s, p in index.search("zebra", visible, 10)], [(p.doc, s) for s, p in index.similar(QUERY, visible, 10)])

    before = results()
    # U1 ranks [a1, a2], U2 ranks [b1, b2]; fused with k=60: a1 1/61, b1 1/61, a2 1/62, b2 1/62; U1 sorts first
    assert [d for d, _ in before[0]] == ["a1", "b1", "a2", "b2"]
    assert [s for _, s in before[0]] == pytest.approx([1 / 61, 1 / 61, 1 / 62, 1 / 62])
    for unit in ("U2", "U1", "U2"):
        stuffed = f"H{unit}{len(h.repo.documents())}"
        h.repo.put_document(doc(unit, stuffed))
        h.repo.ingest_version(unit, stuffed, 1, b"x", [passage(0, "zebra " * 80 + "alpha beta", 0.0)])
        assert results() == before, unit


def test_keyword_statistics_are_per_unit_and_never_pooled(h):
    """A keyword score is computed from the passages of one unit. Adding many documents to U2 that use the query word changes no score
    in U1, and not the order of U1's documents among themselves in a merged result."""
    two_units(h.repo)
    index = PartitionedIndex(h.repo)
    alone = reference([("a1", 1, "zebra zebra zebra", 0.1), ("a2", 1, "zebra alpha", 0.5)]).search("zebra", {"a1": 1, "a2": 1}, 10)  # U1's passages and no others
    in_u1 = lambda: [(p.doc, s) for s, p in h.repo.index("U1").search("zebra", {"a1": 1, "a2": 1}, 10)]
    assert in_u1() == [(p.doc, s) for s, p in alone]
    visible = {"a1": 1, "a2": 1, "b1": 1, "b2": 1}
    for i in range(60):  # a crowd in U2, all visible, all full of the word: pooled statistics would move U1's scores
        name = f"crowd{i}"
        h.repo.put_document(doc("U2", name))
        h.repo.ingest_version("U2", name, 1, b"x", [passage(0, "zebra zebra zebra zebra common words", 0.9)])
        visible[name] = 1
    assert in_u1() == [(p.doc, s) for s, p in alone]  # U1's own scores are exactly what U1's passages alone give
    merged = [p.doc for _, p in index.search("zebra", visible, 100)]
    assert [d for d in merged if d in ("a1", "a2")] == ["a1", "a2"]  # and U1's documents keep their order among themselves
    assert merged[0] == "a1"  # rank 1 in U1 ties with rank 1 in U2 and U1 sorts first


# ── A world saved to a repository and loaded back ──────────────────────

WORKFLOW_FILES = {n: WORKFLOWS / f"{n}.yml" for n in ("knowledge_asset",)}


def test_the_ministry_world_gives_the_same_answers_from_the_repository(h, gw):
    """Save the world of test_access_control.py, load it back, and ask the service: every channel still says what the
    table says, and the history, states and guidance are the same."""
    save_world(gw.world, h.repo, gw.corpus, WORKFLOW_FILES)
    world2 = load_world(h.repo)
    service2 = DocumentService(world2, PartitionedIndex(h.repo))
    assert list(world2.documents) == list(gw.world.documents) and list(world2.users) == list(gw.world.users)
    for doc_id, original in gw.world.documents.items():
        assert world2.documents[doc_id].timeline.entries == original.timeline.entries
        assert world2.state_of(doc_id) == gw.world.state_of(doc_id)
    for user in PEOPLE:
        want = expected_for(user)
        assert {d.id: d.version for d in service2.list_documents(user)} == want, user
        assert {(x.doc, x.version) for x in service2.search(user, "zebra", 50)} == set(want.items()), user
        assert {(x.doc, x.version) for x in service2.similar(user, QUERY, 50)} == set(want.items()), user
        assert {(x.doc, x.version) for x in service2.hybrid(user, "zebra", QUERY, 50)} == set(want.items()), user
        for doc_id in ORACLE:
            if doc_id in want:
                got = service2.get_document(user, doc_id)
                assert got["version"] == want[doc_id] and got["guidance"] == gw.world.guidance(doc_id, user)
                assert [p.content for p in got["passages"]] == [p.content for p in gw.get_document(user, doc_id)["passages"]]
            else:
                with pytest.raises(NotFound):
                    service2.get_document(user, doc_id)
        original = [(x.doc, x.version, x.passage.seq, round(x.score, 9)) for x in gw.search(user, "zebra", 50)]
        assert [(x.doc, x.version, x.passage.seq, round(x.score, 9)) for x in service2.search(user, "zebra", 50)] == original, user  # scores too


SCENARIOS = sorted((ROOT / "scenarios").glob("*.yml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_world_survives_a_round_trip(h, path):
    """Run each scenario, save the world it ended in, load it back: who can open and find what, the states, the guidance each person
    reads, and the and the histories are the same."""
    import yaml
    from docflow.stores.workflow_definitions import _Loader

    sc = yaml.load(path.read_text(), Loader=_Loader)
    runner = scenario.Runner(sc, ROOT / "workflows")
    assert runner.run().ok
    world, corpus = runner.world, runner.corpus
    save_world(world, h.repo, corpus, {n: ROOT / "workflows" / f"{n}.yml" for n in sc["workflows"]})
    world2 = load_world(h.repo)
    assert list(world2.documents) == list(world.documents)
    assert world2.orgs == world.orgs and world2.users == world.users and world2.workspaces == world.workspaces
    for doc_id, d in world.documents.items():
        d2 = world2.documents[doc_id]
        assert (d2.workspace, d2.owner, d2.workflow, d2.invited) == (d.workspace, d.owner, d.workflow, d.invited), doc_id
        assert d2.timeline.entries == d.timeline.entries, doc_id
    service, service2 = DocumentService(world, corpus), DocumentService(world2, PartitionedIndex(h.repo))
    for user in world.users:
        for doc_id in world.documents:
            for search in (False, True):
                assert world2.readable_version(doc_id, user, for_search=search) == world.readable_version(doc_id, user, for_search=search), (user, doc_id)
            assert world2.guidance(doc_id) == world.guidance(doc_id), doc_id
            assert world2.guidance(doc_id, user) == world.guidance(doc_id, user), (user, doc_id)
            assert world2.state_of(doc_id) == world.state_of(doc_id)
        assert [(d.id, d.version, d.guidance) for d in service2.list_documents(user)] == [(d.id, d.version, d.guidance) for d in service.list_documents(user)], user
        assert [(x.doc, x.version) for x in service2.search(user, "text", 50)] == [(x.doc, x.version) for x in service.search(user, "text", 50)], user


def test_saving_again_adds_only_what_is_new_and_refuses_a_history_that_differs(h, gw):
    save_world(gw.world, h.repo, gw.corpus, WORKFLOW_FILES)
    unit = "U1"
    assert len(h.repo.events(unit, "revised")) == len(gw.world.documents["revised"].timeline.entries)
    save_world(gw.world, h.repo, gw.corpus, WORKFLOW_FILES)  # nothing new: nothing added
    assert len(h.repo.events(unit, "revised")) == len(gw.world.documents["revised"].timeline.entries)
    gw.world.act("shared", "ann", "upload")  # one new event on one document
    gw.corpus.add("shared", 2, ["shared zebra beta two"], [vec(0.7)])
    before = {d: len(h.repo.events(unit, d)) for d in ORACLE}
    save_world(gw.world, h.repo, gw.corpus, WORKFLOW_FILES)
    after = {d: len(h.repo.events(unit, d)) for d in ORACLE}
    assert {d: after[d] - before[d] for d in ORACLE} == {"draft": 0, "shared": 1, "public": 0, "unit": 0, "revised": 0}
    assert h.repo.read_version(unit, "shared", 2) == b"shared zebra beta two"
    # a world whose first event differs from the stored one is not an extension of it
    doc = gw.world.documents["draft"]
    doc.timeline = Timeline(doc.timeline.machine, (("version", 7), *doc.timeline.entries[1:]))
    with pytest.raises(HistoryDiverged):
        save_world(gw.world, h.repo, gw.corpus, WORKFLOW_FILES)


def test_keyword_search_loads_only_the_passages_of_the_visible_versions(pg, monkeypatch):
    """The database hands the application the candidate passages of the visible versions and no others: a hidden document's text
    is not even read out of the database for this search."""
    from docflow.stores import postgres_repository

    fetched = []

    class Spy(Corpus):
        def __init__(self, passages, *a, **k):
            fetched.append([(p.doc, p.version) for p in passages])
            super().__init__(passages, *a, **k)

    monkeypatch.setattr(postgres_repository, "Corpus", Spy)
    fill(pg.repo, ITEMS[:3])
    pg.repo.put_document(doc("U1", "HIDDEN"))
    pg.repo.ingest_version("U1", "HIDDEN", 1, b"x", [passage(0, "zebra alpha " * 50, 0.0)])
    pg.repo.put_document(doc("U1", "A2"))
    pg.repo.ingest_version("U1", "A2", 1, b"x", [passage(0, "zebra other version", 0.0)])
    pg.repo.ingest_version("U1", "A", 2, b"x", [passage(0, "zebra a second version of A", 0.0)])
    pg.repo.index("U1").search("zebra", {"A": 1, "B": 1}, 10)
    assert fetched == [[("A", 1), ("B", 1)]]  # not HIDDEN, not A2, and not version 2 of A
