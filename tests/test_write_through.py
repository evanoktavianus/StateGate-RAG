"""The world writes through to a repository after every accepted action, on both implementations.

What is checked is what a reader of the repository alone would see: a world thrown away and loaded back, a refused action that
wrote nothing, an action that failed halfway that left nothing, and a world loaded from the repository that goes on writing."""

from pathlib import Path

import pytest
import yaml

from docflow import Machine, Refused
from docflow.document_service.service import DocumentService
from docflow.document_service.retrieve import PartitionedIndex
from docflow.stores.repository import BadRecord, PassageIn
from docflow.stores.workflow_definitions import _Loader
from docflow.stores.world import World
from docflow.stores.world_store import MissingContent, WriteThrough, load_world
from docflow.support import scenario
from conftest import WORKFLOWS
from conftest_ministry import world  # noqa: F401
from test_repository import h, pg, vec  # noqa: F401  (the repository fixture, on both implementations)

ROOT = Path(__file__).parent.parent
FILES = {n: WORKFLOWS / f"{n}.yml" for n in ("knowledge_asset",)}
COMMITTEE = ["c1", "c2", "c3"]


def provider(doc_id, version):
    return [PassageIn(0, (0, 0), "", f"{doc_id} v{version} text", tuple(vec(0.1)))]


@pytest.fixture
def live(world, h):
    """The ministry world, writing through to the repository; `d1` is a working draft of ann's."""
    world.store = WriteThrough(h.repo, FILES, provider)
    world.add_document("d1", "ws", "ann", "knowledge_asset")  # the first flush writes the people, the unit, the workflows and the document
    return world, h


def counts(repo, subjects):
    return {s: len(repo.events("U1", s)) for s in subjects}


def watch(monkeypatch, repo):
    """Counts every call that would write, so a test can say nothing was written."""
    calls = []
    for name in ("append_event", "put_document", "put_person", "put_workspace", "put_unit", "put_org", "ingest_version",
                 "put_workflow", "transaction"):
        original = getattr(repo, name)
        monkeypatch.setattr(repo, name, lambda *a, _o=original, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
    return calls


# ── A world thrown away and loaded back ────────────────────────────────

def snapshot(world, service):
    """Everything a reader can ask of a world, as plain data."""
    out = {"documents": list(world.documents), "states": {}, "entries": {}, "open": {}, "guidance": {}, "list": {}, "search": {}}
    for doc_id, d in world.documents.items():
        out["states"][doc_id] = world.state_of(doc_id)
        out["entries"][doc_id] = d.timeline.entries
    for user in world.users:
        for doc_id in world.documents:
            out["open"][(user, doc_id)] = (world.readable_version(doc_id, user, for_search=False), world.readable_version(doc_id, user, for_search=True))
            out["guidance"][(user, doc_id)] = world.guidance(doc_id)
        out["list"][user] = [(d.id, d.version, d.state, d.guidance) for d in service.list_documents(user)]
        out["search"][user] = [(x.doc, x.version) for x in service.search(user, "text", 50)]
    return out


@pytest.mark.parametrize("path", sorted((ROOT / "scenarios").glob("*.yml")), ids=lambda p: p.stem)
def test_a_world_that_writes_through_loads_back_from_the_repository_alone(h, path):
    """Run a scenario with the world writing through, throw the world away, load a new one from the repository (no workflow file, no
    corpus), and get the same states, versions, guidance and answers."""
    sc = yaml.load(path.read_text(), Loader=_Loader)
    runner = scenario.Runner(sc, ROOT / "workflows", repository=h.repo)
    assert runner.run().ok  # the scenario's own expectations hold on the live world
    assert runner.world.store.in_sync(runner.world)  # nothing is waiting to be saved
    before = snapshot(runner.world, DocumentService(runner.world, runner.corpus))
    del runner
    reloaded = load_world(h.repo)
    after = snapshot(reloaded, DocumentService(reloaded, PartitionedIndex(h.repo)))
    assert after == before


# ── A refused action writes nothing ────────────────────────────────────

@pytest.mark.parametrize("attempt", [
    lambda w: w.act("d1", "ext", "submit", input={"committee": COMMITTEE}),    # someone with no standing
    lambda w: w.act("d1", "ann", "nonsense"),                                   # an action the state does not have
    lambda w: w.act("d1", "ann", "approve"),                                    # the right person, the wrong state
], ids=["no standing", "no such action", "wrong state"])
def test_a_refused_action_writes_nothing(live, monkeypatch, attempt):
    world, h = live
    before_events, before_entries = counts(h.repo, ["d1"]), world.documents["d1"].timeline.entries
    calls = watch(monkeypatch, h.repo)
    with pytest.raises(Refused):
        attempt(world)
    assert calls == []  # no write, not even an empty transaction
    assert counts(h.repo, ["d1"]) == before_events and world.documents["d1"].timeline.entries == before_entries
    assert world.store.in_sync(world)


def test_an_accepted_action_writes_its_new_entries_and_nothing_else(live, monkeypatch):
    world, h = live
    calls = watch(monkeypatch, h.repo)
    world.act("d1", "ann", "submit", input={"committee": COMMITTEE})
    # one transaction; the document's new history entries; no person, no workflow, no version
    assert calls.count("transaction") == 1 and set(calls) <= {"transaction", "append_event"}, calls
    assert [e["entry"] for e in h.repo.events("U1", "d1")][-1] == ["state", "quality_assurance"]


# ── A failure halfway leaves nothing ───────────────────────────────────

def released(world):
    world.act("d1", "ann", "submit", input={"committee": COMMITTEE})
    world.act("d1", "c1", "set_level", input={"level": "public", "readers": []})
    world.act("d1", "c1", "approve")
    world.act("d1", "c2", "approve")


def fail_on_nth_call(monkeypatch, repo, name, n, error):
    original, seen = getattr(repo, name), []

    def wrapper(*a, **k):
        seen.append(1)
        if len(seen) == n:
            raise error
        return original(*a, **k)

    monkeypatch.setattr(repo, name, wrapper)


def test_a_failure_halfway_through_an_action_leaves_no_partial_entries(live, monkeypatch):
    """`submit` adds two entries: the committee chosen, then the move to quality assurance. The repository fails while the second is
    being appended."""
    world, h = live
    before = counts(h.repo, ["d1"])
    entries, state = world.documents["d1"].timeline.entries, world.state_of("d1")
    fail_on_nth_call(monkeypatch, h.repo, "append_event", 2, RuntimeError("the disk is full"))
    with pytest.raises(RuntimeError, match="disk is full"):
        world.act("d1", "ann", "submit", input={"committee": COMMITTEE})
    assert counts(h.repo, ["d1"]) == before  # the first entry that was appended is gone too
    assert world.documents["d1"].timeline.entries == entries and world.state_of("d1") == state  # the world agrees with the repository
    assert world.store.in_sync(world) and load_world(h.repo).state_of("d1") == state
    monkeypatch.undo()
    world.act("d1", "ann", "submit", input={"committee": COMMITTEE})  # and the same action goes through once the failure is gone
    assert world.state_of("d1") == "quality_assurance" and load_world(h.repo).state_of("d1") == "quality_assurance"


def test_a_new_version_is_written_with_its_file_hash_and_passages_or_not_at_all(live, monkeypatch):
    world, h = live
    assert [(v.version, v.size) for v in h.repo.versions("U1", "d1")] == [(1, len(b"d1 v1 text"))]
    world.act("d1", "ann", "upload")
    v2 = h.repo.versions("U1", "d1")[1]
    assert (v2.version, v2.size) == (2, len(b"d1 v2 text")) and h.repo.read_version("U1", "d1", 2) == b"d1 v2 text"
    assert [p.content for p in h.repo.index("U1").passages_of("d1", 2)] == ["d1 v2 text"]
    assert [e["entry"] for e in h.repo.events("U1", "d1")][-1] == ["version", 2]
    # version 3 with passages the repository will refuse (a vector of the wrong size), after the entry was appended
    before = counts(h.repo, ["d1"])
    monkeypatch.setattr(world.store.writer, "passages_for", lambda doc, v: [PassageIn(0, (0, 0), "", "bad", (1.0, 2.0))])
    with pytest.raises(BadRecord):
        world.act("d1", "ann", "upload")
    assert counts(h.repo, ["d1"]) == before and [v.version for v in h.repo.versions("U1", "d1")] == [1, 2]
    assert not h.has_file("U1", "d1", 3) and h.repo.index("U1").passages_of("d1", 3) == []
    assert world.documents["d1"].timeline.version == 2  # the world was put back too
    monkeypatch.undo()
    world.act("d1", "ann", "upload")  # and version 3 can be made afterwards
    assert h.repo.read_version("U1", "d1", 3) == b"d1 v3 text" and world.documents["d1"].timeline.version == 3


def test_a_version_the_host_cannot_supply_content_for_is_refused_and_undone(live, monkeypatch):
    world, h = live
    before = counts(h.repo, ["d1"])
    monkeypatch.setattr(world.store.writer, "passages_for", lambda doc, v: None)
    with pytest.raises(MissingContent, match="d1 version 2"):
        world.act("d1", "ann", "upload")
    assert counts(h.repo, ["d1"]) == before and world.documents["d1"].timeline.version == 1 and not h.has_file("U1", "d1", 2)


def test_a_file_left_over_from_a_run_that_died_does_not_block_the_version(pg, live):
    """A file with no row is left over from a run that died before it committed; nothing refers to it."""
    world, h = live
    from docflow.stores.repository import relative_path
    leftover = h.root / relative_path("U1", "d1", 2, ".txt")
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"junk from a crash")
    world.act("d1", "ann", "upload")
    assert h.repo.read_version("U1", "d1", 2) == b"d1 v2 text"


# ── A world loaded from the repository goes on writing ─────────────────

def test_a_world_loaded_from_the_repository_goes_on_writing_through(live):
    """A restart in the middle: the live world is dropped, a new one is loaded from the repository alone and given a store; what it does
    next is appended to what was there, once, and a third load sees all of it."""
    world, h = live
    released(world)
    first = counts(h.repo, ["d1"])
    state_before = world.state_of("d1")
    del world
    world2 = load_world(h.repo)
    assert world2.state_of("d1") == state_before
    world2.store = WriteThrough(h.repo, FILES, provider)
    world2.act("d1", "c3", "approve")
    after = counts(h.repo, ["d1"])
    assert after["d1"] > first["d1"] and world2.state_of("d1") == "disseminated"
    world3 = load_world(h.repo)
    assert world3.documents["d1"].timeline.entries == world2.documents["d1"].timeline.entries and world3.state_of("d1") == "disseminated"
    world2.act("d1", "ann", "start_update")
    # knowledge_asset.yml: `start_update` in `disseminated` goes back to `working` and makes no version; `upload` then stays in `working`.
    # Each adds one entry. Nothing is written twice.
    assert counts(h.repo, ["d1"])["d1"] == after["d1"] + 1
    assert [e["entry"] for e in h.repo.events("U1", "d1")][-1:] == [["state", "working"]]
    world2.act("d1", "ann", "upload")
    assert counts(h.repo, ["d1"])["d1"] == after["d1"] + 2
    assert [e["entry"] for e in h.repo.events("U1", "d1")][-1:] == [["version", 2]]
    assert h.repo.read_version("U1", "d1", 2) == b"d1 v2 text"


def regulation_live(h):
    """Regulations of unit U1 writing through to the repository: T (in force) and S1 (in review), and W, a working draft."""
    files = {"regulation_single": WORKFLOWS / "regulation_single.yml"}
    w = World({"regulation_single": Machine.load(files["regulation_single"])}, WriteThrough(h.repo, files, provider))
    w.add_org("O1")
    w.add_workspace("ws", "O1")
    w.add_user("lead", "O1", unit="U1", positions=("leadership",))
    w.add_user("team", "O1", unit="U1")
    for rid in ("T", "S1"):
        w.add_document(rid, "ws", "team", "regulation_single", creator="lead", assign={"drafting_team": ["team"], "invited": []})
        w.act(rid, "team", "submit_for_review")
    w.fire_event("T", "promulgate", "promulgated")
    w.add_document("W", "ws", "team", "regulation_single", creator="lead", assign={"drafting_team": ["team"], "invited": []})  # still a working draft: its team may upload
    return w


def test_an_outside_event_is_one_accepted_action_and_one_transaction(h, monkeypatch):
    """`fire_event` acts on the document through `act`, one call inside another. It is still one flush and one transaction, and the
    repository holds the move, so a reload agrees."""
    w = regulation_live(h)
    before = counts(h.repo, ["T", "S1"])
    calls, flushes = watch(monkeypatch, h.repo), []
    original = w.store.flush
    monkeypatch.setattr(w.store, "flush", lambda world: (flushes.append(1), original(world))[1])
    w.fire_event("S1", "promulgate", "promulgated")
    w.fire_event("T", "amend", "amendment_in_force")
    assert flushes == [1, 1] and calls.count("transaction") == 2, (flushes, calls)  # the call inside did not write by itself
    after = counts(h.repo, ["T", "S1"])
    assert after["S1"] > before["S1"] and after["T"] > before["T"]
    assert (w.state_of("T"), w.state_of("S1")) == ("amended", "in_force")
    monkeypatch.undo()
    again = load_world(h.repo)
    assert (again.state_of("T"), again.state_of("S1")) == ("amended", "in_force")


def test_a_rejected_event_writes_nothing(h, monkeypatch):
    """An event the state does not leave through is rejected and logged, and changes nothing in the repository."""
    w = regulation_live(h)
    before, calls = counts(h.repo, ["T"]), watch(monkeypatch, h.repo)
    assert w.fire_event("T", "promulgate", "promulgated") is None  # T is in force already
    assert calls == [] and counts(h.repo, ["T"]) == before and w.log[-1]["kind"] == "rejected"


def test_a_failure_after_the_file_was_written_takes_the_file_back(h, monkeypatch):
    """Uploading a new version of a working draft's text writes the file and its rows; then the connection drops. The file is removed too."""
    w = regulation_live(h)
    versions, events = h.repo.versions("U1", "W"), counts(h.repo, ["W"])
    original = h.repo.ingest_version

    def then_fail(*a, **k):
        original(*a, **k)
        raise RuntimeError("the connection dropped")

    monkeypatch.setattr(h.repo, "ingest_version", then_fail)
    with pytest.raises(RuntimeError, match="connection dropped"):
        w.act("W", "team", "upload")
    assert h.repo.versions("U1", "W") == versions and counts(h.repo, ["W"]) == events
    assert not h.has_file("U1", "W", 2) and h.repo.index("U1").passages_of("W", 2) == []
    assert w.documents["W"].timeline.version == 1  # the world was put back too
    monkeypatch.undo()
    w.act("W", "team", "upload")
    assert h.repo.read_version("U1", "W", 2) == b"W v2 text"


def test_a_call_that_changes_nothing_writes_nothing(live, monkeypatch):
    world, h = live
    calls = watch(monkeypatch, h.repo)
    world.add_org("O1")  # already there
    world.add_user("ann", "O1", unit="U1")  # the same person again
    assert calls == []
