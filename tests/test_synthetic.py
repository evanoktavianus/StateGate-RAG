"""The paper's synthetic dataset, replayed. Needs the synthetic data in eval/data/documents (skipped without it)."""

import os
from collections import Counter
from pathlib import Path

import pytest

from docflow import Machine
from docflow.stores.search_index import Corpus
from docflow.support.synthetic import build_world, load
from conftest import FIXTURES

DATA = Path(os.environ.get("DOCFLOW_DATA", str(Path(__file__).parent.parent / "eval/data/documents")))
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="export not present")

STATE = {"working": "working", "in_qa": "under_review", "released": "released"}


@pytest.fixture(scope="module")
def docs_identity():
    return load(DATA / "docs.json")


@pytest.fixture(scope="module")
def world(docs_identity):
    docs, identity = docs_identity
    return build_world({"policy_analysis": Machine.load(FIXTURES / "policy_analysis.yml")}, docs, identity)


@pytest.fixture(scope="module")
def corpus():
    return Corpus.load(DATA)


def test_every_document_ends_where_the_manifest_says(world, docs_identity):
    docs, _ = docs_identity
    wrong = []
    for d in docs:
        tl = world.documents[d["id"]].timeline
        if (tl.state, tl.version) != (STATE[d["state"]], int(d["n"])):
            wrong.append((d["id"], d["state"], d["n"], tl.state, tl.version))
    assert not wrong, wrong[:5]


def test_releases_match(world, docs_identity):
    docs, _ = docs_identity
    for d in docs:
        tl = world.documents[d["id"]].timeline
        assert tl.entries.count(("state", "released")) == len(d["releases"]), d["id"]  # one arrival in `released` per release of the manifest
        if d["releases"]:
            assert tl.version_at("released") == d["releases"][-1]["file_version"]


def test_every_version_has_passages(world, corpus):
    for doc in world.documents.values():
        for v in range(1, doc.timeline.version + 1):
            assert corpus.by_version.get((doc.id, v)), (doc.id, v)


def test_the_askers_of_each_document(world, docs_identity):
    """Each document's named askers get the answer its state and audience imply (derived here, not given by the dataset)."""
    docs, _ = docs_identity
    wrong, checked = [], 0
    for d in docs:
        a, tl = d["askers"], world.documents[d["id"]].timeline
        home = world.workspaces[d["workspace"]].org
        level = tl.choices().get("audience")
        for role in ("owner", "collaborator"):
            if role in a:
                checked += 1
                if world.readable_version(d["id"], a[role]) != tl.version:  # the people working on it open the newest
                    wrong.append((d["id"], role, "open"))
        for role, user in a.items():
            if role in ("owner", "collaborator") or d["personal"]:
                continue
            reaches = {
                "unit_member": level in ("unit", "organization", "public"),
                "org_member": level in ("organization", "public") and world.users[user].org == home or level == "public",
                "outsider": level == "public",
                "qa_member": True,  # the QA group is opened to every workspace, so they hold a role
            }.get(role)
            if reaches is None:
                continue
            checked += 1
            want = tl.version_at("released") if reaches else None
            got = world.readable_version(d["id"], user, for_search=True)
            if got != want:
                wrong.append((d["id"], role, level, got, want))
    assert checked > 3000
    assert not wrong, (len(wrong), wrong[:8])


def test_an_outsider_finds_only_public_releases(world, corpus, docs_identity):
    visible = world.visible("user37")  # external, no organization
    assert visible, "an outsider should find the public releases"
    for doc_id, version in visible.items():
        tl = world.documents[doc_id].timeline
        assert tl.choices()["audience"] == "public" and version == tl.version_at("released")
    for _, p in corpus.search("fiscal", visible, limit=20):
        assert (p.doc, p.version) in {(d, v) for d, v in visible.items()}


def test_visibility_summary(world, capsys):
    counts = {u: len(world.visible(u)) for u in ("user01", "user07", "user10", "user21", "user37")}
    with capsys.disabled():
        print("\ndocuments searchable per person:", counts)
    assert counts["user37"] < counts["user10"] < len(world.documents)
