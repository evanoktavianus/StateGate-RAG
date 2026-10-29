"""The channel sweep over the full synthetic world: sampled people × all documents × every channel."""

import os
from pathlib import Path

import pytest

from docflow import Machine
from docflow.stores.search_index import Corpus
from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.support.synthetic import build_world, load
from conftest import FIXTURES

DATA = Path(os.environ.get("DOCFLOW_DATA", str(Path(__file__).parent.parent / "eval/data/documents")))
pytestmark = pytest.mark.skipif(not (DATA / "chunks.jsonl").exists(), reason="corpus not built")

PEOPLE = ["user01", "user03", "user07", "user10", "user21", "user37"]  # owner, unit member, QA, other unit, org, outsider


@pytest.fixture(scope="module")
def gw():
    docs, identity = load(DATA / "docs.json")
    world = build_world({"policy_analysis": Machine.load(FIXTURES / "policy_analysis.yml")}, docs, identity)
    return DocumentService(world, Corpus.load(DATA, embeddings=False))


def test_every_version_has_simple_chunks_and_a_canary(gw):
    for doc in gw.world.documents.values():
        for v in range(1, doc.timeline.version + 1):
            rows = gw.corpus.by_version.get((doc.id, v))
            assert rows, (doc.id, v)
            assert any(f"CANARY-{doc.id}-v{v}." in gw.corpus.passages[i].content for i in rows), (doc.id, v)


def test_list_get_and_search_agree_with_the_workflow_for_everyone_sampled(gw):
    world, bad = gw.world, []
    for user in PEOPLE:
        listed = {d.id: d.version for d in gw.list_documents(user)}
        for doc_id in world.documents:
            opened = world.readable_version(doc_id, user, for_search=False)
            if listed.get(doc_id) != opened:
                bad.append((user, doc_id, "list"))
            try:
                got = gw.get_document(user, doc_id)["version"]
            except NotFound:
                got = None
            if got != opened:
                bad.append((user, doc_id, "get"))
    assert not bad, bad[:10]


def test_a_canary_search_never_returns_a_version_the_person_may_not_find(gw):
    """Each version carries its own canary, so a hit's text names the version it came from."""
    world, leaks = gw.world, []
    sample = list(world.documents)[::25]
    for user in PEOPLE:
        for doc_id in sample:
            for h in gw.search(user, f"CANARY {doc_id}", limit=10):
                if world.readable_version(h.doc, user, for_search=True) != h.version:
                    leaks.append((user, h.doc, h.version))
    assert not leaks, leaks[:5]
