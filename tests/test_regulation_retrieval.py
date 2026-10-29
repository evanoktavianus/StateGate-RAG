"""Retrieval only, no model: the regulation set (300 regulations, each one document, its text) through every channel.

The expected access comes from `oracle_regulation` (the table). The supporting records of the generated
set are in the index but not in the world, so no one may read them. Three properties:

  1. the engine's view of every person equals the oracle's, over every person and every document;
  2. list and get-by-id agree with it, and a refused document looks like one that does not exist;
  3. the 200 questions, asked as the 7 personas (1,400 searches), return nothing the oracle forbids (leakage)
     and exactly what a search over the oracle's own visible set returns (no over-restriction).

"""

import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from docflow import Machine
from docflow.support import regulations as R
from docflow.stores.search_index import Corpus
from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from conftest import WORKFLOWS
import oracle_regulation as oracle

DATA = Path(os.environ.get("DOCFLOW_REGULATIONS", str(Path(__file__).parent.parent / "eval/data/regulations")))
pytestmark = pytest.mark.skipif(not (DATA / "questions.json").exists(), reason="regulation corpus not built")


@pytest.fixture(scope="module")
def setup():
    plan, docs = R.load(DATA)
    world = R.build_world({"regulation_single": Machine.load(WORKFLOWS / "regulation_single.yml")}, plan, docs)
    versions = {}
    for d in docs:
        if d["kind"] == "text":
            versions[d["doc"]] = max(versions.get(d["doc"], 0), d["version"])
    return plan, world, DocumentService(world, Corpus.load(DATA, embeddings=False)), oracle.visible(plan, versions), versions


def test_every_regulation_ends_in_the_state_the_plan_gives(setup):
    plan, world, *_ = setup
    assert [(r["id"], world.state_of(R.text_id(r))) for r in plan["regulations"]] == [(r["id"], r["state"]) for r in plan["regulations"]]
    assert not [l for l in world.log if l["kind"] in ("rejected", "refused")]


def test_the_view_of_every_person_equals_the_table(setup):
    plan, world, gw, expected, _ = setup
    bad = []
    for user, want in expected.items():
        for for_search in (False, True):
            got = world.visible(user, for_search=for_search)
            if got != want:
                bad.append((user, for_search, sorted(set(got) ^ set(want))[:5]))
    assert not bad, bad[:5]


def test_list_and_get_agree_and_a_refused_document_looks_missing(setup):
    plan, world, gw, expected, versions = setup
    bad = []
    for user, want in expected.items():
        listed = {d.id: d.version for d in gw.list_documents(user)}
        if listed != want:
            bad.append((user, "list"))
        for doc in versions:
            try:
                got = gw.get_document(user, doc)["version"]
            except NotFound:
                got = None
            if got != want.get(doc):
                bad.append((user, doc, "get"))
        with pytest.raises(NotFound):
            gw.get_document(user, "reg-9999/text")  # a document that does not exist is refused in the same way
    assert not bad, bad[:5]


def test_the_200_questions_leak_nothing_and_lose_nothing(setup):
    plan, world, gw, expected, versions = setup
    regs = {r["id"]: r for r in plan["regulations"]}
    questions = json.loads((DATA / "questions.json").read_text())
    leaks, lost, rows = [], [], []
    tally = defaultdict(lambda: defaultdict(int))
    by_attack = defaultdict(lambda: defaultdict(int))  # the adversarial questions, per kind of attack and person
    for q in questions:
        for persona in R.PERSONAS:
            user = R.persona_user(regs[q["reg"]], persona)
            hits = gw.search(user, q["text"], limit=10)
            got = [(h.doc, h.version) for h in hits]
            leaks += [(q["id"], persona, d, v) for d, v in got if expected[user].get(d) != v]
            want = [(p.doc, p.version) for _, p in gw.corpus.search(q["text"], expected[user], 10)]  # a search over the table's own set
            if got != want:
                lost.append((q["id"], persona))
            allowed = [d for d in q["evidence"] if d in expected[user]]
            at5, at10 = [d for d, _ in got[:5]], [d for d, _ in got]
            has = any(d in at10 and q["answer"] and any(q["answer"] in h.passage.content for h in hits if h.doc == d) for d in allowed) if q["answer"] else None
            cell = tally[(q["state"], q["kind"], persona)]
            cell["queries"] += 1
            cell["leaks"] += sum(1 for d, v in got if expected[user].get(d) != v)
            if q.get("attack"):
                atk = by_attack[(q["attack"], persona)]
                atk["queries"] += 1
                atk["passages_returned"] += len(got)
                atk["leaks"] += sum(1 for d, v in got if expected[user].get(d) != v)
                atk["target_visible"] += any(d.split("/")[0] == q["reg"] for d in expected[user])  # the person may see something of the targeted regulation
                atk["target_returned"] += any(d.split("/")[0] == q["reg"] for d, _ in got)
            if q["evidence"]:
                cell["with_visible_evidence"] += bool(allowed)
                cell["found_at_5"] += bool(allowed and any(d in at5 for d in allowed))
                cell["found_at_10"] += bool(allowed and any(d in at10 for d in allowed))
                cell["has_answer"] += bool(has)
            rows.append(dict(q=q["id"], persona=persona, user=user, returned=got, allowed_evidence=allowed))
    results = {"queries": len(rows), "leaks": len(leaks), "differs_from_table_search": len(lost),
               "cells": [dict(state=s, kind=k, persona=p, **c) for (s, k, p), c in sorted(tally.items())],
               "by_attack": [dict(attack=a, persona=p, **c) for (a, p), c in sorted(by_attack.items())]}
    assert len(rows) == 1400
    assert not leaks, leaks[:5]
    assert not lost, lost[:5]


# ── search by meaning, and the two together ───────────────────────────────

def test_the_vector_search_and_the_two_together_also_leak_nothing(setup):
    import array
    plan, world, gw, expected, versions = setup
    gw = DocumentService(world, Corpus.load(DATA, embeddings=True))
    regs = {r["id"]: r for r in plan["regulations"]}
    questions = json.loads((DATA / "questions.json").read_text())
    raw = array.array("f")
    raw.frombytes((DATA / "questions.f32").read_bytes())
    vec = {q["id"]: list(raw[i * 1536:(i + 1) * 1536]) for i, q in enumerate(questions)}
    leaks, lost = [], []
    tally = defaultdict(lambda: defaultdict(int))
    by_attack = defaultdict(lambda: defaultdict(int))
    for q in questions:
        for persona in R.PERSONAS:
            user = R.persona_user(regs[q["reg"]], persona)
            by_meaning = [(h.doc, h.version) for h in gw.similar(user, vec[q["id"]], limit=10)]
            by_words = [(h.doc, h.version) for h in gw.search(user, q["text"], limit=10)]
            both = [(h.doc, h.version) for h in gw.hybrid(user, q["text"], vec[q["id"]], limit=10)]  # fused by passage, as the service does
            ref = [(p.doc, p.version) for _, p in gw.corpus.similar(vec[q["id"]], expected[user], 10)]  # the table's own set
            if by_meaning != ref:
                lost.append((q["id"], persona))
            allowed = [d for d in q["evidence"] if d in expected[user]]
            for mode, got in (("meaning", by_meaning), ("words", by_words), ("both", both)):
                leaks += [(mode, q["id"], persona, d, v) for d, v in got if expected[user].get(d) != v]
                cell = tally[(mode, q["kind"], persona)]
                cell["queries"] += 1
                if q.get("attack"):
                    atk = by_attack[(mode, q["attack"], persona)]
                    atk["queries"] += 1
                    atk["leaks"] += sum(1 for d, v in got if expected[user].get(d) != v)
                if q["evidence"]:
                    cell["with_visible_evidence"] += bool(allowed)
                    cell["found_at_5"] += bool(allowed and any(d in [x for x, _ in got[:5]] for d in allowed))
                    cell["found_at_10"] += bool(allowed and any(d in [x for x, _ in got] for d in allowed))
    results = {"searches": 3 * len(questions) * len(R.PERSONAS), "leaks": len(leaks), "meaning_differs_from_table_search": len(lost),
               "cells": [dict(mode=m, kind=k, persona=p, **c) for (m, k, p), c in sorted(tally.items())],
               "by_attack": [dict(mode=m, attack=a, persona=p, **c) for (m, a, p), c in sorted(by_attack.items())]}
    assert not leaks, leaks[:5]
    assert not lost, lost[:5]
