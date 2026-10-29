"""The synthetic set replayed under the nine workflows: documents in every state, in every unit, with attribute values on both sides of every condition."""

import json
import os
from pathlib import Path

import pytest

from evalkit import corpus as C
from evalkit import paths as P
from evalkit.oracle import Cast, Doc
from evalkit.oracle.policy import load_policy
from evalkit.worlds import load_spec
from docflow.stores.search_index import Corpus

ROOT = Path(__file__).parent.parent
DATA = Path(os.environ.get("DOCFLOW_DATA", str(Path(__file__).parent.parent / "eval/data/documents")))


def mini_docs(per_unit=120):
    docs = []
    for u in C.DOC_UNITS:
        for i in range(per_unit):
            docs.append({"id": f"doc-{u}-{i:03d}", "unit": u, "n": (3, 3, 4, 2, 1)[i % 5]})
    return docs


def mini_corpus(docs):
    c = Corpus([])
    for d in docs:
        for v in range(1, d["n"] + 1):
            c.add(d["id"], v, [f"text of {d['id']} version {v}"])
    return c


@pytest.fixture(scope="module")
def built():
    docs = mini_docs()
    return C.build(docs, mini_corpus(docs)), docs


def test_the_fixture_is_moved_into_a_unit_and_nobody_is_shared_between_units():
    spec = load_spec("policy_analysis_other_unit_single")
    a, b = C.unit_spec(spec, "U1"), C.unit_spec(spec, "U2")
    assert a.people[a.owner]["unit"] == "U1" and b.people[b.owner]["unit"] == "U2" and a.owner == "U1.an" and b.owner == "U2.an"
    assert not set(a.people) & set(b.people)
    assert {p["org"] for p in a.people.values() if p["org"]} == {"O1", "O2"}  # its own organization and the other
    assert a.people["U1.req"]["org"] == "O2" and a.people["U1.req"]["unit"] in C.UNITS["O2"]
    assert a.attrs["requesting_unit"] == [a.people["U1.req"]["unit"], a.people["U1.req2"]["unit"]]  # the attribute follows the units
    assert a.attrs["group"] == ["g1-u1", "g2-u1"] and a.people["U1.gh"]["groups"] == ["g1-u1"]  # a group label belongs to its unit
    u4 = C.unit_spec(spec, "U4")
    assert u4.people[u4.owner]["org"] == "O2" and u4.people["U4.req"]["org"] == "O1"


def test_every_document_is_placed_once_and_keeps_its_unit(built):
    b, docs = built
    assert sorted(a["doc"] for a in b.assignment) == sorted(d["id"] for d in docs)
    unit = {d["id"]: d["unit"] for d in docs}
    assert all(a["unit"] == unit[a["doc"]] for a in b.assignment)


def test_every_workflow_has_documents_in_every_state_in_every_unit(built):
    b, _ = built
    cov = b.coverage()
    for wf in P.WORKFLOWS:
        model = P.generate(wf).model
        states = cov["workflows"][wf]["states"]
        assert set(states) == set(model.states), wf
        for s, per_unit in states.items():
            assert set(per_unit) == set(C.DOC_UNITS), (wf, s)  # a document of every unit


def test_three_units_of_one_organization_and_a_second_organization(built):
    b, _ = built
    cov = b.coverage()
    assert cov["units_per_organization"] == {"O1": ["U1", "U2", "U3"], "O2": ["U4"]}


def test_attribute_values_are_on_both_sides_of_every_condition(built):
    b, _ = built
    wf = b.coverage()["workflows"]
    assert set(wf["card7_official_document"]["attribute_values"]["secrecy"]) == {"B", "SR", "R", "T"}  # `B` against the rest
    assert len(wf["policy_analysis_other_unit_single"]["attribute_values"]["requesting_unit"]) >= 2  # a unit that matches and one that does not
    for w in ("card2_policy_analysis_individual", "policy_analysis_team_single", "policy_analysis_other_unit_single", "card5_policy_analysis_sensitive"):
        assert len(wf[w]["attribute_values"]["group"]) >= 2, w
    levels = {a["choices"].get("level") for a in b.assignment if a["workflow"] == "knowledge_asset" and a["state"] == "disseminated"}
    assert levels == {"named", "unit", "organization", "public"}


def test_the_oracle_gives_every_person_the_answer_the_replayed_world_gives(built):
    """An independent check of the replay: the oracle follows the same steps and must serve the same version to everyone."""
    b, _ = built
    seen, bad = 0, []
    for wf, world in b.worlds.items():
        people = {pid: {"org": u.org, "unit": u.unit, "groups": list(u.groups), "positions": list(u.positions), "attrs": dict(u.attrs), "org_admin": u.org_admin}
                  for pid, u in world.users.items()}
        cast = Cast.from_people(people)
        policy = load_policy(P.generate(wf).spec.workflow_path)
        for a in [x for x in b.assignment if x["workflow"] == wf][::7]:
            spec, cell = b.placed[a["doc"]]
            ws = spec.workspaces[spec.workspace]
            create = {k: v for k, v in spec.assign.items() if policy.create and k in (policy.create.get("assign") or {})}
            od = Doc.create(policy, cast, ws, ws.get("org"), a["doc"], spec.owner, spec.creator, create or None, cell.attrs or None, spec.invited)
            for s in cell.path.steps[: cell.index]:
                assert od.apply(s.actor, s.action, C.rebind(s.input, a["doc"], od.version), s.event).allowed, (a["doc"], s)
            for person in cast:
                for find in (False, True):
                    seen += 1
                    got = world.readable_version(a["doc"], person, for_search=find)
                    if got != od.access(person, find=find):
                        bad.append((a["doc"], person, find, got, od.access(person, find=find)))
    assert seen > 5000 and not bad, bad[:5]


@pytest.mark.skipif(not (DATA / "docs.json").exists() or not (DATA / "chunks.jsonl").exists(), reason="the synthetic set is not here (eval/data/documents)")
def test_the_real_set_is_replayed():
    docs = json.loads((DATA / "docs.json").read_text())["docs"]
    corpus = Corpus.load(DATA, embeddings=False)
    b = C.build(docs, corpus)
    assert len(b.assignment) == 1024 and b.coverage()["units_per_organization"] == {"O1": ["U1", "U2", "U3"], "O2": ["U4"]}


def test_a_recorded_placement_is_replayed_and_a_dropped_workflow_leaves_its_documents_out(built):
    """The paper's placement was made over nine workflows; with a workflow dropped, its documents are left out and nobody else moves."""
    b, docs = built
    dropped = "card7_official_document"
    rows = [dict(a) for a in b.assignment if a["workflow"] != dropped]
    workflows = [w for w in b.worlds if w != dropped]
    again = C.build(docs, mini_corpus(docs), workflows=workflows, placement=rows)
    assert [(a["doc"], a["workflow"], a["unit"], a["state"], a["path"], a["steps"]) for a in again.assignment] == \
        [(a["doc"], a["workflow"], a["unit"], a["state"], a["path"], a["steps"]) for a in b.assignment if a["workflow"] != dropped]
    assert dropped not in again.worlds and len(again.assignment) < len(b.assignment)


def test_the_placement_of_the_runs_of_the_paper_is_in_the_repository():
    """E5 to E7 put the documents where this file says; without it the documents are spread differently."""
    rows = json.loads(C.PLACEMENT.read_text())
    assert len(rows) == 1100 and len({r["doc"] for r in rows}) == 1100 and {r["workflow"] for r in rows} <= set(P.WORKFLOWS)
