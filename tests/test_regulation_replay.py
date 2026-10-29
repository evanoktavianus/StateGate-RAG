"""The replay of a regulation plan into single documents, on a small plan (the full set needs eval/data/regulations, see
test_regulation_retrieval.py). Each regulation is one document, its text; the host fires the events that follow a promulgation."""

import pytest

from docflow import Machine
from docflow.support import regulations as R
from conftest import WORKFLOWS
import oracle_regulation as oracle


def reg(id, state, org="O1", unit="LEG", amending=(), repeals=None, amends=()):
    return {"id": id, "state": state, "org": org, "unit": unit, "lang": "en", "amending": list(amending), "repeals": repeals, "amends": list(amends)}


PLAN = {"regulations": [
    reg("reg-1", "working_draft"),
    reg("reg-2", "review"),
    reg("reg-3", "amended", amends=[{"by": "reg-4"}]),
    reg("reg-4", "in_force", amending=[{"target": "reg-3", "article": 3}]),
    reg("reg-5", "repealed", org="O2", unit="BPP", repeals="reg-6"),
    reg("reg-6", "in_force", org="O2", unit="BPP"),
    reg("reg-7", "in_force"),
]}
VERSIONS = {"reg-1": 2, "reg-2": 1, "reg-3": 3, "reg-4": 1, "reg-5": 1, "reg-6": 2, "reg-7": 1}
DOCS = [{"doc": f"{r}/text", "kind": "text", "version": v} for r, n in VERSIONS.items() for v in range(1, n + 1)]
DOCS += [{"doc": "reg-1/records", "kind": "records", "version": 1}]  # not part of the single document: ignored


@pytest.fixture(scope="module")
def world():
    return R.build_world({"regulation_single": Machine.load(WORKFLOWS / "regulation_single.yml")}, PLAN, DOCS)


def test_every_regulation_ends_in_the_state_the_plan_gives(world):
    assert [(r["id"], world.state_of(R.text_id(r))) for r in PLAN["regulations"]] == [(r["id"], r["state"]) for r in PLAN["regulations"]]
    assert not [e for e in world.log if e["kind"] in ("rejected", "refused")]
    assert "reg-1/records" not in world.documents


def test_every_text_has_all_its_versions(world):
    assert {d: world.documents[d].timeline.version for d in world.documents} == {f"{r}/text": n for r, n in VERSIONS.items()}


@pytest.mark.parametrize("for_search", [False, True])
def test_the_view_of_every_person_equals_the_oracle(world, for_search):
    expected = oracle.visible(PLAN, {f"{r}/text": n for r, n in VERSIONS.items()})
    got = {user: world.visible(user, for_search=for_search) for user in expected}
    assert got == expected
    assert expected["cit"] == {f"{r}/text": VERSIONS[r] for r in ("reg-3", "reg-4", "reg-5", "reg-6", "reg-7")}  # not drafts, not the review
