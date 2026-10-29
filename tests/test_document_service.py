"""Every channel gives the same answer as the workflow: list, get by id, a named version, search, the agent's tools."""

import pytest

from docflow import Refused
from docflow.stores.search_index import Corpus
from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller
from conftest_ministry import world  # noqa: F401

COMMITTEE = ["c1", "c2", "c3"]


@pytest.fixture
def gw(world):
    corpus = Corpus([])
    gw = DocumentService(world, corpus)

    def doc(doc_id, owner, state_steps, texts):
        world.add_document(doc_id, "ws", owner, "knowledge_asset", invited=["bob"] if doc_id == "shared" else [])
        for v, text in enumerate(texts, 1):
            corpus.add(doc_id, v, [text])
            if v > 1:
                world.act(doc_id, owner, "upload")
        state_steps(doc_id)

    def released(d, level="public"):
        world.act(d, "ann", "submit", input={"committee": COMMITTEE})
        world.act(d, "c1", "set_level", input={"level": level, "readers": []})
        for c in COMMITTEE:
            world.act(d, c, "approve")

    doc("draft", "ann", lambda d: None, ["draft zebra alpha"])
    doc("shared", "ann", lambda d: None, ["shared zebra beta"])
    doc("public", "ann", released, ["public zebra gamma"])
    doc("unit", "ann", lambda d: released(d, "unit"), ["unit zebra delta"])
    # released v1 with a newer draft v2 in progress
    doc("revised", "ann", released, ["revised zebra epsilon old"])
    world.act("revised", "ann", "start_update")
    world.act("revised", "ann", "upload")
    corpus.add("revised", 2, ["revised zebra epsilon new"])
    return gw


def test_every_channel_agrees_with_the_workflow(gw, world):
    """For every person and every document: list, get, a named version and search say what the workflow says."""
    bad = []
    for user in world.users:
        listed = {d.id: d.version for d in gw.list_documents(user)}
        found = {h.doc: h.version for h in gw.search(user, "zebra", limit=50)}
        for doc_id in world.documents:
            opened = world.readable_version(doc_id, user, for_search=False)
            searched = world.readable_version(doc_id, user, for_search=True)
            if listed.get(doc_id) != opened:
                bad.append((user, doc_id, "list", listed.get(doc_id), opened))
            if found.get(doc_id) != searched:
                bad.append((user, doc_id, "search", found.get(doc_id), searched))
            try:
                got = gw.get_document(user, doc_id)["version"]
            except NotFound:
                got = None
            if got != opened:
                bad.append((user, doc_id, "get", got, opened))
            for v in (1, 2, 3):  # naming a version works only for the one the person opens
                try:
                    named = gw.get_document(user, doc_id, v)["version"]
                except NotFound:
                    named = None
                if named != (v if v == opened else None):
                    bad.append((user, doc_id, f"get v{v}", named, opened))
    assert not bad, bad[:10]


def test_a_denied_document_looks_like_one_that_does_not_exist(gw):
    with pytest.raises(NotFound) as denied:
        gw.get_document("ext", "draft")
    with pytest.raises(NotFound) as missing:
        gw.get_document("ext", "no-such-doc")
    assert type(denied.value) is type(missing.value)
    assert str(denied.value) == "draft"  # carries the id asked for, nothing about why


def test_an_outsider_lists_only_public_releases(gw):
    assert {d.id for d in gw.list_documents("ext")} == {"public", "revised"}
    assert {h.doc for h in gw.search("ext", "zebra")} == {"public", "revised"}


def test_a_released_document_with_a_new_draft(gw):
    assert gw.get_document("ext", "revised")["version"] == 1  # everyone else: the release
    assert gw.get_document("ann", "revised")["version"] == 2  # the author opens the draft...
    assert {(h.doc, h.version) for h in gw.search("ann", "epsilon")} == {("revised", 2)}  # authors find their draft too
    assert {(h.doc, h.version) for h in gw.search("ext", "epsilon")} == {("revised", 1)}  # everyone else, the release only
    assert "old" in gw.search("ext", "epsilon")[0].passage.content


def test_listing_and_hits_carry_the_guidance_of_the_state(gw):
    by_id = {d.id: d for d in gw.list_documents("ann")}
    assert by_id["draft"].guidance.startswith("Unapproved working document")
    assert by_id["public"].guidance == "Approved asset; mention its level (public)."
    assert by_id["revised"].guidance.startswith("Unapproved working document")  # back in working after start_update: the author's text
    assert {d.id: d for d in gw.list_documents("ext")}["revised"].guidance == "Approved asset; mention its level (public)."  # a reader of the release, same state
    hit = next(h for h in gw.search("ext", "gamma") if h.doc == "public")
    assert hit.guidance == "Approved asset; mention its level (public)."


def test_the_agents_tools_are_the_same_channels_bound_to_the_person(gw):
    tools = tool_controller.tools(gw, "m2")  # a ministry employee with no role in the workspace
    assert {d["id"] for d in tools["list_documents"]()["documents"]} == {"public", "revised"}
    assert {h["id"] for h in tools["search"]("zebra")["hits"]} == {"public", "revised"}
    assert tools["get_document"]("public")["version"] == 1
    assert tools["get_document"]("unit") == {"error": "not_found"}  # released to unit U1 only; m2 is in U2
    assert {d["id"] for d in tool_controller.tools(gw, "u1b")["list_documents"]()["documents"]} == {"public", "revised", "unit"}


def test_a_workspace_filter_never_widens_access(gw, world):
    world.add_workspace("other", "O1", {"ann": "contributor"})
    assert gw.list_documents("ext", workspace="other") == []
    assert {d.id for d in gw.list_documents("ann", workspace="ws")} >= {"draft", "shared"}


def test_access_follows_a_change_at_once(gw, world):
    assert gw.search("ext", "alpha") == []  # a draft
    world.act("draft", "ann", "submit", input={"committee": COMMITTEE})
    world.act("draft", "c1", "set_level", input={"level": "public", "readers": []})
    for c in COMMITTEE:
        world.act("draft", c, "approve")
    assert [h.doc for h in gw.search("ext", "alpha")] == ["draft"]  # released: found by the very next call
    world.act("draft", "km1", "set_level", input={"level": "unit", "readers": []})
    assert gw.search("ext", "alpha") == []  # moved to a narrower level: gone at once
    with pytest.raises(NotFound):
        gw.get_document("ext", "draft")
