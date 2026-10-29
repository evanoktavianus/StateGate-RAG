"""P1: the access control is the one way in to documents, on every channel.

Where a table of who may open what appears, it was read off the workflow file
(`workflows/knowledge_asset.yml`, `workflows/regulation_single.yml`), never taken from the engine.
"""

import array
import ast
import json
import re
from pathlib import Path

import pytest

from docflow import Machine
from docflow.access_control import enforcement
from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller
from docflow.stores import search_index
from docflow.stores.search_index import DIMS, Corpus
from docflow.stores.world import World
from conftest import WORKFLOWS
from conftest_ministry import world  # noqa: F401

SRC = Path(__file__).parent.parent / "src" / "docflow"
COMMITTEE = ["c1", "c2", "c3"]
PEOPLE = ["ann", "bob", "cat", "u1b", "u1x", "m2", "km1", "c1", "c2", "c3", "c4", "c5", "ext"]


def vec(a):
    """A passage's vector: along the first axis, tilted by `a`. Against QUERY its cosine is 1 / sqrt(1 + a*a): the larger a, the lower."""
    return [1.0, a] + [0.0] * (DIMS - 2)


QUERY = [1.0] + [0.0] * (DIMS - 1)

# Who opens which document, and at which version. Read off knowledge_asset.yml:
#   draft    state `working`: the author (ann) and the unit's knowledge-management actors (km1, unit U1) only
#   shared   state `working`, bob invited: ann, bob, km1
#   public   `disseminated` at level public: everyone
#   unit     `disseminated` at level unit: the author's unit U1 (ann, bob, u1b, u1x, km1); not cat, m2, the committee (U2), ext
#   revised  released public (v1), then `start_update` back to `working` and a new draft v2: its author and the unit's knowledge-management actors (km1) see v2, everyone else the release v1
EVERYONE = {p: 1 for p in PEOPLE}
ORACLE = {
    "draft": {"ann": 1, "km1": 1},
    "shared": {"ann": 1, "bob": 1, "km1": 1},
    "public": EVERYONE,
    "unit": {p: 1 for p in ("ann", "bob", "u1b", "u1x", "km1")},
    "revised": {**EVERYONE, "ann": 2, "km1": 2},
}
# The passage vectors tilt (see vec): this is the order by meaning, nearest first, among the passages a person may read.
MEANING_ORDER = ["draft", "shared", "public", "unit", "revised"]


@pytest.fixture
def gw(world):
    corpus = Corpus([], array.array("f"))
    gw = DocumentService(world, corpus)

    def released(d, level="public"):
        world.act(d, "ann", "submit", input={"committee": COMMITTEE})
        world.act(d, "c1", "set_level", input={"level": level, "readers": []})
        for c in COMMITTEE:
            world.act(d, c, "approve")

    def doc(doc_id, steps, passages, invited=()):
        world.add_document(doc_id, "ws", "ann", "knowledge_asset", invited=invited)
        for v, (text, a) in enumerate(passages, 1):
            corpus.add(doc_id, v, [text], [vec(a)])
            if v > 1:
                world.act(doc_id, "ann", "upload")
        steps(doc_id)

    doc("draft", lambda d: None, [("draft zebra alpha", 0.1)])
    doc("shared", lambda d: None, [("shared zebra beta", 0.2)], invited=["bob"])
    doc("public", released, [("public zebra gamma", 0.3)])
    doc("unit", lambda d: released(d, "unit"), [("unit zebra delta", 0.4)])
    doc("revised", released, [("revised zebra epsilon old", 0.5)])
    world.act("revised", "ann", "start_update")  # a new draft starts: back in `working`, where versions are uploaded
    world.act("revised", "ann", "upload")
    corpus.add("revised", 2, ["revised zebra epsilon new"], [vec(0.6)])
    return gw


def expected_for(user):
    return {d: m[user] for d, m in ORACLE.items() if user in m}


def test_the_table_covers_everyone_in_the_world(world):
    assert set(world.users) == set(PEOPLE)


# ── 1. A single choke point ────────────────────────────────────────────

FORBIDDEN_MODULES = {"stores"}
FORBIDDEN_NAMES = {"World", "Corpus", "Passage", "Timeline", "regulations", "synthetic", "scenario"}
FORBIDDEN_ATTRS = {"world", "corpus", "documents", "groups", "workspaces", "users", "timeline"}


def reaches_storage(source: str) -> list[str]:
    """What in `source` imports a store or holds a reference to a world, a corpus or an index."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            parts = set((node.module or "").split("."))
            if parts & FORBIDDEN_MODULES or parts & FORBIDDEN_NAMES or {a.name for a in node.names} & (FORBIDDEN_MODULES | FORBIDDEN_NAMES):
                found.append(f"import from {'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if set(a.name.split(".")) & (FORBIDDEN_MODULES | FORBIDDEN_NAMES):
                    found.append(f"import {a.name}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            found.append(f"attribute .{node.attr}")
    return found


def test_the_agent_harness_and_the_interface_never_reach_a_store():
    files = sorted((SRC / "agent_harness").glob("*.py")) + [p for p in [SRC / "interface_api.py", SRC / "interface_api_http.py"] if p.exists()]
    assert SRC / "agent_harness" / "tool_controller.py" in files
    bad = {str(f.relative_to(SRC)): reaches_storage(f.read_text()) for f in files}
    assert not any(bad.values()), bad


def test_the_choke_point_check_can_fail():
    """The static check must be able to fail, or a pass means nothing."""
    assert reaches_storage("from ..stores.world import World")
    assert reaches_storage("from ..stores import search_index")
    assert reaches_storage("def f(service):\n    return service.world.documents")
    assert not reaches_storage("from ..access_control import enforcement\ndef f(service):\n    return service.search('x')")


# ── 2. Every channel says what the policy says ─────────────────────────

def test_list_matches_the_table(gw):
    for user in PEOPLE:
        assert {d.id: d.version for d in gw.list_documents(user)} == expected_for(user), user


def test_get_by_id_and_a_named_version_match_the_table(gw):
    for user in PEOPLE:
        for doc in ORACLE:
            want = expected_for(user).get(doc)
            try:
                got = gw.get_document(user, doc)["version"]
            except NotFound:
                got = None
            assert got == want, (user, doc, "latest")
            for v in (1, 2, 3):  # a named version opens only if it is the one the person opens
                try:
                    named = gw.get_document(user, doc, v)["version"]
                except NotFound:
                    named = None
                assert named == (v if v == want else None), (user, doc, v)


def pairs(hits):
    return {(h.doc, h.version) for h in hits}


def test_keyword_search_matches_the_table(gw):
    for user in PEOPLE:
        assert pairs(gw.search(user, "zebra", limit=50)) == set(expected_for(user).items()), user


def test_search_by_meaning_matches_the_table(gw):
    for user in PEOPLE:
        assert pairs(gw.similar(user, QUERY, limit=50)) == set(expected_for(user).items()), user


def test_search_by_words_and_meaning_matches_the_table(gw):
    for user in PEOPLE:
        assert pairs(gw.hybrid(user, "zebra", QUERY, limit=50)) == set(expected_for(user).items()), user


def test_search_by_meaning_ranks_only_what_the_person_may_read(gw):
    """Order by cosine, nearest first, from the tilts in `vec`; the second one is the author, who reads revised at v2."""
    assert [h.doc for h in gw.similar("ext", QUERY, 50)] == ["public", "revised"]
    assert [h.doc for h in gw.similar("u1b", QUERY, 50)] == ["public", "unit", "revised"]
    ann = gw.similar("ann", QUERY, 50)
    assert [h.doc for h in ann] == MEANING_ORDER
    assert [h.version for h in ann if h.doc == "revised"] == [2]
    assert [h.score for h in gw.similar("ext", QUERY, 50)] == pytest.approx([1 / (1 + 0.3**2) ** 0.5, 1 / (1 + 0.5**2) ** 0.5])


def test_the_pure_python_cosine_gives_the_same_order_as_numpy(gw, monkeypatch):
    with_numpy = {u: [(h.doc, h.version, round(h.score, 6)) for h in gw.similar(u, QUERY, 50)] for u in PEOPLE}
    monkeypatch.setattr(search_index, "_np", None)
    gw.corpus._matrix = None
    without = {u: [(h.doc, h.version, round(h.score, 6)) for h in gw.similar(u, QUERY, 50)] for u in PEOPLE}
    assert with_numpy == without


def test_fusing_words_and_meaning_orders_by_reciprocal_rank(gw):
    """ext reads public and revised v1. The word `epsilon` is only in revised; by meaning, public is nearer.
    Words: [revised]. Meaning: [public, revised]. Fused with k=60 (a passage ranked r from the top adds 1 / (60 + r + 1)):
    revised 1/61 + 1/62, public 1/61, so revised first."""
    hits = gw.hybrid("ext", "epsilon", QUERY, 50)
    assert [h.doc for h in hits] == ["revised", "public"]
    assert [h.score for h in hits] == pytest.approx([1 / 61 + 1 / 62, 1 / 61])


# ── 3. The decision comes before ranking ───────────────────────────────

def snapshot(gw, user):
    return {
        "words": [(h.doc, h.version, h.passage.seq, h.score) for h in gw.search(user, "zebra alpha epsilon", 50)],
        "meaning": [(h.doc, h.version, h.passage.seq, h.score) for h in gw.similar(user, QUERY, 50)],
        "both": [(h.doc, h.version, h.passage.seq, h.score) for h in gw.hybrid(user, "zebra alpha epsilon", QUERY, 50)],
    }


def test_a_hidden_document_full_of_matching_words_changes_nothing(gw, world):
    """`stuffed` is a working draft of ann's: ext, m2 and cat cannot read it. It repeats the query words and has the query's own
    vector (cosine 1). If it touched a score, an order or a count, the three snapshots would differ."""
    before = {u: snapshot(gw, u) for u in ("ext", "m2", "cat")}
    world.add_document("stuffed", "ws", "ann", "knowledge_asset")
    gw.corpus.add("stuffed", 1, ["zebra " * 40 + "alpha epsilon gamma"], [vec(0.0)])
    after = {u: snapshot(gw, u) for u in ("ext", "m2", "cat")}
    assert after == before
    # the control: the author does read it. It is found by words, and by meaning it is first (its vector is the query, cosine 1),
    # so had it been counted for the others the snapshots would have moved.
    ann = snapshot(gw, "ann")
    assert "stuffed" in [d for d, *_ in ann["words"]]
    assert ann["meaning"][0][0] == "stuffed"


# ── 4. A refusal looks like "not found" ────────────────────────────────

def refusal(gw, user, doc_id, version=None):
    start = len(gw.log)
    with pytest.raises(NotFound) as e:
        gw.get_document(user, doc_id, version)
    (entry,) = gw.log[start:]
    return e.value, entry


def test_a_hidden_document_is_refused_as_one_that_does_not_exist(gw):
    cases = {
        "hidden": ("ext", "draft", None),
        "missing": ("ext", "no-such-doc", None),
        "named version not opened": ("ext", "revised", 2),  # ext opens revised at v1
        "named version of a missing document": ("ext", "no-such-doc", 2),
    }
    seen = {name: refusal(gw, *args) for name, args in cases.items()}
    shapes = set()
    for name, (error, entry) in seen.items():
        doc_id = cases[name][1]
        assert type(error) is NotFound, name
        assert error.args == (doc_id,) and str(error) == doc_id and vars(error) == {}, name
        assert entry["channel"] == "get" and entry["returned"] == [] and entry["user"] == "ext", name
        shapes.add(frozenset(entry))
    assert len(shapes) == 1  # the same fields in the log
    # nothing that says why reaches the caller
    for error, entry in seen.values():
        assert entry["reason"] not in repr(error) and entry["reason"] not in str(error)


def test_the_log_keeps_the_reason_the_caller_never_gets(gw):
    assert refusal(gw, "ext", "draft")[1]["reason"] == "not_opened"
    assert refusal(gw, "ext", "no-such-doc")[1]["reason"] == "no_such_document"
    assert refusal(gw, "ext", "revised", 2)[1]["reason"] == "version_not_opened"


def test_the_tools_refuse_alike(gw):
    bound = tool_controller.tools(gw, "ext")
    results = [bound["get_document"]("draft"), bound["get_document"]("no-such-doc"), bound["get_document"]("revised", 2),
               bound["get_document"]("no-such-doc", 2)]
    assert results == [{"error": "not_found"}] * 4


# ── 5. Guidance is the state's own text, the same for everyone ─────────

def regulation_world():
    """Regulation T amended and S1 in force, S2 a draft, S3 in review, each one document of unit U1, under regulation_single.yml."""
    w = World({"regulation_single": Machine.load(WORKFLOWS / "regulation_single.yml")})
    w.add_org("O1"), w.add_org("O2")
    w.add_workspace("ws", "O1")
    w.add_user("lead", "O1", unit="U1", positions=("leadership",))
    w.add_user("team", "O1", unit="U1")
    w.add_user("staff", "O1", unit="U1")
    w.add_user("inv", "O1", unit="INV")
    w.add_user("emp", "O1", unit="OTH")
    w.add_user("ag", "O2", unit="AGY")
    w.add_user("cit")
    corpus = Corpus([])
    for rid in ("T", "S1", "S2", "S3"):
        w.add_document(rid, "ws", "team", "regulation_single", creator="lead", assign={"drafting_team": ["team"], "invited": ["inv"]})
        corpus.add(rid, 1, [f"regulation {rid} text"])
    for rid in ("T", "S1", "S3"):
        w.act(rid, "team", "submit_for_review")
    w.fire_event("T", "promulgate", "promulgated")
    w.fire_event("S1", "promulgate", "promulgated")
    w.fire_event("T", "amend", "amendment_in_force")  # the host fires what S1 coming into force does to T
    return w, DocumentService(w, corpus)


USERS = ("lead", "team", "staff", "inv", "emp", "ag", "cit")


def test_the_world_is_in_the_states_the_test_assumes():
    w, _ = regulation_world()
    # from regulation_single.yml: promulgate -> in_force; amend (an outside event) -> amended
    assert [w.state_of(r) for r in ("T", "S1", "S2", "S3")] == ["amended", "in_force", "working_draft", "review"]


def test_guidance_is_one_fixed_text_on_every_channel_and_names_no_other_document():
    """The amended text's guidance is fixed in the workflow: it says the regulation was amended, and names no regulation or article."""
    _, gw = regulation_world()
    want = "This regulation has been amended; the rest is unchanged. Present both."
    for user in USERS:
        listed = {d.id: d for d in gw.list_documents(user)}
        assert listed["T"].guidance == want, user  # `anyone` reads the amended text
        hit = next(h for h in gw.search(user, "regulation", 50) if h.doc == "T")
        assert hit.guidance == want, user
        assert gw.get_document(user, "T")["guidance"] == want, user
        for d in listed.values():
            assert not re.search(r"\bS[123]\b|Pasal|Article", d.guidance or ""), (user, d.id)


def test_two_people_in_one_state_served_different_versions_get_different_guidance(gw):
    """`revised` is in `working` for everyone: its author is served the new draft and told it is unapproved; the readers are served the
    release (their access rule says so) and are told it is an approved asset, at its level. The state's own text is the author's."""
    own = "Unapproved working document; do not present it as a knowledge asset."
    release = "Approved asset; mention its level (public)."
    assert gw.world.state_of("revised") == "working"
    assert (gw.get_document("ann", "revised")["version"], gw.get_document("ext", "revised")["version"]) == (2, 1)
    for user, want in (("ann", own), ("ext", release), ("u1b", release), ("m2", release)):
        listed = {d.id: d for d in gw.list_documents(user)}["revised"]
        hit = next(h for h in gw.search(user, "revised", 50) if h.doc == "revised")
        tools = tool_controller.tools(gw, user)
        tool_hit = next(h for h in tools["search"]("revised", 50)["hits"] if h["id"] == "revised")
        got = {listed.guidance, hit.guidance, gw.get_document(user, "revised")["guidance"], gw.world.guidance("revised", user),
               tools["get_document"]("revised")["guidance"], tool_hit["guidance"],
               next(d["guidance"] for d in tools["list_documents"]()["documents"] if d["id"] == "revised")}
        assert got == {want}, user  # the same on every channel
    assert gw.world.guidance("revised") == own  # the host's own view, with no person, is the state's


def test_a_person_a_rule_without_guidance_covers_is_given_the_states(gw):
    """The first rule that covers a person decides; if it has no guidance, the state's applies. km1 (a knowledge-management actor of the unit)
    is covered by the first rule of `working`, which has none."""
    assert gw.get_document("km1", "revised")["guidance"] == gw.world.guidance("revised")
    assert gw.get_document("km1", "revised")["version"] == 2


def test_a_refused_person_gets_what_they_get_for_a_missing_document(gw):
    """No document and no guidance reach a person no rule opens: the refusal and the tool result are those of a missing document."""
    start = len(gw.log)
    seen = {}
    for name, doc_id in (("hidden", "draft"), ("missing", "no-such-doc")):
        with pytest.raises(NotFound) as e:
            gw.get_document("ext", doc_id)
        seen[name] = (type(e.value), e.value.args, vars(e.value))
    assert seen["hidden"][0] is seen["missing"][0] and seen["hidden"][2] == seen["missing"][2] == {}
    assert seen["hidden"][1] == ("draft",) and seen["missing"][1] == ("no-such-doc",)  # only the id asked for
    tools = tool_controller.tools(gw, "ext")
    assert tools["get_document"]("draft") == tools["get_document"]("no-such-doc") == {"error": "not_found"}
    assert all("draft zebra" not in repr(h) and "Unapproved" not in repr(h) for h in gw.search("ext", "alpha draft", 50))
    assert not any("Unapproved" in str(d.guidance) for d in gw.list_documents("ext") if d.id in ("draft", "shared"))
    assert len(gw.log) > start


# ── 6. What a tool returns, field by field ─────────────────────────────

LIST_FIELDS = {"documents", "truncated"}
LISTED_FIELDS = {"id", "version", "state", "workflow", "guidance"}
GET_FIELDS = {"id", "version", "state", "guidance", "passages", "truncated"}
PASSAGE_FIELDS = {"seq", "pages", "text"}
SEARCH_FIELDS = {"hits"}
HIT_FIELDS = {"id", "version", "state", "guidance", "score", "passage"}


def all_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from all_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_keys(v)


def test_every_field_a_tool_returns_comes_from_a_document_the_person_opens(gw):
    for user in PEOPLE:
        bound = tool_controller.tools(gw, user)
        listed = bound["list_documents"]()
        searched = bound["search"]("zebra", 50)
        assert set(listed) == LIST_FIELDS and set(searched) == SEARCH_FIELDS
        assert {d["id"]: d["version"] for d in listed["documents"]} == expected_for(user)
        assert {h["id"]: h["version"] for h in searched["hits"]} == expected_for(user)
        for d in listed["documents"]:
            assert set(d) == LISTED_FIELDS
        for h in searched["hits"]:
            assert set(h) == HIT_FIELDS and set(h["passage"]) == PASSAGE_FIELDS
        for doc in expected_for(user):
            got = bound["get_document"](doc)
            assert set(got) == GET_FIELDS and all(set(p) == PASSAGE_FIELDS for p in got["passages"])
        # no total, count or number of matches anywhere: such a figure could include what is hidden
        keys = set(all_keys(listed)) | set(all_keys(searched)) | set(all_keys(bound["get_document"](next(iter(expected_for(user)), "draft"))))
        assert not [k for k in keys if re.search(r"total|count|matches|num|hidden", k)], keys


# ── 7. Results are model-ready ─────────────────────────────────────────

def test_tool_results_are_plain_json(gw):
    for user in ("ann", "ext"):
        bound = tool_controller.tools(gw, user)
        for result in (bound["list_documents"](), bound["search"]("zebra"), bound["get_document"]("public"), bound["get_document"]("draft")):
            assert json.loads(json.dumps(result)) == result


def test_each_tool_has_a_json_schema_that_matches_its_arguments(gw):
    names = [s["function"]["name"] for s in tool_controller.SCHEMAS]
    assert names == ["list_documents", "get_document", "search"] == list(tool_controller.tools(gw, "ann"))
    by_name = {s["function"]["name"]: s["function"]["parameters"] for s in tool_controller.SCHEMAS}
    assert all(p["type"] == "object" and p["additionalProperties"] is False for p in by_name.values())
    assert by_name["list_documents"]["properties"].keys() == {"workspace"} and by_name["list_documents"]["required"] == []
    assert by_name["get_document"]["properties"].keys() == {"doc_id", "version"} and by_name["get_document"]["required"] == ["doc_id"]
    assert by_name["search"]["properties"].keys() == {"query", "limit"} and by_name["search"]["required"] == ["query"]
    assert by_name["search"]["properties"]["limit"]["maximum"] == enforcement.MAX_SEARCH_LIMIT


def test_a_bad_call_gets_a_fixed_refusal_not_a_stack_trace(gw):
    bound = tool_controller.tools(gw, "ann")
    for name, arguments in (("search", {}), ("search", {"query": 5}), ("search", {"query": "x", "limit": "3"}),
                            ("search", {"query": "x", "limit": True}), ("get_document", {"doc_id": None}),
                            ("get_document", {"doc_id": "public", "version": "1"}), ("list_documents", {"workspace": 3}),
                            ("search", {"query": "x", "user": "lead_U1"}), ("search", "x")):
        assert tool_controller.call(bound, name, arguments) == {"error": "bad_request"}, (name, arguments)
    assert tool_controller.call(bound, "get_document", {"doc_id": "public"})["version"] == 1


# ── 8. Limits ──────────────────────────────────────────────────────────

def test_the_limits_are_named_constants_in_the_enforcement():
    for name in ("MAX_SEARCH_LIMIT", "MAX_LIST", "MAX_GET_PASSAGES", "MAX_PASSAGE_CHARS"):
        assert isinstance(getattr(enforcement, name), int) and getattr(enforcement, name) > 0


def test_a_search_never_returns_more_than_the_limit(gw, monkeypatch):
    monkeypatch.setattr(enforcement, "MAX_SEARCH_LIMIT", 2)
    assert len(gw.search("ann", "zebra", limit=10_000)) == 2  # ann reads five
    assert len(gw.similar("ann", QUERY, limit=10_000)) == 2
    assert len(gw.hybrid("ann", "zebra", QUERY, limit=10_000)) == 2
    assert len(gw.search("ann", "zebra", limit=0)) == 1 and len(gw.search("ann", "zebra", limit=-5)) == 1
    assert len(tool_controller.tools(gw, "ann")["search"]("zebra", 10_000)["hits"]) == 2


def test_a_list_and_a_document_are_cut_at_the_limits(gw, world, monkeypatch):
    monkeypatch.setattr(enforcement, "MAX_LIST", 2)
    listed = tool_controller.tools(gw, "ann")["list_documents"]()
    assert len(listed["documents"]) == 2 and listed["truncated"] is True
    assert tool_controller.tools(gw, "ext")["list_documents"]()["truncated"] is False  # ext reads two: nothing cut
    world.add_document("long", "ws", "ann", "knowledge_asset")
    gw.corpus.add("long", 1, ["a" * 50 + str(i) for i in range(5)], [vec(0.9)] * 5)
    monkeypatch.setattr(enforcement, "MAX_GET_PASSAGES", 3)
    monkeypatch.setattr(enforcement, "MAX_PASSAGE_CHARS", 10)
    got = tool_controller.tools(gw, "ann")["get_document"]("long")
    assert len(got["passages"]) == 3 and all(len(p["text"]) == 10 for p in got["passages"]) and got["truncated"] is True


# ── 9. The decision log ────────────────────────────────────────────────

def test_every_call_is_logged_with_channel_person_arguments_and_what_came_back(gw):
    gw.list_documents("ext", "ws")
    gw.get_document("ext", "public")
    gw.search("ext", "zebra", 3)
    gw.similar("ext", QUERY, 3)
    gw.hybrid("ext", "zebra", QUERY, 3)
    log = gw.log
    assert [e["channel"] for e in log] == ["list", "get", "search", "similar", "hybrid"]
    assert all(e["user"] == "ext" for e in log)
    assert log[0]["arguments"] == {"workspace": "ws"} and log[1]["arguments"] == {"doc_id": "public", "version": None}
    assert log[2]["arguments"] == {"query": "zebra", "limit": 3}
    # public is released (disseminated); revised is released too but has a newer draft: its state is `working` again
    assert log[0]["returned"] == [{"doc": "public", "version": 1, "state": "disseminated"}, {"doc": "revised", "version": 1, "state": "working"}]
    assert log[1]["returned"] == [{"doc": "public", "version": 1, "state": "disseminated"}]
    assert {(r["doc"], r["version"]) for r in log[2]["returned"]} == {("public", 1), ("revised", 1)}
    assert all("reason" not in e for e in log)  # only a refusal has one


# ── 10. The agent cannot change anything ───────────────────────────────

NOT_READ = ["act", "fire_event", "add_org", "add_user",
            "add_workspace", "add_document", "upload", "submit", "approve", "set_level", "similar", "hybrid",
            "visible", "readable_version", "guidance", "state_of", "log", "world", "corpus", "_logged", "__init__", "__class__",
            "__dict__", "", "list_documents ", "SEARCH", "../search"]


def test_no_tool_changes_a_document(gw, world):
    bound = tool_controller.tools(gw, "ann")
    assert set(bound) == {"list_documents", "get_document", "search"}
    state_before, log_before = world.documents["draft"].timeline.state, len(world.log)
    for name in NOT_READ:
        assert tool_controller.call(bound, name, {"doc_id": "draft", "user": "ann", "action": "submit"}) == {"error": "unknown_tool"}, name
    assert tool_controller.call(bound, 7, {}) == {"error": "unknown_tool"}
    assert world.documents["draft"].timeline.state == state_before and len(world.log) == log_before


def test_the_names_that_change_things_are_not_among_the_bound_tools(gw, world):
    changing = {n for n in dir(world) if not n.startswith("_") and callable(getattr(world, n))} - {"visible", "readable_version", "guidance", "state_of",
                                                                                                      "person", "members", "role_in", "context"}
    assert changing >= {"act", "fire_event", "add_document", "add_user", "add_workspace"}
    assert not changing & set(tool_controller.tools(gw, "ann"))
