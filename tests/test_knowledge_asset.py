"""The knowledge-asset table: who acts, who finds and opens, which version, what the agent is told."""

import pytest

from docflow import Refused
from conftest_ministry import world  # noqa: F401

COMMITTEE = ["c1", "c2", "c3"]


def submitted(world, committee=COMMITTEE):
    world.add_document("d", "ws", "ann", "knowledge_asset")
    world.act("d", "ann", "submit", input={"committee": committee})


def released(world, level="public", readers=()):
    submitted(world)
    world.act("d", "c1", "set_level", input={"level": level, "readers": list(readers)})
    for c in COMMITTEE:
        world.act("d", c, "approve")


def opened(world, doc="d"):
    return {u: world.readable_version(doc, u, for_search=False) for u in world.users}


# ── Working document ────────────────────────────────────────────────────────

def test_a_working_document_is_for_its_authors_and_km_actors_only(world):
    world.add_document("d", "ws", "ann", "knowledge_asset", invited=["bob"])
    reads = {u: v for u, v in opened(world).items() if v}
    assert reads == {"ann": 1, "bob": 1, "km1": 1}
    assert world.visible("ann") == {"d": 1}  # authors find their draft; nobody else does
    assert world.visible("u1b") == {} and world.visible("c1") == {}


def test_a_working_document_carries_its_guidance(world):
    world.add_document("d", "ws", "ann", "knowledge_asset")
    assert world.guidance("d") == "Unapproved working document; do not present it as a knowledge asset."


def test_only_the_author_or_km_actors_submit(world):
    world.add_document("d", "ws", "ann", "knowledge_asset")
    for who in ("u1b", "c1", "ext"):
        with pytest.raises(Refused) as e:
            world.act("d", who, "submit", input={"committee": COMMITTEE})
        assert e.value.reason == "not_permitted"
    world.act("d", "km1", "submit", input={"committee": COMMITTEE})
    assert world.documents["d"].timeline.state == "quality_assurance"


# ── Quality assurance ───────────────────────────────────────────────────────

@pytest.mark.parametrize("committee, why", [
    (["c1", "c2"], "assignment_needed"),
    (["c1", "c2", "c3", "c4"], "assignment_invalid"),  # even
    (["c1", "c2", "ann"], "assignment_invalid"),  # an author can't judge their own work
    (["c1", "c2", "u1b"], "assignment_invalid"),  # not of the committee
])
def test_the_committee_is_odd_at_least_three_and_not_the_authors(world, committee, why):
    world.add_document("d", "ws", "ann", "knowledge_asset")
    with pytest.raises(Refused) as e:
        world.act("d", "ann", "submit", input={"committee": committee})
    assert e.value.reason == why
    assert world.documents["d"].timeline.state == "working"


def test_five_is_odd_too(world):
    submitted(world, ["c1", "c2", "c3", "c4", "c5"])
    assert world.documents["d"].timeline.state == "quality_assurance"


def test_qa_is_read_by_the_committee_and_the_authors_only(world):
    submitted(world)
    reads = {u: v for u, v in opened(world).items() if v}
    assert reads == {"ann": 1, "c1": 1, "c2": 1, "c3": 1}  # the committee holds no other role on it
    assert "c4" not in reads and "km1" not in reads
    assert world.guidance("d").startswith("Unapproved working document")


def test_the_committee_sets_the_level_not_the_author(world):
    submitted(world)
    with pytest.raises(Refused) as e:
        world.act("d", "ann", "set_level", input={"level": "public", "readers": []})
    assert e.value.reason == "not_permitted"


def test_approval_waits_for_a_level(world):
    submitted(world)
    with pytest.raises(Refused) as e:
        world.act("d", "c1", "approve")
    assert (e.value.reason, e.value.detail["name"]) == ("choice_needed", "level")
    assert world.documents["d"].timeline.given("approve") == []  # nothing recorded


def test_all_three_must_approve(world):
    submitted(world)
    world.act("d", "c1", "set_level", input={"level": "unit", "readers": []})
    world.act("d", "c1", "approve"), world.act("d", "c2", "approve")
    assert world.documents["d"].timeline.state == "quality_assurance"
    with pytest.raises(Refused) as e:
        world.act("d", "c2", "approve")
    assert e.value.reason == "already_given"
    world.act("d", "c3", "approve")
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version_at("disseminated")) == ("disseminated", 1)


def test_a_revision_goes_back_to_working_and_restarts_the_approvals(world):
    submitted(world)
    world.act("d", "c1", "set_level", input={"level": "unit", "readers": []})
    world.act("d", "c1", "approve"), world.act("d", "c2", "approve")
    with pytest.raises(Refused) as e:
        world.act("d", "ann", "upload")  # a version is not uploaded while the committee reads
    assert e.value.reason == "not_accepted"
    world.act("d", "c3", "request_changes")
    world.act("d", "ann", "upload")  # authors revise the one document, in working
    world.act("d", "ann", "submit", input={"committee": COMMITTEE})
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version, tl.given("approve")) == ("quality_assurance", 2, [])
    for c in COMMITTEE:
        world.act("d", c, "approve")
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version_at("disseminated")) == ("disseminated", 2)  # all three approved the latest revision


def test_changing_the_level_restarts_the_approvals_too(world):
    submitted(world)
    world.act("d", "c1", "set_level", input={"level": "unit", "readers": []})
    world.act("d", "c1", "approve"), world.act("d", "c2", "approve")
    world.act("d", "c3", "set_level", input={"level": "public", "readers": []})
    assert world.documents["d"].timeline.given("approve") == []


def test_request_changes_sends_it_back(world):
    submitted(world)
    world.act("d", "c2", "request_changes")
    assert world.documents["d"].timeline.state == "working"


def test_an_outsider_cannot_upload_a_revision(world):
    submitted(world)
    for who in ("c1", "ext", "km1"):
        with pytest.raises(Refused):
            world.act("d", who, "upload")


# ── Disseminated, by level ──────────────────────────────────────────────────

ORG = {"ann", "bob", "cat", "u1b", "u1x", "m2", "km1", "c1", "c2", "c3", "c4", "c5"}  # everyone of the ministry (O1)
LEVELS = {
    # the author always reads it; everyone else, the km actors and the committee included, by level (card 6)
    "named": {"ann", "m2"},  # the individuals the committee named
    "unit": {"ann", "bob", "u1b", "u1x", "km1"},  # everyone of the producing unit (U1), whatever their role
    "organization": ORG,
    "public": ORG | {"ext"},  # anyone signed in
}


@pytest.mark.parametrize("level", list(LEVELS))
def test_each_level_reaches_its_audience(world, level):
    released(world, level, readers=["m2"] if level == "named" else [])
    reads = {u for u, v in opened(world).items() if v == 1}
    assert reads == LEVELS[level]


@pytest.mark.parametrize("level", list(LEVELS))
def test_search_finds_the_release_for_the_same_audience(world, level):
    released(world, level, readers=["m2"] if level == "named" else [])
    found = {u for u in world.users if world.visible(u) == {"d": 1}}
    assert found == LEVELS[level]


def test_the_guidance_says_the_asset_is_approved_and_its_level(world):
    released(world, "unit")
    assert world.guidance("d") == "Approved asset; mention its level (unit)."


def test_km_actors_can_move_a_released_asset_to_another_level(world):
    released(world, "unit")
    world.act("d", "km1", "set_level", input={"level": "public", "readers": []})
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version_at("disseminated")) == ("disseminated", 1)
    assert tl.entries.count(("state", "disseminated")) == 1  # changing the level is not a new arrival
    assert world.readable_version("d", "ext") == 1


# ── Update ──────────────────────────────────────────────────────────────────

def test_an_update_leaves_the_release_standing_and_the_draft_to_its_authors(world):
    released(world, "public")
    world.act("d", "ann", "start_update")
    world.act("d", "ann", "upload")
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version, tl.version_at("disseminated")) == ("working", 2, 1)
    assert world.readable_version("d", "ann", for_search=False) == 2  # the draft: authors
    assert world.readable_version("d", "ext") == 1 and world.readable_version("d", "m2") == 1  # the release
    assert world.readable_version("d", "u1b", for_search=False) == 1
    assert "napproved working document" in world.guidance("d")


def test_an_update_goes_through_qa_again_and_its_version_is_then_the_one_served(world):
    released(world, "public")
    world.act("d", "ann", "start_update")
    world.act("d", "ann", "upload")
    world.act("d", "ann", "submit", input={"committee": ["c3", "c4", "c5"]})
    world.act("d", "c3", "approve"), world.act("d", "c4", "approve"), world.act("d", "c5", "approve")  # the level stays
    tl = world.documents["d"].timeline
    assert (tl.state, tl.version_at("disseminated")) == ("disseminated", 2)
    assert tl.entries.count(("state", "disseminated")) == 2
    assert world.readable_version("d", "ext") == 2
