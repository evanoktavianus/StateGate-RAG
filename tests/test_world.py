"""Several organizations, several workflows, documents added as we go: the host side, on small data."""

import pytest

from docflow import Machine, Refused
from docflow.stores.search_index import Corpus
from docflow.stores.world import World
from conftest import FIXTURES


@pytest.fixture
def world():
    w = World({n: Machine.load(FIXTURES / f"{n}.yml") for n in ("policy_analysis", "report")})
    for org in ("acme", "globex"):
        w.add_org(org)
    for u, org in (("ann", "acme"), ("bea", "acme"), ("cy", "acme"), ("gil", "globex"), ("hal", "globex"), ("ext", None)):
        w.add_user(u, org)
    w.add_user("qa1", "acme", groups=("QA team",))
    w.add_workspace("acme-ws", "acme", {"ann": "contributor", "bea": "viewer", "cy": "editor"}, {"QA team": "editor"})
    w.add_workspace("globex-ws", "globex", {"gil": "contributor", "hal": "viewer"})
    return w


def release(w, doc, audience):
    w.act(doc, "ann", "submit", input={"reviewer": ["qa1"]})
    w.act(doc, "qa1", "publish", input={"audience": audience})


@pytest.mark.parametrize(
    "audience, reads",
    [
        ("unit", {"ann": 1, "bea": 1, "cy": 1, "gil": None, "ext": None}),
        ("organization", {"ann": 1, "bea": 1, "cy": 1, "gil": None, "ext": None}),
        ("public", {"ann": 1, "bea": 1, "cy": 1, "gil": 1, "hal": 1, "ext": 1}),
    ],
)
def test_who_a_release_reaches(world, audience, reads):
    world.add_document("d", "acme-ws", "ann", "policy_analysis")
    release(world, "d", audience)
    assert {u: world.readable_version("d", u) for u in reads} == reads


def test_a_draft_is_visible_to_nobody_but_its_people(world):
    world.add_document("d", "acme-ws", "ann", "policy_analysis", invited=["bea"])
    opened = lambda u: world.visible(u, for_search=False)
    assert opened("ann") == {"d": 1}
    assert opened("bea") == {"d": 1}  # invited
    for other in ("cy", "gil", "ext", "qa1"):
        assert opened(other) == {}
    assert world.visible("ann") == {}  # a draft is in nobody's search, its author's included


def test_organization_audience_reaches_a_member_with_no_role_in_the_workspace(world):
    world.add_user("zed", "acme")  # same organization, no role in acme-ws
    world.add_document("d", "acme-ws", "ann", "policy_analysis")
    release(world, "d", "organization")
    assert world.readable_version("d", "zed") == 1
    assert world.readable_version("d", "gil") is None  # another organization


def test_documents_can_be_added_under_other_workflows(world):
    world.add_document("r", "acme-ws", "ann", "report")
    world.act("r", "ann", "submit", input={"reviewers": ["cy"]})
    world.act("r", "cy", "approve")
    assert world.documents["r"].timeline.state == "editor_approved"
    # report has no access rules: anyone with a role reads the latest, nobody else
    assert world.visible("bea") == {"r": 1} and world.visible("gil") == {}
    # and the same world holds a policy document at once
    world.add_document("p", "acme-ws", "ann", "policy_analysis")
    assert world.visible("bea") == {"r": 1}


def test_a_refused_move_changes_nothing(world):
    world.add_document("d", "acme-ws", "ann", "policy_analysis")
    before = world.documents["d"].timeline
    with pytest.raises(Refused):
        world.act("d", "bea", "submit", input={"reviewer": ["qa1"]})  # not the author
    assert world.documents["d"].timeline is before


def test_search_only_scores_what_the_person_may_read(world):
    corpus = Corpus([])
    corpus.add("secret", 1, ["the launch code is swordfish"])
    corpus.add("open", 1, ["the picnic is on friday, bring fish"])
    world.add_document("secret", "acme-ws", "ann", "policy_analysis")
    world.add_document("open", "acme-ws", "ann", "policy_analysis")
    release(world, "open", "public")
    hits = lambda user: [p.doc for _, p in corpus.search("swordfish fish", world.visible(user))]
    assert hits("ann") == ["open"]  # the draft is out of search even for its author
    assert hits("gil") == ["open"]
    world.act("secret", "ann", "upload")  # still a draft, still nobody's search
    assert hits("cy") == ["open"]


def test_a_new_draft_does_not_change_what_search_finds_for_others(world):
    corpus = Corpus([])
    corpus.add("d", 1, ["version one says alpha"])
    world.add_document("d", "acme-ws", "ann", "policy_analysis")
    release(world, "d", "public")
    world.act("d", "ann", "start_update")
    world.act("d", "ann", "upload")
    corpus.add("d", 2, ["version two says beta"])
    found = lambda user, q: [(p.version) for _, p in corpus.search(q, world.visible(user))]
    assert found("gil", "alpha") == [1] and found("gil", "beta") == []
    assert found("ann", "alpha") == [1] and found("ann", "beta") == []  # search finds the release, opening shows the draft
    assert world.readable_version("d", "ann") == 2
