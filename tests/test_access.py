"""Policy analysis: private until released, then open to the audience chosen at release."""

from docflow import Context, Timeline, readable_version, resolve
from conftest import load, person

M = load("policy_analysis")

AUTHOR = person("ann", "contributor")
QA = person("qa", "editor", groups=("qa team",))
MEMBER = person("mem", "viewer")  # has a role in the workspace
ORG = person("org", None, in_org=True)  # same organization, no role here
OUTSIDER = person("out", None)  # signed in, another organization

OTHERS = (QA, MEMBER, ORG, OUTSIDER)


def ctx(actor=None, **kw):
    return Context(actor=actor, author_id="ann", members=(AUTHOR, QA, MEMBER), **kw)


def drafted():
    return Timeline.new(M)


def in_review():
    return drafted().act("submit", ctx(AUTHOR, input={"reviewer": ["qa"]}))


def released(audience):
    return in_review().act("publish", ctx(QA, input={"audience": audience}))


def reads(tl, who, *, search=False):
    return readable_version(tl, who, Context(author_id="ann"), for_search=search)


def test_a_draft_is_private():
    tl = drafted()
    assert reads(tl, AUTHOR) == 1
    assert [reads(tl, p) for p in OTHERS] == [None, None, None, None]


def test_the_chosen_reviewer_opens_the_draft_while_it_is_reviewed():
    tl = in_review()
    assert reads(tl, QA) == 1  # assigned comes from the timeline, not the caller
    assert [reads(tl, p) for p in (MEMBER, ORG, OUTSIDER)] == [None, None, None]


def test_someone_not_assigned_does_not_get_in():
    assert reads(in_review(), person("other_qa", "editor")) is None


def test_search_finds_nothing_until_a_release_even_for_the_author():
    # the draft is open to its authors, but `find: { from: released }` keeps it out of search
    tl = in_review()
    assert reads(tl, AUTHOR) == 1
    assert reads(tl, AUTHOR, search=True) is None


def test_released_to_the_unit_reaches_people_with_a_role_only():
    tl = released("unit")
    assert tl.state == "released" and tl.version_at("released") == 1
    assert reads(tl, MEMBER) == 1
    assert reads(tl, AUTHOR) == 1
    assert reads(tl, ORG) is None and reads(tl, OUTSIDER) is None


def test_released_to_the_organization():
    tl = released("organization")
    assert reads(tl, MEMBER) == 1
    assert reads(tl, ORG) == 1
    assert reads(tl, OUTSIDER) is None


def test_released_to_the_public():
    tl = released("public")
    assert all(reads(tl, p) == 1 for p in (AUTHOR, MEMBER, ORG, OUTSIDER))


def test_search_finds_the_release_for_everyone_it_reaches():
    tl = released("public")
    assert all(reads(tl, p, search=True) == 1 for p in (AUTHOR, MEMBER, ORG, OUTSIDER))


def test_a_new_draft_does_not_take_the_release_away():
    tl = released("public").act("start_update", ctx(AUTHOR)).act("upload", ctx(AUTHOR))  # released -> working, v2
    assert (tl.state, tl.version, tl.version_at("released")) == ("working", 2, 1)
    assert reads(tl, AUTHOR) == 2  # the author works on the newest
    assert reads(tl, OUTSIDER) == 1  # everyone else keeps release 1
    assert reads(tl, MEMBER) == 1
    assert reads(tl, OUTSIDER, search=True) == 1
    assert reads(tl, AUTHOR, search=True) == 1  # search finds what was released, for the author too


def test_while_the_new_draft_is_reviewed_the_release_still_stands():
    tl = released("public").act("start_update", ctx(AUTHOR)).act("upload", ctx(AUTHOR)).act("submit", ctx(AUTHOR, input={"reviewer": ["qa"]}))
    assert tl.state == "under_review"
    assert reads(tl, QA) == 2 and reads(tl, OUTSIDER) == 1


def test_a_narrower_release_takes_the_document_from_those_it_no_longer_reaches():
    tl = released("public")
    assert reads(tl, OUTSIDER) == 1
    tl = tl.act("start_update", ctx(AUTHOR)).act("upload", ctx(AUTHOR)).act("submit", ctx(AUTHOR, input={"reviewer": ["qa"]}))
    tl = tl.act("publish", ctx(QA, input={"audience": "unit"}))
    assert tl.version_at("released") == 2
    assert reads(tl, OUTSIDER) is None
    assert reads(tl, MEMBER) == 2


def test_the_rule_that_applied_is_reported():
    access = resolve(released("public"), OUTSIDER)
    assert (access.role, access.serve, access.find) == ("viewer", 1, 1)
    assert resolve(drafted(), AUTHOR, Context(author_id="ann")).serve == 1
    # no rule speaks of a member of the unit in a released-to-unit document: their own role stands
    assert resolve(released("unit"), MEMBER) is None


def test_a_workflow_without_access_rules_gives_the_latest_to_anyone_with_a_role():
    tl = Timeline.new(load("report"))
    c = Context(author_id="ann")
    assert readable_version(tl, MEMBER, c) == 1
    assert readable_version(tl, OUTSIDER, c) is None
