import pytest

from docflow import Context, Refused
from conftest import load


def test_step_follows_a_listed_action(ctx):
    m = load("report")
    assert m.step("draft", "invalidate", ctx("ann")) == "invalid"


def test_an_unlisted_action_is_refused():
    m = load("report")
    with pytest.raises(Refused) as e:
        m.step("invalid", "upload")
    assert e.value.reason == "not_accepted"
    assert e.value.detail == {"action": "upload", "state": "invalid"}


def test_unknown_state():
    with pytest.raises(Refused) as e:
        load("report").step("limbo", "upload")
    assert e.value.reason == "unknown_state"


def test_by_author_only(ctx):
    m = load("report")
    with pytest.raises(Refused) as e:
        m.step("draft", "invalidate", ctx("bob"))
    assert e.value.reason == "not_permitted"


def test_by_needs_an_actor():
    with pytest.raises(Refused):
        load("report").step("draft", "invalidate", Context())


def test_assign_needs_enough_people(ctx):
    m = load("report")
    with pytest.raises(Refused) as e:
        m.step("draft", "submit", ctx("ann", input={"reviewers": []}))
    assert (e.value.reason, e.value.detail) == ("assignment_needed", {"name": "reviewers", "min": 1})


def test_assign_rejects_people_who_cannot_be_chosen(ctx):
    m = load("report")
    # dan is a viewer; ann is the author and is excluded
    for bad in (["dan"], ["ann"], ["stranger"]):
        with pytest.raises(Refused) as e:
            m.step("draft", "submit", ctx("ann", input={"reviewers": bad}))
        assert e.value.reason == "assignment_invalid"
    assert m.step("draft", "submit", ctx("ann", input={"reviewers": ["bob"]})) == "pending_review"


def test_assign_input_must_be_a_list(ctx):
    with pytest.raises(Refused) as e:
        load("report").step("draft", "submit", ctx("ann", input={"reviewers": "bob"}))
    assert e.value.reason == "assignment_invalid"


def test_choice_is_required(ctx):
    m = load("policy_analysis")
    c = ctx("bob", assigned={"reviewer": ("bob",)})
    with pytest.raises(Refused) as e:
        m.step("under_review", "publish", c)
    assert e.value.reason == "choice_needed"
    assert e.value.detail["options"] == ["unit", "organization", "public"]


def test_next_states():
    m = load("report")
    assert m.next_states("draft") == ["invalid", "pending_review", "draft"]  # by action name; upload stays
