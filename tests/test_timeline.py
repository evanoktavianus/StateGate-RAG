import pytest
from dataclasses import replace

from docflow import Context, Refused, Timeline
from conftest import load


def submit(tl, ctx, reviewers=("bob", "cat")):
    return tl.act("submit", ctx("ann", input={"reviewers": list(reviewers)}))


def test_new_document_starts_in_the_initial_state():
    tl = Timeline.new(load("report"))
    assert (tl.state, tl.version, tl.version_at("published")) == ("draft", 1, None)
    assert tl.entries == (("version", 1), ("state", "draft"))


def test_a_refused_action_adds_nothing(ctx):
    tl = Timeline.new(load("report"))
    with pytest.raises(Refused):
        tl.act("approve", ctx("bob"))
    assert tl.state == "draft" and len(tl.entries) == 2


def test_upload_adds_a_version_and_stays(ctx):
    tl = Timeline.new(load("report")).act("upload", ctx("ann"))
    assert (tl.state, tl.version) == ("draft", 2)


def test_upload_may_name_its_version(ctx):
    tl = Timeline.new(load("report")).act("upload", ctx("ann", version=7))
    assert tl.version == 7


def test_submit_records_who_was_chosen_before_the_move(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    assert tl.entries[-2:] == (("assigned", "reviewers", ("bob", "cat")), ("state", "pending_review"))
    assert tl.assigned() == {"reviewers": ("bob", "cat")}


def test_approval_pass_all_waits_for_every_reviewer(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    tl = tl.act("approve", ctx("bob"))
    assert tl.state == "pending_review"  # bob's part is recorded, the doc waits
    assert tl.given("approve") == ["bob"]
    with pytest.raises(Refused) as e:
        tl.act("approve", ctx("bob"))
    assert e.value.reason == "already_given"
    tl = tl.act("approve", ctx("cat"))
    assert tl.state == "editor_approved"


def test_only_assigned_reviewers_may_approve(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    with pytest.raises(Refused) as e:
        tl.act("approve", ctx("dan"))
    assert e.value.reason == "not_permitted"


def test_a_new_version_starts_the_approval_count_again(ctx):
    tl = submit(Timeline.new(load("report")), ctx).act("approve", ctx("bob"))
    tl = tl.act("upload", ctx("ann"))
    assert tl.given("approve") == []
    assert tl.state == "pending_review"


def test_reviewers_stay_chosen_when_the_document_goes_back(ctx):
    tl = submit(Timeline.new(load("report")), ctx).act("request_changes", ctx("bob"))
    assert tl.state == "draft" and tl.assigned()["reviewers"] == ("bob", "cat")


def test_the_version_as_of_a_state_is_set_on_arriving_there(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    tl = tl.act("approve", ctx("bob")).act("approve", ctx("cat")).act("publish", ctx("ann"))
    assert (tl.state, tl.version_at("published")) == ("published", 1)


def test_the_version_as_of_a_state_stays_while_the_working_line_goes_on(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    tl = tl.act("approve", ctx("bob")).act("approve", ctx("cat")).act("publish", ctx("ann"))
    tl = tl.act("start_revision", ctx("ann")).act("upload", ctx("ann"))  # published -> pending_review, version 2
    assert (tl.state, tl.version, tl.version_at("published")) == ("pending_review", 2, 1)
    tl = tl.act("approve", ctx("bob")).act("approve", ctx("cat")).act("publish", ctx("ann"))
    assert tl.version_at("published") == 2  # arriving again takes the version it holds then


def test_an_upload_adds_a_version_and_never_moves_the_state(ctx):
    tl = submit(Timeline.new(load("report")), ctx)
    tl = tl.act("approve", ctx("bob")).act("approve", ctx("cat"))
    assert tl.state == "editor_approved"
    assert "upload" not in load("report").actions("editor_approved")  # leaving the approval is its own action
    tl = tl.act("start_revision", ctx("ann"))
    assert (tl.state, tl.version) == ("pending_review", 1)  # no version was made by starting
    before = tl.state
    assert tl.act("upload", ctx("ann")).state == before


def test_version_at_is_the_version_when_the_state_was_last_entered(ctx):
    tl = Timeline.new(load("report"))
    tl = tl.act("upload", ctx("ann")).act("upload", ctx("ann"))  # draft, v3
    tl = submit(tl, ctx)
    assert tl.version_at("pending_review") == 3
    assert tl.version_at("published") is None
    tl = tl.act("upload", ctx("ann"))  # v4 in review
    assert tl.version_at("pending_review") == 3  # entered at v3, an upload doesn't re-enter
    tl = tl.act("request_changes", ctx("bob")).act("upload", ctx("ann"))
    tl = submit(tl, ctx)
    assert tl.version_at("pending_review") == 5


def test_pass_a_number():
    from docflow import Machine, Person
    m = Machine.from_dict({
        "name": "n", "initial": "a",
        "states": {"a": {"on": {"ok": {"to": "b", "requires": {"type": "approval", "pass": 2}}}}, "b": {}},
    })
    people = [Person(f"p{i}", role="editor") for i in range(3)]
    tl = Timeline.new(m)
    tl = tl.act("ok", Context(actor=people[0]))
    assert tl.state == "a"
    tl = tl.act("ok", Context(actor=people[1]))
    assert tl.state == "b"


def test_given_without_an_actor_is_just_refused():
    from docflow import Machine
    m = Machine.from_dict({
        "name": "n", "initial": "a",
        "states": {"a": {"on": {"ok": {"to": "b", "requires": {"type": "approval"}}}}, "b": {}},
    })
    # approval needs 1 and nobody has given: with no actor nothing can be recorded
    with pytest.raises(Refused) as e:
        Timeline.new(m).act("ok", Context())
    assert e.value.reason == "requirement_unmet"
