import pytest

from docflow import InvalidWorkflow, Machine
from conftest import load


@pytest.mark.parametrize("name", ["report", "policy_analysis"])
def test_shipped_workflows_load(name):
    assert load(name).initial


def test_on_is_a_key_not_a_boolean():
    m = Machine.from_dict({"name": "x", "initial": "a", "states": {"a": {"on": {"go": {"to": "b"}}}, "b": {}}})
    assert m.actions("a") == ["go"]


def problems(raw):
    with pytest.raises(InvalidWorkflow) as e:
        Machine.from_dict(raw)
    return e.value.problems


def mk(**states):
    return {"name": "x", "initial": "a", "states": states}


def test_needs_states():
    assert problems({"name": "x"}) == ["a workflow needs at least one state under `states`"]


def test_every_problem_is_listed_at_once():
    found = problems(
        {"states": {"a": {"on": {"go": {"to": "nowhere"}, "undo": {"to": "a"}}}, "orphan": {}}, "initial": "zz"}
    )
    text = "\n".join(found)
    assert "needs a `name`" in text
    assert "`initial` must name one of the states" in text
    assert "`to` must name one of the states, got 'nowhere'" in text
    assert "`undo` is a name the host uses itself" in text


def test_unreachable_state():
    found = problems(mk(a={"on": {"go": {"to": "b"}}}, b={}, island={}))
    assert "state island can never be reached from a" in found


def test_bare_number_name_must_be_quoted():
    assert any("put it in quotes" in p for p in problems(mk(a={}, **{}) | {"states": {1: {}, "a": {}}}))


def test_unknown_who_and_role():
    found = problems(
        mk(a={"on": {"x": {"to": "a", "by": ["nobody", {"role": "king"}]}}})
    )
    assert any("must be `anyone`" in p for p in found)
    assert any("there is no role 'king'" in p for p in found)


def test_assigned_must_be_assigned_somewhere():
    found = problems(mk(a={"on": {"x": {"to": "a", "by": {"assigned": "ghosts"}}}}))
    assert any("no action assigns 'ghosts'" in p for p in found)


def test_assign_max_below_min():
    found = problems(mk(a={"on": {"x": {"to": "a", "assign": {"r": {"from": "anyone", "min": 3, "max": 1}}}}}))
    assert any("`max` is less than `min`" in p for p in found)


def test_unknown_requirement_and_bad_pass():
    found = problems(
        mk(
            a={
                                "on": {
                    "x": {"to": "a", "requires": {"type": "telepathy"}},
                    "y": {"to": "a", "requires": {"type": "approval", "pass": 0}},
                },
            }
        )
    )
    assert any("unknown requirement type 'telepathy'" in p for p in found)
    assert any("`pass` must be `any`, `all` or a whole number" in p for p in found)


def test_access_rule_checks():
    found = problems(
        mk(
            a={
                                "access": [
                    {"who": "public", "role": "king"},
                    {"who": "public", "serve": {"from": "nowhere"}},
                    {"who": "public", "when": {"audience": "unit"}},
                    {"who": "public", "colour": "red"},
                ],
            }
        )
    )
    text = "\n".join(found)
    assert "`role` is `editor`, `contributor`, `viewer` or `none`" in text
    assert "names a state that isn't there" in text
    assert "`when` names 'audience', which no action chooses" in text
    assert "there is no 'colour'" in text


def test_when_value_must_be_offered():
    raw = mk(
        a={
                        "access": [{"who": "public", "when": {"audience": "galaxy"}}],
            "on": {"p": {"to": "a", "choose": {"audience": ["unit", "public"]}}},
        }
    )
    assert any("which no action offers (unit, public)" in p for p in problems(raw))


def test_the_knowledge_workflow_loads():
    assert load("knowledge_asset").initial


def test_new_words_are_checked():
    raw = mk(
        a={
                        "guidance": 5,
            "on": {
                "z": {"to": "a", "by": "unit", "assign": {"r": {"from": "unit", "min": 3, "odd": True}}},
            },
        }
    )
    text = "\n".join(problems(raw))
    assert "`guidance` is text" in text
    assert "must be `anyone`" not in text  # `unit` is a who


def test_guidance_fills_in_choices():
    m = load("knowledge_asset")
    assert m.guidance_for("disseminated", {"level": "public"}) == "Approved asset; mention its level (public)."
    assert m.guidance_for("working") .startswith("Unapproved")


def test_a_word_the_language_does_not_have_is_refused():
    found = problems(mk(a={"colour": "red", "on": {"x": {"to": "a", "bye": "ann"}}}) | {"flavour": 1})
    text = "\n".join(found)
    assert "there is no 'flavour'" in text and "there is no 'colour'" in text and "there is no 'bye'" in text


def test_attributes_declare_what_rules_may_test():
    raw = mk(a={"access": [
        {"who": "anyone", "when": {"secrecy": ["R", "X"]}},
        {"who": {"same": {"unit": "nowhere"}}},
        {"who": "anyone", "when": {"ghost": "1"}},
    ]})
    raw["attributes"] = {"secrecy": {"values": ["SR", "R", "T", "B"]}}
    text = "\n".join(problems(raw))
    assert "gives secrecy the value 'X'" in text
    assert "`same` names 'nowhere'" in text
    assert "names 'ghost'" in text and "R" in text


def test_create_may_set_only_declared_attributes():
    raw = mk(a={})
    raw["create"] = {"by": "anyone", "set": ["secrecy"]}
    assert any("`attributes` does not declare" in p for p in problems(raw))


def test_final_is_not_a_word_of_the_language():
    """No state is marked as the end or the release: a workflow is states and the actions between them."""
    assert any("there is no 'final'" in p for p in problems(mk(a={"final": True})))


def test_the_words_for_groups_of_documents_are_gone():
    """A workflow governs single documents. The words that made a group, a package or a link are refused like any other unknown word."""
    text = "\n".join(problems(mk(a={"members": {"add": {"kinds": ["t"], "by": "anyone"}}, "emits": [],
                                   "on": {"x": {"to": "a", "document": {"name": "m", "workflow": "w"}}},
                                   "access": [{"who": "anyone", "kind": "text"}]}) | {"applies_to": "group", "links": {}}))
    for word in ("applies_to", "links", "members", "emits", "document", "kind"):
        assert f"there is no '{word}'" in text, word
    assert any("unknown requirement type 'members_in_state'" in p
               for p in problems(mk(a={"on": {"x": {"to": "a", "requires": {"type": "members_in_state", "state": "a"}}}})))


def test_the_vocabulary_reference_and_schema_are_generated_from_one_definition():
    from docflow.access_control.access_policy_engine import dsl
    schema = dsl.schema()
    assert set(schema["properties"]) == set(dsl.VOCAB["workflow"])
    assert set(schema["properties"]["states"]["additionalProperties"]["properties"]) == set(dsl.VOCAB["state"])
    assert "`attributes`" in dsl.reference() and "signature" in dsl.reference()


def test_an_upload_must_lead_to_the_state_it_is_in():
    found = problems(mk(a={"on": {"upload": {"to": "b"}}}, b={"on": {"upload": {"to": "b"}}}))
    assert any("a: `upload` must lead to 'a', not 'b'" in p for p in found) and not any(p.startswith("b:") for p in found)
    good = Machine.from_dict(mk(a={"on": {"upload": {"to": "a"}, "go": {"to": "b"}}}, b={}))
    assert good.actions("a") == ["upload", "go"]


def test_every_shipped_workflow_follows_the_upload_rule():
    """A file that broke the rule would not load; this makes the reason plain if one is added."""
    from conftest import WORKFLOWS
    for path in sorted(WORKFLOWS.glob("*.yml")):
        Machine.load(path)


def test_an_access_rule_may_carry_guidance_and_it_must_be_text():
    good = Machine.from_dict(mk(a={"access": [{"who": "author", "guidance": "for authors"}, {"who": "anyone"}]}))
    assert [r.guidance for r in good.access_rules("a")] == ["for authors", None]
    found = problems(mk(a={"access": [{"who": "author", "guidance": 5}]}))
    assert any("`guidance` is text, got 5" in p for p in found)


def test_a_rules_guidance_is_filled_in_like_a_states():
    m = load("knowledge_asset")
    rule = next(r for r in m.access_rules("working") if r.guidance)
    assert m.guidance_for("working", {"level": "unit"}, rule.guidance) == "Approved asset; mention its level (unit)."
