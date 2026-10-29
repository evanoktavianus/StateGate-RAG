"""The model of a workflow, the paths that cover it, and the situations they replay into. The expected counts are those of the workflow files."""

from pathlib import Path

import pytest
import yaml

from evalkit import model as model_mod
from evalkit import paths
from evalkit.replay import Step, Unfireable, replay
from evalkit.worlds import Spec, load_spec

# states, edges (transitions), edge-pairs: counted from the workflow files
HAND = {
    "card1_policy_analysis_simple": (2, 2, 2),
    "card2_policy_analysis_individual": (4, 6, 10),  # `return` added in submitted: pairs 3 edges into working_document x2 + 1 into review x2 + 1 into submitted x2
    "policy_analysis_team_single": (4, 6, 10),
    "policy_analysis_other_unit_single": (5, 7, 11),  # 3 edges into working_document x2, then 1x2, 1x2, 1 into handed_over x1 (share)
    "card5_policy_analysis_sensitive": (5, 7, 11),
    "knowledge_asset": (3, 7, 16),  # states working, quality_assurance, disseminated; pairs 3 into working x2 + 2 into QA x3 + 2 into disseminated x2
    "card7_official_document": (5, 7, 11),  # draft, under_approval, approved, signed, stored; 3 edges into draft x2, 1x2, 1x2, 1 into signed x1
    "regulation_single": (5, 8, 12),
}


@pytest.fixture(scope="module")
def sets():
    return {n: paths.generate(n) for n in HAND}


@pytest.mark.parametrize("name", HAND)
def test_the_model_has_the_states_edges_and_pairs_counted_by_hand(name, sets):
    m = sets[name].model
    assert (len(m.states), len(m.edges), len(m.pairs())) == HAND[name]


@pytest.mark.parametrize("name", HAND)
def test_every_state_edge_and_pair_is_covered_and_nothing_is_dropped(name, sets):
    ps = sets[name]
    row = ps.table_row()
    assert row["states"][0] == row["states"][1] and row["edges"][0] == row["edges"][1] and row["edge_pairs"][0] == row["edge_pairs"][1]
    assert not ps.uncoverable_edges and not ps.uncoverable_pairs
    assert row["unexplained"] == {"edges": [], "pairs": [], "options": []}


@pytest.mark.parametrize("name", HAND)
def test_every_path_replays_without_a_refusal_and_ends_where_it_fired(name, sets):
    ps = sets[name]
    total = 0
    for p in ps.paths:
        seen = list(paths.situations(ps, p))
        total += len(seen)
        assert len(seen) == len(p.steps) + 1
        assert seen[-1][0].state == p.firings[-1].after
    assert total == sum(len(p.steps) + 1 for p in ps.paths)


def test_card1_by_hand(sets):
    ps = sets["card1_policy_analysis_simple"]
    assert ps.edges_covered() == {("working_document", "upload"), ("working_document", "mark_finalized")}
    assert ps.pairs_covered() == {
        (("working_document", "upload"), ("working_document", "upload")), (("working_document", "upload"), ("working_document", "mark_finalized")),
    }


def test_every_option_of_the_level_choice_is_made_and_carried_on(sets):
    ps = sets["knowledge_asset"]
    levels = {f.option for p in ps.paths for f in p.firings if f.option}
    assert levels == {"named", "unit", "organization", "public"}
    # a path reaches `disseminated` at each level
    reached = {dict(s.choices).get("level") for p in ps.paths for s, _ in paths.situations(ps, p) if s.state == "disseminated"}
    assert reached == {"named", "unit", "organization", "public"}


def test_every_value_of_an_attribute_set_at_creation_is_tried(sets):
    ps = sets["card7_official_document"]
    assert {p.attrs["secrecy"] for p in ps.paths} == {"B", "SR", "R", "T"}
    for level in ("B", "SR", "R", "T"):
        assert any(p.attrs["secrecy"] == level and p.firings[-1].after == "stored" for p in ps.paths), level


def test_an_outside_event_is_fired_by_the_host(sets):
    ps = sets["regulation_single"]
    steps = [s for p in ps.paths for s in p.steps if s.event]
    assert {s.event for s in steps} == {"promulgated", "amendment_in_force", "repeal_in_force"} and all(s.actor is None for s in steps)


def test_a_recorded_step_the_system_refuses_is_an_error():
    ps = paths.generate("card1_policy_analysis_simple")
    bad = [Step("z_outsider", "mark_finalized")]
    with pytest.raises(Unfireable):
        list(replay(ps.spec, ps.model, ps.paths[0].attrs, bad))


def test_what_cannot_be_fired_is_reported_with_the_reason_and_not_dropped(tmp_path):
    """An action only a person the cast does not have may take: uncoverable, and everything behind it with it."""
    f = tmp_path / "nobody.yml"
    f.write_text(yaml.dump({"name": "n", "initial": "a", "default_access": "deny", "states": {
        "a": {"access": [{"who": "author"}], "on": {"go": {"to": "b", "by": {"user": "nobody@example.test"}}, "stay": {"to": "a", "by": "author"}}},
        "b": {"access": [{"who": "author"}]}}}))
    spec = load_spec("card1_policy_analysis_simple")
    spec = Spec(**{**spec.__dict__, "workflow": "nobody", "workflow_file": f})
    ps = paths.generate("nobody", spec)
    assert ("a", "go") in ps.uncoverable_edges and "not_permitted" in ps.uncoverable_edges[("a", "go")]
    assert ("a", "stay") in ps.edges_covered()
    assert ps.table_row()["states"] == [1, 2]  # `b` is never reached


def test_every_state_of_every_planned_workflow_has_a_stage_and_the_targets_add_up():
    """The stage table of the question plan names every state of its workflows with one of the thirteen stages; the plan has the compositions of the evaluation plan."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("qplan", Path(__file__).parent.parent / "eval" / "evalkit" / "qplan.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    assert len(q.STAGES) == 15 and q.STATUS["card1_policy_analysis_simple"]["finalized"] == "final" and q.STATUS["card7_official_document"]["stored"] == "diarsipkan"
    for wf, table in q.STATUS.items():
        states = set(paths.generate(wf).model.states)
        assert states == set(table), (wf, states ^ set(table))
        assert set(table.values()) <= set(q.STAGES), (wf, set(table.values()) - set(q.STAGES))
    assert len(q.STATUS_Q) >= 6
    for stem in q.STATUS_Q:  # no wording gives an answer away
        assert not any(stage in stem.lower() for stage in q.STAGES), stem
    t = q.targets()
    assert sum(t.values()) == 1200
    assert {k: sum(v for (w, kk), v in t.items() if kk == k) for k in q.KINDS} == {"content": 660, "version": 300, "status": 240}
    for bucket, (n, wfs) in q.BUCKETS.items():
        assert sum(v for (w, k), v in t.items() if w in wfs) == n, bucket


def test_a_pair_that_differs_by_a_missing_piece_is_not_a_version_pair():
    import importlib.util
    spec = importlib.util.spec_from_file_location("qplan", Path(__file__).parent.parent / "eval" / "evalkit" / "qplan.py")
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    assert q.related("27 Mei 2027", "27 Mei") and q.related("12", "12,5%") and q.related("ARD 8", "ARD 8,06 miliar")
    assert not q.related("9,60%", "8,04%") and not q.related("20 Desember 2027", "20 November 2027") and not q.related("ARD 8,06 miliar", "ARD 1,30 miliar")
