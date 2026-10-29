"""The mutation harness, the report, the utility measures and the probes run on small fixtures (smoke), and notice faults."""

from pathlib import Path

import pytest

from evalkit import mutate, probes, random_baseline, report, system_mutants, utility
from evalkit.worlds import ROOT

CARD1 = "card1_policy_analysis_simple"
PATH = ROOT / "workflows" / f"{CARD1}.yml"


def test_every_operator_has_sites_on_the_simple_workflow_or_is_explained():
    ops = {m["op"] for m in mutate.generate_mutants(PATH)}
    assert {"widen_who", "narrow_who", "swap", "change_version", "drop_rule", "condition", "default_flip", "transition_guard", "wrong_target", "upload"} - ops <= {"condition"}
    assert mutate.NOT_EXPRESSIBLE  # the negate operator is skipped, and says why


def test_a_mutant_that_does_not_load_is_discarded():
    with pytest.raises(Exception):
        mutate.load_mutant("name: x\ninitial: nowhere\nstates: {}\n")


def test_a_few_mutants_are_killed_and_the_oracle_stays_on_the_original(tmp_path):
    r = mutate.analyse(CARD1, seed=3, limit=5, cache=tmp_path)
    assert r["total"]["judged"] == 5 and r["total"]["killed_by_suite"] >= 3
    assert (tmp_path / f"{CARD1}.json").exists()
    again = mutate.analyse(CARD1, seed=3, limit=5, cache=tmp_path)  # cached results give the same answer
    assert again["total"] == r["total"]


def test_the_random_baseline_is_reproducible_and_bounded():
    pool = [("p", 0, f"x{i}", "get", "d") for i in range(100)]
    differing = {"m1": {pool[0]}, "m2": set(pool[:50])}
    a = random_baseline.repetitions(pool, 10, differing, seed=1, reps=30)
    assert a == random_baseline.repetitions(pool, 10, differing, seed=1, reps=30) and len(a) == 30
    assert sum(r["m2"] for r in a) > sum(r["m1"] for r in a)


def test_zero_failures_carry_an_upper_bound_not_a_bare_zero():
    assert report.rate(0, 300).startswith("0/300 (upper ") and report.wilson_upper(0, 300) == pytest.approx(0.0126, abs=0.002)
    assert report.rate(2, 300) == "2/300"


def test_the_utility_measures_run_with_the_scripted_model():
    u = utility.measure("card7_official_document", 6, 5)
    assert u["fill_mean"] == 1.0 and u["same_order_share"] == 1.0 and 0 <= u["authorized_yield_mean"] <= 1
    assert utility.scripted_ask_seconds(CARD1) > 0


@pytest.fixture(scope="module")
def results():
    return {r.probe: r for r in probes.run_all("card7_official_document", http=False)}


@pytest.mark.parametrize("probe", ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P7b"])
def test_each_probe_passes_on_the_clean_engine(results, probe):
    assert results[probe].status == "pass", results[probe]


def test_out_of_scope_items_are_marked_not_hidden(results):
    assert results["G1"].status == results["G2"].status == "not_applicable" and results["P7b"].status in ("pass", "not_run")


def test_the_probe_for_hidden_documents_fails_when_statistics_are_pooled():
    assert system_mutants.run_probe7("card7_official_document", None).startswith("P7 pass")
    assert system_mutants.run_probe7("card7_official_document", "pooled_keyword_statistics").startswith("P7 fail")


def test_every_system_mutant_patch_matches_exactly_once(tmp_path):
    import shutil
    for name, *_ in system_mutants.MUTANTS:  # each in a fresh copy: two patches touch the same file
        d = tmp_path / name
        shutil.copytree(ROOT / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__"))
        system_mutants.apply(name, d)  # raises if a patch does not match once


# ── the upload rule of the path generator ──
def _actions(path):
    return [s.action for s in path.steps]


@pytest.mark.parametrize("workflow, back", [("card7_official_document", "return"), ("card2_policy_analysis_individual", "return"), ("knowledge_asset", "start_update")])
def test_every_way_back_into_a_drafting_state_is_followed_by_an_upload(workflow, back):
    from evalkit import paths as P
    ps = P.generate(workflow)
    followed = [a for a in map(_actions, ps.paths) if any(x == back and i + 1 < len(a) and a[i + 1] == "upload" for i, x in enumerate(a))]
    assert followed, f"{workflow}: no path uploads right after {back}"
    assert not ps.uncoverable_edges


def test_the_grid_tells_a_version_as_of_a_state_from_the_latest_when_a_policy_can_differ():
    from evalkit.run import Run
    from evalkit import mutate as M
    raw = M.load_raw(ROOT / "workflows" / "card7_official_document.yml")
    raw["states"]["signed"]["access"][0]["serve"] = {"from": "draft"}   # the version when the document last entered `draft`, not the one signed
    run = Run("card7_official_document", http=False, machine=M.load_mutant(M.dump(raw)), everyone=True, fast=True).run()
    assert run.disagreements() or run.step_findings


def test_the_latest_version_fault_is_killed_where_a_policy_can_show_it():
    # in knowledge_asset a version is uploaded after the asset was disseminated, while readers still get the disseminated one
    n, _ = system_mutants.run_grid("knowledge_asset", "serve_latest_not_version_as_of_state")
    assert n > 0


def test_the_path_set_does_not_depend_on_the_hash_seed():
    import os, subprocess, sys
    code = ("import json,hashlib;from evalkit import paths;"
            "print(hashlib.md5(json.dumps([p.to_json() for p in paths.generate('card1_policy_analysis_simple').paths],sort_keys=True,default=str).encode()).hexdigest())")
    outs = {subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=ROOT,
                           env={**os.environ, "PYTHONHASHSEED": str(s), "PYTHONPATH": os.pathsep.join(sys.path)}).stdout.strip() for s in (1, 2, 3)}
    assert len(outs) == 1 and "" not in outs


def test_the_stratified_sample_is_proportional_with_a_floor_of_one_per_operator():
    work = [{"op": o, "hash": f"{o}{i}"} for o, n in (("a", 50), ("b", 30), ("c", 15), ("d", 1)) for i in range(n)]
    s = mutate.stratified_sample(work, 20, seed=1)
    ops = [m["op"] for m in s]
    assert len(s) == 20 and len({m["hash"] for m in s}) == 20
    assert set(ops) == {"a", "b", "c", "d"} and ops.count("a") > ops.count("b") > ops.count("c")
    assert [m["hash"] for m in s] == [m["hash"] for m in mutate.stratified_sample(work, 20, seed=1)]
    assert len(mutate.stratified_sample(work, 20, seed=2)) == 20


def test_the_equivalence_check_reads_the_graph():
    from docflow import Machine
    from evalkit import equivalence, model as M
    card1 = M.from_machine("card1_policy_analysis_simple", Machine.load(ROOT / "workflows" / "card1_policy_analysis_simple.yml"))
    assert equivalence.decide(card1, 'finalized.access[0]: serve {"from": "finalized"} -> "latest"')["verdict"] == "equivalent"  # no upload after finalized
    card7 = M.from_machine("card7_official_document", Machine.load(ROOT / "workflows" / "card7_official_document.yml"))
    assert equivalence.decide(card7, 'signed.access[0]: serve {"from": "signed"} -> {"from": "draft"}')["verdict"] == "gap"  # uploads in draft, and a return to it
    assert equivalence.decide(card7, 'signed.access[0]: serve {"from": "signed"} -> {"from": "approved"}')["verdict"] == "equivalent"
    assert equivalence.decide(card7, 'signed.access[0]: serve {"from": "signed"} -> "nothing"') is None
