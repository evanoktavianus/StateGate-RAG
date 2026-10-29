"""The E6 pieces, checked with the scripted model: the three conditions, the runner (results on disk, resume, stop rules), the table. No call to the model service."""

import json

import pytest

from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller

from evalkit import utility
from evalkit.e6 import runner, sample

WF = "knowledge_asset"


def test_the_stratified_200_has_the_composition_asked_for():
    qs = json.loads((sample.__file__ and __import__("pathlib").Path(sample.__file__).resolve().parents[3] / "eval/data/questions" / "questions.json").read_text())
    picked = sample.draw(qs)
    assert len(picked) == 200 and len({q["id"] for q in picked}) == 200
    reg = [q for q in picked if q["workflow"] == "regulation_single"]
    assert len(reg) == 54
    from collections import Counter
    assert Counter(q["kind"] for q in picked if q not in reg) == {"content": 88, "version": 25, "status": 33}
    assert [q["id"] for q in sample.draw(list(reversed(qs)))] == [q["id"] for q in picked]


@pytest.fixture(scope="module")
def setup():
    world, corpus, oracle, spec = utility.build(WF, 8)
    person = next(p for p in sorted(spec.people) if 0 < sum(od.access(p) is not None for od in oracle.values()) < len(oracle))
    hidden = next(d for d, od in oracle.items() if od.access(person) is None)
    shown = next(d for d, od in oracle.items() if od.access(person) is not None)
    return world, corpus, oracle, spec, person, hidden, shown


def test_the_conditions(setup):
    world, corpus, oracle, spec, person, hidden, shown = setup
    seen = {}
    for cond in ("no_access_control", "filtering", "filtering_guidance"):
        tools = tool_controller.tools(DocumentService(runner.CondWorld(world, cond), corpus), person)
        seen[cond] = tools["list_documents"]()
    ids = lambda c: {d["id"] for d in seen[c]["documents"]}
    assert hidden in ids("no_access_control") and hidden not in ids("filtering") and hidden not in ids("filtering_guidance")
    assert ids("filtering") == ids("filtering_guidance") == {d for d, od in oracle.items() if od.access(person) is not None}
    assert all(d["guidance"] is None for d in seen["filtering"]["documents"]) and all(d["guidance"] is None for d in seen["no_access_control"]["documents"])
    assert any(d["guidance"] for d in seen["filtering_guidance"]["documents"])
    got = tool_controller.tools(DocumentService(runner.CondWorld(world, "no_access_control"), corpus), person)["get_document"](hidden)
    assert "passages" in got and "error" not in got
    assert tool_controller.tools(DocumentService(runner.CondWorld(world, "filtering"), corpus), person)["get_document"](hidden) == {"error": "not_found"}


def make_runs(setup, n, cond="filtering", role="non_reader"):
    world, corpus, oracle, spec, person, hidden, shown = setup
    return [{"key": f"q{i}|{cond}|{role}|m", "qid": f"q{i}", "kind": "content", "workflow": WF, "state": "x", "regulation": False, "condition": cond, "role": role, "model": "m",
             "question": "Berapa kode referensi dokumen itu?", "user": person, "category": "c"} for i in range(n)]


def go(setup, tmp_path, runs, **over):
    world, corpus, *_ = setup
    cfg = {"out": str(tmp_path), "scripted": True, "workers": 2, "threads": 2, "cap": 10, "start": 4, "max": 4, "models": ["m"], "latency": 0.0, "fail_rate": 0.0, "ramp_seconds": 1000, "monitor_seconds": 0.05, **over}
    return runner.Runner(runs, tmp_path, cfg, lambda: {WF: (world, corpus)}).run()


def test_results_are_written_as_they_finish_and_a_restart_skips_them(setup, tmp_path):
    runs = make_runs(setup, 6)
    s = go(setup, tmp_path, runs)
    assert s["stop"] is None and s["finished"] == 6
    rows = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert len(rows) == 6 and all(not ({"answer", "score", "events", "finished"} & set(r)) for r in rows)  # a record holds the tool calls and what they returned, not the answer
    assert all(s["tool_results"] and "reply" not in s for r in rows for s in r["steps"][:1])
    assert rows[0]["calls"] == 2 and rows[0]["prompt_tokens"] == 1300 and rows[0]["cost"] > 0
    assert list((tmp_path / "calls").glob("calls-*.jsonl"))
    s = go(setup, tmp_path, runs)  # nothing left to do
    assert s["finished"] == 6 and len(open(tmp_path / "results.jsonl").readlines()) == 6
    s = go(setup, tmp_path, runs + make_runs(setup, 8)[6:])  # two new ones only
    assert len(open(tmp_path / "results.jsonl").readlines()) == 8


def test_a_run_of_failed_runs_stops_the_run(setup, tmp_path):
    s = go(setup, tmp_path, make_runs(setup, 120), fail_rate=1.0, latency=0.05)
    assert s["stop"] is not None and s["stop"]["cause"] == "run_failures"


def test_the_spending_cap_stops_the_run(setup, tmp_path):
    s = go(setup, tmp_path, make_runs(setup, 40), cap=0.0002, latency=0.05)
    assert s["stop"] is not None and s["stop"]["cause"] == "spending"


def test_the_concurrency_is_held_when_asked(setup, tmp_path):
    s = go(setup, tmp_path, make_runs(setup, 20), hold=True, ramp_seconds=0.01, start=4, max=8, latency=0.05)
    assert s["stop"] is None and all(h["target"] == 4 for h in s["target_history"])


def test_wilson_interval():
    from evalkit.e6 import analysis as A
    lo, hi = A.wilson(0, 100)
    assert lo == 0.0 and abs(hi - 0.0370) < 1e-3
    lo, hi = A.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_the_table_counts_runs_with_a_context_leak_per_model_and_condition():
    from evalkit.e6 import analysis as A
    runs = [{"model": "m", "condition": "no_access_control", "context_leaks": [{"doc": "d"}]}] * 3 + [{"model": "m", "condition": "no_access_control", "context_leaks": []}] \
        + [{"model": "m", "condition": "filtering", "context_leaks": []}] * 4
    assert A.table(runs) == [["m", "no_access_control", 4, 3, "75.0%", "30.1% to 95.4%"], ["m", "filtering", 4, 0, "0.0%", "0.0% to 49.0%"]]
