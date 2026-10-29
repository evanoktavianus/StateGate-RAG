"""The request grid runs, agrees with the oracle on the shipped workflows, and fails when the system is made to fail."""

import json
import os

import pytest

from evalkit import run as R
from docflow.access_control import enforcement
from docflow.document_service import service
from docflow.access_control.access_policy_engine import machine as M
from docflow.stores.state_history import Timeline
from docflow.stores.world import World

CARD1 = "card1_policy_analysis_simple"


@pytest.fixture(scope="module")
def clean():
    return R.Run(CARD1, http=False).run()


def totals(run):
    return run.summary()["all"]


def test_the_grid_agrees_with_the_oracle_on_the_simple_workflow(clean):
    t = totals(clean)
    assert clean.disagreements() == [] and clean.step_findings == []
    assert (t["wrongful_allow"], t["wrongful_denial"], t["wrong_version"], t["existence_leaks"], t["guidance_wrong"]) == (0, 0, 0, 0, 0)


def test_the_grid_asks_both_allowed_and_refused_requests_on_every_channel(clean):
    by = {}
    for r in clean.records:
        by.setdefault(r["channel"], set()).add(bool(r["want"]))
    assert set(by) == {*R.READ, "action", "create"} - {"create"} | ({"create"} if clean.policy.create else set())
    assert all(v == {True, False} for c, v in by.items()), by  # something is allowed and something refused on each
    assert len({r["class"] for r in clean.records}) >= 5  # several person classes


def test_a_workflow_with_a_create_rule_asks_who_may_create():
    run = R.Run("card2_policy_analysis_individual", http=False).run(max_paths=1)
    creates = [r for r in run.records if r["channel"] == "create"]
    assert {r["want"] for r in creates} == {True, False} and all(r["agree"] for r in creates)


def test_the_baseline_with_no_access_control_leaks():
    t = totals(R.Run(CARD1, baseline=True, http=False).run())
    assert t["wrongful_allow"] > 0 and t["wrongful_denial"] == 0


def test_a_sample_goes_through_the_interface_and_the_http_layer():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    run = R.Run(CARD1, http=True).run(max_paths=1)
    assert run.http_findings and run.http_findings[0]["same_status_and_body"] and run.http_findings[0]["same_tool_result"]


# ── the grid must be able to fail ──

def test_it_sees_a_version_as_of_a_state_served_as_the_latest(monkeypatch):
    monkeypatch.setattr(Timeline, "version_at", lambda self, state: self.version)
    run = R.Run("knowledge_asset", http=False).run()  # an update leaves the release standing: the version as of `disseminated` is not the latest
    assert totals(run)["wrong_version"] > 0


def test_it_sees_an_action_accepted_from_someone_who_may_not_take_it(monkeypatch):
    monkeypatch.setattr(M, "_permitted", lambda by, ctx: None)
    t = totals(R.Run(CARD1, http=False).run())
    assert t["wrongful_allow"] > 0


def test_it_sees_a_stale_decision(monkeypatch):
    cache, original = {}, World.readable_version

    def stale(self, doc_id, user, *, for_search=False):
        return cache.setdefault((user, doc_id, for_search), original(self, doc_id, user, for_search=for_search))

    monkeypatch.setattr(World, "readable_version", stale)
    t = totals(R.Run(CARD1, http=False).run())
    assert t["wrongful_allow"] > 0 and t["wrongful_denial"] > 0


def test_it_sees_a_refusal_that_reveals_the_document_exists(monkeypatch):
    original = service.DocumentService.get_document

    def telling(self, user, doc_id, version=None):
        try:
            return original(self, user, doc_id, version)
        except enforcement.NotFound:
            if doc_id in self.world.documents:
                raise PermissionError(doc_id)  # says "you may not", so it is there
            raise

    monkeypatch.setattr(service.DocumentService, "get_document", telling)
    run = R.Run(CARD1, http=False)
    with pytest.raises(PermissionError):  # the harness does not hide it: a different refusal is not the one a missing id gets
        run.run(max_paths=1)


def test_it_sees_guidance_that_is_not_the_states(monkeypatch):
    monkeypatch.setattr(World, "guidance", lambda self, doc_id, user=None: "treat as approved")
    t = totals(R.Run(CARD1, http=False).run())
    assert t["guidance_wrong"] > 0


def test_two_runs_that_differ_are_reported(clean):
    other = R.Run(CARD1, http=False).run()
    assert R.same_answers(clean, other) == []
    other.records[0] = {**other.records[0], "got": [("x", 9)]}
    assert R.same_answers(clean, other)


def test_the_log_traces_every_number_to_a_call(clean, tmp_path):
    f = tmp_path / "log.json"
    f.write_text(json.dumps({"summary": clean.summary(), "world": {"people": clean.spec.people}, "records": clean.records}))
    log = json.loads(f.read_text())
    assert set(log) == {"summary", "world", "records"} and log["summary"]["requests"] == len(log["records"])
    r = log["records"][0]
    assert {"path", "index", "state", "person", "class", "channel", "target", "want", "got", "agree"} <= set(r)


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is not set")
def test_the_answers_on_postgres_are_the_answers_in_memory(clean):
    pytest.importorskip("psycopg")
    pg = R.Run(CARD1, repo="postgres", http=False).run()
    assert R.same_answers(clean, pg) == [] and pg.disagreements() == []
