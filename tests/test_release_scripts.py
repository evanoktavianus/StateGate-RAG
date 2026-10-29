"""The small scripts around the results: the default store of the grid and the log reader that takes `.json.gz`. No network."""

import gzip
import json
from pathlib import Path

from evalkit import report
from evalkit import run as run_mod

ROOT = Path(__file__).parent.parent


def _store_of_a_grid_run(monkeypatch, argv, database_url):
    seen = {}

    class Fake:
        def __init__(self, workflow, baseline=False, repo="memory", http=True):
            seen["repo"] = repo

        def run(self, max_paths=None):
            return self

        def summary(self):
            return {}

        def disagreements(self):
            return []

    monkeypatch.setattr(run_mod, "Run", Fake)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    if database_url:
        monkeypatch.setenv("DATABASE_URL", database_url)
    run_mod.main(argv)
    return seen["repo"]


def test_the_grid_reads_from_postgresql_when_there_is_a_database_and_from_memory_otherwise(monkeypatch, capsys):
    assert _store_of_a_grid_run(monkeypatch, ["wf"], "postgresql://x") == "postgres"
    assert _store_of_a_grid_run(monkeypatch, ["wf"], None) == "memory"
    assert _store_of_a_grid_run(monkeypatch, ["wf", "--repo", "memory"], "postgresql://x") == "memory"  # the option wins


def test_a_grid_log_is_read_from_its_gz_copy_when_the_plain_file_is_not_there(tmp_path):
    (tmp_path / "a.json.gz").write_bytes(gzip.compress(json.dumps({"records": [1]}).encode()))
    assert report.load(tmp_path, "a.json") == {"records": [1]}
    (tmp_path / "a.json").write_text(json.dumps({"records": [2]}))
    assert report.load(tmp_path, "a.json") == {"records": [2]}  # the plain file first
    assert report.load(tmp_path, "b.json") is None


def test_the_default_run_adds_e6_and_e7_only_when_a_model_can_be_reached(monkeypatch, capsys):
    from evalkit import reproduce
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert reproduce.chosen(None, False) == set(reproduce.NO_MODEL.split(","))
    assert "skipped" in capsys.readouterr().out
    assert reproduce.chosen(None, True) >= {"e6", "e7"}  # the fake model
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert reproduce.chosen(None, False) >= {"e6", "e7"} and reproduce.chosen("grid", False) == {"grid"}


def test_the_unfiltered_retrieval_reads_every_version_and_the_filtered_one_only_what_may_be_read():
    """`--no-filter` of E5: the search is given every (document, version) of the corpus; a hit the asker may not read is counted as hidden."""
    from docflow.stores.search_index import Corpus
    from evalkit import retrieval_utility as U

    corpus = Corpus([], __import__("array").array("f"))
    corpus.add("a", 1, ["zebra alpha old"], None)
    corpus.add("a", 2, ["zebra alpha new"], None)
    corpus.add("b", 1, ["zebra beta secret"], None)
    visible = {"a": 2}  # what one person may read
    assert [(p.doc, p.version) for _, p in corpus.search("zebra", visible, 10)] == [("a", 2)]
    everything = U._everything(corpus)
    found = {(p.doc, p.version) for _, p in corpus.search("zebra", everything, 10)}
    assert found == {("a", 1), ("a", 2), ("b", 1)}
    assert {hit for hit in found if visible.get(hit[0]) != hit[1]} == {("a", 1), ("b", 1)}  # what `hidden_in_top10` counts


def test_a_static_policy_is_scored_like_a_request_of_the_grid():
    """The static baseline scores an outcome as the grid does: a wrongful allow, a wrongful denial, and a wrong version (which counts in both of the others)."""
    from evalkit import static_baseline as B

    assert B.score([["d", 1]], [["d", 1]]) == (False, False, False)  # served as the oracle says
    assert B.score([], [["d", 2]]) == (True, False, False)  # allowed, the oracle denies
    assert B.score([["d", 1]], []) == (False, True, False)  # denied, the oracle allows
    assert B.score([["d", 1]], [["d", 2]]) == (True, True, True)  # the newest version where the oracle serves an earlier one
