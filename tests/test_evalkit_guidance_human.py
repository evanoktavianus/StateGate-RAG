"""The human check of the guidance judges: the blind sample, the check of the labels, and the agreement with each judge. No model call, no network."""

import csv

import pytest

from evalkit.e6 import guidance as G
from evalkit.e6 import guidance_human as H


def records(n_per_cell=3):
    answers, verdicts = [], []
    for k, kind in enumerate(H.KINDS):
        for c, cond in enumerate(H.CONDITIONS):
            for r in range(n_per_cell):
                doc = f"reg-{k}{c}{r}"
                answers.append({"pair": f"q{k}{c}{r}|reader|m", "condition": cond, "kind": kind, "qid": "q1", "answer": f"Menurut {doc} berlaku",
                                "items": [{"doc": doc, "version": 1, "state": "s", "guidance": "g", "kind": kind}]})
                verdicts.append({"pair": answers[-1]["pair"], "condition": cond, "kind": kind, "error": None,
                                 "items": [{"doc": doc, "uses_document": True, "follows_guidance": r != 0, "reason": "because"}]})
    return answers, verdicts


def test_the_sample_takes_the_same_number_from_every_cell_and_hides_the_condition():
    answers, verdicts = records()
    rows, sample = H.build(answers, verdicts, verdicts, {"q1": "pertanyaan"}, seed=5, per_cell=2)
    assert len(rows) == len(sample["items"]) == 4 * 3 * 2
    assert {(c["kind"], c["condition"]): c["taken"] for c in sample["cells"]} == {(k, c): 2 for k in H.KINDS for c in H.CONDITIONS}
    assert [r["id"] for r in rows] == list(range(1, 25)) and all(r["label"] == "" for r in rows)
    assert not any(cond in str(r) for r in rows for cond in H.CONDITIONS)
    again = H.build(answers, verdicts, verdicts, {"q1": "pertanyaan"}, seed=5, per_cell=2)[1]["items"]
    assert [i["doc"] for i in again] == [i["doc"] for i in sample["items"]]  # a fixed seed gives the same sample


def test_a_cell_with_too_few_items_is_reported_as_a_shortfall():
    answers, verdicts = records(n_per_cell=1)
    _, sample = H.build(answers, verdicts, verdicts, {"q1": "q"}, seed=1, per_cell=3)
    assert all(c["taken"] == 1 and c["shortfall"] == 2 for c in sample["cells"])


def write_labels(path, labels):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "note"])
        for i, label in labels.items():
            w.writerow([i, label, ""])
    return path


def test_the_check_refuses_an_empty_or_unknown_label_and_warns_about_a_fill(tmp_path):
    with pytest.raises(ValueError):
        H.read_labels(write_labels(tmp_path / "labels_a.csv", {1: "KEEPS", 2: ""}))
    with pytest.raises(ValueError):
        H.read_labels(write_labels(tmp_path / "labels_b.csv", {1: "KEEPS", 2: "MAYBE"}))
    labels = H.read_labels(write_labels(tmp_path / "labels_c.csv", {i: "KEEPS" for i in range(1, 11)}))
    assert any(line.startswith("WARNING") for line in H.check(labels))
    assert H.read_labels(write_labels(tmp_path / "labels_d.csv", {1: "keeps", 2: "Breaks", 3: "not used"})) == {1: "KEEPS", 2: "BREAKS", 3: "NOT USED"}


def test_the_score_counts_a_judge_that_agrees_with_every_label_as_perfect():
    answers, verdicts = records()
    _, sample = H.build(answers, verdicts, verdicts, {"q1": "q"}, seed=2, per_cell=2)
    human = {i["id"]: H.verdict(i["glm"]) for i in sample["items"]}
    text = H.score(sample, human)
    assert text.count("agreement 24/24 (100.0%), kappa 1.000") == 2
    flipped = dict(human)
    flipped[1] = "BREAKS" if human[1] != "BREAKS" else "KEEPS"
    assert "agreement 23/24" in H.score(sample, flipped) and "Disagreements on the binary label (1)" in H.score(sample, flipped)
    assert "Two labellers" in H.score(sample, human, flipped)


def test_kappa_is_one_for_equal_labels_and_undefined_when_all_labels_are_equal():
    assert H.kappa([True, False, True], [True, False, True]) == 1.0
    assert H.kappa([False, False], [False, False]) is None
