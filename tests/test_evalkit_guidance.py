"""The guidance judge: which documents a run holds, the pairing with its control, the reading of the judge's reply, the rates and the agreement. No model call, no network."""

import json

from docflow import model_client as mc
from evalkit.e6 import guidance as G

REPEALED = "This regulation has been repealed. Do not present as current."
APPROVED = "Approved asset; mention its level (unit)."


def run(cond, guidance, answer="jawaban", key="k", doc="reg-1"):
    hit = {"id": doc, "version": 1, "state": "s", "guidance": guidance if cond != "filtering" else None, "text": "t"}
    return {"key": f"{key}|{cond}", "qid": "q1", "role": "reader", "model": "m", "user": "u", "condition": cond, "answer": answer,
            "steps": [{"tool_results": [{"name": "search", "arguments": {}, "result": {"hits": [hit]}}]}]}


def test_kinds_are_told_apart_by_the_text_of_the_guidance():
    assert G.kind_of(REPEALED) == "repealed" and G.kind_of(APPROVED) == "approved_level" and G.kind_of("Draft under review, not in force.") == "draft_or_review"
    assert G.kind_of("something else") is None


def test_a_run_holds_the_documents_whose_tools_returned_guidance_once():
    r = run("filtering_guidance", REPEALED)
    r["steps"].append(r["steps"][0])
    assert [(i["doc"], i["kind"]) for i in G.items(r)] == [("reg-1", "repealed")]
    assert G.items(run("filtering", REPEALED)) == []


def test_select_pairs_each_guided_run_with_its_control_and_drops_runs_without_an_answer():
    runs = [run("filtering_guidance", REPEALED), run("filtering", REPEALED, answer="ada"), run("filtering_guidance", REPEALED, answer=None, key="x")]
    rows = G.select(runs, 100)
    assert [r["condition"] for r in rows] == ["filtering", "filtering_guidance"] and {r["pair"] for r in rows} == {"q1|reader|m"}
    assert all(r["items"][0]["guidance"] == REPEALED for r in rows)
    assert G.select([run("filtering_guidance", REPEALED)], 100) == []  # no control run


def test_the_judge_reply_is_matched_to_the_documents_asked_even_when_a_suffix_differs():
    held = [{"doc": "reg-1/text"}]
    reply = json.dumps({"items": [{"doc": "reg-1", "uses_document": True, "follows_guidance": False, "reason": "r"}]})
    got = G.parse("noise " + reply, held)
    assert [(g["doc"], g["uses_document"], g["follows_guidance"], g["reason"]) for g in got] == [("reg-1/text", True, False, "r")]
    try:
        G.parse(reply, [{"doc": "reg-2"}])
    except ValueError:
        pass
    else:
        raise AssertionError("another document must be refused")


def test_rates_count_runs_not_documents_and_keep_approved_level_apart():
    answers = [{"pair": "p", "condition": "filtering", "items": [{"doc": "a", "kind": "repealed"}, {"doc": "b", "kind": "approved_level"}]}]
    item = lambda d, uses, follows: {"doc": d, "uses_document": uses, "follows_guidance": follows, "reason": ""}
    verdicts = [{"pair": "p", "condition": "filtering", "error": None, "items": [item("a", True, False), item("b", True, False)]}]
    stat = G.rates(answers, verdicts)
    assert stat[("filtering", "repealed")] == [1, 1] and stat[("filtering", "restrictions")] == [1, 1] and stat[("filtering", "approved_level")] == [1, 1]
    assert G.rates(answers, [{**verdicts[0], "error": "unreadable"}]) == {}


def test_kappa_and_the_wilson_interval():
    assert G.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert G.kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert G.kappa([], []) is None
    lo, hi = G.wilson(0, 100)
    assert lo == 0.0 and 0.03 < hi < 0.04


def test_nostate_answers_join_the_guided_documents_of_their_pair():
    answers = G.select([run("filtering_guidance", REPEALED), run("filtering", REPEALED, answer="ada")], 100)
    out = G.add_nostate(answers, [{"qid": "q1", "role": "reader", "model": "m", "user": "u", "answer": "tanpa status"}, {"qid": "q1", "role": "reader", "model": "m", "user": "u", "answer": None}])
    assert [a["condition"] for a in out] == ["filtering", "filtering_guidance", "filtering_nostate"] and out[-1]["items"] == answers[1]["items"]


def test_a_document_the_answer_does_not_name_is_not_asked_of_the_judge():
    answers = G.select([run("filtering_guidance", REPEALED), run("filtering", REPEALED, answer="ada")], 100)
    client = mc.ModelClient(mc.Settings(model="test"), transport=mc.Scripted([]))  # a call would raise: the script is empty
    verdicts = G.judge(answers, client, {"q1": "pertanyaan"})
    assert [v["error"] for v in verdicts] == [None, None] and all(not v["items"][0]["uses_document"] for v in verdicts)


def test_the_scripted_judge_runs_end_to_end_and_the_rubric_has_a_version():
    answers = G.select([run("filtering_guidance", REPEALED, answer="Menurut reg-1 berlaku"), run("filtering", REPEALED, answer="Menurut reg-1 berlaku")], 100)
    reply = mc.chat_response(json.dumps({"items": [{"doc": "reg-1", "uses_document": True, "follows_guidance": True, "reason": "ok"}]}))
    client = mc.ModelClient(mc.Settings(model="test"), transport=mc.Scripted([reply, reply]))
    verdicts = G.judge(answers, client, {"q1": "pertanyaan"})
    assert [v["error"] for v in verdicts] == [None, None] and G.rates(answers, verdicts)[("filtering_guidance", "repealed")] == [0, 1]
    assert len(G.rubric_digest()) == 12 and G.rubric_digest() in G.report([], [])


def test_the_nostate_arm_asks_the_questions_of_the_sampled_pairs_under_the_filtering_plan():
    from evalkit.e6 import nostate
    plan = [{"condition": "filtering", "qid": "q1", "role": "reader", "model": "m", "key": "q1|reader|filtering|m"},
            {"condition": "filtering", "qid": "q2", "role": "reader", "model": "m", "key": "q2|reader|filtering|m"},
            {"condition": "filtering_guidance", "qid": "q1", "role": "reader", "model": "m", "key": "q1|reader|filtering_guidance|m"}]
    runs = nostate.runs_of([{"pair": "q1|reader|m"}], plan)
    assert [(r["condition"], r["key"]) for r in runs] == [("filtering_nostate", "q1|reader|filtering_nostate|m")]


def test_the_tables_show_the_four_status_kinds_and_the_approved_level_goes_to_its_own_section():
    answers = [{"pair": "p", "condition": "filtering", "items": [{"doc": "a", "kind": "repealed"}, {"doc": "b", "kind": "approved_level"}]}]
    item = lambda d: {"doc": d, "uses_document": True, "follows_guidance": False, "reason": ""}
    verdicts = [{"pair": "p", "condition": "filtering", "kind": "repealed", "error": None, "items": [item("a"), item("b")]}]
    shown = G.show(G.table(answers, verdicts))
    assert "approved_level" not in shown and "restrictions" in shown
    text = G.report(answers, [("one", verdicts), ("two", verdicts)])
    closing = text.split("Judged but not reported in the paper: approved level")
    assert len(closing) == 2 and "1/1" in closing[1] and closing[1].count("one ") == 3 and closing[1].count("two ") == 3  # three conditions for each of the two judges
    assert G.agreement(answers, verdicts, verdicts)[1][1] == "1"  # only the document of a status kind is compared
