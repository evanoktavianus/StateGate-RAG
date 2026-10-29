"""P4: the agent loop, with a scripted model. No network, no paid call. Who may
read what comes from the table in test_access_control.py, the passage texts and the guidance from its fixture and from
workflows/knowledge_asset.yml."""

import json

import pytest

from docflow import model_client as mc
from docflow.agent_harness import agent_loop, tool_controller
from docflow.model_client import ModelClient, Scripted, Settings, chat_response
from test_access_control import ORACLE, PEOPLE, expected_for, gw, world  # noqa: F401  (the ministry fixture and its table)

QUESTION = "What does the zebra document say?"

# The passage of every (document, version) in the fixture, and the guidance of each document's state (knowledge_asset.yml).
TEXT = {("draft", 1): "draft zebra alpha", ("shared", 1): "shared zebra beta", ("public", 1): "public zebra gamma",
        ("unit", 1): "unit zebra delta", ("revised", 1): "revised zebra epsilon old", ("revised", 2): "revised zebra epsilon new"}
GUIDANCE = {
    "draft": "Unapproved working document; do not present it as a knowledge asset.",
    "shared": "Unapproved working document; do not present it as a knowledge asset.",
    "public": "Approved asset; mention its level (public).",
    "unit": "Approved asset; mention its level (unit).",
    "revised": "Unapproved working document; do not present it as a knowledge asset.",  # back in working after start_update: what its authors are told
}
RELEASE = "Approved asset; mention its level (public)."  # the readers of `revised` are served the release, and their rule says so


def guidance_for(user, doc):
    """What the person is told: for `revised` the author reads the draft; everyone else the release, under the reader rule's own text."""
    return RELEASE if doc == "revised" and user != "ann" else GUIDANCE[doc]


def run(gw, user, replies, **kw):
    script = Scripted(replies)
    client = ModelClient(Settings(), script)
    result = agent_loop.run(client, tool_controller.tools(gw, user), QUESTION, who=user, **kw)
    return result, script, client


def everything_sent(script):
    """Every string in every request the model was sent."""
    return json.dumps([body["messages"] for _, body in script.requests], ensure_ascii=False)


def received(script):
    """What the model was told by others: the system prompt, the question and the tool results, not its own replies."""
    return json.dumps([m for _, body in script.requests for m in body["messages"] if m["role"] != "assistant"], ensure_ascii=False)


# ── The loop ───────────────────────────────────────────────────────────

def test_the_system_prompt_is_read_from_its_file_and_stored_by_hash():
    text = agent_loop.load_system_prompt()
    assert text.startswith("You answer questions about documents, using only what your tools return.")
    assert "# System prompt" not in text and "---" not in text  # the file is the prompt and nothing else
    assert "guidance" in text and "repealed" in text and "not found" in text


def test_an_answer_without_tools_ends_the_run(gw):
    result, script, _ = run(gw, "ext", [chat_response("There is nothing to look up.")])
    assert (result.answer, result.stop_reason) == ("There is nothing to look up.", "answered")
    assert len(script.requests) == 1
    sent = script.requests[0][1]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"] and sent["messages"][1]["content"] == QUESTION
    assert [t["function"]["name"] for t in sent["tools"]] == ["list_documents", "get_document", "search"]


def test_a_tool_call_is_run_and_its_result_goes_back_to_the_model(gw):
    result, script, _ = run(gw, "ext", [chat_response(None, calls=[("search", {"query": "gamma"})]), chat_response("It is public.")])
    assert result.stop_reason == "answered" and result.answer == "It is public."
    second = script.requests[1][1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["function"]["name"] == "search"
    tool = second[3]
    assert tool["tool_call_id"] == "call_0"
    returned = json.loads(tool["content"])
    assert [h["id"] for h in returned["hits"]] == ["public"] and returned["hits"][0]["passage"]["text"] == "public zebra gamma"
    assert result.log["steps"][0]["tool_results"][0]["result"] == returned


def test_several_calls_in_one_reply_are_all_answered_in_order(gw):
    result, script, _ = run(gw, "ext", [chat_response(None, calls=[("list_documents", {}), ("get_document", {"doc_id": "public"})]), chat_response("done")])
    tools = [m for m in script.requests[1][1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["call_0", "call_1"]
    assert [d["id"] for d in json.loads(tools[0]["content"])["documents"]] == ["public", "revised"]
    assert json.loads(tools[1]["content"])["version"] == 1


def test_a_tool_the_model_invents_or_misuses_gets_a_fixed_refusal(gw):
    result, script, _ = run(gw, "ann", [
        chat_response(None, calls=[("act", {"doc_id": "draft", "action": "submit"}), ("search", {"query": "x", "user": "lead"}), ("search", {"query": 5})]),
        chat_response("I cannot do that.")])
    contents = [m["content"] for m in script.requests[1][1]["messages"] if m["role"] == "tool"]
    assert contents == ['{"error": "unknown_tool"}', '{"error": "bad_request"}', '{"error": "bad_request"}']
    assert result.stop_reason == "answered"


def test_arguments_that_are_not_an_object_are_a_bad_request(gw):
    reply = chat_response(None, calls=[("search", {"query": "x"})])
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    result, script, _ = run(gw, "ext", [reply, chat_response("ok")])
    assert json.loads(script.requests[1][1]["messages"][-1]["content"]) == {"error": "bad_request"}


def test_a_hidden_document_and_a_missing_one_look_the_same_to_the_model(gw):
    result, script, _ = run(gw, "ext", [chat_response(None, calls=[("get_document", {"doc_id": "draft"}), ("get_document", {"doc_id": "no-such-doc"})]), chat_response("not found")])
    hidden, missing = [m["content"] for m in script.requests[1][1]["messages"] if m["role"] == "tool"]
    assert hidden == missing == '{"error": "not_found"}'


def test_the_run_stops_at_the_step_limit(gw):
    loop = [chat_response(None, calls=[("search", {"query": "zebra"})]) for _ in range(5)]
    result, script, _ = run(gw, "ext", loop, step_limit=3)
    assert (result.answer, result.stop_reason) == (None, "step_limit")
    assert len(script.requests) == 3 and len(result.log["steps"]) == 3 and result.log["step_limit"] == 3


def test_an_empty_reply_and_a_failing_service_end_the_run_without_a_crash(gw):
    empty, _, _ = run(gw, "ext", [chat_response("   ")])
    assert (empty.answer, empty.stop_reason) == ("", "empty_reply")
    failing, _, _ = run(gw, "ext", [mc.HttpError(503, "unavailable")])
    assert (failing.answer, failing.stop_reason) == (None, "model_error")
    assert failing.log["steps"][0]["error"] == "HTTP 503: unavailable" and failing.log["stop_reason"] == "model_error"


def test_the_log_records_every_step_with_tokens_and_cost(gw):
    result, _, client = run(gw, "ext", [chat_response(None, calls=[("search", {"query": "gamma"})], prompt_tokens=100, completion_tokens=10, cost=0.001),
                                        chat_response("answer", prompt_tokens=150, completion_tokens=20, cost=0.002)])
    log = result.log
    assert json.loads(json.dumps(log)) == log  # plain JSON
    assert log["who"] == "ext" and log["question"] == QUESTION and log["step_limit"] == 8
    assert log["settings"]["model"] == "z-ai/glm-5.3-flash" and log["settings"]["temperature"] == 0.0 and log["settings"]["reasoning_effort"] == "medium"
    assert [s["step"] for s in log["steps"]] == [1, 2] and [s["messages_sent"] for s in log["steps"]] == [2, 4]
    assert log["steps"][0]["usage"] == {"prompt_tokens": 100, "completion_tokens": 10, "reasoning_tokens": 0, "cost_usd": 0.001}
    totals = log["totals"]
    assert (totals["calls"], totals["prompt_tokens"], totals["completion_tokens"]) == (2, 250, 30)
    assert totals["cost_usd"] == pytest.approx(0.003) and totals["cost_rupiah"] == pytest.approx(54.0)  # 0.003 USD at Rp 18,000
    assert log["answer"] == "answer" and log["stop_reason"] == "answered" and len(log["system_prompt_sha256"]) == 64
    assert [m["role"] for m in log["messages"]] == ["system", "user", "assistant", "tool", "assistant"]


def test_the_loop_is_given_tools_and_never_the_person():
    """`run` takes the model, the bound tools and the question. There is no parameter through which the model's words
    could change who is asking."""
    import inspect
    assert list(inspect.signature(agent_loop.run).parameters)[:3] == ["client", "tools", "question"]


# ── Three people, one question (the acceptance test) ───────────────────

def asks_then_reads_everything(gw, user):
    """The model searches, then opens each document the search found, then answers."""
    found = [d for d in ORACLE if user in ORACLE[d]]
    return run(gw, user, [chat_response(None, calls=[("search", {"query": "zebra", "limit": 50})]),
                          chat_response(None, calls=[("get_document", {"doc_id": d}) for d in found]),
                          chat_response("Summary.")])


def asks_for_everything(gw, user):
    """The model lists, searches, and asks for every document in the world, hidden ones and other versions too."""
    asks = [("list_documents", {}), ("search", {"query": "zebra", "limit": 50})]
    asks += [("get_document", {"doc_id": d}) for d in ORACLE] + [("get_document", {"doc_id": "revised", "version": 1}), ("get_document", {"doc_id": "revised", "version": 2})]
    return run(gw, user, [chat_response(None, calls=asks), chat_response("Summary.")])


@pytest.mark.parametrize("user", ["ann", "u1b", "ext"])  # the author (reads drafts), a unit viewer and an outsider (releases only)
def test_what_the_model_is_sent_holds_nothing_the_person_may_not_read(gw, user):
    result, script, _ = asks_for_everything(gw, user)
    assert result.stop_reason == "answered"
    sent = everything_sent(script)
    readable = {(d, v) for d, v in expected_for(user).items()}
    for (doc, version), text in TEXT.items():
        if (doc, version) in readable:
            assert text in sent, (user, doc, version, "should have been sent")  # not vacuous: what they may read does arrive
        else:
            assert text not in sent, (user, doc, version, "must not be sent")
    # nor is the id of a document they cannot open, in anything the model was told (its own requests name what it asked for)
    told = received(script)
    for doc in ORACLE:
        if user not in ORACLE[doc]:
            assert f'"{doc}"' not in told and f'\\"{doc}\\"' not in told, (user, doc)


@pytest.mark.parametrize("user", ["ann", "u1b", "ext"])
def test_the_guidance_of_each_document_the_model_was_shown_is_in_what_it_was_sent(gw, user):
    result, script, _ = asks_for_everything(gw, user)
    sent = everything_sent(script)
    for doc in expected_for(user):
        assert json.dumps(guidance_for(user, doc), ensure_ascii=False)[1:-1] in sent, (user, doc)
    shown = {guidance_for(user, d) for d in expected_for(user)}
    for doc in (d for d in ORACLE if user not in ORACLE[d]):
        if GUIDANCE[doc] not in shown:  # a sentence two documents share (draft and shared) can be shown for the one they may read
            assert GUIDANCE[doc] not in sent, (user, doc)


def test_the_three_people_see_what_the_table_says_and_the_same_prompt_otherwise(gw):
    prompts = {u: asks_then_reads_everything(gw, u)[1].requests[0][1]["messages"] for u in ("ann", "u1b", "ext")}
    assert prompts["ann"] == prompts["u1b"] == prompts["ext"]  # before any tool is called, the three prompts are the same
    docs_read = {}
    for u in ("ann", "u1b", "ext"):
        result, _, _ = asks_then_reads_everything(gw, u)
        docs_read[u] = {r["result"]["id"]: r["result"]["version"] for s in result.log["steps"] for r in s["tool_results"] if r["name"] == "get_document"}
    assert docs_read["ann"] == {"draft": 1, "shared": 1, "public": 1, "unit": 1, "revised": 2}  # the author reads the new draft of revised
    assert docs_read["u1b"] == {"public": 1, "unit": 1, "revised": 1}  # a unit viewer: releases only
    assert docs_read["ext"] == {"public": 1, "revised": 1}


@pytest.mark.parametrize("user", ["ann", "u1b", "ext"])
def test_every_tool_result_carries_the_guidance_of_its_documents_state(gw, user):
    """Each place a document appears (a list entry, a search hit, an opened document) says how to treat it."""
    result, _, _ = asks_for_everything(gw, user)
    seen = {"list_documents": [], "search": [], "get_document": []}
    for r in (r for s in result.log["steps"] for r in s["tool_results"]):
        out = r["result"]
        if r["name"] == "list_documents":
            seen["list_documents"] += [(d["id"], d["guidance"]) for d in out["documents"]]
        elif r["name"] == "search":
            seen["search"] += [(h["id"], h["guidance"]) for h in out["hits"]]
        elif "error" not in out:
            seen["get_document"].append((out["id"], out["guidance"]))
    visible = set(expected_for(user))
    for tool, entries in seen.items():
        assert {doc for doc, _ in entries} == visible - ({"x"} if tool else set()), (user, tool)  # exactly what they may read, no more
        assert all(guidance == guidance_for(user, doc) for doc, guidance in entries), (user, tool)


@pytest.mark.parametrize("user", ["ann", "u1b", "ext"])
def test_a_document_the_person_cannot_open_is_refused_the_same_way_whatever_it_is(gw, user):
    result, _, _ = asks_for_everything(gw, user)
    refused = [r for s in result.log["steps"] for r in s["tool_results"] if r["name"] == "get_document" and "error" in r["result"]]
    assert {json.dumps(r["result"]) for r in refused} == ({'{"error": "not_found"}'} if refused else set())
    hidden = [d for d in ORACLE if user not in ORACLE[d]]
    asked_hidden = [r for r in refused if r["arguments"].get("doc_id") in hidden and "version" not in r["arguments"]]
    assert len(asked_hidden) == len(hidden)  # every hidden document was asked for, and refused


@pytest.mark.parametrize("user", ["ann", "u1b", "ext"])
def test_a_named_version_opens_only_if_it_is_the_version_the_person_is_served(gw, user):
    """The model asks for revised at version 1 and at version 2. The author is served 2, the others 1 (the table)."""
    result, _, _ = asks_for_everything(gw, user)
    named = {r["arguments"]["version"]: r["result"] for s in result.log["steps"] for r in s["tool_results"]
             if r["name"] == "get_document" and r["arguments"].get("version") is not None}
    served = ORACLE["revised"][user]
    for version, out in named.items():
        if version == served:
            assert out["id"] == "revised" and out["version"] == version
            assert TEXT[("revised", version)] == out["passages"][0]["text"]
        else:
            assert out == {"error": "not_found"}, (user, version)
