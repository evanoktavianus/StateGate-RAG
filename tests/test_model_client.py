"""Model client: scripted responses only. No test opens a network connection (a guard fails any that tries), and the
API key in these tests is a made-up sentinel that must appear nowhere it should not."""

import array
import hashlib
import io
import json
import socket
import urllib.error

import pytest

from docflow import model_client as mc
from docflow.model_client import (HttpError, LiveTransport, MissingKey, ModelClient, ModelClientError, NotRecorded, Recording, Replay,
                                  ScriptExhausted, Scripted, Settings, chat_response, embedding_response, tool_calls)

SENTINEL = "sk-or-v1-SENTINEL-do-not-leak-0123456789"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.delenv(mc.KEY_VARIABLE, raising=False)


def f32(values):
    return array.array("f", values).tobytes()


# ── Chat ───────────────────────────────────────────────────────────────

def test_a_chat_reply_carries_the_message_the_usage_and_what_was_asked():
    script = Scripted([chat_response("Hello", prompt_tokens=12, completion_tokens=3, cost=0.0004)])
    client = ModelClient(Settings(temperature=0.2, max_tokens=500), script)
    reply = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "search"}}])
    assert reply.message == {"role": "assistant", "content": "Hello"} and reply.cached is False and reply.finish_reason == "stop"
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens, reply.usage.cost_usd) == (12, 3, 15, 0.0004)
    (path, body), = script.requests
    assert path == "/chat/completions"
    # the settings the paper reports are in the request: model, reasoning effort medium, temperature, token limit, cost reporting
    assert body["model"] == "z-ai/glm-5.3-flash" and body["reasoning"] == {"effort": "medium"} and body["temperature"] == 0.2
    assert body["max_tokens"] == 500 and body["usage"] == {"include": True}
    assert body["messages"] == [{"role": "user", "content": "hi"}] and body["tools"][0]["function"]["name"] == "search"


def test_tool_calls_are_read_with_their_arguments():
    script = Scripted([chat_response(None, calls=[("search", {"query": "rate"}), ("get_document", {"doc_id": "d1", "version": 2})])])
    reply = ModelClient(Settings(), script).chat([{"role": "user", "content": "q"}])
    assert reply.finish_reason == "tool_calls"
    assert tool_calls(reply.message) == [mc.ToolCall("call_0", "search", {"query": "rate"}), mc.ToolCall("call_1", "get_document", {"doc_id": "d1", "version": 2})]
    broken = {"tool_calls": [{"id": "x", "function": {"name": "search", "arguments": "{not json"}}, {"id": "y", "function": {"name": "search", "arguments": "[1]"}}]}
    assert [c.arguments for c in tool_calls(broken)] == [None, None]  # not a JSON object: the loop answers bad_request
    assert tool_calls({"role": "assistant", "content": "done"}) == []


def test_a_repeat_comes_from_the_cache_and_counts_no_new_tokens():
    script = Scripted([chat_response("one", prompt_tokens=100, completion_tokens=20, cost=0.01)])
    client = ModelClient(Settings(), script)
    first = client.chat([{"role": "user", "content": "q"}])
    again = client.chat([{"role": "user", "content": "q"}])
    assert again.cached is True and again.message == first.message
    assert (again.usage.total_tokens, again.usage.cost_usd) == (0, 0.0)
    assert len(script.requests) == 1  # the second never reached the transport
    t = client.totals
    assert (t.calls, t.cache_hits, t.prompt_tokens, t.completion_tokens, t.cost_usd) == (1, 1, 100, 20, 0.01)


def test_a_different_request_is_a_different_answer(tmp_path):
    cache = tmp_path / "cache.sqlite"
    client = ModelClient(Settings(), Scripted([chat_response("a"), chat_response("b")]), cache=cache)
    assert client.chat([{"role": "user", "content": "q1"}]).message["content"] == "a"
    assert client.chat([{"role": "user", "content": "q2"}]).message["content"] == "b"
    # a change in a setting is a different request too: the same question under another reasoning effort is not served from the cache
    other = ModelClient(Settings(reasoning_effort="low"), Scripted([chat_response("c")]), cache=cache)
    reply = other.chat([{"role": "user", "content": "q1"}])
    assert reply.message["content"] == "c" and reply.cached is False


def test_the_cache_is_kept_for_good_on_disk(tmp_path):
    first = ModelClient(Settings(), Scripted([chat_response("kept", prompt_tokens=7)]), cache=tmp_path / "cache.sqlite")
    first.chat([{"role": "user", "content": "q"}])
    later = ModelClient(Settings(), Scripted([]), cache=tmp_path / "cache.sqlite")  # a script with nothing left would raise
    reply = later.chat([{"role": "user", "content": "q"}])
    assert reply.cached and reply.message["content"] == "kept" and later.totals.prompt_tokens == 0


def test_cost_is_reported_computed_or_unknown_and_shown_in_rupiah():
    reported = ModelClient(Settings(), Scripted([chat_response("a", cost=0.001)]))
    reported.chat([{"role": "user", "content": "1"}])
    assert reported.totals.cost_usd == 0.001 and reported.totals.cost_rupiah == pytest.approx(18.0)  # 1 USD = Rp 18,000
    priced = ModelClient(Settings(prompt_usd_per_million=1.0, completion_usd_per_million=4.0), Scripted([chat_response("a", prompt_tokens=2000, completion_tokens=500)]))
    result = priced.chat([{"role": "user", "content": "1"}])
    assert result.usage.cost_usd == pytest.approx(2000 * 1.0 / 1e6 + 500 * 4.0 / 1e6)  # 0.002 + 0.002 = 0.004 USD
    unknown = ModelClient(Settings(), Scripted([chat_response("a", prompt_tokens=50)]))
    assert unknown.chat([{"role": "user", "content": "1"}]).usage.cost_usd is None
    assert unknown.totals.unpriced_calls == 1 and unknown.totals.cost_usd == 0.0  # not counted as free: it is reported as unpriced


def test_an_answer_with_no_choices_is_an_error_not_a_crash():
    client = ModelClient(Settings(), Scripted([{"error": {"message": "provider unavailable"}}]))
    with pytest.raises(ModelClientError, match="provider unavailable"):
        client.chat([{"role": "user", "content": "q"}])


def test_an_exhausted_script_says_so():
    with pytest.raises(ScriptExhausted):
        ModelClient(Settings(), Scripted([])).chat([{"role": "user", "content": "q"}])


# ── Embeddings ─────────────────────────────────────────────────────────

def test_embeddings_come_back_in_order_as_float32_bytes():
    script = Scripted([embedding_response([[0.5, 1.0], [2.0, 4.0]], tokens=9)])
    client = ModelClient(Settings(), script)
    assert client.embed(["first", "second"]) == [f32([0.5, 1.0]), f32([2.0, 4.0])]
    assert script.requests == [("/embeddings", {"model": "openai/text-embedding-3-small", "input": ["first", "second"]})]
    assert client.totals.embedding_tokens == 9 and client.totals.cost_usd == pytest.approx(9 * 0.02 / 1e6)


def test_only_new_texts_are_sent_and_a_repeated_text_is_sent_once():
    script = Scripted([embedding_response([[1.0]]), embedding_response([[2.0], [3.0]])])
    client = ModelClient(Settings(), script)
    client.embed(["a"])
    assert client.embed(["a", "b", "c", "b"]) == [f32([1.0]), f32([2.0]), f32([3.0]), f32([2.0])]
    assert script.requests[1][1]["input"] == ["b", "c"]  # "a" is cached and "b" appears twice
    assert client.totals.cache_hits == 1


def test_an_existing_embedding_cache_keeps_working():
    """The build scripts keyed their cache by sha256 of the model, a NUL and the text; `embeddings.sqlite` files made by
    them must not be paid for again."""
    key = hashlib.sha256(b"openai/text-embedding-3-small\x00hello").hexdigest()
    assert mc.embedding_digest("openai/text-embedding-3-small", "hello") == key
    client = ModelClient(Settings(), Scripted([]))  # nothing may be sent
    client._db.execute("insert into emb values (?,?)", (key, f32([9.0, 8.0])))
    assert client.embed(["hello"]) == [f32([9.0, 8.0])] and client.totals.embedding_tokens == 0


def test_a_long_list_is_sent_in_batches_with_the_order_kept():
    texts = [f"t{i}" for i in range(100)]
    script = Scripted([embedding_response([[float(i)] for i in range(96)]), embedding_response([[float(i)] for i in range(96, 100)])])
    client = ModelClient(Settings(), script, workers=1)
    out = client.embed(texts)
    assert [len(r[1]["input"]) for r in script.requests] == [96, 4]
    assert out == [f32([float(i)]) for i in range(100)]


def test_batches_run_together_without_losing_the_order():
    texts = [f"t{i}" for i in range(200)]

    class ByInput:  # answers each request from its own input, so the order of arrival does not matter
        def post(self, path, body):
            return embedding_response([[float(t[1:])] for t in body["input"]])

    out = ModelClient(Settings(), ByInput(), workers=4).embed(texts)
    assert out == [f32([float(i)]) for i in range(200)]


def test_one_bad_text_fails_only_itself_when_a_batch_is_refused():
    script = Scripted([HttpError(400, "invalid input"), embedding_response([[1.0]]), embedding_response([[2.0]])])
    assert ModelClient(Settings(), script).embed(["a", "b"]) == [f32([1.0]), f32([2.0])]
    assert [r[1]["input"] for r in script.requests] == [["a", "b"], ["a"], ["b"]]
    with pytest.raises(HttpError):  # a single text that is refused stays an error
        ModelClient(Settings(), Scripted([HttpError(400, "invalid input")])).embed(["a"])


def test_vectors_are_floats_for_the_search_index():
    assert ModelClient(Settings(), Scripted([embedding_response([[0.25, 0.5]])])).vectors(["q"]) == [[0.25, 0.5]]


# ── The key ────────────────────────────────────────────────────────────

def test_no_key_is_needed_until_a_real_call_is_made(tmp_path):
    transport = LiveTransport()  # no key in the environment: fine so far
    client = ModelClient(Settings(), transport, cache=tmp_path / "c.sqlite")
    with pytest.raises(MissingKey, match=mc.KEY_VARIABLE):
        client.chat([{"role": "user", "content": "q"}])
    # an answer already in the cache needs no key and no call
    warm = ModelClient(Settings(), Scripted([chat_response("cached")]), cache=tmp_path / "c.sqlite")
    warm.chat([{"role": "user", "content": "q2"}])
    cold = ModelClient(Settings(), LiveTransport(), cache=tmp_path / "c.sqlite")
    assert cold.chat([{"role": "user", "content": "q2"}]).cached is True


def fake_urlopen(monkeypatch, outcomes, seen):
    def urlopen(req, timeout=None):
        seen.append(req)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(json.dumps(outcome).encode())

    monkeypatch.setattr(mc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)


def http_error(code, body=b""):
    return urllib.error.HTTPError("https://openrouter.ai/api/v1/x", code, "msg", {}, io.BytesIO(body))


def test_the_live_transport_sends_the_key_in_the_header_only_and_retries_what_is_worth_retrying(monkeypatch):
    seen = []
    fake_urlopen(monkeypatch, [http_error(429), urllib.error.URLError("reset"), chat_response("ok")], seen)
    transport = LiveTransport(key=SENTINEL)
    assert transport.post("/chat/completions", {"model": "m"})["choices"][0]["message"]["content"] == "ok"
    assert transport.retries == 2 and len(seen) == 3
    assert seen[0].get_header("Authorization") == f"Bearer {SENTINEL}"  # the header carries it...
    assert SENTINEL not in seen[0].data.decode()  # ...the body does not
    assert seen[0].full_url == "https://openrouter.ai/api/v1/chat/completions"


def test_an_error_that_is_not_worth_retrying_is_raised_at_once_without_the_key(monkeypatch):
    seen = []
    fake_urlopen(monkeypatch, [http_error(401, f'{{"error": "bad key {SENTINEL}"}}'.encode())], seen)
    with pytest.raises(HttpError) as raised:
        LiveTransport(key=SENTINEL).post("/chat/completions", {})
    assert raised.value.code == 401 and len(seen) == 1
    assert SENTINEL not in str(raised.value) and "[redacted]" in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__  # no chained error to print


def test_a_service_that_keeps_failing_gives_up(monkeypatch):
    fake_urlopen(monkeypatch, [http_error(503)] * 3, [])
    with pytest.raises(ModelClientError, match="3 attempts"):
        LiveTransport(key=SENTINEL, attempts=3).post("/x", {})


def test_the_key_comes_from_the_environment_and_nowhere_prints_it(monkeypatch):
    monkeypatch.setenv(mc.KEY_VARIABLE, SENTINEL)
    transport = LiveTransport()
    client = ModelClient(Settings(), transport)
    for text in (repr(transport), repr(client), str(client), repr(Settings()), json.dumps(Settings().as_log())):
        assert SENTINEL not in text
    assert transport.scrub(f"it said {SENTINEL}") == "it said [redacted]"
    seen = []
    fake_urlopen(monkeypatch, [chat_response("ok")], seen)
    transport.post("/chat/completions", {})
    assert seen[0].get_header("Authorization") == f"Bearer {SENTINEL}"


def test_the_key_is_in_neither_the_cache_nor_a_recorded_file(monkeypatch, tmp_path):
    monkeypatch.setenv(mc.KEY_VARIABLE, SENTINEL)
    seen = []
    fake_urlopen(monkeypatch, [chat_response("answer"), embedding_response([[1.0, 2.0]])], seen)
    recorded = tmp_path / "recorded.jsonl"
    client = ModelClient(Settings(), Recording(LiveTransport(), recorded), cache=tmp_path / "cache.sqlite")
    client.chat([{"role": "user", "content": "q"}])
    client.embed(["text"])
    assert all(r.get_header("Authorization") == f"Bearer {SENTINEL}" for r in seen)  # it was used, so the checks below mean something
    del client
    for path in (tmp_path / "cache.sqlite", recorded):
        assert path.stat().st_size > 0 and SENTINEL.encode() not in path.read_bytes(), path


# ── Recorded responses ─────────────────────────────────────────────────

def test_a_recording_replays_by_request(tmp_path):
    path = tmp_path / "recorded.jsonl"
    live = ModelClient(Settings(), Recording(Scripted([chat_response("one"), chat_response("two")]), path))
    live.chat([{"role": "user", "content": "q1"}])
    live.chat([{"role": "user", "content": "q2"}])
    replay = ModelClient(Settings(), Replay(path))
    assert replay.chat([{"role": "user", "content": "q2"}]).message["content"] == "two"  # in any order
    assert replay.chat([{"role": "user", "content": "q1"}]).message["content"] == "one"
    with pytest.raises(NotRecorded):
        replay.chat([{"role": "user", "content": "never asked"}])
