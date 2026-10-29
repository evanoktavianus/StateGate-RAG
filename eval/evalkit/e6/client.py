"""The model side of E6, in the harness: a transport that reuses connections and records every call, a client with the chat cache off, and a scripted transport.

Every call (one row per request that ended, after its retries) is appended to a file of its own process: run key, step, model, provider the service named, status,
attempts, attempts refused with 429 or 402, seconds, finish reason, tokens, cost, and the reply (text and tool calls). The cost of a run is the sum over its steps
(`log["steps"][i]["usage"]`), never a difference of the client's shared counters.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

import httpx

from docflow import model_client as mc

CTX = threading.local()  # .key, .step, .events (list)
REFUSED = (429, 402)
RETRY = (402, 408, 429, 500, 502, 503, 504)  # a refusal is retried with a wait, like an error


def events() -> list:
    if not hasattr(CTX, "events"):
        CTX.events = []
    return CTX.events


class CallLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock:
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")


class HttpTransport:
    """OpenRouter over HTTPS with one connection pool per process. Retries 408, 429, 5xx and network errors with a jittered wait of up to 30 s."""

    def __init__(self, key: str, calllog: CallLog, provider: dict | None = None, base_url: str = mc.BASE_URL, timeout: float = 120, attempts: int = 8, pool: int = 80) -> None:
        self._key, self.calllog, self.provider, self.base_url, self.attempts = key, calllog, provider, base_url, attempts
        self.http = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool))

    def scrub(self, text: str) -> str:
        return text.replace(self._key, "[redacted]") if self._key else text

    def post(self, path: str, body: dict) -> dict:
        if self.provider:
            body = {**body, "provider": self.provider}
        refused = tries = 0
        last = None
        t0 = time.time()
        for attempt in range(self.attempts):
            tries += 1
            try:
                r = self.http.post(self.base_url + path, json=body, headers={"Authorization": f"Bearer {self._key}"})
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = type(e).__name__
            else:
                if r.status_code == 200:
                    data = r.json()
                    CTX.last_meta = {"attempts": tries, "refused_attempts": refused, "seconds": round(time.time() - t0, 2), "provider": data.get("provider")}
                    return data
                last = r.status_code
                if r.status_code in REFUSED:
                    refused += 1
                if r.status_code not in RETRY:
                    CTX.last_meta = {"attempts": tries, "refused_attempts": refused, "seconds": round(time.time() - t0, 2), "provider": None}
                    raise mc.HttpError(r.status_code, self.scrub(r.text[:300]))
            time.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
        CTX.last_meta = {"attempts": tries, "refused_attempts": refused, "seconds": round(time.time() - t0, 2), "provider": None}
        raise mc.ModelClientError(f"the model service did not answer after {tries} attempts (last: {last})")


class NoCacheClient(mc.ModelClient):
    """`ModelClient` with the chat cache off: every request is sent, and every call is logged."""

    def __init__(self, settings, transport, calllog: CallLog, exclude_reasoning: bool = False) -> None:
        super().__init__(settings, transport)
        self.calllog, self.exclude_reasoning = calllog, exclude_reasoning

    def chat(self, messages, tools=None):
        s = self.settings
        body = {"model": s.model, "messages": messages, "max_tokens": s.max_tokens, "reasoning": {"effort": s.reasoning_effort}, "usage": {"include": True}}
        if getattr(CTX, "plain", False) and tools:  # plain RAG: the tool descriptions do not speak of a state either
            tools = [{**t, "function": {**t["function"], "description": t["function"]["description"].replace(", each with its state and how to treat it", "")}} for t in tools]
            body["tools"] = tools
        if self.exclude_reasoning:
            body["reasoning"]["exclude"] = True
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if tools:
            body["tools"] = tools
        CTX.last_meta = {}
        CTX.step = getattr(CTX, "step", 0) + 1
        t0 = time.time()
        row = {"t": round(t0, 2), "key": getattr(CTX, "key", None), "step": getattr(CTX, "step", None), "model": s.model}
        try:
            response = self.transport.post("/chat/completions", body)
            if not response.get("choices"):
                raise mc.ModelClientError("the service returned no answer: " + self._scrubbed(json.dumps(response.get("error", response))[:300]))
        except mc.ModelClientError as e:
            meta = getattr(CTX, "last_meta", {})
            events().append({"t": t0, "outcome": "failed", "attempts": meta.get("attempts", 1), "refused_attempts": meta.get("refused_attempts", 0),
                             "refused": isinstance(e, mc.HttpError) and e.code in REFUSED})
            self.calllog.write({**row, **meta, "outcome": "failed", "error": self._scrubbed(str(e))[:300]})
            raise
        result = self._result(response, cached=False)
        meta = getattr(CTX, "last_meta", {})
        msg, finish = result.message, result.finish_reason
        outcome = "cutoff" if finish == "length" else "empty" if (not (msg.get("content") or "").strip() and not msg.get("tool_calls")) else "ok"
        u = result.usage
        events().append({"t": t0, "outcome": outcome, "attempts": meta.get("attempts", 1), "refused_attempts": meta.get("refused_attempts", 0), "refused": False})
        self.calllog.write({**row, **meta, "outcome": outcome, "finish": finish, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                            "reasoning_tokens": u.reasoning_tokens, "cost": u.cost_usd, "reply": msg})
        return result


class ScriptedTransport:
    """A model that searches once with the first words of the question, then declines, after `latency` seconds."""

    def __init__(self, latency: float = 0.0, fail_rate: float = 0.0) -> None:
        self.latency, self.fail_rate = latency, fail_rate

    def scrub(self, text: str) -> str:
        return text

    def post(self, path: str, body: dict) -> dict:
        time.sleep(self.latency)
        CTX.last_meta = {"attempts": 1, "refused_attempts": 0, "seconds": self.latency, "provider": "scripted"}
        if self.fail_rate and random.random() < self.fail_rate:
            raise mc.ModelClientError("scripted failure")
        used_tool = any(m.get("role") == "tool" for m in body["messages"])
        if not used_tool:
            question = next(m["content"] for m in body["messages"] if m["role"] == "user")
            return mc.chat_response(calls=[("search", {"query": " ".join(question.split()[:6])})], prompt_tokens=400, completion_tokens=20, cost=0.00001)
        text = "Maaf, saya tidak menemukan dokumen itu."
        return mc.chat_response(content=text, prompt_tokens=900, completion_tokens=30, cost=0.00002)
