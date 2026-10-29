"""Model client: chat with tool calling, and embeddings, both against OpenRouter.

    client = ModelClient(Settings(), LiveTransport(), cache="cache.sqlite", workers=32)
    reply = client.chat(messages, tools=SCHEMAS)      # reply.message, reply.usage, reply.cached
    vectors = client.embed(["a passage", "another"])   # one float32 vector (bytes) per text, in order

The transport is what talks to the network, so the same client runs against a live service, a script of responses
(`Scripted`, for tests and demonstrations: free and deterministic), or a file of recorded responses (`Replay`).
`Recording` wraps a live transport and writes each response to a file for `Replay`; nothing in the tests runs it.

Every response is kept for good in a sqlite cache, keyed by the whole request, so a repeat costs nothing and counts no
tokens. Embeddings are cached per text under the same key the build scripts always used (sha256 of the model, a NUL
and the text), so an existing `embeddings.sqlite` keeps working.

The API key is read from the environment variable OPENROUTER_API_KEY, only when a real call is attempted. It is
never written to the cache, a recorded file, a log or an error message: it is not part of a request body, and the
transport removes it from anything it reports.
"""

from __future__ import annotations

import array
import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

KEY_VARIABLE = "OPENROUTER_API_KEY"
BASE_URL = "https://openrouter.ai/api/v1"
RUPIAH_PER_USD = 18_000
RETRY_CODES = (408, 429, 500, 502, 503, 504)
EMBED_BATCH = 96  # texts per embeddings request


class ModelClientError(Exception):
    """Something went wrong talking to the model; the message never contains the API key."""


class MissingKey(ModelClientError):
    pass


class HttpError(ModelClientError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"HTTP {code}: {message}")
        self.code = code


class NotRecorded(ModelClientError):
    pass


class ScriptExhausted(ModelClientError):
    pass


@dataclass(frozen=True)
class Settings:
    """Everything that changes what the model says or what it costs; stored with a run's log."""

    model: str = "z-ai/glm-5.3-flash"
    reasoning_effort: str = "medium"  # sent as reasoning.effort; "enabled: false" is refused by the service
    temperature: float | None = 0.0
    max_tokens: int = 4000
    embedding_model: str = "openai/text-embedding-3-small"
    prompt_usd_per_million: float | None = None  # unknown until set: a call whose cost the service does not report is then unpriced
    completion_usd_per_million: float | None = None
    embedding_usd_per_million: float | None = 0.02  # OpenAI's list price for text-embedding-3-small; check against the invoice

    def as_log(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Usage:
    """What one call billed. A call served from the cache billed nothing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = 0.0  # None: the service did not report a cost and no price is set

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Totals:
    calls: int = 0  # requests sent
    cache_hits: int = 0  # chat replies and embedded texts served from the cache
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0  # the calls whose cost is known
    unpriced_calls: int = 0  # billed calls whose cost is not known

    @property
    def cost_rupiah(self) -> float:
        return self.cost_usd * RUPIAH_PER_USD


@dataclass(frozen=True)
class ChatResult:
    message: dict  # {"role": "assistant", "content": str | None, "tool_calls": [...]}
    usage: Usage
    cached: bool
    finish_reason: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the model's arguments were not a JSON object


def tool_calls(message: dict) -> list[ToolCall]:
    """The calls the model asked for, with their arguments read as JSON (None if they are not an object)."""
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = None
        out.append(ToolCall(call.get("id", ""), fn.get("name", ""), args if isinstance(args, dict) else None))
    return out


def request_hash(path: str, body: dict) -> str:
    return hashlib.sha256((path + "\0" + json.dumps(body, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()


def embedding_digest(model: str, text: str) -> str:
    """The key an embedding is cached under; the same as the build scripts have always used."""
    return hashlib.sha256((model + "\0" + text).encode()).hexdigest()


# ── Transports: what talks to the network (or stands in for it) ────────


class LiveTransport:
    """OpenRouter over HTTPS. Retries what is worth retrying; the key is read when a call is made, from the environment."""

    def __init__(self, key: str | None = None, base_url: str = BASE_URL, timeout: float = 120, attempts: int = 8) -> None:
        self._key, self.base_url, self.timeout, self.attempts = key, base_url, timeout, attempts
        self.retries = 0

    def __repr__(self) -> str:
        return f"LiveTransport(base_url={self.base_url!r}, key=<hidden>)"

    def _secret(self) -> str | None:
        return self._key or os.environ.get(KEY_VARIABLE) or None

    def scrub(self, text: str) -> str:
        secret = self._secret()
        return text.replace(secret, "[redacted]") if secret else text

    def post(self, path: str, body: dict) -> dict:
        key = self._secret()
        if not key:
            raise MissingKey(f"{KEY_VARIABLE} is not set, so no call can be made to the model service")
        data = json.dumps(body).encode()
        for attempt in range(self.attempts):
            req = urllib.request.Request(self.base_url + path, data, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_CODES:
                    try:
                        detail = e.read()[:300].decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    raise HttpError(e.code, self.scrub(detail)) from None
            except (urllib.error.URLError, TimeoutError):
                pass
            self.retries += 1
            time.sleep(min(2**attempt, 30))
        raise ModelClientError(f"the model service did not answer after {self.attempts} attempts")


class Scripted:
    """Answers from a list, in order, and keeps every request it was given, so a test can read what the model was shown.
    An item that is an exception is raised instead of returned."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict]] = []

    def post(self, path: str, body: dict) -> dict:
        self.requests.append((path, body))
        if not self.responses:
            raise ScriptExhausted("the script has no response left for this request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Replay:
    """Answers from a file written by `Recording`, by request, in any order."""

    def __init__(self, path: str | Path) -> None:
        self.recorded: dict[str, dict] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.recorded[row["hash"]] = row["response"]

    def post(self, path: str, body: dict) -> dict:
        h = request_hash(path, body)
        if h not in self.recorded:
            raise NotRecorded(f"no recorded response for this request ({h[:12]})")
        return self.recorded[h]


class Recording:
    """Wraps a transport and appends each request and response to a file. Never run by the tests with a live transport."""

    def __init__(self, inner, path: str | Path) -> None:
        self.inner, self.path = inner, Path(path)

    def scrub(self, text: str) -> str:
        return self.inner.scrub(text) if hasattr(self.inner, "scrub") else text

    def post(self, path: str, body: dict) -> dict:
        response = self.inner.post(path, body)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": request_hash(path, body), "path": path, "request": body, "response": response}, ensure_ascii=False) + "\n")
        return response


def chat_response(content: str | None = None, calls: list | tuple = (), prompt_tokens: int = 10, completion_tokens: int = 5,
                  cost: float | None = None, finish_reason: str | None = None) -> dict:
    """A chat completion as the service sends it, for a script. `calls` are (name, arguments) pairs."""
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": message, "finish_reason": finish_reason or ("tool_calls" if calls else "stop")}], "usage": usage}


def embedding_response(vectors: list, tokens: int | None = None) -> dict:
    """An embeddings reply as the service sends it, for a script."""
    return {"data": [{"index": i, "embedding": list(v)} for i, v in enumerate(vectors)],
            "usage": {"total_tokens": tokens if tokens is not None else len(vectors)}}


# ── The client ─────────────────────────────────────────────────────────


class ModelClient:
    def __init__(self, settings: Settings | None = None, transport=None, cache: str | Path | None = None, workers: int = 1) -> None:
        self.settings = settings or Settings()
        self.transport = transport if transport is not None else LiveTransport()
        self.workers = max(1, workers)
        self.totals = Totals()
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(cache) if cache else ":memory:", check_same_thread=False)
        self._db.execute("create table if not exists emb (h text primary key, v blob)")
        self._db.execute("create table if not exists chat (h text primary key, request text, response text)")
        self._db.commit()

    def __repr__(self) -> str:
        return f"ModelClient(model={self.settings.model!r}, transport={self.transport!r})"

    def _scrubbed(self, text: str) -> str:
        scrub = getattr(self.transport, "scrub", None)
        return scrub(text) if scrub else text

    # ── Chat ───────────────────────────────────────────────────────────

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        s = self.settings
        body: dict = {"model": s.model, "messages": messages, "max_tokens": s.max_tokens,
                      "reasoning": {"effort": s.reasoning_effort}, "usage": {"include": True}}
        if s.temperature is not None:
            body["temperature"] = s.temperature
        if tools:
            body["tools"] = tools
        h = request_hash("/chat/completions", body)
        with self._lock:
            row = self._db.execute("select response from chat where h = ?", (h,)).fetchone()
        if row is not None:
            with self._lock:
                self.totals.cache_hits += 1
            return self._result(json.loads(row[0]), cached=True)
        response = self.transport.post("/chat/completions", body)
        if not response.get("choices"):
            raise ModelClientError("the model service returned no answer: " + self._scrubbed(json.dumps(response.get("error", response))[:300]))
        with self._lock:
            self._db.execute("insert or replace into chat values (?,?,?)", (h, json.dumps(body, ensure_ascii=False), json.dumps(response, ensure_ascii=False)))
            self._db.commit()
        return self._result(response, cached=False)

    def _result(self, response: dict, cached: bool) -> ChatResult:
        choice = response["choices"][0]
        raw = choice.get("message") or {}
        message = {"role": raw.get("role", "assistant"), "content": raw.get("content")}
        if raw.get("tool_calls"):
            message["tool_calls"] = raw["tool_calls"]
        usage = Usage()
        if not cached:
            u = response.get("usage") or {}
            prompt, completion = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            cost = u.get("cost")
            s = self.settings
            if cost is None and s.prompt_usd_per_million is not None and s.completion_usd_per_million is not None:
                cost = (prompt * s.prompt_usd_per_million + completion * s.completion_usd_per_million) / 1e6
            usage = Usage(prompt, completion, (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0), cost)
            with self._lock:
                t = self.totals
                t.calls += 1
                t.prompt_tokens += prompt
                t.completion_tokens += completion
                t.reasoning_tokens += usage.reasoning_tokens
                if cost is None:
                    t.unpriced_calls += 1
                else:
                    t.cost_usd += cost
        return ChatResult(message, usage, cached, choice.get("finish_reason"))

    # ── Embeddings ─────────────────────────────────────────────────────

    def embed(self, texts: list[str], progress=None) -> list[bytes]:
        """One float32 vector (as bytes) per text, in order. Only texts not already cached are sent, once each."""
        model = self.settings.embedding_model
        hs = [embedding_digest(model, t) for t in texts]
        have = self._cached(list(set(hs)))
        with self._lock:
            self.totals.cache_hits += sum(1 for h in hs if h in have)
        todo = list(dict.fromkeys(t for t, h in zip(texts, hs) if h not in have))
        batches = [todo[i : i + EMBED_BATCH] for i in range(0, len(todo), EMBED_BATCH)]
        done = 0

        def run(batch: list[str]) -> None:
            nonlocal done
            for text, vector in zip(batch, self._embed_batch(batch)):
                with self._lock:
                    self._db.execute("insert or replace into emb values (?,?)", (embedding_digest(model, text), array.array("f", vector).tobytes()))
            with self._lock:
                self._db.commit()
                done += len(batch)
            if progress:
                progress(done, len(todo))

        if self.workers > 1:
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(run, batches))
        else:
            for batch in batches:
                run(batch)
        have = self._cached(list(set(hs)))
        return [have[h] for h in hs]

    def vectors(self, texts: list[str]) -> list[list[float]]:
        """The same, as lists of floats (what `Corpus.similar` takes)."""
        return [list(array.array("f", b)) for b in self.embed(texts)]

    def _cached(self, hs: list[str]) -> dict[str, bytes]:
        with self._lock:
            out: dict[str, bytes] = {}
            for i in range(0, len(hs), 500):
                part = hs[i : i + 500]
                out.update(self._db.execute("select h, v from emb where h in (%s)" % ",".join("?" * len(part)), part).fetchall())
            return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.transport.post("/embeddings", {"model": self.settings.embedding_model, "input": texts})
        except HttpError as e:
            if e.code == 400 and len(texts) > 1:  # one bad input fails the batch: send them one at a time
                return [self._embed_batch([t])[0] for t in texts]
            raise
        data = sorted(response.get("data") or [], key=lambda d: d["index"])
        if len(data) != len(texts):
            raise ModelClientError(f"asked for {len(texts)} embeddings, got {len(data)}")
        u = response.get("usage") or {}
        tokens = u.get("total_tokens", 0)
        cost = u.get("cost")
        price = self.settings.embedding_usd_per_million
        if cost is None and price is not None:
            cost = tokens * price / 1e6
        with self._lock:
            t = self.totals
            t.calls += 1
            t.embedding_tokens += tokens
            if cost is None:
                t.unpriced_calls += 1
            else:
                t.cost_usd += cost
        return [d["embedding"] for d in data]
