"""Agent loop: a plain ReAct-style loop, written here and not taken from a framework.

The model is given the system prompt (`system_prompt.txt`, next to this file) and the person's question. It either answers or asks for
tools; each tool call is run through the tool controller, the result goes back to the model, and this repeats until
the model answers, the step limit is reached, or the model service fails. The loop is handed the tools already bound
to one person and never sees who that is, the service, or any store.

Where the agent is told how to treat a document (DR3): every tool result carries the document's state and its
guidance in a `guidance` field. The system prompt tells the model to follow it. That field is the one place.

Every step is logged: the messages sent, the reply, each tool call with its arguments and result, tokens and cost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .. import model_client
from . import tool_controller

SYSTEM_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.txt"
DEFAULT_STEP_LIMIT = 8


def load_system_prompt(path: str | Path = SYSTEM_PROMPT_FILE) -> str:
    """The text of the file: it is sent as it is, without the blank lines around it."""
    return Path(path).read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class RunResult:
    answer: str | None  # None when the run did not end in an answer
    stop_reason: str  # answered | step_limit | empty_reply | model_error
    log: dict  # JSON-serialisable: settings, messages, steps, totals


def run(client: model_client.ModelClient, tools: dict, question: str, *, who: str | None = None,
        step_limit: int = DEFAULT_STEP_LIMIT, system_prompt: str | None = None) -> RunResult:
    """Runs the loop for one question. `who` only labels the log; it is not shown to the model."""
    system = system_prompt if system_prompt is not None else load_system_prompt()
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    log = {
        "who": who, "question": question, "step_limit": step_limit,
        "settings": client.settings.as_log(), "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "steps": [], "messages": messages, "answer": None, "stop_reason": None, "totals": None,
    }
    before = _snapshot(client)
    answer, stop = None, "step_limit"
    for number in range(1, step_limit + 1):
        step = {"step": number, "messages_sent": len(messages), "reply": None, "usage": None, "cached": None, "tool_results": []}
        log["steps"].append(step)
        try:
            reply = client.chat(list(messages), tools=tool_controller.SCHEMAS)  # a copy: what was sent stays as it was sent
        except model_client.ModelClientError as error:
            step["error"] = str(error)  # the client has already removed the key from it
            stop = "model_error"
            break
        step["reply"], step["cached"] = reply.message, reply.cached
        step["usage"] = {"prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens,
                         "reasoning_tokens": reply.usage.reasoning_tokens, "cost_usd": reply.usage.cost_usd}
        messages.append(reply.message)
        calls = model_client.tool_calls(reply.message)
        if not calls:
            answer = (reply.message.get("content") or "").strip()
            stop = "answered" if answer else "empty_reply"
            break
        for call in calls:
            result = tool_controller.call(tools, call.name, call.arguments)
            step["tool_results"].append({"id": call.id, "name": call.name, "arguments": call.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    log["answer"], log["stop_reason"], log["totals"] = answer, stop, _spent(client, before)
    return RunResult(answer, stop, log)


def _snapshot(client: model_client.ModelClient) -> dict:
    t = client.totals
    return {"calls": t.calls, "cache_hits": t.cache_hits, "prompt_tokens": t.prompt_tokens, "completion_tokens": t.completion_tokens,
            "reasoning_tokens": t.reasoning_tokens, "cost_usd": t.cost_usd, "unpriced_calls": t.unpriced_calls}


def _spent(client: model_client.ModelClient, before: dict) -> dict:
    """What this run added to the client's totals, in USD and rupiah (1 USD = Rp 18,000)."""
    now = _snapshot(client)
    out = {k: now[k] - before[k] for k in now}
    out["cost_rupiah"] = out["cost_usd"] * model_client.RUPIAH_PER_USD
    return out
