"""Interface and API: the one entry point for a person who asks a question.

    answer = interface.ask(token, "What is the rate in Article 3 of ...?")

`ask` checks the token, fixes the person from it, gives the agent loop that person's read-only tools, and returns the
answer with a reference to the run's log. This is the real interface; the HTTP layer and the evaluation call it.

It holds the directory, the service and the model client it was given, and nothing else: no store, no world.
"""

from __future__ import annotations

import json
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .agent_harness import agent_loop, tool_controller
from .authentication_directory import Unauthenticated  # noqa: F401  (raised by ask)

MAX_QUESTION_CHARS = 4000


class BadQuestion(Exception):
    """The question is not text, is empty, or is too long."""


@dataclass(frozen=True)
class Answer:
    answer: str | None  # None when the run did not end in an answer
    stop_reason: str  # answered | step_limit | empty_reply | model_error
    run_id: str  # the reference to the run's log


class Interface:
    def __init__(self, directory, service, client, *, step_limit: int = agent_loop.DEFAULT_STEP_LIMIT,
                 system_prompt: str | None = None, log_dir: str | Path | None = None, keep: int = 100) -> None:
        self.directory, self.service, self.client = directory, service, client
        self.step_limit, self.system_prompt, self.log_dir, self.keep = step_limit, system_prompt, log_dir, keep
        self._runs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def ask(self, token, question) -> Answer:
        user = self.directory.authenticate(token)  # first: nobody unknown learns anything else
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise BadQuestion()
        tools = tool_controller.tools(self.service, user)  # the person is fixed here, by the host
        result = agent_loop.run(self.client, tools, question, who=user, step_limit=self.step_limit, system_prompt=self.system_prompt)
        run_id = secrets.token_hex(8)
        with self._lock:
            self._runs[run_id] = result.log
            while len(self._runs) > self.keep:
                self._runs.popitem(last=False)
        if self.log_dir is not None:
            (Path(self.log_dir) / f"{run_id}.json").write_text(json.dumps(result.log, ensure_ascii=False, indent=1), encoding="utf-8")
        return Answer(result.answer, result.stop_reason, run_id)

    def run_log(self, run_id: str) -> dict:
        """The log of a run, for the host (not reachable through the HTTP layer)."""
        with self._lock:
            return self._runs[run_id]
