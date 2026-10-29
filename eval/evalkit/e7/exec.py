"""Running one E7 prompt: the agent loop of the engine over tools bound to the asker, over a service that records the documents it returns."""

from __future__ import annotations

import time

from docflow.document_service.service import DocumentService
from docflow.agent_harness import agent_loop, tool_controller

from .. import pgstore
from ..e6 import client as C
from ..e6.runner import CondWorld, STEP_LIMIT


class TraceService(DocumentService):
    """A service that keeps, for every call, the documents and versions it returned (`find` is true for a search: the version a search shows, not the one a person opens)."""

    def __init__(self, world, corpus) -> None:
        super().__init__(world, corpus)
        self.returned: list[dict] = []

    def list_documents(self, user, workspace=None):
        out = super().list_documents(user, workspace)
        self.returned += [{"tool": "list_documents", "doc": d.id, "version": d.version, "find": False} for d in out]
        return out

    def get_document(self, user, doc_id, version=None):
        doc = super().get_document(user, doc_id, version)
        self.returned.append({"tool": "get_document", "doc": doc["id"], "version": doc["version"], "find": False})
        return doc

    def search(self, user, query, limit=5):
        hits = super().search(user, query, limit)
        self.returned += [{"tool": "search", "doc": h.doc, "version": h.version, "find": True} for h in hits]
        return hits


def make_services(S) -> dict:
    """(workflow, condition, corpus variant) -> (the wrapped world, the corpus)."""
    out = {}
    for (wf, variant), corpus in S.corpus.items():
        index = pgstore.wrap(S.worlds[wf], corpus, f"{wf}|{variant}")  # PostgreSQL when DATABASE_URL is set (loaded on first use), else the in-memory corpus
        for cond in ("no_access_control", "filtering", "filtering_guidance"):
            out[(wf, cond, variant)] = (CondWorld(S.worlds[wf], cond), index)
    return out


def execute(run: dict, clients: dict, services: dict) -> dict:
    C.CTX.key, C.CTX.step, C.CTX.events = run["key"], 0, []
    t0 = time.time()
    world, corpus = services[(run["workflow"], run["condition"], run["corpus"])]
    gw = TraceService(world, corpus)
    tools = tool_controller.tools(gw, run["user"])
    result = agent_loop.run(clients[run["model"]], tools, run["question"], who=run["user"], step_limit=STEP_LIMIT)
    usage = [s.get("usage") or {} for s in result.log["steps"]]
    calls = [{"step": s["step"], "name": t["name"], "arguments": t["arguments"]} for s in result.log["steps"] for t in s["tool_results"]]
    return {"key": run["key"], "prompt_id": run["prompt_id"], "type": run["type"], "placement": run["placement"], "role": run["role"], "condition": run["condition"],
            "corpus": run["corpus"], "workflow": run["workflow"], "model": run["model"], "user": run["user"], "stop_reason": result.stop_reason, "tool_calls": calls,
            "returned": gw.returned, "calls": len(usage), "cost": sum((u.get("cost_usd") or 0.0) for u in usage), "seconds": round(time.time() - t0, 2),
            "events": list(C.CTX.events), "finished": time.time()}
