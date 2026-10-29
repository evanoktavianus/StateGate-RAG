"""E6 context leaks, measured by document identity, offline (no model call).

    python -m evalkit.e6.context_leaks --results eval/results/runs/e6/results.jsonl --out eval/results/runs/e6

Every (document, version) that a tool returned to the agent in a run is compared with what the independent oracle says the asker may read at the state the document is in. The
oracle is `eval/evalkit/oracle` on the original policy for the synthetic set (the document is carried to its state by the steps of its cell), and the table of
`tests/oracle_regulation.py` for the regulations; nothing here calls the engine's access code to decide. A search returns the version the asker is *found* at, `get_document` and
`list_documents` the version the asker *opens* (`find` and `serve` of the rule). A context leak is a document the oracle does not let the asker read (`document outside`) or a
version other than the one it lets them read (`version outside`).

The saved trace holds only the first five documents of a `list_documents` result, but a tool result is a function of the asker, the condition and the world, so each
`list_documents` call is run again here through the same service (the same condition wrapper) and its documents are compared; the first five and the count must equal the saved ones.
Writes `context_leaks.json.gz`; `python -m evalkit.e6.analysis` prints the table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller

from .. import corpus as C
from .. import paths as paths_mod
from .. import retrieval_utility as U
from ..oracle import Cast, Doc
from ..oracle.policy import load_policy
from .runner import CondWorld

ROOT = Path(__file__).resolve().parents[3]


def _oracle_reg():
    spec = importlib.util.spec_from_file_location("oracle_regulation", ROOT / "tests" / "oracle_regulation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Oracle:
    """What each person may read of each document, by the oracle: serve (opens) and find (is shown by a search)."""

    def __init__(self) -> None:
        self.b = U.G["built"]
        self._wf: dict = {}
        self._doc: dict = {}
        reg = _oracle_reg()
        plan, docs = U.G["reg_plan"], U.G["reg_docs"]
        versions: dict[str, int] = {}
        for d in docs:
            if d["kind"] == "text":
                versions[d["doc"]] = max(versions.get(d["doc"], 0), d["version"])
        self.reg_visible = reg.visible(plan, versions)

    def _workflow(self, wf: str):
        if wf not in self._wf:
            world = self.b.worlds[wf]
            people = {pid: {"org": u.org, "unit": u.unit, "groups": list(u.groups), "positions": list(u.positions), "attrs": dict(u.attrs), "org_admin": u.org_admin}
                      for pid, u in world.users.items()}
            self._wf[wf] = (Cast.from_people(people), load_policy(paths_mod.generate(wf).spec.workflow_path))
        return self._wf[wf]

    def _oracle_doc(self, wf: str, doc_id: str):
        key = (wf, doc_id)
        if key not in self._doc:
            cast, policy = self._workflow(wf)
            spec, cell = self.b.placed[doc_id]
            ws = spec.workspaces[spec.workspace]
            create = {k: v for k, v in spec.assign.items() if policy.create and k in (policy.create.get("assign") or {})}
            od = Doc.create(policy, cast, ws, ws.get("org"), doc_id, spec.owner, spec.creator, create or None, cell.attrs or None, spec.invited)
            for s in cell.path.steps[: cell.index]:
                if not od.apply(s.actor, s.action, C.rebind(s.input, doc_id, od.version), s.event).allowed:
                    raise ValueError(f"the oracle refuses a step of {doc_id}: {s}")
            self._doc[key] = od
        return self._doc[key]

    def known(self, wf: str, doc_id: str) -> bool:
        return (doc_id.endswith("/text") and wf == "regulation_single") or (wf != "regulation_single" and doc_id in self.b.placed and self.b.assignment_wf.get(doc_id) == wf)

    def access(self, wf: str, doc_id: str, person: str, find: bool) -> int | None:
        if wf == "regulation_single":
            return self.reg_visible.get(person, {}).get(doc_id)  # the table serves the same version to find and to open
        return self._oracle_doc(wf, doc_id).access(person, find=find)


def tool_entries(run: dict, services: dict, list_cache: dict, report: Counter) -> list[dict]:
    """[{tool, step, doc, version, find}] for every document a tool returned in the run (lists run again through the service)."""
    out = []
    wf = run["workflow"]
    if "gateway_trace" in run:  # a run with the full trace: what the service returned, in full
        for g in run["gateway_trace"]:
            if g["call"] == "list_documents":
                out += [{"tool": "list_documents", "step": 0, "doc": d["id"], "version": d["version"], "find": False} for d in g["documents"]]
            elif g["call"] == "get_document" and "document" in g:
                out.append({"tool": "get_document", "step": 0, "doc": g["document"]["id"], "version": g["document"]["version"], "find": False})
            elif g["call"] == "search":
                out += [{"tool": "search", "step": 0, "doc": h["id"], "version": h["version"], "find": True} for h in g["hits"]]
        return out
    for s in run["steps"]:
        for t in s["tool_results"]:
            res, name = t["result"], t["name"]
            if name == "search":
                out += [{"tool": "search", "step": s["step"], "doc": h["id"], "version": h["version"], "find": True} for h in res.get("hits", [])]
            elif name == "get_document" and "id" in res:
                out.append({"tool": "get_document", "step": s["step"], "doc": res["id"], "version": res["version"], "find": False})
            elif name == "list_documents" and "documents" in res:
                key = (wf, run["condition"], run["user"], json.dumps(t["arguments"], sort_keys=True))
                if key not in list_cache:
                    tools = tool_controller.tools(services[(wf, run["condition"])], run["user"])
                    list_cache[key] = tools["list_documents"](**t["arguments"])
                full = list_cache[key]["documents"]
                ok = [tuple(d) for d in res["documents"]] == [(d["id"], d["version"], d["state"]) for d in full[:5]] and res["n"] == len(full)
                report["list_calls"] += 1
                report["list_replay_equal"] += ok
                out += [{"tool": "list_documents", "step": s["step"], "doc": d["id"], "version": d["version"], "find": False} for d in full]
    return out


def main(argv: list[str]) -> None:
    args = dict(zip(argv[::2], argv[1::2]))
    results, outdir = Path(args["--results"]), Path(args.get("--out", "."))
    U.setup()
    from .analysis import load
    rows = load(results)
    U.G["assignment_wf"] = {d: a["workflow"] for d, a in U.G["syn"][2].items()}
    oracle = Oracle()
    oracle.b.assignment_wf = U.G["assignment_wf"]
    services = {}
    worlds = dict(U.G["syn"][0])
    worlds["regulation_single"] = U.G["reg"][0]
    corpora = {wf: U.G["syn"][1] for wf in U.G["syn"][0]} | {"regulation_single": U.G["reg"][1]}
    for wf, w in worlds.items():
        for cond in ("no_access_control", "filtering", "filtering_guidance"):
            services[(wf, cond)] = DocumentService(CondWorld(w, cond), corpora[wf])
    list_cache: dict = {}
    replay = Counter()
    per_run = []
    for r in rows:
        wf, person = r["workflow"], r["user"]
        leaks, n_entries, unknown = [], 0, 0
        for e in tool_entries(r, services, list_cache, replay):
            n_entries += 1
            doc = e["doc"]
            if wf != "regulation_single" and not oracle.known(wf, doc):
                unknown += 1
                continue
            acc = oracle.access(wf, doc, person, e["find"])
            if acc is None:
                leaks.append({**e, "kind": "document outside", "allowed": None})
            elif acc != e["version"]:
                leaks.append({**e, "kind": "version outside", "allowed": acc})
        per_run.append({"key": r["key"], "condition": r["condition"], "model": r["model"], "role": r["role"], "tool_results": n_entries, "unknown_to_oracle": unknown,
                        "context_leaks": leaks})
    import gzip
    outdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(outdir / "context_leaks.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"list_replay": dict(replay), "runs": per_run}, f, ensure_ascii=False)
    print(f"{len(per_run)} runs; {sum(bool(r['context_leaks']) for r in per_run)} with a context leak; "
          f"list calls run again: {replay['list_replay_equal']} of {replay['list_calls']} equal to the saved ones")


if __name__ == "__main__":
    main(sys.argv[1:])
