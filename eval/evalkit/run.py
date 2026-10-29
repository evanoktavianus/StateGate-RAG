"""The request grid: for every situation of every path, every person class, over every channel and action, ask the system and the oracle.

The system is asked through the Access control and the Tool controller (and, for a sample, the interface and the HTTP layer). The oracle is
asked about the same person, document and operation, and tracks the state from the same steps with its own interpreter; it never asks the system.
Two errors are counted, everything else is a slice of them:

  wrongful allow   the system allows (returns a passage, accepts an action), the oracle denies
  wrongful denial  the system denies, the oracle allows

A request is one (situation, person class representative, channel or action). Each call and result is kept in the log, so every number traces to a call.

    python -m evalkit.run <workflow> [--baseline] [--paths N] [--repo memory|postgres] [--log file.json]   (--repo defaults to postgres when DATABASE_URL is set, else memory; --compare also runs the other one and lists differences)
"""

from __future__ import annotations

import array
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller
from docflow.stores.search_index import DIMS, Corpus
from docflow.support.model import Refused

from . import paths as paths_mod
from .oracle import Cast, Decision, Doc
from .oracle.policy import load_policy
from .people import classify, policy_terms, representatives
from .replay import Unfireable, input_for
from .worlds import Spec, build_world

QUERY_WORDS = "evalkit"
QUERY_VECTOR = [1.0] + [0.0] * (DIMS - 1)
GHOST = "no-such-document"
LIMIT = 20
READ = ("list", "get", "get_version", "keyword", "semantic", "hybrid", "tool_list", "tool_get", "tool_get_version", "tool_search")


def text_of(doc: str, version: int) -> str:
    return f"{QUERY_WORDS} text of {doc} version {version}"


def vector_of(version: int) -> list[float]:
    return [1.0, 0.1 * version] + [0.0] * (DIMS - 2)


def clone_world(world):
    """A world that can take an action without changing the original (the documents' histories are replaced, never edited)."""
    w = copy.copy(world)
    w.documents = {k: copy.copy(d) for k, d in world.documents.items()}
    w.store, w.log, w._depth = None, [], 0
    return w


def unrestricted(world) -> None:
    """The baseline with no access control: every person is served the newest version of everything. Inside the harness only; `src/` is untouched."""
    world.readable_version = lambda doc_id, user, *, for_search=False: world.documents[doc_id].timeline.version


class Run:
    def __init__(self, workflow: str, *, baseline: bool = False, repo: str = "memory", http: bool = True, ps=None, machine=None,
                 everyone: bool = False, fast: bool = False) -> None:
        """`ps` reuses the paths of another run; `machine` is a mutant policy the system runs on (the oracle stays on the original);
        `everyone` asks for every person of the cast, not one per person class; `fast` stops at the first path that shows a difference."""
        self.workflow, self.baseline, self.repo_kind, self.http = workflow, baseline, repo, http
        self.everyone, self.fast = everyone, fast
        self.ps = ps or paths_mod.generate(workflow)
        if machine is not None:
            self.ps = copy.copy(self.ps)
            self.ps.model = replace(self.ps.model, machine=machine)
        self.spec: Spec = self.ps.spec
        self.model = self.ps.model
        self.policy = load_policy(self.spec.workflow_path)
        self.terms = policy_terms(self.spec.workflow_path)
        self.cast = Cast.from_people(self.spec.people)
        self.ws = self.spec.workspaces[self.spec.workspace]
        self.records: list[dict] = []
        self.step_findings: list[dict] = []
        self.http_findings: list[dict] = []
        self.repository = None
        self._tmp = None

    # ── the repository the world writes through to (PostgreSQL), if asked ──
    def _store(self):
        if self.repo_kind == "memory":
            return None, None
        import secrets
        from docflow.stores.postgres_repository import PostgresRepository
        from docflow.stores.repository import PassageIn
        from docflow.stores.world_store import WriteThrough

        if self.repository is None:
            self._tmp = tempfile.TemporaryDirectory()
            self.repository = PostgresRepository(os.environ["DATABASE_URL"], Path(self._tmp.name) / "files", namespace="ev" + secrets.token_hex(4))
        else:  # a fresh namespace for each path: the document id is the same in all of them
            self.repository.drop_all()
            self.repository.close()
            self.repository = PostgresRepository(os.environ["DATABASE_URL"], Path(self._tmp.name) / "files", namespace="ev" + secrets.token_hex(4))
        content = lambda doc, v: [PassageIn(0, (0, 0), "", text_of(doc, v), tuple(vector_of(v)))]
        return WriteThrough(self.repository, {self.spec.workflow: self.spec.workflow_path}, content), self.repository

    # ── the grid ──
    def run(self, max_paths: int | None = None) -> "Run":
        for path in self.ps.paths[:max_paths]:
            self._path(path)
            if self.fast and (self.step_findings or self.disagreements()):
                break
        if self.http:
            self._http_sample()
        if self.repository is not None:
            self.repository.drop_all()
            self.repository.close()
        return self

    def _oracle_doc(self, attrs: dict) -> Doc | Decision:
        s = self.spec
        assign = {k: v for k, v in s.assign.items() if self.policy.create and k in (self.policy.create.get("assign") or {})}
        return Doc.create(self.policy, self.cast, self.ws, self.ws.get("org"), s.doc_id, s.owner, s.creator, assign or None, attrs or None, s.invited)

    def _path(self, path) -> None:
        try:
            self._path_inner(path)
        except Unfireable as u:  # a mutant policy refuses a step the original accepted
            self.step_findings.append({"path": path.id, "index": len(path.steps), "what": f"the system refuses a recorded step: {u.reason}"})

    def _path_inner(self, path) -> None:
        store, repo = self._store()
        odoc = self._oracle_doc(path.attrs)
        if isinstance(odoc, Decision):
            self.step_findings.append({"path": path.id, "index": 0, "what": f"the oracle refuses to create the document: {odoc.reason}"})
            return
        corpus = Corpus([], array.array("f"))
        for sit, world in paths_mod.situations(self.ps, path, store):
            if sit.index > 0:
                step = path.steps[sit.index - 1]
                d = odoc.apply(step.actor, step.action, step.input, step.event)
                if not d.allowed:
                    self.step_findings.append({"path": path.id, "index": sit.index, "what": f"the oracle refuses a step the system accepted: {step.actor} {step.action} ({d.reason})"})
                    return
            if (sit.state, sit.version) != (odoc.state, odoc.version):
                self.step_findings.append({"path": path.id, "index": sit.index, "what": f"state or version differs: system {sit.state} v{sit.version}, oracle {odoc.state} v{odoc.version}"})
            for v in range(1, sit.version + 1):
                if (self.spec.doc_id, v) not in corpus.by_version:
                    corpus.add(self.spec.doc_id, v, [text_of(self.spec.doc_id, v)], [vector_of(v)])
            if self.baseline:
                unrestricted(world)
            gw = DocumentService(world, paths_mod_index(repo) if repo else corpus)
            classes = classify(self.cast, odoc, self.terms)
            reps = [(c, p) for p, c in sorted(classes.items())] if self.everyone else list(representatives(classes).items())
            for cls, person in reps:
                ctx = {"workflow": self.workflow, "path": path.id, "index": sit.index, "state": odoc.state, "class": cls, "person": person}
                self._reads(gw, odoc, ctx)
                self._actions(world, odoc, ctx)
                if sit.index == 0:
                    self._create(odoc, ctx, path.attrs)
        # the path is over

    # ── read channels ──
    def _record(self, ctx: dict, channel: str, target, want, got, seconds: float, **extra) -> dict:
        want_s, got_s = set(want), set(got)
        leak = bool(got_s - want_s)
        denial = bool(want_s - got_s)
        wrong_version = any(d == d2 and v != v2 for d, v in got_s - want_s for d2, v2 in want_s - got_s)
        rec = {**ctx, "channel": channel, "target": target, "want": sorted(want_s), "got": sorted(got_s), "agree": not leak and not denial,
               "leak": leak, "denial": denial, "wrong_version": wrong_version, "ms": round(seconds * 1000, 3), **extra}
        self.records.append(rec)
        return rec

    def _reads(self, gw: DocumentService, od: Doc, ctx: dict) -> None:
        p, doc, serve, find = ctx["person"], self.spec.doc_id, od.access(ctx["person"]), od.access(ctx["person"], find=True)
        guidance = od.guidance(p)
        opened = {(doc, serve)} if serve is not None else set()
        found = {(doc, find)} if find is not None else set()

        def timed(f):
            t = time.perf_counter()
            try:
                out = f()
            except NotFound:
                out = None
            return out, time.perf_counter() - t

        def gsig(f):  # what a refusal looks like to the caller
            try:
                f()
                return ("opened",)
            except NotFound:
                return ("NotFound",)

        tools = tool_controller.tools(gw, p)
        out, dt = timed(lambda: gw.list_documents(p))
        self._record(ctx, "list", doc, opened, {(d.id, d.version) for d in out}, dt, guidance_ok=_g(guidance, [d.guidance for d in out]))
        out, dt = timed(lambda: gw.get_document(p, doc))
        got = {(out["id"], out["version"])} if out else set()
        rec = self._record(ctx, "get", doc, opened, got, dt, guidance_ok=_g(guidance, [out["guidance"]] if out else []))
        self._existence(rec, serve is None, gsig(lambda: gw.get_document(p, doc)), gsig(lambda: gw.get_document(p, GHOST)))
        for v in range(1, od.version + 1):
            out, dt = timed(lambda: gw.get_document(p, doc, v))
            rec = self._record(ctx, "get_version", f"{doc}@{v}", {(doc, v)} if serve == v else set(), {(out["id"], out["version"])} if out else set(), dt)
            self._existence(rec, serve != v, gsig(lambda: gw.get_document(p, doc, v)), gsig(lambda: gw.get_document(p, GHOST, v)))
        for channel, call in (("keyword", lambda: gw.search(p, QUERY_WORDS, LIMIT)), ("semantic", lambda: gw.similar(p, QUERY_VECTOR, LIMIT)),
                              ("hybrid", lambda: gw.hybrid(p, QUERY_WORDS, QUERY_VECTOR, LIMIT))):
            out, dt = timed(call)
            self._record(ctx, channel, QUERY_WORDS, found, {(h.doc, h.version) for h in out}, dt, guidance_ok=_g(guidance, [h.guidance for h in out]))
        out, dt = timed(lambda: tools["list_documents"]())
        self._record(ctx, "tool_list", doc, opened, {(d["id"], d["version"]) for d in out["documents"]}, dt, guidance_ok=_g(guidance, [d["guidance"] for d in out["documents"]]))
        out, dt = timed(lambda: tools["get_document"](doc))
        got = {(out["id"], out["version"])} if "error" not in out else set()
        rec = self._record(ctx, "tool_get", doc, opened, got, dt, guidance_ok=_g(guidance, [out["guidance"]] if "error" not in out else []))
        self._existence(rec, serve is None, tools["get_document"](doc) if "error" in out else ("opened",), tools["get_document"](GHOST))
        for v in range(1, od.version + 1):
            out, dt = timed(lambda: tools["get_document"](doc, v))
            rec = self._record(ctx, "tool_get_version", f"{doc}@{v}", {(doc, v)} if serve == v else set(), {(out["id"], out["version"])} if "error" not in out else set(), dt)
            self._existence(rec, serve != v, tools["get_document"](doc, v) if "error" in out else ("opened",), tools["get_document"](GHOST, v))
        out, dt = timed(lambda: tools["search"](QUERY_WORDS, LIMIT))
        self._record(ctx, "tool_search", QUERY_WORDS, found, {(h["id"], h["version"]) for h in out["hits"]}, dt, guidance_ok=_g(guidance, [h["guidance"] for h in out["hits"]]))

    def _existence(self, rec: dict, denied: bool, signature, ghost) -> None:
        """A refused request must look exactly like one for an id that does not exist."""
        rec["existence_leak"] = bool(denied and signature != ("opened",) and signature != ghost)
        rec["existence_checked"] = denied

    # ── actions and creation ──
    def _actions(self, world, od: Doc, ctx: dict) -> None:
        state, p, doc = od.state, ctx["person"], self.spec.doc_id
        names = od.available()
        other = next((a for s, acts in self.policy.states.items() if s != state for a in (acts.get("on") or acts.get(True) or {}) if a not in names), "no_such_action")
        for action in [*names, other]:
            try:
                e = self.model.edge(state, action)
                inp = input_for(self.model, world, self.spec, e)
            except StopIteration:
                inp = {}
            want = od.decide(p, action, inp)
            w2, t = clone_world(world), time.perf_counter()
            try:
                w2.act(doc, p, action, input=inp)
                got, reason = True, None
            except Refused as r:
                got, reason = False, r.reason
            dt = time.perf_counter() - t
            leak, denial = got and not want.allowed, (not got) and want.allowed
            self.records.append({**ctx, "channel": "action", "target": action, "want": want.allowed, "got": got, "agree": got == want.allowed,
                                 "leak": leak, "denial": denial, "wrong_version": False, "ms": round(dt * 1000, 3),
                                 "reason_system": reason, "reason_oracle": want.reason})

    def _create(self, od: Doc, ctx: dict, attrs: dict) -> None:
        if self.policy.create is None:
            return
        s = self.spec
        want = self._oracle_create(ctx["person"], attrs)
        t = time.perf_counter()
        try:
            build_world(replace(s, creator=ctx["person"]), attrs, self.model.machine)
            got, reason = True, None
        except Refused as r:
            got, reason = False, r.reason
        self.records.append({**ctx, "channel": "create", "target": s.workflow, "want": want.allowed, "got": got, "agree": got == want.allowed,
                             "leak": got and not want.allowed, "denial": (not got) and want.allowed, "wrong_version": False,
                             "ms": round((time.perf_counter() - t) * 1000, 3), "reason_system": reason, "reason_oracle": want.reason})

    def _oracle_create(self, person: str, attrs: dict) -> Decision:
        s = self.spec
        assign = {k: v for k, v in s.assign.items() if k in (self.policy.create.get("assign") or {})}
        made = Doc.create(self.policy, self.cast, self.ws, self.ws.get("org"), s.doc_id, s.owner, person, assign or None, attrs or None, s.invited)
        return made if isinstance(made, Decision) else Decision(True)

    # ── a sample through the interface and the HTTP layer ──
    def _http_sample(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from docflow.authentication_directory import AuthenticationDirectory
            from docflow.interface_api import Interface
            from docflow.interface_api_http import create_app
            from docflow.model_client import ModelClient, Scripted, Settings, chat_response
        except ImportError:
            self.http_findings.append({"skipped": "fastapi or httpx is not installed"})
            return
        denied = next((r for r in self.records if r["channel"] == "get" and r.get("existence_checked")), None)
        if denied is None:
            self.http_findings.append({"skipped": "no refused request in this run"})
            return
        path = next(p for p in self.ps.paths if p.id == denied["path"])
        for sit, world in paths_mod.situations(self.ps, path):
            if sit.index == denied["index"]:
                break
        corpus = Corpus([], array.array("f"))
        for v in range(1, sit.version + 1):
            corpus.add(self.spec.doc_id, v, [text_of(self.spec.doc_id, v)], [vector_of(v)])
        gw = DocumentService(world, corpus)
        answers = {}
        for label, doc in (("denied", self.spec.doc_id), ("missing", GHOST)):
            script = Scripted([chat_response(None, calls=[("get_document", {"doc_id": doc})]), chat_response("Scripted reply: the run is over.")])
            directory = AuthenticationDirectory(world)
            interface = Interface(directory, gw, ModelClient(Settings(), script))
            r = TestClient(create_app(interface)).post("/ask", json={"question": "q"}, headers={"Authorization": f"Bearer {directory.issue(denied['person'])}"})
            body = {k: v for k, v in r.json().items() if k != "run_id"}
            log = interface.run_log(r.json()["run_id"])
            tool = [t["result"] for st in log["steps"] for t in st["tool_results"]]
            answers[label] = (r.status_code, body, tool)
        self.http_findings.append({"person": denied["person"], "state": denied["state"], "same_status_and_body": answers["denied"][:2] == answers["missing"][:2],
                                   "same_tool_result": answers["denied"][2] == answers["missing"][2], "status": answers["denied"][0]})

    # ── results ──
    def summary(self) -> dict:
        def tally(rows):
            n = len(rows)
            la = sum(r["leak"] for r in rows)
            wd = sum(r["denial"] for r in rows)
            return {"n": n, "wrongful_allow": la, "wrongful_denial": wd, "wrong_version": sum(r["wrong_version"] for r in rows),
                    "existence_leaks": sum(bool(r.get("existence_leak")) for r in rows), "existence_checked": sum(bool(r.get("existence_checked")) for r in rows),
                    "guidance_checked": sum(r.get("guidance_ok") is not None for r in rows), "guidance_wrong": sum(r.get("guidance_ok") is False for r in rows),
                    "disagreements": sum(not r["agree"] for r in rows)}
        out = {"workflow": self.workflow, "baseline": self.baseline, "repo": self.repo_kind, "requests": len(self.records), "all": tally(self.records),
               "by_channel": {c: tally([r for r in self.records if r["channel"] == c]) for c in sorted({r["channel"] for r in self.records})},
               "by_state": {c: tally([r for r in self.records if r["state"] == c]) for c in sorted({r["state"] for r in self.records})},
               "by_class": {c: tally([r for r in self.records if r["class"] == c]) for c in sorted({r["class"] for r in self.records})},
               "step_findings": self.step_findings, "http": self.http_findings}
        lat = defaultdict(list)
        for r in self.records:
            lat[r["channel"]].append(r["ms"])
        out["latency_ms"] = {c: {"median": round(statistics.median(v), 3), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 3)} for c, v in lat.items()}
        return out

    def disagreements(self) -> list[dict]:
        return [r for r in self.records if not r["agree"] or r.get("existence_leak") or r.get("guidance_ok") is False]


def same_answers(a: "Run", b: "Run") -> list[str]:
    """The differences between two runs of the same workflow (say, on the two repositories): every request must have the same expected and returned result."""
    key = lambda r: (r["path"], r["index"], r["person"], r["channel"], str(r["target"]))
    ra, rb = {key(r): r for r in a.records}, {key(r): r for r in b.records}
    out = [f"only in one run: {k}" for k in sorted(set(ra) ^ set(rb))]
    for k in sorted(set(ra) & set(rb)):
        x, y = ra[k], rb[k]
        if (x["want"], x["got"], x.get("existence_leak"), x.get("guidance_ok")) != (y["want"], y["got"], y.get("existence_leak"), y.get("guidance_ok")):
            out.append(f"{k}: {x['got']} against {y['got']}")
    return out


def paths_mod_index(repo):
    from docflow.document_service.retrieve import PartitionedIndex
    return PartitionedIndex(repo)


def _g(want: str | None, got: list) -> bool | None:
    """Whether every guidance returned is the text the person's rule or, failing that, the state gives (None: nothing was returned to check)."""
    return None if not got else all(g == want for g in got)


def main(argv: list[str]) -> None:
    args = list(argv)
    opts = {"--paths": None, "--repo": "postgres" if os.environ.get("DATABASE_URL") else "memory", "--log": None}  # PostgreSQL when DATABASE_URL is set, as in the paper; else memory
    flags = {"--baseline": False, "--no-http": False, "--brief": False, "--compare": False}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i : i + 2]
    for k in flags:
        if k in args:
            flags[k] = True
            args.remove(k)
    t0 = time.time()
    run = Run(args[0], baseline=flags["--baseline"], repo=opts["--repo"], http=not flags["--no-http"])
    run.run(int(opts["--paths"]) if opts["--paths"] else None)
    s = run.summary()
    s["seconds"] = round(time.time() - t0, 1)
    bad = run.disagreements()
    if flags["--compare"]:  # the same run on the other repository: every answer must be the same
        other = Run(args[0], baseline=flags["--baseline"], repo="memory" if opts["--repo"] != "memory" else "postgres", http=False).run(int(opts["--paths"]) if opts["--paths"] else None)
        diff = same_answers(run, other)
        print(f"compared {run.repo_kind} with {other.repo_kind}: {len(run.records)} and {len(other.records)} requests, {len(diff)} differences" + ("" if not diff else f", first: {diff[:3]}"))
    if flags["--brief"]:
        a = s["all"]
        print(f"{s['workflow']}{' (baseline)' if s['baseline'] else ''} [{s['repo']}]: {s['requests']} requests, wrongful allow {a['wrongful_allow']}, wrongful denial {a['wrongful_denial']}, "
              f"wrong version {a['wrong_version']}, existence leaks {a['existence_leaks']}/{a['existence_checked']}, guidance wrong {a['guidance_wrong']}/{a['guidance_checked']}, "
              f"step findings {len(s['step_findings'])}, http {[h.get('same_status_and_body', h.get('skipped')) for h in s['http']]}, {s['seconds']} s")
    else:
        print(json.dumps({k: v for k, v in s.items() if k in ("workflow", "baseline", "repo", "requests", "all", "by_channel", "step_findings", "http", "seconds")}, indent=1))
    print(f"{len(bad)} disagreements" + ("" if not bad else f", first: {bad[0]}"))
    if opts["--log"]:
        Path(opts["--log"]).write_text(json.dumps({"summary": s, "world": {"people": run.spec.people, "workspaces": run.spec.workspaces, "document": run.spec.doc_id,
                                                                            "owner": run.spec.owner, "assign": run.spec.assign},
                                                   "records": run.records}, indent=0))


if __name__ == "__main__":
    main(sys.argv[1:])
