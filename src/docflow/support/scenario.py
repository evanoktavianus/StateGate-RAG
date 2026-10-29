"""Runs a scenario file against a workflow file: people, moves and expected outcomes, all data.

Adding a workflow to a test suite is adding two YAML files; no Python. A scenario names the workflows it uses,
the people, and a list of steps. Steps either do something (create, add, upload, act …) or expect something.
Every `expect_access` is checked on every channel (list, get by id, search) and against a document that does not exist,
so the expected outcomes are the *oracle*: a table that is never computed by the engine.

    python -m docflow.support.scenario scenarios/card1_policy_analysis_simple.yml --workflows workflows/

Steps (each a one-key mapping):
    add_document {id, workspace, owner, workflow, [invited, creator, assign, attrs]}
    upload {doc, by}          act {doc, by, action, [input]}         event {doc, action, event}
    expect_refused {step: <any step>, [reason]}
    expect_state {doc, state}
    expect_access {docs: {letter: id}, table: {person: "letters"}, [find: {...}], [versions: {letter: n}],
                   [guidance: {letter: text | null}]}
    expect_log {kind, [count], [reason]}
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..access_control.access_policy_engine.machine import Machine
from ..access_control.enforcement import NotFound
from ..document_service.service import DocumentService
from ..stores.search_index import Corpus
from ..stores.workflow_definitions import _Loader
from ..stores.world import World
from .model import Refused


@dataclass
class Result:
    name: str
    steps: int = 0
    checks: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def token(doc: str, version: int) -> str:
    """The word each version's text is made of, so a search for it says exactly which version it found."""
    return "tok" + re.sub(r"[^0-9a-z]", "", doc.lower()) + "v" + str(version)


class Runner:
    def __init__(self, scenario: dict, workflows_dir: Path, repository=None) -> None:
        """With a `repository`, the world writes through to it after every accepted step; the text of each version is its token."""
        self.s = scenario
        names = scenario["workflows"]
        store = None
        if repository is not None:
            from ..stores.repository import PassageIn
            from ..stores.world_store import WriteThrough

            store = WriteThrough(repository, {n: Path(workflows_dir) / f"{n}.yml" for n in names},
                                 lambda doc, version: [PassageIn(0, (0, 0), "", f"{token(doc, version)} text of {doc} version {version}")])
        self.world = World({n: Machine.load(workflows_dir / f"{n}.yml") for n in names}, store)
        self.corpus = Corpus([])
        self.gw = DocumentService(self.world, self.corpus)
        self.result = Result(scenario.get("name", "scenario"))
        for org in {p.get("org") for p in scenario["people"].values() if p.get("org")}:
            self.world.add_org(org)
        for pid, p in scenario["people"].items():
            self.world.add_user(pid, p.get("org"), p.get("groups", ()), p.get("org_admin", False), p.get("unit"), p.get("positions", ()), p.get("attrs"))
        for wid, w in scenario.get("workspaces", {}).items():
            self.world.add_workspace(wid, w.get("org"), w.get("members"), w.get("group_roles"))

    def _sync(self) -> None:
        for doc in self.world.documents:
            d = self.world.documents[doc]
            top = d.timeline.version
            for v in range(1, top + 1):
                if (doc, v) not in self.corpus.by_version:
                    self.corpus.add(doc, v, [f"{token(doc, v)} text of {doc} version {v}"])

    # ── steps ──
    def do(self, step: dict) -> None:
        (kind, a), = step.items()
        w = self.world
        match kind:
            case "add_document":
                w.add_document(a["id"], a.get("workspace", ""), a["owner"], a["workflow"], a.get("version", 1), a.get("invited", ()), a.get("creator"), a.get("assign"), a.get("attrs"))
            case "upload":
                w.act(a["doc"], a["by"], "upload")
            case "act":
                w.act(a["doc"], a["by"], a["action"], input=a.get("input") or {})
            case "event":
                w.fire_event(a["doc"], a["action"], a["event"])  # the host fires it; no person takes it
            case _:
                raise ValueError(f"unknown step {kind!r}")
        self._sync()

    def check(self, ok: bool, message: str) -> None:
        self.result.checks += 1
        if not ok:
            self.result.failures.append(f"step {self.result.steps}: {message}")

    def expect(self, step: dict) -> bool:
        (kind, a), = step.items()
        match kind:
            case "expect_refused":
                try:
                    self.do(a["step"])
                except Refused as r:
                    self.check(a.get("reason") in (None, r.reason), f"refused with {r.reason}, expected {a.get('reason')}")
                else:
                    self.check(False, f"expected a refusal, got success: {a['step']}")
            case "expect_state":
                got = self.world.state_of(a["doc"])
                self.check(got == a["state"], f"{a['doc']} is in {got}, expected {a['state']}")
            case "expect_log":
                n = sum(1 for e in self.world.log if e.get("kind") == a["kind"] and (a.get("reason") is None or e.get("reason") == a["reason"]))
                self.check(n == a.get("count", 1) if "count" in a else n >= 1, f"log has {n} {a['kind']} entries, expected {a.get('count', 'some')}")
            case "expect_access":
                self.access(a)
            case _:
                return False
        return True

    def access(self, a: dict) -> None:
        docs, table = a["docs"], a["table"]
        find = a.get("find", table)
        allowed = lambda v: set(v) if isinstance(v, (list, str)) else set()  # "TM" is T and M; a list names longer labels (SR, R)
        for person, letters in table.items():
            found = allowed(find.get(person, letters))
            for letter, doc in docs.items():
                self._open(person, doc, letter, letter in allowed(letters), a)
                self._search(person, doc, letter in found, a.get("versions", {}).get(letter))
        for person in table:  # a document that does not exist answers exactly like one that may not be opened
            try:
                self.gw.get_document(person, "no-such-document")
            except NotFound as e:
                self.check(type(e) is NotFound, "missing document")

    def _open(self, person: str, doc: str, letter: str, opened: bool, a: dict) -> None:
        listed = {d.id: d for d in self.gw.list_documents(person)}
        self.check((doc in listed) == opened, f"{person} list of {doc}: {doc in listed}, expected {opened}")
        try:
            got = self.gw.get_document(person, doc)
        except NotFound:
            self.check(not opened, f"{person} get {doc}: not found, expected to open it")
            return
        self.check(opened, f"{person} get {doc}: opened, expected refusal")
        want = a.get("versions", {}).get(letter)
        if want is not None:
            self.check(got["version"] == want, f"{person} {doc} version {got['version']}, expected {want}")
        if letter in a.get("guidance", {}):
            self.check(got["guidance"] == a["guidance"][letter], f"{person} {doc} guidance {got['guidance']!r}, expected {a['guidance'][letter]!r}")
        self.check(self.world.guidance(doc, person) == got["guidance"], f"{person} {doc} guidance differs between channels")

    def _search(self, person: str, doc: str, found: bool, version: int | None) -> None:
        top = max(v for d, v in self.corpus.by_version if d == doc)
        seen = {}
        for v in range(1, top + 1):
            for h in self.gw.search(person, token(doc, v), limit=20):
                if h.doc == doc:
                    seen.setdefault(h.version, True)
        self.check(bool(seen) == found, f"{person} search for {doc}: found versions {sorted(seen)}, expected {'some' if found else 'none'}")
        if seen and version is not None:
            self.check(set(seen) == {version}, f"{person} search for {doc}: versions {sorted(seen)}, expected only {version}")

    def run(self) -> Result:
        for step in self.s["steps"]:
            self.result.steps += 1
            try:
                if not self.expect(step):
                    self.do(step)
            except Refused as r:
                self.result.failures.append(f"step {self.result.steps}: refused {r} on {step}")
                break
            except Exception as e:  # a scenario that cannot run is a failed scenario, with the reason
                self.result.failures.append(f"step {self.result.steps}: {type(e).__name__}: {e} on {step}")
                break
        return self.result


def run(path: str | Path, workflows_dir: str | Path) -> Result:
    scenario = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_Loader)
    return Runner(scenario, Path(workflows_dir)).run()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="+")
    ap.add_argument("--workflows", default="workflows")
    code = 0
    args = ap.parse_args()
    for sc in args.scenario:
        r = run(sc, args.workflows)
        print(f"{'PASS' if r.ok else 'FAIL'} {r.name}: {r.steps} steps, {r.checks} checks")
        for f in r.failures:
            print("   ", f)
        code |= 0 if r.ok else 1
    sys.exit(code)
