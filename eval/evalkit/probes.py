"""The targeted probes (P1 to P8), as named functions with a pass or fail each. Smoke runs only; no model, no network.

    python -m evalkit.probes <workflow> [--md file] [--json file]

Each probe returns a Result: status `pass`, `fail`, `not_run` (a needed part is missing, with the reason) or `not_applicable`.
Known gaps are marked, not hidden: a person changing position or unit after documents were handled (out of scope: the fixtures fix each
person's unit and positions), and concurrent actions (out of scope: the repository uses one connection with a lock).
"""

from __future__ import annotations

import array
import json
import os
import sys
from dataclasses import dataclass, field

from docflow.access_control.enforcement import NotFound
from docflow.document_service.service import DocumentService
from docflow.agent_harness import tool_controller
from docflow.stores.search_index import Corpus

from . import paths as paths_mod
from . import utility
from .run import GHOST, Run


@dataclass
class Result:
    probe: str
    name: str
    status: str
    detail: str = ""
    counts: dict = field(default_factory=dict)


def p1_wrong_person(run: Run) -> Result:
    bad = run.disagreements()
    return Result("P1", "A document is shown to the wrong person in some state", "pass" if not bad else "fail",
                  "covered by the request grid (every state, person class and channel)", {"requests": len(run.records), "disagreements": len(bad)})


def p2_old_access_survives(run: Run) -> Result:
    """After each transition, every person the oracle no longer lets read the document gets nothing from the read channels; a superseded version is not served."""
    prev: dict[tuple, dict] = {}
    checked = lost = bad = 0
    for r in run.records:
        if r["channel"] not in ("list", "get", "keyword", "tool_get", "tool_search"):
            continue
        k = (r["path"], r["person"], r["channel"], str(r["target"]))
        before = prev.get(k)
        if before is not None and before["want"] and not r["want"]:
            lost += 1
            checked += 1
            bad += bool(r["got"])
        prev[k] = r
    stale = sum(1 for r in run.records if r["channel"] in ("get_version", "tool_get_version") and r["got"] and not r["want"])
    return Result("P2", "Old access survives a change of state", "pass" if bad == 0 and stale == 0 else "fail",
                  "persons who lost access at a transition, and named older versions asked after it", {"lost_access_cases": lost, "still_served": bad, "superseded_version_served": stale})


def p3_existence(run: Run) -> Result:
    ex = [r for r in run.records if r.get("existence_checked")]
    leaks = sum(bool(r.get("existence_leak")) for r in ex)
    http = run.http_findings
    if not http:
        http_note = "the HTTP layer was not sampled in this run"
    elif "skipped" in http[0]:
        http_note = "HTTP layer: not run (" + http[0]["skipped"] + ")"
    else:
        http_note = f"HTTP layer: same status and body {http[0]['same_status_and_body']}, same tool result {http[0]['same_tool_result']}"
        if not (http[0]["same_status_and_body"] and http[0]["same_tool_result"]):
            leaks += 1
    return Result("P3", "A refusal reveals that the document exists", "pass" if leaks == 0 else "fail", http_note,
                  {"refused_requests_checked": len(ex), "existence_leaks": leaks})


def p4_cross_unit(workflow: str) -> Result:
    """Scripted tool calls from a person of another unit of the same organization and from another organization: another unit's id, a guessed id, a named version, arguments naming another person."""
    world, corpus, oracle, spec = utility.build(workflow, 8)
    gw = DocumentService(world, corpus)
    n = bad = 0
    for person in ("z_same_org_other_unit", "z_other_org", "z_outsider"):
        if person not in spec.people:
            continue
        bound = tool_controller.tools(gw, person)
        ghost = tool_controller.call(bound, "get_document", {"doc_id": GHOST})
        for doc, od in oracle.items():
            hidden = od.access(person) is None
            for args in ({"doc_id": doc}, {"doc_id": doc, "version": 1}, {"doc_id": doc, "version": 99}):
                out = tool_controller.call(bound, "get_document", args)
                n += 1
                if hidden or args.get("version") == 99:
                    bad += out != ghost
        for guess in ("u99", "U00", "../u00", "u00 OR 1=1"):
            n += 1
            bad += tool_controller.call(bound, "get_document", {"doc_id": guess}) != ghost
        for naming in ({"doc_id": "u00", "user": spec.owner}, {"doc_id": "u00", "person": spec.owner}, {"query": "evalkit", "as_user": spec.owner}):
            name = "search" if "query" in naming else "get_document"
            out = tool_controller.call(bound, name, naming)
            n += 1
            bad += out != tool_controller.call(bound, "get_document", {"doc_id": None})  # a bad request, and never another person's data
        listed = tool_controller.call(bound, "list_documents", {})
        visible = {d for d, od in oracle.items() if od.access(person) is not None}
        n += 1
        bad += {d["id"] for d in listed["documents"]} != visible
    return Result("P4", "Access crosses the unit boundary", "pass" if bad == 0 else "fail",
                  "scripted tool calls by a person of another unit of the same organization, of another organization, and a signed-in outsider",
                  {"calls": n, "wrong": bad})


def p5_actions(run: Run) -> Result:
    acts = [r for r in run.records if r["channel"] in ("action", "create")]
    bad = sum(1 for r in acts if r["got"] and not r["want"])
    refused_ok = sum(1 for r in acts if not r["want"])
    return Result("P5", "An action is accepted from someone who may not take it", "pass" if bad == 0 else "fail",
                  "the action grid, with `create` by someone not allowed", {"actions_and_creates": len(acts), "should_be_refused": refused_ok, "wrongly_accepted": bad})


def p6_legitimate_refused(run: Run, workflow: str) -> Result:
    denials = sum(r["denial"] for r in run.records)
    u = utility.measure(workflow, 8, 5)
    ok = denials == 0 and (u["fill_mean"] in (None, 1.0))
    return Result("P6", "Legitimate users are refused", "pass" if ok else "fail", "wrongful denial over the grid, and the fill of the restricted search",
                  {"requests": len(run.records), "wrongful_denials": denials, "authorized_yield_mean": u["authorized_yield_mean"], "fill_mean": u["fill_mean"]})


def p7_hidden_changes_ranking(workflow: str) -> Result:
    """Scores and order of what a person sees must be identical with and without the documents they cannot read in the corpus (same unit; another unit has no separate index in memory, see P7b)."""
    world, corpus, oracle, spec = utility.build(workflow, 8)
    gw = DocumentService(world, corpus)
    n = bad = 0
    for person in sorted(spec.people):
        visible = {d for d, od in oracle.items() if od.access(person, find=True) is not None}
        if not visible or len(visible) == len(oracle):
            continue
        keep = [p for p in corpus.passages if p.doc in visible]
        small = Corpus([], array.array("f"))
        for d in sorted(visible):
            for v in sorted({p.version for p in keep if p.doc == d}):
                ps = [p for p in corpus.passages if p.doc == d and p.version == v]
                small.add(d, v, [p.content for p in ps], [corpus.vector_of(p) for p in ps])
        gw2 = DocumentService(world, small)
        a = [(h.doc, h.version, round(h.score, 9)) for h in gw.search(person, utility.QUERY, 50)]
        b = [(h.doc, h.version, round(h.score, 9)) for h in gw2.search(person, utility.QUERY, 50)]
        n += 1
        bad += a != b
    status = "pass" if bad == 0 and n else ("not_applicable" if not n else "fail")
    return Result("P7", "A hidden document changes what a person sees", status,
                  "ranking and scores for each person, with every unreadable document in the corpus and with none", {"people_compared": n, "different": bad})


def p7b_cross_unit_ranking() -> Result:
    """The search layer of a repository with two units: scores and order of what a person sees must not change when hidden documents are added to their own unit or to another unit.

    Two repositories are built, one with only the visible documents and one that also holds hidden ones stuffed with the query words and the query's vector. On PostgreSQL when
    DATABASE_URL is set, else in memory (said in the note). The person sees the visible set of one unit, and then of both."""
    import secrets
    import tempfile
    from pathlib import Path

    from docflow.document_service.retrieve import PartitionedIndex
    from docflow.stores.repository import DocumentRecord, InMemoryRepository, PassageIn

    def make():
        if os.environ.get("DATABASE_URL"):
            from docflow.stores.postgres_repository import PostgresRepository
            tmp = tempfile.TemporaryDirectory()
            return PostgresRepository(os.environ["DATABASE_URL"], Path(tmp.name) / "f", namespace="p7" + secrets.token_hex(4)), tmp, "PostgreSQL"
        return InMemoryRepository(), None, "memory"

    def put(repo, unit, doc, text, k):
        repo.put_unit(unit)
        repo.put_document(DocumentRecord(doc, unit, "w", "ann"))
        repo.ingest_version(unit, doc, 1, b"x", [PassageIn(0, (0, 0), "", text, tuple(utility.vec(doc, k)))])

    visible_docs = [("U1", "a1", "zebra zebra zebra", 1), ("U1", "a2", "zebra alpha", 2), ("U2", "b1", "zebra", 3), ("U2", "b2", "zebra beta", 4)]
    hidden_docs = [("U1", "h1", "zebra " * 80 + "alpha beta", 0), ("U2", "h2", "zebra " * 80 + "alpha beta", 0), ("U2", "h3", "zebra " * 40, 0)]
    small, t1, kind = make()
    big, t2, _ = make()
    for u, d, t, k in visible_docs:
        put(small, u, d, t, k)
        put(big, u, d, t, k)
    for u, d, t, k in hidden_docs:
        put(big, u, d, t, k)
    n = bad = 0
    try:
        for visible in ({"a1": 1, "a2": 1}, {"a1": 1, "a2": 1, "b1": 1, "b2": 1}):
            for name, ask in (("keyword", lambda ix: ix.search("zebra", visible, 10)), ("semantic", lambda ix: ix.similar(utility.vec("q", 0), visible, 10))):
                a = [(p.doc, round(sc, 9)) for sc, p in ask(PartitionedIndex(small))]
                b = [(p.doc, round(sc, 9)) for sc, p in ask(PartitionedIndex(big))]
                n += 1
                bad += a != b
    finally:
        for r in (small, big):
            if hasattr(r, "drop_all"):
                r.drop_all()
            if hasattr(r, "close"):
                r.close()
    return Result("P7b", "A hidden document in the same unit or another unit changes a ranking", "pass" if bad == 0 else "fail",
                  f"search layer of a two-unit repository ({kind}); hidden documents stuffed with the query words and the query vector in U1 and U2",
                  {"comparisons": n, "different": bad})


def p8_restart(workflow: str) -> Result:
    if not os.environ.get("DATABASE_URL"):
        return Result("P8", "A restart changes an answer", "not_run", "no DATABASE_URL in this environment")
    from docflow.stores.world_store import load_world

    run = Run(workflow, repo="postgres", http=False)
    ps = run.ps
    n = bad = 0
    for path in ps.paths[:3]:
        store, repo = run._store()
        last = None
        for sit, world in paths_mod.situations(ps, path, store):
            last = world
        before = {(u, d): (last.readable_version(d, u), last.readable_version(d, u, for_search=True)) for u in last.users for d in last.documents}
        states = {d: last.state_of(d) for d in last.documents}
        reloaded = load_world(repo)
        after = {(u, d): (reloaded.readable_version(d, u), reloaded.readable_version(d, u, for_search=True)) for u in reloaded.users for d in reloaded.documents}
        n += len(before)
        bad += sum(before[k] != after.get(k) for k in before) + sum(states[d] != reloaded.state_of(d) for d in states)
    if run.repository is not None:
        run.repository.drop_all()
        run.repository.close()
    return Result("P8", "A restart changes an answer", "pass" if bad == 0 else "fail", "the world is dropped and loaded again from the repository alone; the read answers are compared",
                  {"answers_compared": n, "different": bad})


OUT_OF_SCOPE = [
    Result("G1", "A person changes position or unit after documents were handled", "not_applicable", "out of scope: each fixture person has one unit and fixed positions; the engine reads them at request time, which the grid does not vary over time"),
    Result("G2", "Concurrent actions", "not_applicable", "out of scope: the PostgreSQL repository uses one connection with a lock; ordering is not tested here"),
]


def run_all(workflow: str, http: bool = True) -> list[Result]:
    run = Run(workflow, http=http).run()
    return [p1_wrong_person(run), p2_old_access_survives(run), p3_existence(run), p4_cross_unit(workflow), p5_actions(run), p6_legitimate_refused(run, workflow),
            p7_hidden_changes_ranking(workflow), p7b_cross_unit_ranking(), p8_restart(workflow), *OUT_OF_SCOPE]


def to_markdown(workflow: str, results: list[Result]) -> str:
    out = [f"### Probes on {workflow}", "", "| probe | name | status | counts | note |", "|---|---|---|---|---|"]
    out += [f"| {r.probe} | {r.name} | {r.status} | {json.dumps(r.counts)} | {r.detail} |" for r in results]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    args = list(argv)
    opts = {"--md": None, "--json": None}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i : i + 2]
    res = run_all(args[0])
    text = to_markdown(args[0], res)
    print(text)
    if opts["--md"]:
        open(opts["--md"], "w").write(text + "\n")
    if opts["--json"]:
        open(opts["--json"], "w").write(json.dumps([r.__dict__ for r in res], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
