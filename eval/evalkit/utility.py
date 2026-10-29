"""Utility and cost measurements, built and tested with the scripted model. Answer quality against the unrestricted view needs a real model
and belongs to the later run.

A shared world holds several documents of one workflow, each carried to a different state by the recorded paths; the oracle (original policy)
says what each person may read. Measured, per person of the cast:

  authorized yield   the share of the unrestricted top-k that the person may read (the oracle's visible set)
  fill               the share of what the person may read that the restricted search returns, up to k
  ranking change     whether the restricted ranking equals the unrestricted ranking with the unreadable documents removed (and the top-1 and overlap)

and the latency per channel (median, p95) on both repositories, and the index overhead: time and peak memory of a search with and without the access filter.

    python -m evalkit.utility <workflow> [--docs N] [--k K] [--json file]
"""

from __future__ import annotations

import array
import statistics
import sys
import time
import tracemalloc
import zlib

from docflow.document_service.service import DocumentService
from docflow.stores.search_index import DIMS, Corpus

from . import paths as paths_mod
from .oracle import Cast, Doc
from .oracle.policy import load_policy
from .worlds import build_cast, create_document

QUERY = "evalkit"


def vec(doc: str, v: int) -> list[float]:
    h = zlib.crc32(f"{doc}@{v}".encode())
    return [1.0, (h % 97) / 97.0, (h % 89) / 89.0] + [0.0] * (DIMS - 3)


def _rename(x, old: str, new: str):
    """A signature digest is `<document>@<version>`: the input of a recorded step names the path's own document."""
    if isinstance(x, str):
        return x.replace(old + "@", new + "@")
    if isinstance(x, dict):
        return {k: _rename(v, old, new) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename(v, old, new) for v in x]
    return x


def build(workflow: str, n_docs: int = 8):
    """(world, corpus, oracle docs, cast people, spec): n_docs documents, each carried part-way along one of the recorded paths."""
    ps = paths_mod.generate(workflow)
    spec = ps.spec
    world = build_cast(spec)
    policy, cast = load_policy(spec.workflow_path), Cast.from_people(spec.people)
    ws = spec.workspaces[spec.workspace]
    oracle, corpus = {}, Corpus([], array.array("f"))
    paths = [p for p in ps.paths if p.steps] or ps.paths
    for i in range(n_docs):
        path = paths[i % len(paths)]
        doc_id = f"u{i:02d}"
        stop = len(path.steps) if i % 2 == 0 else max(1, len(path.steps) // 2)
        create_document(world, spec, path.attrs, doc_id)
        assign = {k: v for k, v in spec.assign.items() if policy.create and k in (policy.create.get("assign") or {})}
        od = Doc.create(policy, cast, ws, ws.get("org"), doc_id, spec.owner, spec.creator, assign or None, path.attrs or None, spec.invited)
        for s in path.steps[:stop]:
            s = type(s)(**{**s.__dict__, "input": _rename(s.input, spec.doc_id, doc_id)}) if getattr(s, "input", None) else s
            if s.event is not None:
                world.fire_event(doc_id, s.action, s.event)
            else:
                world.act(doc_id, s.actor, s.action, input=s.input)
            od.apply(s.actor, s.action, s.input, s.event)
        oracle[doc_id] = od
        for v in range(1, od.version + 1):
            corpus.add(doc_id, v, [f"{QUERY} text of {doc_id} version {v}", f"{QUERY} {doc_id} second passage", f"Reference code CANARY-{doc_id}-v{v}."], [vec(doc_id, v), vec(doc_id, v + 50), vec(doc_id, v + 70)])
    return world, corpus, oracle, spec


def measure(workflow: str, n_docs: int = 8, k: int = 5) -> dict:
    world, corpus, oracle, spec = build(workflow, n_docs)
    gw = DocumentService(world, corpus)
    everything = {d: world.documents[d].timeline.version for d in world.documents}
    unrestricted = [p.doc for _, p in corpus.search(QUERY, everything, 50)]
    rows = []
    for person in sorted(spec.people):
        allowed = {d for d, od in oracle.items() if od.access(person, find=True) is not None}
        top = list(dict.fromkeys(unrestricted))[:k]
        got = list(dict.fromkeys(h.doc for h in gw.search(person, QUERY, 50)))[:k]
        want_order = [d for d in dict.fromkeys(unrestricted) if d in allowed][:k]
        rows.append({"person": person, "visible_docs": len(allowed),
                     "yield": (sum(d in allowed for d in top) / len(top)) if top else None,
                     "fill": (len(got) / min(k, len(allowed))) if allowed else None,
                     "same_order": got == want_order, "same_top1": (got[:1] == want_order[:1]),
                     "overlap": (len(set(got) & set(want_order)) / len(set(want_order) | set(got))) if (got or want_order) else 1.0})
    def avg(key):
        v = [r[key] for r in rows if r[key] is not None]
        return round(sum(v) / len(v), 3) if v else None
    out = {"workflow": workflow, "documents": n_docs, "people": len(rows), "k": k, "authorized_yield_mean": avg("yield"), "fill_mean": avg("fill"),
           "same_order_share": round(sum(r["same_order"] for r in rows) / len(rows), 3), "same_top1_share": round(sum(r["same_top1"] for r in rows) / len(rows), 3),
           "overlap_mean": avg("overlap"), "rows": rows}
    out["overhead"] = overhead(world, corpus, everything, sorted(spec.people), gw)
    return out


def overhead(world, corpus, everything, people, gw, rounds: int = 20) -> dict:
    """Time and peak memory of a keyword search without the access filter (every document, no visibility) and with it (the service)."""
    def timed(f):
        ts = []
        for _ in range(rounds):
            t = time.perf_counter()
            f()
            ts.append((time.perf_counter() - t) * 1000)
        return round(statistics.median(ts), 4), round(sorted(ts)[int(0.95 * (len(ts) - 1))], 4)
    def peak(f):
        tracemalloc.start()
        f()
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return round(p / 1024, 1)
    plain = lambda: corpus.search(QUERY, everything, 5)
    filtered = lambda: [gw.search(p, QUERY, 5) for p in people]
    n = len(people)
    return {"without_filter_ms_median_p95": timed(plain), "with_filter_ms_median_p95_all_people": timed(filtered), "people": n,
            "peak_kib_without_filter": peak(plain), "peak_kib_with_filter_all_people": peak(filtered)}


def latency(run) -> dict:
    """Latency per channel (median, p95) of a finished grid run, on whichever repository it ran."""
    return run.summary()["latency_ms"]


def scripted_ask_seconds(workflow: str = "card1_policy_analysis_simple") -> float:
    """One question through the interface with a scripted model: the round trip of the harness itself, no model call."""
    from docflow.authentication_directory import AuthenticationDirectory
    from docflow.interface_api import Interface
    from docflow.model_client import ModelClient, Scripted, Settings, chat_response

    world, corpus, _, spec = build(workflow, 4)
    directory = AuthenticationDirectory(world)
    script = Scripted([chat_response(None, calls=[("search", {"query": QUERY})]), chat_response("Scripted reply: done.")])
    interface = Interface(directory, DocumentService(world, corpus), ModelClient(Settings(), script))
    t = time.perf_counter()
    interface.ask(directory.issue(spec.owner), "q")
    return time.perf_counter() - t


def main(argv: list[str]) -> None:
    import json
    args = list(argv)
    opts = {"--docs": "8", "--k": "5", "--json": None}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i : i + 2]
    r = measure(args[0], int(opts["--docs"]), int(opts["--k"]))
    r["scripted_ask_seconds"] = round(scripted_ask_seconds(args[0]), 4)
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    if opts["--json"]:
        open(opts["--json"], "w").write(json.dumps(r, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
