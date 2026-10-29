"""Overhead of the access decision. In one process, in memory, no model call.

    python -m evalkit.bench [--only decision,overhead,search,change] [--reps 5] [--pairs 300] [--out eval/results/runs/bench/bench.json]

  decision  the time of every request of the request grid (all eight workflow files, the same situations, people and channels as E1 and E2), after a warm-up pass of the whole
            grid, over `--reps` passes; median, 95th percentile and maximum in microseconds, and requests per second; by group of channels. The time of one request is the
            time of the call of the access service (or of the agent's tool) that the grid makes. The time of the engine's own decision (`World.visible` and
            `World.readable_version`) is measured on its own, in a separate pass.
  overhead  the same requests with the access check on and switched off, in one process, alternating, paired by request: the extra time the check adds
  search    keyword, meaning and hybrid search with the filter (the asker's visible set, whose computation is timed apart and added) and without it (every version of every
            document), per query, over the original questions of E5 as written, with the stored vectors; `--pairs` (question, asker) pairs, spread evenly over the set.
  change    the cost of a state change: for every accepted step of every path of the grid, the time of the action and of the next requests (open the document, search), and
            whether those next requests already agree with the oracle for every person class of the new situation.
"""

from __future__ import annotations

import array
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from docflow.access_control.enforcement import NotFound
from docflow.document_service import retrieve
from docflow.document_service.service import DocumentService
from docflow.stores.search_index import Corpus
from docflow.stores.world import World

from . import paths as paths_mod
from . import retrieval_utility as U
from .paths import WORKFLOWS
from .people import classify, representatives
from .run import LIMIT, QUERY_VECTOR, QUERY_WORDS, READ, Run, text_of, vector_of

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "direct: list, open, open version": ("list", "get", "get_version"),
    "direct: keyword, meaning, hybrid search": ("keyword", "semantic", "hybrid"),
    "agent tools: list, open, open version": ("tool_list", "tool_get", "tool_get_version"),
    "agent tools: search": ("tool_search",),
    "actions": ("action",),
}


def stats(xs: list[float], unit: float = 1.0) -> dict:
    xs = sorted(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": round(statistics.median(xs) / unit, 1), "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))] / unit, 1), "max": round(xs[-1] / unit, 1)}


def hardware() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "cores": os.cpu_count()}
    try:
        if sys.platform == "darwin":
            out["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            out["cores_physical"] = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True))
            out["ram_gb"] = round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30, 1)
        else:
            out["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
            out["ram_gb"] = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20, 1)
    except Exception as e:  # not known: say so, do not guess
        out["hardware_note"] = f"not read: {e}"
    return out


# ── the access decision over the whole grid ──────────────────────────────────────────────

def one_pass() -> list[dict]:
    records: list[dict] = []
    for wf in WORKFLOWS:
        records += Run(wf, http=False).run().records
    return records


def part_decision(reps: int) -> dict:
    one_pass()  # warm-up: imports, caches, the allocator
    by_group: dict[str, list[float]] = {g: [] for g in GROUPS}
    per_rep_total = []
    for _ in range(reps):
        recs = one_pass()
        per_rep_total.append(len(recs))
        for r in recs:
            for g, chans in GROUPS.items():
                if r["channel"] in chans:
                    by_group[g].append(r["ms"] * 1000)
    res = {"repetitions": reps, "requests_per_pass": per_rep_total[0], "groups": {g: stats(v) for g, v in by_group.items()}}
    reads = [x for g, v in by_group.items() if g != "actions" for x in v]
    everything = [x for v in by_group.values() for x in v]
    res["all reads"] = {**stats(reads), "requests_per_second": round(len(reads) / (sum(reads) / 1e6))}
    res["all requests"] = {**stats(everything), "requests_per_second": round(len(everything) / (sum(everything) / 1e6))}
    # the engine's own decision, timed where the access service asks for it
    calls: dict[str, list[float]] = {"visible": [], "readable_version": []}
    orig = {n: getattr(World, n) for n in calls}

    def timed(name):
        f = orig[name]

        def wrapper(self, *a, **k):
            t = time.perf_counter()
            out = f(self, *a, **k)
            calls[name].append(time.perf_counter() - t)
            return out
        return wrapper

    for n in calls:
        setattr(World, n, timed(n))
    try:
        one_pass()
    finally:
        for n, f in orig.items():
            setattr(World, n, f)
    res["engine_decision"] = {n: stats(v, 1e-6) for n, v in calls.items()}
    res["engine_decision_note"] = "one pass of the grid with the two engine calls wrapped in a timer; the timer itself adds about 0.2 microseconds to each call"
    return res


# ── the access check against no check ────────────────────────────────────────────────────

def one_mode_pass(check: bool) -> list[dict]:
    records: list[dict] = []
    for wf in WORKFLOWS:
        records += Run(wf, baseline=not check, http=False).run().records
    return records


def part_overhead(reps: int) -> dict:
    """The same requests of the grid with the access check on and switched off (`unrestricted`: every person is served the newest version of everything), in one process,
    alternating, after a warm-up of both. A request is paired with itself: the median and the 95th percentile of the extra time per request, and the two medians."""
    one_mode_pass(True)
    one_mode_pass(False)
    on: list[list[float]] = []
    off: list[list[float]] = []
    for rep in range(reps):
        order = (True, False) if rep % 2 == 0 else (False, True)  # which mode goes first alternates
        done = {}
        for check in order:
            done[check] = one_mode_pass(check)
        for check, sink in ((True, on), (False, off)):
            sink.append([r["ms"] * 1000 for r in done[check] if r["channel"] in READ])
    n = len(on[0])
    mean = lambda runs: [sum(run[i] for run in runs) / len(runs) for i in range(n)]  # a request's time: the mean of its repetitions
    t_on, t_off = mean(on), mean(off)
    extra = [a - b for a, b in zip(t_on, t_off)]
    return {"repetitions": reps, "read_requests": n, "with_check_us": stats(t_on), "check_off_us": stats(t_off), "extra_per_request_us": stats(extra),
            "ratio_of_medians": round(statistics.median(t_on) / statistics.median(t_off), 2),
            "requests_per_second": {"with_check": round(n / (sum(t_on) / 1e6)), "check_off": round(n / (sum(t_off) / 1e6))},
            "note": "each request's time is the mean over the repetitions; the extra time is paired by request; negative values are noise of the machine"}


# ── search with and without the filter ───────────────────────────────────────────────────

def part_search(pairs: int) -> dict:
    U.setup()
    scored = [(i, q) for i, q in enumerate(U.G["questions"]) if q["kind"] != "status"]
    stride = max(1, round(len(scored) * 2 / pairs))  # up to two askers a question; looking an asker up costs the visible set of every person, so only the sampled questions are looked up
    chosen, items = [], len(scored) * 2
    for i, q in scored[::stride]:
        for cat, user, world, corpus in U.askers(q):
            chosen.append((i, q, cat, user, world, corpus))
        if len(chosen) >= pairs:
            break
    chosen = chosen[:pairs]
    sizes = {"synthetic passages": len(U.G["syn"][1].passages), "regulation passages": len(U.G["reg"][1].passages)}
    acc: dict = {}

    def add(key, seconds):
        acc.setdefault(key, []).append(seconds)

    def hybrid(corpus, text, vec, scope):
        return retrieve.fuse(corpus.search(text, scope, retrieve.HYBRID_DEPTH), corpus.similar(vec, scope, retrieve.HYBRID_DEPTH), limit=10)

    warmed = set()
    visible_passages = []
    for n, (i, q, cat, user, world, corpus) in enumerate(chosen):
        if n % 25 == 0:
            print(f"  search: pair {n} of {len(chosen)}", flush=True)
        reg = q["workflow"] == "regulation_single"
        text, vec = q["text"], U.G["vec"][(q["id"], "question")]
        everything = U._everything(corpus)
        if id(corpus) not in warmed:  # the first call builds the matrix of vectors
            corpus.search(text, everything, 10)
            corpus.similar(vec, everything, 10)
            warmed.add(id(corpus))
        t = time.perf_counter()
        vis = world.visible(user, for_search=True)
        t_vis = time.perf_counter() - t
        visible_passages.append(sum(len(corpus.by_version.get(k, ())) for k in vis.items()))
        s = "regulation" if reg else "synthetic"
        add((s, "visible set", "filter"), t_vis)
        for mode, call in (("keyword", lambda sc: corpus.search(text, sc, 10)), ("semantic", lambda sc: corpus.similar(vec, sc, 10)), ("hybrid", lambda sc: hybrid(corpus, text, vec, sc))):
            t = time.perf_counter()
            call(vis)
            dt = time.perf_counter() - t
            add((s, mode, "filter, search only"), dt)
            add((s, mode, "filter, with the visible set"), dt + t_vis)
            t = time.perf_counter()
            call(everything)
            add((s, mode, "no filter"), time.perf_counter() - t)
    out = {"questions_as_written": True, "pairs": len(chosen), "of_about": items, "corpus": sizes, "store": "memory (in-process corpus); PostgreSQL was not used here",
           "visible_passages_median": statistics.median(visible_passages), "timings_ms": {}}
    for (s, mode, how), v in sorted(acc.items()):
        out["timings_ms"].setdefault(s, {}).setdefault(mode, {})[how] = stats(v, 1e-3)
    return out


# ── the cost of a state change ───────────────────────────────────────────────────────────

def part_change() -> dict:
    last: list[float] = []
    orig = {n: getattr(World, n) for n in ("act", "fire_event")}

    def timed(name):
        f = orig[name]
        depth = [0]

        def wrapper(self, *a, **k):
            depth[0] += 1
            t = time.perf_counter()
            try:
                return f(self, *a, **k)
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    last.append(time.perf_counter() - t)
        return wrapper

    for n in orig:
        setattr(World, n, timed(n))
    act_us, open_us, search_us, per_step_max = [], [], [], []
    checks = mismatches = steps = steps_new_version = 0
    try:
        for wf in WORKFLOWS:
            r = Run(wf, http=False)
            for path in r.ps.paths:
                odoc = r._oracle_doc(path.attrs)
                if not hasattr(odoc, "access"):
                    continue
                corpus = Corpus([], array.array("f"))
                seen_version = 0
                for sit, world in paths_mod.situations(r.ps, path, None):
                    new_version = sit.version > seen_version
                    n_before = len(corpus.passages)
                    for v in range(1, sit.version + 1):
                        if (r.spec.doc_id, v) not in corpus.by_version:
                            corpus.add(r.spec.doc_id, v, [text_of(r.spec.doc_id, v)], [vector_of(v)])
                    seen_version = sit.version
                    if sit.index == 0:
                        continue
                    step = path.steps[sit.index - 1]
                    odoc.apply(step.actor, step.action, step.input, step.event)
                    acted = last[-1]
                    steps += 1
                    steps_new_version += new_version and len(corpus.passages) > n_before
                    gw = DocumentService(world, corpus)
                    classes = classify(r.cast, odoc, r.terms)
                    worst = 0.0
                    for cls, person in representatives(classes).items():
                        serve, find = odoc.access(person), odoc.access(person, find=True)
                        t = time.perf_counter()
                        try:
                            got = gw.get_document(person, r.spec.doc_id)
                            got_open = (got["id"], got["version"])
                        except NotFound:
                            got_open = None
                        d_open = time.perf_counter() - t
                        t = time.perf_counter()
                        hits = {(h.doc, h.version) for h in gw.search(person, QUERY_WORDS, LIMIT)}
                        d_search = time.perf_counter() - t
                        checks += 2
                        mismatches += (got_open != ((r.spec.doc_id, serve) if serve is not None else None)) + (hits != ({(r.spec.doc_id, find)} if find is not None else set()))
                        open_us.append(d_open * 1e6)
                        search_us.append(d_search * 1e6)
                        worst = max(worst, d_open, d_search)
                    act_us.append(acted * 1e6)
                    per_step_max.append(worst * 1e6)
    finally:
        for n, f in orig.items():
            setattr(World, n, f)
    return {"accepted_steps": steps, "of_which_the_step_added_a_version": steps_new_version, "checks_after_a_step": checks, "disagreements_with_the_oracle": mismatches,
            "action_us": stats(act_us), "next_request_open_us": stats(open_us), "next_request_search_us": stats(search_us), "slowest_of_the_next_requests_per_step_us": stats(per_step_max),
            "index_writes_by_an_action": 0,
            "note": "the action call receives no index: it appends to the document's history. The first request after it is already answered from the new history for every person class."}


def main(argv: list[str]) -> int:
    args = dict(zip(argv[::2], argv[1::2]))
    only = set((args.get("--only") or "decision,overhead,search,change").split(","))
    res = {"hardware": hardware(), "model_calls": 0, "database_url_set": bool(os.environ.get("DATABASE_URL"))}
    t0 = time.time()
    for name, fn in (("decision", lambda: part_decision(int(args.get("--reps", 5)))), ("overhead", lambda: part_overhead(int(args.get("--reps", 5)))), ("search", lambda: part_search(int(args.get("--pairs", 300)))), ("change", part_change)):
        if name in only:
            t = time.time()
            res[name] = fn()
            res[name]["seconds"] = round(time.time() - t, 1)
            print(f"{name}: {res[name]['seconds']} s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(args.get("--out", ROOT / "eval" / "results" / "runs" / "bench" / "bench.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
