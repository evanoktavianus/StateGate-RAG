"""Reproduce the experiments and compare each with the stored result. Choose what to run with `--only`.

    python -m evalkit.reproduce                         # everything without a model, plus a small E6 and E7 sample if OPENROUTER_API_KEY is set
    python -m evalkit.reproduce --show                 # first: the stored results of the paper as tables (runs nothing)
    python -m evalkit.reproduce [--only grid,mutation,probes,e5,e6-tools] [--out work/reproduced] [--quick] [--workers 7]
    python -m evalkit.reproduce --only e6 [--e6-questions 5] [--scripted]     # agent runs with a model (cost: about USD 0.002 a run), a few only
    python -m evalkit.reproduce --only e7 [--e7-runs 30] [--scripted]         # prompt injection with a model, a few only

No model, no cost (the default, `--only grid,probes,mutation,e5,e6-tools`):
    grid       E1 and E2 request grids (with and without the access control)   stored: eval/results/runs/e1, e2
    mutation   mutation analysis, seed 1 (60 mutants a workflow, all of card1)  stored: eval/results/runs/e1, e2 (mut-*.json)
    probes     E4 property checks P1 to P8                                       stored: eval/results/runs/e4
    e5         retrieval utility, stored vectors                                 stored: eval/results/runs/e5/utility.json
    e6-tools   every tool call of the stored E6 runs, run again on this code     stored: eval/results/runs/e6/results.jsonl
With a model (run by default when OPENROUTER_API_KEY is set or `--scripted` is given, else skipped with a note; `--only e6` names it; sample sizes `--e6-questions` 5 and `--e7-runs` 10):
    e6         a sample of the E6 runs (`--e6-questions` questions, each under no_access_control, filtering and filtering_guidance); the answers differ from
               run to run, what is checked is what the tools returned: no document or version the independent oracle does not let the asker read may be
               returned under filtering or filtering_guidance (stored: 0 of 1,911 runs); under no_access_control some are, as stored.
    e7         the first `--e7-runs` runs of the injection experiment; the same check on what the tools returned.

Everything is written to `--out`; `comparison.json` there and the last lines printed say, per experiment, how many items were compared and
how many differ. `--quick` uses card1 only, the first 150 questions and 400 runs (a few minutes). Run from the repository root with PYTHONPATH=src:eval.
"""

from __future__ import annotations

import gzip
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .paths import WORKFLOWS

ROOT = Path(__file__).resolve().parents[2]
STORED = ROOT / "eval" / "results" / "runs"
FIRST = WORKFLOWS[0]


def stored(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    z = path.with_name(path.name + ".gz")
    return json.loads(gzip.open(z, "rt", encoding="utf-8").read()) if z.exists() else None


def home(wf: str) -> Path:
    return STORED / ("e1" if wf == FIRST else "e2")


def sh(*args: str) -> None:
    out = subprocess.run([sys.executable, "-W", "ignore", "-m", *args], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(["src", "eval", os.environ.get("PYTHONPATH", "")])})
    if out.returncode:
        raise RuntimeError(f"{' '.join(args)}: {out.stderr[-400:]}")


LINES: list[str] = []


def table(title: str, header: list[str], rows: list[list]) -> None:
    """A small table, printed and kept for `results.txt`."""
    cells = [header] + [[str(c) for c in r] for r in rows]
    width = [max(len(r[i]) for r in cells) for i in range(len(header))]
    text = [f"\n{title}"] + ["  " + "  ".join(c.ljust(w) for c, w in zip(r, width)) for r in cells]
    LINES.extend(text)
    print("\n".join(text))


def compare_grid(out: Path, wfs: list[str]) -> dict:
    n = bad = 0
    rows = []
    for wf in wfs:
        same = {}
        for suffix in ("", ".baseline"):
            a, b = json.loads((out / f"{wf}{suffix}.json").read_text())["summary"], stored(home(wf) / f"{wf}{suffix}.json")["summary"]
            same[suffix] = a["all"]
            ok = True
            for k in ("requests", "all", "by_channel", "by_state", "by_class", "step_findings"):
                n += 1
                if a[k] != b[k]:
                    bad += 1
                    ok = False
            same[suffix + "ok"] = ok
        f, bl = same[""], same[".baseline"]
        rows.append([wf, f["n"], f["wrongful_allow"], f["wrongful_denial"], f["wrong_version"], f"{f['existence_leaks']}/{f['existence_checked']}",
                     f"{f['guidance_wrong']}/{f['guidance_checked']}", bl["wrongful_allow"], "yes" if same["ok"] and same[".baselineok"] else "NO"])
    table("E1, E2: request grid against the independent oracle (the last two columns: wrongful allows with the filter disabled; same as stored)",
          ["workflow", "requests", "wrongful allow", "wrongful denial", "wrong version", "existence leaks", "guidance wrong", "filter off: allows", "same"], rows)
    return {"compared": n, "differences": bad}


def compare_mutation(out: Path, wfs: list[str]) -> dict:
    n = bad = 0
    rows = []
    for wf in wfs:
        a = {r["hash"]: (r["status"], r["killed_by_suite"]) for r in json.loads((out / f"mut-{wf}.json").read_text())["records"]}
        b = {r["hash"]: (r["status"], r["killed_by_suite"]) for r in stored(home(wf) / f"mut-{wf}.json")["records"]}
        d = [h for h in a.keys() | b.keys() if a.get(h) != b.get(h)]
        n += len(b)
        bad += len(d)
        rows.append([wf, len(a), sum(v[0] == "killed" for v in a.values()), sum(v[0] == "survived" for v in a.values()), "yes" if not d else f"NO ({len(d)})"])
    table("E2: mutation analysis (seed 1)", ["workflow", "mutants", "killed", "survived", "same"], rows)
    return {"compared": n, "differences": bad}


def compare_probes(out: Path, wfs: list[str]) -> dict:
    """Status and counts of every probe; the wording of the details and a probe that needs a database (not run without `DATABASE_URL`) are left out."""
    n = bad = 0
    rows = []
    for wf in wfs:
        a, b = json.loads((out / f"probes-{wf}.json").read_text()), stored(STORED / "e4" / f"probes-{wf}.json")
        d = 0
        for x, y in zip(a, b):
            if x.get("status") == "not_run":
                continue
            n += 1
            if (x.get("status"), x.get("counts")) != (y.get("status"), y.get("counts")):
                bad += 1
                d += 1
        rows.append([wf, sum(x["status"] == "pass" for x in a), sum(x["status"] == "fail" for x in a), sum(x["status"] == "not_run" for x in a), "yes" if not d else f"NO ({d})"])
    table("E4: property checks P1 to P8 (not run: the one that needs a database, when DATABASE_URL is not set)", ["workflow", "pass", "fail", "not run", "same"], rows)
    return {"compared": n, "differences": bad}


def compare_e5(out: Path, limit: int | None) -> dict:
    new, old = json.loads((out / "e5.json").read_text())["rows"], stored(STORED / "e5" / "utility.json")["rows"]
    key = lambda r: (r["id"], r["asker"], r["form"], r["role"], r["mode"])
    ids = {r["id"] for r in new}
    A, B = {key(r): r for r in new}, {key(r): r for r in old if r["id"] in ids}
    rows = []
    for mode in ("keyword", "semantic", "hybrid"):
        rs = [r for r in new if r["mode"] == mode and r["form"] == "original" and r.get("evidence_scored")]
        rows.append([mode, len(rs), f"{sum(r['doc_recall@5'] for r in rs) / max(len(rs), 1):.3f}", f"{sum(r['evidence_recall@5'] for r in rs) / max(len(rs), 1):.3f}"])
    table("E5: retrieval utility, original questions, as the asker sees them", ["mode", "rows", "document recall@5", "evidence recall@5"], rows)
    diff = sum(1 for k in A.keys() | B.keys() if A.get(k) != B.get(k))
    return {"compared": len(B), "differences": diff}


def e6_tools(limit: int | None) -> dict:
    """Every tool call of the stored E6 runs, run again; the result in the compact form the results file keeps, compared with the stored one."""
    from docflow.agent_harness import tool_controller
    from docflow.document_service.service import DocumentService

    from . import retrieval_utility as U
    from .e6 import runner as R

    U.setup()
    src = {wf: (w, U.G["syn"][1]) for wf, w in U.G["syn"][0].items()}
    src["regulation_single"] = (U.G["reg"][0], U.G["reg"][1])
    services = {(wf, c): DocumentService(R.CondWorld(w, c), corpus) for wf, (w, corpus) in src.items() for c in ("no_access_control", "filtering", "filtering_guidance")}
    n = bad = 0
    for line in itertools.islice(open(STORED / "e6" / "results.jsonl"), limit):
        run = json.loads(line)
        svc = services.get(("regulation_single" if run["regulation"] else run["workflow"], run["condition"]))
        if svc is None:
            continue
        tools = tool_controller.tools(svc, run["user"])
        for step in run["steps"]:
            for t in step["tool_results"]:
                got = R.compact({"steps": [{"step": 0, "usage": None, "error": None, "reply": None, "tool_results": [{"name": t["name"], "arguments": t["arguments"],
                                 "result": tool_controller.call(tools, t["name"], t["arguments"])}]}]})[0]["tool_results"][0]["result"]
                n += 1
                bad += json.loads(json.dumps(got)) != json.loads(json.dumps(t["result"]))
    return {"compared": n, "differences": bad}


CORE = ("no_access_control", "filtering", "filtering_guidance")


def sample_plan(plan: list[dict], questions: int) -> list[dict]:
    """`questions` questions of the E6 plan on the main model, each under the three core conditions: two thirds as non-readers (where a leak could show), the rest
    as readers, spread evenly over the plan."""
    model = next(r["model"] for r in plan if r["model"].startswith("z-ai/"))
    groups: dict = {}
    for r in plan:
        if r["model"] == model and r["condition"] in CORE:
            groups.setdefault((r["qid"], r["role"]), {})[r["condition"]] = r
    full = [g for g in groups.values() if len(g) == 3]
    pick = lambda pool, k: [pool[i * len(pool) // max(k, 1)] for i in range(k)] if pool else []
    n_non = -(-questions * 2 // 3)
    chosen = pick([g for g in full if g["filtering"]["role"] == "non_reader"], n_non) + pick([g for g in full if g["filtering"]["role"] == "reader"], questions - n_non)
    keys = {r["key"] for g in chosen for r in g.values()}
    return [r for r in plan if r["key"] in keys]


def needs_key(scripted: bool) -> None:
    if not scripted and not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("this experiment calls a model: set OPENROUTER_API_KEY in the environment (the key is never written anywhere), or add --scripted to use a fake model")


def e6_sample(out: Path, questions: int, scripted: bool) -> dict:
    needs_key(scripted)
    plan = sample_plan(json.loads((STORED / "e6" / "plan.json").read_text()), questions)
    work = out / "e6"
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps(plan))
    sh("evalkit.e6.runner", "--plan", str(work / "plan.json"), "--out", str(work), "--cap", "0.5", "--workers", "4", "--threads", "4", *(("--scripted",) if scripted else ()))
    sh("evalkit.e6.context_leaks", "--results", str(work / "results.jsonl"), "--out", str(work))
    runs = stored(work / "context_leaks.json")["runs"]
    by = {c: [r for r in runs if r["condition"] == c] for c in CORE}
    table("E6 sample: runs that were returned a document or version the independent oracle does not let the asker read (filtering: must be 0)",
          ["condition", "runs", "runs with such a leak"], [[c, len(rs), sum(bool(r["context_leaks"]) for r in rs)] for c, rs in by.items()])
    return {"compared": sum(len(by[c]) for c in CORE[1:]), "differences": sum(bool(r["context_leaks"]) for c in CORE[1:] for r in by[c])}


def e7_sample(out: Path, runs: int, scripted: bool) -> dict:
    needs_key(scripted)
    work = out / "e7"
    sh("evalkit.e7.run", "--out", str(work), "--cap", "0.5", "--limit", str(runs), "--workers", "2", "--threads", "8", *(("--scripted",) if scripted else ()))
    sh("evalkit.e7.score", str(work / "results.jsonl"), str(work))
    scores = stored(work / "scores.json")
    by = {c: [s for s in scores if s["condition"] == c] for c in {s["condition"] for s in scores}}
    table("E7 sample: prompts whose tools returned a document or version the oracle does not let the asker read (filtering_guidance: must be 0)",
          ["condition", "prompts", "prompts with such a leak"], [[c, len(ss), sum(bool(x["any_context_leak"]) for x in ss)] for c, ss in sorted(by.items())])
    filtered = [s for c, ss in by.items() if c != "no_access_control" for s in ss]
    return {"compared": len(filtered), "differences": sum(bool(s["any_context_leak"]) for s in filtered)}


def show(out: Path) -> int:
    """The stored results as the same tables, without running anything: the files of the paper are copied under the names the comparison reads, and compared with themselves."""
    stage = out / "stored"
    stage.mkdir(parents=True, exist_ok=True)

    def put(name: str, path: Path) -> None:
        data = stored(path)
        if data is not None:
            (stage / name).write_text(json.dumps(data))

    for wf in WORKFLOWS:
        put(f"{wf}.json", home(wf) / f"{wf}.json")
        put(f"{wf}.baseline.json", home(wf) / f"{wf}.baseline.json")
        put(f"mut-{wf}.json", home(wf) / f"mut-{wf}.json")
        put(f"probes-{wf}.json", STORED / "e4" / f"probes-{wf}.json")
    put("e5.json", STORED / "e5" / "utility.json")
    print("The stored results of the paper (nothing is run; `same` is the file compared with itself).")
    compare_grid(stage, WORKFLOWS)
    compare_mutation(stage, WORKFLOWS)
    compare_probes(stage, WORKFLOWS)
    compare_e5(stage, None)
    leaks = stored(STORED / "e6" / "context_leaks.json")
    if leaks:
        runs = leaks["runs"]
        table("E6 (stored, all runs): runs that were returned a document or version the independent oracle does not let the asker read",
              ["model", "condition", "runs", "runs with such a leak"],
              [[m, c, len(rs), sum(bool(r["context_leaks"]) for r in rs)] for m in sorted({r["model"] for r in runs}) for c in ("no_access_control", "filtering", "filtering_guidance")
               for rs in [[r for r in runs if r["model"] == m and r["condition"] == c]] if rs])
    scores = stored(STORED / "e7" / "scores.json")
    if scores:
        table("E7 (stored): prompts whose tools returned a document or version the oracle does not let the asker read",
              ["condition", "prompts", "prompts with such a leak"],
              [[c, len(ss), sum(bool(x["any_context_leak"]) for x in ss)] for c in sorted({x["condition"] for x in scores}) for ss in [[x for x in scores if x["condition"] == c]]])
    where = [
        ("E1, E2 grid", "runs/e1|e2/<workflow>.json[.gz]", "one record per request: who asked, by which channel, what the oracle says, what the system returned"),
        ("E1, E2 grid, filter off", "runs/e1|e2/<workflow>.baseline.json[.gz]", "the same requests with the access filter disabled"),
        ("E2 mutation", "runs/e1|e2/mut-<workflow>.json", "one record per mutant: the change made, killed or survived"),
        ("E4 properties", "runs/e4/probes-<workflow>.json", "one record per probe: status and counts"),
        ("E5 retrieval", "runs/e5/utility.json", "one row per question, asker, wording and search mode"),
        ("E6 agent runs", "runs/e6/results.jsonl", "one line per run: the user, the condition, every tool call with the documents it returned"),
        ("E6 leak check", "runs/e6/context_leaks.json.gz", "one record per run: each document or version returned that the oracle does not allow"),
        ("E7 injection", "runs/e7/results.jsonl.gz", "one line per run: every tool call and the documents the tools returned"),
        ("E7 scores", "runs/e7/scores.json.gz", "one record per prompt: target called, target returned, any document outside the visible set"),
    ]
    rows = []
    for name, path, what in where:
        first = path.split(" ")[0].rstrip(",").rstrip("/").replace("|e2", "").replace("<workflow>", FIRST).replace("[.gz]", "")
        base = STORED / first.replace("runs/", "", 1)
        found = base.exists() or base.with_name(base.name + ".gz").exists()
        rows.append([name, "eval/results/" + path, "yes" if found else ("not fetched yet" if "fetch" in path else "MISSING"), what])
    table("Where the raw records behind the tables are (all under the repository)", ["experiment", "file", "there", "one record is"], rows)
    (out / "stored_results.txt").write_text("\n".join(LINES) + "\n")
    print(f"\nwritten to {out / 'stored_results.txt'}")
    return 0


NO_MODEL = "grid,mutation,probes,e5,e6-tools"


def chosen(only: str | None, scripted: bool) -> set[str]:
    """What to run. Named with `--only`: exactly that. Otherwise everything that needs no model and, if a key is set (or `--scripted`), a small sample of E6 and E7 as well."""
    if only:
        return set(only.split(","))
    names = set(NO_MODEL.split(","))
    if os.environ.get("OPENROUTER_API_KEY") or scripted:
        names |= {"e6", "e7"}
    else:
        print("E6 and E7 (agent with a model) are skipped: no OPENROUTER_API_KEY. Set it to add a small sample (15 + 10 runs, a few cents), or add --scripted for a fake model.")
    return names


NEEDS = {
    "grid": ["chunks"], "mutation": ["chunks"], "probes": [], "e5": ["chunks", "vectors"], "e6-tools": ["chunks", "vectors"], "e6": ["chunks", "vectors"], "e7": ["chunks", "vectors"],
}
VECTORS = ("eval/data/documents/embeddings.f32", "eval/data/regulations/embeddings.f32")
CHUNKS = "eval/data/documents/chunks.jsonl"


def preflight(only: set[str]) -> list[str]:
    """What is missing for the experiments asked for, each with the command that fixes it; and which store the run will use."""
    needed = {n for e in only for n in NEEDS.get(e, [])}
    missing = []
    if "vectors" in needed:
        absent = [v for v in VECTORS if not (ROOT / v).exists()]
        if absent:
            missing.append(f"the vector files {', '.join(absent)}:  python3 scripts/join_vectors.py")
    if "chunks" in needed and not (ROOT / CHUNKS).exists():
        missing.append(f"{CHUNKS}:  gunzip -k {CHUNKS}.gz")
    store = "PostgreSQL (DATABASE_URL is set)" if os.environ.get("DATABASE_URL") and os.environ.get("DOCFLOW_STORE") != "memory" else "memory (set DATABASE_URL to use PostgreSQL, as in the paper)"
    print(f"store: {store}")
    return missing


def main(argv: list[str]) -> int:
    flags = ("--quick", "--scripted", "--show")
    args = dict(zip([a for a in argv if a not in flags][::2], [a for a in argv if a not in flags][1::2]))
    quick, scripted = "--quick" in argv, "--scripted" in argv
    if "--show" in argv:
        return show(Path(args.get("--out", "work/reproduced")) if Path(args.get("--out", "work/reproduced")).is_absolute() else ROOT / args.get("--out", "work/reproduced"))
    out = Path(args.get("--out", "work/reproduced"))
    out = out if out.is_absolute() else ROOT / out
    only = chosen(args.get("--only"), scripted)
    workers = int(args.get("--workers", 7))
    wfs = [FIRST] if quick else list(WORKFLOWS)
    missing = preflight(only)
    if missing:
        print("Not ready. Missing:\n  " + "\n  ".join(missing))
        return 2
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    t0 = time.time()

    def pool(jobs):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda j: sh(*j), jobs))

    if "grid" in only:
        pool([j for wf in wfs for j in ((f"evalkit.run", wf, "--brief", "--log", str(out / f"{wf}.json")), ("evalkit.run", wf, "--baseline", "--brief", "--log", str(out / f"{wf}.baseline.json")))])
        results["grid"] = compare_grid(out, wfs)
    if "probes" in only:
        pool([("evalkit.probes", wf, "--json", str(out / f"probes-{wf}.json")) for wf in wfs])
        results["probes"] = compare_probes(out, wfs)
    if "mutation" in only:
        pool([("evalkit.mutate", wf, "--seed", "1", *(() if wf == FIRST else ("--limit", "60")), "--json", str(out / f"mut-{wf}.json")) for wf in wfs])
        results["mutation"] = compare_mutation(out, wfs)
    limit = 150 if quick else None
    if "e5" in only:
        sh("evalkit.retrieval_utility", "--json", str(out / "e5.json"), *(("--limit", str(limit)) if limit else ()))
        results["e5"] = compare_e5(out, limit)
    if "e6-tools" in only:
        results["e6-tools"] = e6_tools(400 if quick else None)
        table("E6: every tool call of the stored runs, run again", ["tool calls run again", "differ from the stored result"], [[results["e6-tools"]["compared"], results["e6-tools"]["differences"]]])
    if "e6" in only:
        results["e6"] = e6_sample(out, int(args.get("--e6-questions", 5)), scripted)
    if "e7" in only:
        results["e7"] = e7_sample(out, int(args.get("--e7-runs", 10)), scripted)
    results["seconds"] = round(time.time() - t0)
    (out / "comparison.json").write_text(json.dumps(results, indent=1))
    (out / "results.txt").write_text("\n".join(LINES) + "\n")
    differences = sum(r["differences"] for r in results.values() if isinstance(r, dict))
    print("REPRODUCED: every result is identical to the stored one" if not differences else f"DIFFERENCES: {differences} (see above)")
    if "e6" in only or "e7" in only:
        print("(for e6 and e7 the count is the context leaks under filtering, which must be 0)")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
