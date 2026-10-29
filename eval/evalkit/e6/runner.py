"""The E6 runner: executes the runs of a plan with many agents at a time, writing each result to disk as it finishes.

    python -m evalkit.e6.runner --plan runs/e6/plan.json --out runs/e6 [--sample] [--scripted] [--workers 24] [--threads 24] [--cap 10] [--start 300] [--max 500]

Processes: the parent loads the worlds and corpora once (`retrieval_utility.setup`), then forks the workers, which share those pages; the parent keeps the plan, hands
out runs and collects results. A run is executed in a worker thread: the agent loop of the engine (read only), over the tools of the engine's tool controller, bound to the
asker, over a service whose world is wrapped for the condition. The engine is not changed.

Concurrency: starts at `--start` runs at a time; every minute it goes up by 100 (to `--max`) while, over the last minute, fewer than 2% of the attempts were refused (429, 402)
and the machine's CPU stayed under 85%, and goes down by 100 when either fails.

Refusals and errors of the service are retried with a wait (up to 8 attempts); the concurrency is not lowered (`--hold`: it stays at `--start`).
Stops at once (no new run is started; runs in flight are given 60 s to end and are kept if they do) and writes `STOP.json` with the cause when:
  more than 10% of the runs finished in 5 minutes (50 at least) failed after the retries (`model_error`);
  a harness error occurs, or a worker dies, or free memory or free disk is under 2 GB;
  the spending reaches the cap.
What the tools returned to each run is checked against the oracle afterwards, by `evalkit.e6.context_leaks`.
A restart skips the runs already in `results.jsonl`.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from docflow.document_service.service import DocumentService
from docflow.agent_harness import agent_loop, tool_controller
from docflow.model_client import Settings

from .. import pgstore
from . import client as C

STEP_LIMIT = agent_loop.DEFAULT_STEP_LIMIT
GLM_PROVIDERS = {"order": ["Open Inference", "Relace", "Wafer", "Sail Research", "Novita"], "allow_fallbacks": True}


class CondWorld:
    """The world as one condition shows it. `no_access_control`: every person is served the newest version of everything; the other two: the world as it is.
    Guidance is left out of every tool result except under `filtering_guidance`. Everything else is the world's own."""

    def __init__(self, world, condition: str) -> None:
        self._w, self.condition = world, condition

    def __getattr__(self, name):
        return getattr(self._w, name)

    def readable_version(self, doc_id, user, *, for_search=False):
        if self.condition == "no_access_control":
            return self._w.documents[doc_id].timeline.version
        return self._w.readable_version(doc_id, user, for_search=for_search)

    def visible(self, user, *, for_search=True):
        if self.condition == "no_access_control":
            return {d: doc.timeline.version for d, doc in self._w.documents.items()}
        return self._w.visible(user, for_search=for_search)

    def guidance(self, doc_id, user=None):
        return self._w.guidance(doc_id, user) if self.condition == "filtering_guidance" else None

    def state_of(self, doc_id):
        return None if self.condition == "filtering_nostate" else self._w.state_of(doc_id)  # `filtering_nostate`: filtering as it is, with neither the state nor the guidance shown


def compact(log: dict) -> list[dict]:
    """The trace of a run for the results file: what was called, with what, and which documents came back (text cut to 160 characters)."""
    steps = []
    for s in log["steps"]:
        results = []
        for r in s["tool_results"]:
            res = r["result"]
            if "hits" in res:
                res = {"hits": [{"id": h["id"], "version": h["version"], "state": h["state"], "guidance": h["guidance"], "text": h["passage"]["text"][:160]} for h in res["hits"]]}
            elif "passages" in res:
                res = {k: res[k] for k in ("id", "version", "state", "guidance")} | {"text": " ".join(p["text"] for p in res["passages"])[:160]}
            elif "documents" in res:
                res = {"documents": [(d["id"], d["version"], d["state"]) for d in res["documents"][:5]], "n": len(res["documents"]), "truncated": res.get("truncated")}
            results.append({"name": r["name"], "arguments": r["arguments"], "result": res})
        steps.append({"step": s["step"], "usage": s.get("usage"), "error": s.get("error"), "tool_results": results,
                      "tool_calls": len((s["reply"] or {}).get("tool_calls") or [])})
    return steps


KEEP_ANSWERS = False  # `--keep-answers`: the record also holds the final answer of the agent (for `evalkit.e6.guidance`)


def execute(run: dict, clients: dict, services: dict) -> dict:
    C.CTX.key, C.CTX.step, C.CTX.events = run["key"], 0, []
    t0 = time.time()
    gw = services[(run["workflow"] if not run["regulation"] else "regulation_single", run["condition"])]
    tools = tool_controller.tools(gw, run["user"])
    result = agent_loop.run(clients[run["model"]], tools, run["question"], who=run["user"], step_limit=STEP_LIMIT)
    steps = compact(result.log)
    usage = [s["usage"] or {} for s in steps]
    cost = sum((u.get("cost_usd") or 0.0) for u in usage)
    record = {"key": run["key"], "qid": run["qid"], "kind": run["kind"], "workflow": run["workflow"], "regulation": run["regulation"], "condition": run["condition"], "role": run["role"],
            "model": run["model"], "user": run["user"], "category": run["category"], "stop_reason": result.stop_reason, "steps": steps,
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage), "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "calls": len(usage), "cost": cost, "seconds": round(time.time() - t0, 2), "events": list(C.CTX.events), "finished": time.time()}
    if KEEP_ANSWERS:
        record["answer"] = result.answer
    return record


def worker_main(in_q, out_q, cfg: dict, source) -> None:
    global KEEP_ANSWERS
    KEEP_ANSWERS = bool(cfg.get("keep_answers"))
    calllog = C.CallLog(Path(cfg["out"]) / "calls" / f"calls-{os.getpid()}.jsonl")
    if cfg["scripted"]:
        transport = C.ScriptedTransport(cfg.get("latency", 0.0), cfg.get("fail_rate", 0.0))
    else:
        transport = C.HttpTransport(os.environ["OPENROUTER_API_KEY"], calllog)
    clients = {}
    for m in cfg["models"]:
        tr = transport
        if not cfg["scripted"] and m.startswith("z-ai/"):
            tr = C.HttpTransport(os.environ["OPENROUTER_API_KEY"], calllog, provider=GLM_PROVIDERS)
        qwen = m.startswith("qwen/")  # reasoning left out of the reply, room for 8000 tokens
        clients[m] = C.NoCacheClient(Settings(model=m, max_tokens=8000) if qwen else Settings(model=m), tr, calllog, exclude_reasoning=qwen)
    mod = importlib.import_module(cfg["module"]) if cfg.get("module") else None  # another experiment (E7) brings its own services and its own `execute`
    if mod:
        services, execute_fn = mod.make_services(source()), mod.execute
    else:
        services, execute_fn = {}, execute
        for wf, (world, corpus) in source().items():
            index = pgstore.wrap(world, corpus, wf)  # PostgreSQL when DATABASE_URL is set (loaded on first use), else the in-memory corpus
            for cond in ("no_access_control", "filtering", "filtering_guidance"):
                services[(wf, cond)] = DocumentService(CondWorld(world, cond), index)

    def loop():
        while True:
            run = in_q.get()
            if run is None:
                return
            try:
                out_q.put(execute_fn(run, clients, services))
            except BaseException:
                out_q.put({"key": run["key"], "harness_error": traceback.format_exc()})

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(cfg["threads"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def cpu_busy(prev: tuple) -> tuple[float, tuple]:
    """The share of the processor in use since `prev`; 0.0 where there is no /proc (macOS): the run is then paced by its worker count alone."""
    if not Path("/proc/stat").exists():
        return 0.0, prev
    f = open("/proc/stat").readline().split()[1:]
    vals = list(map(int, f))
    idle, total = vals[3] + vals[4], sum(vals)
    d_total, d_idle = total - prev[1], idle - prev[0]
    return (1 - d_idle / d_total if d_total else 0.0), (idle, total)


def mem_available_gb() -> float:
    if not Path("/proc/meminfo").exists():
        return 99.0  # not known (macOS): no memory stop
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return 99.0


class Runner:
    def __init__(self, plan: list[dict], out: Path, cfg: dict, source) -> None:
        self.plan, self.out, self.cfg, self.source = plan, Path(out), cfg, source
        self.results_path = self.out / cfg.get("results", "results.jsonl")
        self.out.mkdir(parents=True, exist_ok=True)
        self.done = {json.loads(l)["key"] for l in open(self.results_path)} if self.results_path.exists() else set()
        self.todo = [r for r in plan if r["key"] not in self.done]
        self.total = len(plan)
        self.events: list[tuple] = []  # (t, outcome, attempts, refused_attempts, refused)
        self.spent = sum(json.loads(l).get("cost", 0.0) for l in open(self.results_path)) if self.results_path.exists() else 0.0
        self.finished = len(self.done)
        self.spent_new = 0.0
        self.inflight = 0
        self.target = cfg["start"]
        self.history: list[dict] = []
        self.run_outcomes: list[tuple] = []
        self.stop: dict | None = None
        self.cpu_samples: list[tuple] = []
        self.last_ramp = time.time()

    # ── stop rules ──
    def window_rate(self, seconds: float, now: float) -> tuple[float, int]:
        ev = [e for e in self.events if e[0] >= now - seconds]
        bad = sum(1 for e in ev if e[1] != "ok" or e[4])
        return (bad / len(ev) if ev else 0.0), len(ev)

    def check(self, now: float, cpu: float) -> None:
        recent = [bad for t, bad in self.run_outcomes if t >= now - 300]
        if len(recent) >= 50 and sum(recent) / len(recent) > 0.10:
            self.stop = {"cause": "run_failures", "detail": f"{sum(recent) / len(recent):.1%} of the {len(recent)} runs finished in the last 5 minutes failed after the retries (the model service did not answer)"}
        if self.spent >= self.cfg["cap"]:
            self.stop = {"cause": "spending", "detail": f"spent {self.spent:.2f} USD, the cap is {self.cfg['cap']}"}
        if mem_available_gb() < 2:
            self.stop = {"cause": "memory", "detail": f"{mem_available_gb():.1f} GB available"}
        if shutil.disk_usage(self.out).free < 2e9:
            self.stop = {"cause": "disk", "detail": f"{shutil.disk_usage(self.out).free / 1e9:.1f} GB free"}

    def ramp(self, now: float, cpu: float) -> None:
        if self.cfg.get("hold") or now - self.last_ramp < self.cfg.get("ramp_seconds", 60):
            return
        self.last_ramp = now
        ev = [e for e in self.events if e[0] >= now - 60]
        attempts = sum(e[2] for e in ev)
        refused = sum(e[3] for e in ev)
        rate = refused / attempts if attempts else 0.0
        old = self.target
        if rate >= 0.02 or cpu >= 0.85:
            self.target = max(100, self.target - 100)
        elif self.target < self.cfg["max"]:
            self.target = min(self.cfg["max"], self.target + 100)
        self.history.append({"t": round(now, 1), "target": self.target, "was": old, "refused_rate": round(rate, 4), "cpu": round(cpu, 3), "inflight": self.inflight, "finished": self.finished, "spent": round(self.spent, 4)})

    def accept(self, res: dict, f) -> None:
        self.inflight -= 1
        if "harness_error" in res:
            self.stop = self.stop or {"cause": "harness_error", "detail": res["harness_error"][-1500:], "run": res["key"]}
            return
        events, finished = res.pop("events"), res.pop("finished")  # what the runner paces by; not part of the record
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()
        self.finished += 1
        self.spent += res["cost"]
        self.spent_new += res["cost"]
        self.events += [(e["t"], e["outcome"], e["attempts"], e["refused_attempts"], e["refused"]) for e in events]
        self.events = [e for e in self.events if e[0] > time.time() - 600]
        self.run_outcomes.append((finished, res["stop_reason"] == "model_error"))
        self.run_outcomes = [o for o in self.run_outcomes if o[0] > time.time() - 900]

    # ── the loop ──
    def run(self) -> dict:
        ctx = mp.get_context("fork")
        in_q, out_q = ctx.Queue(), ctx.Queue()
        procs = [ctx.Process(target=worker_main, args=(in_q, out_q, self.cfg, self.source), daemon=True) for _ in range(self.cfg["workers"])]
        for p in procs:
            p.start()
        t_start = time.time()
        prev = cpu_busy((0, 0))[1]
        cpu_list: list[float] = []
        last_mon = 0.0
        deadline = None
        with open(self.results_path, "a", encoding="utf-8") as f:
            while True:
                now = time.time()
                if self.stop is None:
                    while self.inflight < self.target and self.todo:
                        in_q.put(self.todo.pop(0))
                        self.inflight += 1
                try:
                    while True:
                        self.accept(out_q.get(timeout=0.05), f)
                except queue.Empty:
                    pass
                if now - last_mon >= self.cfg.get("monitor_seconds", 5):
                    last_mon = now
                    cpu, prev = cpu_busy(prev)
                    cpu_list = (cpu_list + [cpu])[-12:]
                    avg = sum(cpu_list) / len(cpu_list)
                    if self.stop is None:
                        self.check(now, avg)
                        self.ramp(now, avg)
                    if any(not p.is_alive() for p in procs) and self.stop is None and (self.inflight or self.todo):
                        self.stop = {"cause": "worker_died", "detail": "a worker process ended"}
                    print(f"[{now - t_start:6.0f}s] finished {self.finished}/{self.total} inflight {self.inflight} target {self.target} cpu {avg:.0%} spent {self.spent:.3f}", flush=True)
                if self.stop is not None and deadline is None:
                    deadline = now + 60
                if (self.stop is None and not self.todo and self.inflight == 0) or (deadline and (self.inflight == 0 or now > deadline)):
                    break
        for _ in procs:
            in_q.put(None)
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        summary = {"finished": self.finished, "total": self.total, "spent": round(self.spent, 4), "seconds": round(time.time() - t_start, 1), "target_history": self.history,
                   "stop": self.stop}
        (self.out / ("STOP.json" if self.stop else "DONE.json")).write_text(json.dumps(summary, indent=1))
        return summary


def sample_plan(plan: list[dict], per_condition: int = 30) -> list[dict]:
    """The first 30 questions (in the plan's order) of the stratified 200, with all their runs, both models."""
    strat = set(json.loads((Path(__file__).resolve().parents[3] / "eval/data/questions" / "stratified200.json").read_text()))
    ids = []
    for r in plan:
        if r["qid"] in strat and r["qid"] not in ids:
            ids.append(r["qid"])
        if len(ids) == per_condition:
            break
    return [r for r in plan if r["qid"] in set(ids)]


def main(argv: list[str]) -> None:
    flags = {"--sample", "--scripted", "--hold", "--keep-answers"}
    args = {a: True for a in argv if a in flags}
    kv = dict(zip([a for a in argv if a not in flags][::2], [a for a in argv if a not in flags][1::2]))
    plan = json.loads(Path(kv["--plan"]).read_text())
    if args.get("--sample"):
        plan = sample_plan(plan)
    if int(kv.get("--limit", 0)):
        plan = plan[: int(kv["--limit"])]
    cfg = {"out": kv.get("--out", "runs/e6"), "scripted": bool(args.get("--scripted")), "workers": int(kv.get("--workers", 24)), "threads": int(kv.get("--threads", 24)),
           "cap": float(kv.get("--cap", 10)), "start": int(kv.get("--start", 300)), "max": int(kv.get("--max", 500)), "latency": float(kv.get("--latency", 0)),
           "fail_rate": float(kv.get("--fail-rate", 0)), "hold": "--hold" in argv, "keep_answers": "--keep-answers" in argv, "ramp_seconds": float(kv.get("--ramp-seconds", 60)), "limit": int(kv.get("--limit", 0)), "models": sorted({r["model"] for r in plan}), "results": "results-sample.jsonl" if args.get("--sample") else "results.jsonl"}
    from .. import retrieval_utility as U
    U.setup()

    def source():
        out = {wf: (w, U.G["syn"][1]) for wf, w in U.G["syn"][0].items()}
        out["regulation_single"] = (U.G["reg"][0], U.G["reg"][1])
        return out

    summary = Runner(plan, Path(cfg["out"]), cfg, source).run()
    print(json.dumps({k: v for k, v in summary.items() if k != "target_history"}, indent=1))
    sys.exit(1 if summary["stop"] else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
