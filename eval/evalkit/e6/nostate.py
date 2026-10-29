"""E6, the arm without the state: `filtering_nostate` is `filtering` with the document's state hidden too (state null, guidance null); the documents, versions and system prompt are the same.

    python -m evalkit.e6.nostate --answers answers.jsonl.gz --out nostate_results.jsonl [--workers 300] [--cap 5] [--limit N]

It re-asks, under `filtering_nostate`, the questions of the pairs that `evalkit.e6.guidance select` chose (same question, reader and model as the `filtering` and `filtering_guidance` runs), all at the
same time in one process (the model service does not limit concurrency), and writes one record per run with the agent's answer, so the result file goes to `guidance judge` like the others. The tool
descriptions are plain (no mention of a state). A restart skips the runs already in the output file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docflow.document_service.service import DocumentService

from .. import pgstore
from . import client as C
from . import guidance as G
from . import runner as R

CONDITION = "filtering_nostate"
ROOT = Path(__file__).resolve().parents[3]


def runs_of(answers: list[dict], plan: list[dict]) -> list[dict]:
    """The `filtering` plan entries of the sampled pairs, as runs under `filtering_nostate`."""
    wanted = {a["pair"] for a in answers}
    out = []
    for r in plan:
        if r["condition"] == "filtering" and f'{r["qid"]}|{r["role"]}|{r["model"]}' in wanted:
            out.append({**r, "condition": CONDITION, "key": r["key"].replace("|filtering|", f"|{CONDITION}|")})
    return out


def main(argv: list[str]) -> int:
    kv = dict(zip(argv[::2], argv[1::2]))
    out_path, workers, cap = Path(kv["--out"]), int(kv.get("--workers", 300)), float(kv.get("--cap", 5))
    answers = G.load(Path(kv["--answers"]))
    runs = runs_of(answers, json.loads((ROOT / "eval" / "results" / "runs" / "e6" / "plan.json").read_text(encoding="utf-8")))
    if int(kv.get("--limit", 0)):
        runs = runs[: int(kv["--limit"])]
    done = {json.loads(l)["key"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in runs if r["key"] not in done]
    print(f"{len(runs)} runs planned, {len(done)} done, {len(todo)} to do, {workers} at once", flush=True)

    from .. import retrieval_utility as U
    U.setup()
    source = {wf: (w, U.G["syn"][1]) for wf, w in U.G["syn"][0].items()}
    source["regulation_single"] = (U.G["reg"][0], U.G["reg"][1])
    services = {(wf, CONDITION): DocumentService(R.CondWorld(world, CONDITION), pgstore.wrap(world, corpus, wf)) for wf, (world, corpus) in source.items()}
    calllog = C.CallLog(out_path.parent / "nostate_calls.jsonl")
    key = os.environ["OPENROUTER_API_KEY"]
    clients = {m: C.NoCacheClient(R.Settings(model=m), C.HttpTransport(key, calllog, provider=R.GLM_PROVIDERS if m.startswith("z-ai/") else None, pool=workers), calllog)
               for m in sorted({r["model"] for r in todo})}
    R.KEEP_ANSWERS = True
    lock, spent, finished, stop = threading.Lock(), [0.0], [0], threading.Event()

    def one(run: dict) -> None:
        if stop.is_set():
            return
        C.CTX.plain = True  # thread-local: the tool descriptions do not speak of a state
        try:
            rec = R.execute(run, clients, services)
        except BaseException as e:  # a failed run is reported and left for a restart
            print("FAILED", run["key"], repr(e)[:200], flush=True)
            return
        rec.pop("events", None), rec.pop("finished", None)
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            spent[0] += rec["cost"]
            finished[0] += 1
            if spent[0] >= cap:
                stop.set()
            if finished[0] % 25 == 0:
                print(f"finished {finished[0]}/{len(todo)} spent USD {spent[0]:.3f}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))
    print(f"done: {finished[0]} runs, USD {spent[0]:.3f}" + (" (spending cap reached)" if stop.is_set() else ""))
    return 1 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
