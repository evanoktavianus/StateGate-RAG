"""E7: runs the prompts with the runner of E6 (500 concurrent, retries with a wait, stops as in E6), on this machine.

    python -m evalkit.e7.run --out runs/e7 [--scripted] [--limit N] [--cap 5] [--only workflow,...]

A run's record in `results.jsonl` holds the tool calls with their arguments and the documents and versions that the tools returned, with the condition, the cost and the stop reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..e6 import runner as R
from . import plan as P
from .world import Shared


def main(argv: list[str]) -> None:
    flags = {"--scripted"}
    args = {a: True for a in argv if a in flags}
    kv = dict(zip([a for a in argv if a not in flags][::2], [a for a in argv if a not in flags][1::2]))
    S = Shared(only=kv["--only"].split(",") if "--only" in kv else None)
    runs = P.build([p for p in S.prompts if p["workflow"] in S.worlds], S.alt_ids)
    if int(kv.get("--limit", 0)):
        runs = runs[: int(kv["--limit"])]
    cfg = {"out": kv.get("--out", "runs/e7"), "scripted": bool(args.get("--scripted")), "workers": int(kv.get("--workers", 4)), "threads": int(kv.get("--threads", 128)),
           "cap": float(kv.get("--cap", 5)), "start": int(kv.get("--start", 500)), "max": int(kv.get("--max", 500)), "hold": True, "latency": float(kv.get("--latency", 0)),
           "fail_rate": float(kv.get("--fail-rate", 0)), "models": sorted({r["model"] for r in runs}), "results": "results.jsonl", "module": "evalkit.e7.exec", "ramp_seconds": 60}
    summary = R.Runner(runs, Path(cfg["out"]), cfg, lambda: S).run()
    print(json.dumps({k: v for k, v in summary.items() if k != "target_history"}, indent=1))
    sys.exit(1 if summary["stop"] else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
