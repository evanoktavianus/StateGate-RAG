"""The random-path baseline of the mutation analysis (no model call).

The generated suite is built from the graph of the workflow (every edge, edge pair and option, the upload rule). Here, instead, for each seed: random walks from the
initial state, as many paths as the generated set and, path by path, as many steps as the generated path (a walk stops early only when no action is accepted from
where it is), the attribute set of each path drawn at random. The requests are issued as usual (one person per class, every channel). A mutant is killed by a seed if
its system answers any request differently from the oracle on the original policy. The mutants are the sample of the generated suite's analysis (`mut-<workflow>.json`);
a survivor judged equivalent there is not run again (it cannot be killed: same behaviour) and counts as not killed.

    python -m evalkit.random_paths <workflow> --mut eval/results/runs/e2/mut-<workflow>.json --out eval/results/runs/e2/random-paths-<workflow>.json [--seeds 30] [--first 1]   (seeds first..seeds; one process per seed: --first 8 --seeds 8 with an output file of its own)

The output is written after each seed, and a restart skips the seeds that are done.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

from . import mutate, paths as paths_mod
from .paths import Path_, PathSet, _run
from .replay import Unfireable
from .report import verdicts
from .run import Run


def random_pathset(template: PathSet, seed: int) -> tuple[PathSet, dict]:
    rng = random.Random(seed)
    spec, model = template.spec, template.model
    variants = spec.variants()
    ps = PathSet(template.workflow, model, spec)
    short = 0
    for n, tp in enumerate(template.paths, 1):
        want = len(tp.steps)
        attrs = rng.choice(variants)
        ops: list = []
        sim = _run(spec, model, attrs, ops)
        while len(sim.steps) < want:
            edges = list(model.edges_from(sim.state))
            rng.shuffle(edges)
            for e in edges:
                option = rng.choice(e.choose[0][1]) if e.choose else None
                try:
                    nxt = _run(spec, model, attrs, ops + [(e, option)])
                except Unfireable:
                    continue
                ops, sim = ops + [(e, option)], nxt
                break
            else:
                break  # nothing is accepted from here
        short += max(0, want - len(sim.steps))
        ps.paths.append(Path_(f"{template.workflow}#r{n:03d}", attrs, sim.steps, sim.firings, f"random walk, seed {seed}"))
    return ps, {"paths": len(ps.paths), "steps": sum(len(p.steps) for p in ps.paths), "steps_wanted": sum(len(p.steps) for p in template.paths), "steps_short": short}


def run(workflow: str, mut_file: Path, out: Path, seeds: int = 30, first: int = 1) -> dict:
    template = paths_mod.generate(workflow)
    mut = json.loads(Path(mut_file).read_text())
    texts = {m["hash"]: m["text"] for m in mutate.generate_mutants(template.spec.workflow_path)}
    v = verdicts().get(workflow, {})
    judged = [r["hash"] for r in mut["records"]]
    equivalent = {h for h in judged if v.get(h, {}).get("verdict") == "equivalent"}
    machines = {h: mutate.load_mutant(texts[h]) for h in judged if h not in equivalent}
    done = json.loads(Path(out).read_text()) if Path(out).exists() else {"workflow": workflow, "judged": judged, "equivalent": sorted(equivalent), "seeds": {}}
    for seed in range(first, seeds + 1):
        if str(seed) in done["seeds"]:
            continue
        t0 = time.time()
        ps, size = random_pathset(template, seed)
        killed = []
        for h, machine in machines.items():
            r = Run(workflow, http=False, ps=ps, machine=machine, fast=True).run()
            if r.step_findings or r.disagreements():
                killed.append(h)
        done["seeds"][str(seed)] = {"killed": killed, "seconds": round(time.time() - t0, 1), **size}
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(done))
        print(f"{workflow} seed {seed}: {len(killed)}/{len(judged)} killed, {size['steps']}/{size['steps_wanted']} steps, {done['seeds'][str(seed)]['seconds']} s", flush=True)
    return done


def main(argv: list[str]) -> None:
    args = dict(zip(argv[1::2], argv[2::2]))
    wf = argv[0]
    run(wf, Path(args["--mut"]), Path(args["--out"]), int(args.get("--seeds", 30)), int(args.get("--first", 1)))


if __name__ == "__main__":
    main(sys.argv[1:])
