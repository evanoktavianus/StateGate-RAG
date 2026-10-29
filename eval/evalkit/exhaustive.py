"""An exhaustive comparison of what each person may read, for a mutant and for the original policy, over the cast and every reachable situation. No model call.

    python -m evalkit.exhaustive <workflow> --verdicts eval/results/survivor_verdicts.json --mut eval/results/runs/e2/mut-<workflow>.json --out file.json

The reachable situations of a workflow are enumerated by breadth-first search with the engine driving a document (`Sim`): from the creation, with each attribute variant, every action
of the model (every option of every choice) that some person of the cast gets the system to accept, up to two uploads after the creation; two histories that leave the document with the same state, the
same version and the same version as of every state, the same choices, assignments and partial approvals are one situation (what a rule reads is exactly that). For each
situation the independent oracle (`eval/evalkit/oracle`) is run on the original policy and on the mutant policy over the same steps, and for every person of the cast the two versions
the policy lets them read (opened, found) are compared, as is whether the mutant policy accepts the steps at all. A mutant is *confirmed equivalent over the cast* if nothing differs.
The cast is the workflow's fixture; people outside it are what the graph check and the arguments cover.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from collections import deque
from pathlib import Path

from . import mutate, paths as paths_mod
from .oracle import Cast, Doc
from .oracle.policy import load_policy
from .replay import Sim, Unfireable
from .worlds import load_spec
from . import model as model_mod

UPLOAD_CAP = 3  # version entries, the creation included: at most two uploads


def key_of(sim: Sim) -> tuple:
    tl, model = sim.tl, sim.model
    given = sorted((e[1], e[2]) for e in tl.entries if e[0] == "given")
    return (tl.state, tl.version, tuple((s, tl.version_at(s)) for s in model.states), json.dumps(tl.choices(), sort_keys=True, default=str),
            json.dumps({k: list(v) for k, v in tl.assigned().items()}, sort_keys=True, default=str), tuple(given))


def fork(sim: Sim) -> Sim:
    new = copy.copy(sim)
    new.world = copy.deepcopy(sim.world)
    new.steps, new.firings = list(sim.steps), list(sim.firings)
    return new


def situations(workflow: str, limit: int = 3000) -> tuple[list[dict], bool]:
    """Every reachable situation: ({attrs, steps}, whether the search was cut at `limit` with situations still to expand)."""
    spec = load_spec(workflow)
    model = model_mod.from_machine(workflow, spec.machine())
    out, seen = [], set()
    for attrs in spec.variants():
        root = Sim(spec, model, attrs)
        queue = deque([root])
        seen.add((json.dumps(attrs, sort_keys=True), key_of(root)))
        out.append({"attrs": attrs, "steps": []})
        while queue and len(out) < limit:
            sim = queue.popleft()
            for e in model.edges_from(sim.state):
                for option in ([None] if not e.choose else list(e.choose[0][1])):
                    if e.action == "upload" and sum(1 for x in sim.tl.entries if x[0] == "version") >= UPLOAD_CAP:
                        continue
                    nxt = fork(sim)
                    try:
                        nxt.fire(e, option)
                    except Unfireable:
                        continue
                    k = (json.dumps(attrs, sort_keys=True), key_of(nxt))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append({"attrs": attrs, "steps": nxt.steps})
                    queue.append(nxt)
    return out, bool(queue)


def replay(policy, cast, spec, attrs, steps):
    ws = spec.workspaces[spec.workspace]
    assign = {k: v for k, v in spec.assign.items() if policy.create and k in (policy.create.get("assign") or {})}
    od = Doc.create(policy, cast, ws, ws.get("org"), spec.doc_id, spec.owner, spec.creator, assign or None, attrs or None, spec.invited)
    if not isinstance(od, Doc):  # the policy refuses to create the document at all
        return None
    for s in steps:
        if not od.apply(s.actor, s.action, s.input, s.event).allowed:
            return None
    return od


def vp(od, people) -> tuple:
    return tuple((p, od.access(p), od.access(p, find=True)) for p in people)


def compare(workflow: str, mutants: dict[str, str], limit: int = 3000) -> dict:
    """{hash: {"confirmed": bool, "difference": ...}} for the mutants {hash: yaml text}, and the number of situations."""
    spec = load_spec(workflow)
    cast = Cast.from_people(spec.people)
    people = sorted(spec.people)
    orig = load_policy(spec.workflow_path)
    sits, cut = situations(workflow, limit)
    base = []
    for s in sits:
        od = replay(orig, cast, spec, s["attrs"], s["steps"])
        base.append(vp(od, people) if od is not None else None)
    result = {}
    with tempfile.TemporaryDirectory() as d:
        for h, text in mutants.items():
            path = Path(d) / f"{h}.yml"
            path.write_text(text, encoding="utf-8")
            pol = load_policy(path)
            diff = None
            for i, s in enumerate(sits):
                od = replay(pol, cast, spec, s["attrs"], s["steps"])
                got = vp(od, people) if od is not None else None
                if got != base[i]:
                    diff = {"situation": i, "steps": [x.action for x in s["steps"]], "mutant_accepts_steps": od is not None,
                            "people_differing": [a[0] for a, b in zip(got or (), base[i] or ()) if a != b][:5]}
                    break
            result[h] = {"confirmed": diff is None, "difference": diff}
    return {"situations": len(sits), "search_cut_at_limit": cut, "people": len(people), "mutants": result}


def main(argv: list[str]) -> None:
    wf = argv[0]
    args = dict(zip(argv[1::2], argv[2::2]))
    verdicts = json.loads(Path(args["--verdicts"]).read_text()).get(wf, {})
    spec = load_spec(wf)
    texts = {m["hash"]: m["text"] for m in mutate.generate_mutants(spec.workflow_path) if "hash" in m}
    if "--killed" in args:  # a check of the method: the mutants the suite killed must differ somewhere
        mut = json.loads(Path(args["--killed"]).read_text())
        wanted = {r["hash"]: texts[r["hash"]] for r in mut["records"] if r["killed_by_suite"] and r["hash"] in texts}
        verdicts = {h: {"desc": r["op"] + ": " + r["desc"], "verdict": "killed"} for r in mut["records"] if (h := r["hash"]) in wanted}
    else:
        wanted = {h: texts[h] for h, v in verdicts.items() if v["verdict"] == "equivalent" and h in texts}
    res = compare(wf, wanted, int(args.get("--limit", 3000)))
    res["workflow"], res["claimed_equivalent"] = wf, len(wanted)
    res["confirmed"] = sum(r["confirmed"] for r in res["mutants"].values())
    for h, r in res["mutants"].items():
        r["desc"] = verdicts[h]["desc"]
    Path(args["--out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(args["--out"]).write_text(json.dumps(res, indent=1))
    what = "killed mutants that differ (the method's check)" if "--killed" in args else "equivalent mutants confirmed"
    print(f"{wf}: {res['situations']} situations{' (search cut at the limit)' if res['search_cut_at_limit'] else ' (all reachable ones)'}, {res['people']} people; " + (f"{res['claimed_equivalent'] - res['confirmed']} of {res['claimed_equivalent']} {what}" if "--killed" in args else f"{res['confirmed']} of {res['claimed_equivalent']} {what}"))


if __name__ == "__main__":
    main(sys.argv[1:])
