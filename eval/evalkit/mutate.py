"""Mutation analysis of a workflow policy.

The policy (the YAML) is mutated, never the engine: one mutant per application site of each operator. Each mutant is written to a temporary
directory and loaded; one that does not load is "invalid, discarded". The request grid (`evalkit.run`) is run against the mutant with the
oracle left on the ORIGINAL policy; a mutant is killed if any answer differs. The grid is run over every person of the cast (the pool); the
suite is the one-per-person-class grid, and a random baseline draws request sets of the suite's size from the pool, 30 times.

    python -m evalkit.mutate <workflow> --seed N [--limit K] [--cache dir] [--json file] [--md file]
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import yaml

from docflow import Machine
from docflow.stores.workflow_definitions import _Loader

from . import paths as paths_mod
from . import random_baseline
from .run import Run

REPS = 30
EXTRA_WHO = ("public", "everyone", "unit", "organization")
VALUE_TERMS = ("position", "group", "assigned", "role")


def key_of(r: dict) -> tuple:
    return (r["path"], r["index"], r["person"], r["channel"], str(r["target"]))


def differs(r: dict) -> bool:
    return (not r["agree"]) or bool(r.get("existence_leak")) or r.get("guidance_ok") is False


def _list(x):
    return list(x) if isinstance(x, list) else [x]


def _terms(who):
    """Every term of a who value, descending into `all`: (container list, index) pairs are not needed, so yields the dicts/strings with a path."""
    def walk(node, path):
        if isinstance(node, list):
            for i, t in enumerate(node):
                yield from walk(t, path + (i,))
        else:
            yield path, node
            if isinstance(node, dict) and "all" in node:
                yield from walk(node["all"], path + ("all",))
    yield from walk(who, ())


def _used(raw: dict) -> dict[str, list]:
    """The values of each value-term the workflow uses anywhere, for swaps."""
    out = defaultdict(set)
    def scan(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in VALUE_TERMS and isinstance(v, (str, list)):
                    out[k].update(_list(v))
                scan(v)
        elif isinstance(x, list):
            for v in x:
                scan(v)
    scan(raw.get("states"))
    scan(raw.get("create"))
    for st in (raw.get("states") or {}).values():
        for t in ((st or {}).get("on") or {}).values():
            for n in (t or {}).get("assign") or {}:
                out["assigned"].add(n)
    return {k: sorted(v) for k, v in out.items()}


def sites(raw: dict):
    """Yields (operator, description, mutate) where mutate(copy_of_raw) edits it in place."""
    states = raw.get("states") or {}
    names = list(states)
    used = _used(raw)
    attrs = {k: (v or {}).get("values") for k, v in (raw.get("attributes") or {}).items()}

    def rule_list(r, s):
        return r["states"][s].setdefault("access", [])

    for s, st in states.items():
        st = st or {}
        rules = st.get("access") or []
        for i, rule in enumerate(rules):
            tag = f"{s}.access[{i}]"
            who = _list(rule.get("who", []))
            # widen who
            for extra in EXTRA_WHO:
                if extra not in who:
                    yield "widen_who", f"{tag}: add {extra}", (lambda r, s=s, i=i, e=extra: r["states"][s]["access"][i].__setitem__("who", _list(r["states"][s]["access"][i]["who"]) + [e]))
            # narrow who
            if len(who) >= 2:
                for j, t in enumerate(who):
                    yield "narrow_who", f"{tag}: remove term {j} ({json.dumps(t)})", (lambda r, s=s, i=i, j=j: _del(r["states"][s]["access"][i], "who", j))
            # swap a value
            for path, t in _terms(rule.get("who", [])):
                if isinstance(t, dict):
                    for k in VALUE_TERMS:
                        if k in t and isinstance(t[k], str):
                            for other in used.get(k, []):
                                if other != t[k]:
                                    yield "swap", f"{tag}: {k} {t[k]} -> {other}", (lambda r, s=s, i=i, p=path, k=k, o=other: _node(r["states"][s]["access"][i]["who"], p).__setitem__(k, o))
                elif t in ("unit", "organization"):
                    other = "organization" if t == "unit" else "unit"
                    yield "swap", f"{tag}: {t} -> {other}", (lambda r, s=s, i=i, p=path, o=other: _put(r["states"][s]["access"][i], "who", p, o))
            # flip role
            if "role" in rule:
                for new in ("none", "viewer", "editor", "contributor"):
                    if new != rule["role"]:
                        yield "flip_role", f"{tag}: role {rule['role']} -> {new}", (lambda r, s=s, i=i, n=new: r["states"][s]["access"][i].__setitem__("role", n))
            # change version
            options = ["latest", "nothing"] + [{"from": n} for n in names]
            for field in ("serve", "find"):
                if field in rule:
                    for o in options:
                        if o != rule[field]:
                            yield "change_version", f"{tag}: {field} {json.dumps(rule[field])} -> {json.dumps(o)}", (lambda r, s=s, i=i, f=field, o=o: r["states"][s]["access"][i].__setitem__(f, copy.deepcopy(o)))
                elif field == "find":
                    for o in options:
                        if o != rule.get("serve"):
                            yield "change_version", f"{tag}: find (absent, = serve) -> {json.dumps(o)}", (lambda r, s=s, i=i, o=o: r["states"][s]["access"][i].__setitem__("find", copy.deepcopy(o)))
            # drop rule, condition
            yield "drop_rule", f"{tag}: delete", (lambda r, s=s, i=i: r["states"][s]["access"].pop(i))
            if "when" in rule:
                yield "condition", f"{tag}: remove when {json.dumps(rule['when'])}", (lambda r, s=s, i=i: r["states"][s]["access"][i].pop("when"))
                for name, val in rule["when"].items():
                    pool = attrs.get(name) or _choices(raw, name)
                    for other in pool or []:
                        if other != val:
                            yield "condition", f"{tag}: when {name}={val} -> {other}", (lambda r, s=s, i=i, n=name, o=other: r["states"][s]["access"][i]["when"].__setitem__(n, o))
        for i in range(len(rules) - 1):
            yield "reorder_rules", f"{s}.access: swap {i} and {i+1}", (lambda r, s=s, i=i: _swap(r["states"][s]["access"], i))
        # transitions
        on = st.get("on") or {}
        for a, t in on.items():
            tag = f"{s}.on.{a}"
            if not isinstance(t, dict):
                continue
            if "by" in t:
                yield "transition_guard", f"{tag}: remove by", (lambda r, s=s, a=a: r["states"][s]["on"][a].pop("by"))
                yield "transition_guard", f"{tag}: widen by to anyone", (lambda r, s=s, a=a: r["states"][s]["on"][a].__setitem__("by", "anyone"))
                by = _list(t["by"])
                if len(by) >= 2:
                    for j in range(len(by)):
                        yield "transition_guard", f"{tag}: narrow by, remove term {j}", (lambda r, s=s, a=a, j=j: _del(r["states"][s]["on"][a], "by", j))
                elif by != ["author"]:
                    yield "transition_guard", f"{tag}: narrow by to author", (lambda r, s=s, a=a: r["states"][s]["on"][a].__setitem__("by", "author"))
            if "requires" in t:
                yield "transition_guard", f"{tag}: remove requires", (lambda r, s=s, a=a: r["states"][s]["on"][a].pop("requires"))
            for n, rule in (t.get("assign") or {}).items():
                if isinstance(rule, dict) and "min" in rule:
                    yield "transition_guard", f"{tag}: remove assign.{n}.min", (lambda r, s=s, a=a, n=n: r["states"][s]["on"][a]["assign"][n].pop("min"))
            if "to" in t and "event" not in t:
                for o in names:
                    if o != t["to"]:
                        yield "wrong_target", f"{tag}: to {t['to']} -> {o}", (lambda r, s=s, a=a, o=o: r["states"][s]["on"][a].__setitem__("to", o))
        # upload
        if "upload" in on:
            yield "upload", f"{s}: remove upload", (lambda r, s=s: r["states"][s]["on"].pop("upload"))
        elif on is not None:
            yield "upload", f"{s}: add upload", (lambda r, s=s: r["states"][s].setdefault("on", {}).__setitem__("upload", {"to": s, "by": "author"}))
    yield "default_flip", "default_access", (lambda r: r.__setitem__("default_access", "own" if r.get("default_access") == "deny" else "deny"))


def _choices(raw: dict, name: str) -> list:
    for st in (raw.get("states") or {}).values():
        for t in ((st or {}).get("on") or {}).values():
            if isinstance(t, dict) and name in (t.get("choose") or {}):
                return list(t["choose"][name])
    return []


def _node(who, path):
    for p in path[:-1]:
        who = who[p] if not isinstance(who, dict) or p != "all" else who["all"]
    return who[path[-1]] if isinstance(who, list) else who


def _put(rule, field, path, value):
    who = rule[field]
    if not isinstance(who, list):
        rule[field] = value
        return
    node = who
    for p in path[:-1]:
        node = node[p] if p != "all" else node["all"]
    node[path[-1]] = value


def _del(d, field, j):
    v = _list(d[field])
    del v[j]
    d[field] = v


def _swap(lst, i):
    lst[i], lst[i + 1] = lst[i + 1], lst[i]


NOT_EXPRESSIBLE = ["negate: the language has no 'everyone except X' term, so the operator has no application site (skipped, stated here)"]


def dump(raw: dict) -> str:
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)


def load_raw(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)


def generate_mutants(workflow_path: Path) -> list[dict]:
    raw = load_raw(workflow_path)
    base = dump(raw)
    out, seen = [], {}
    for op, desc, fn in sites(raw):
        m = copy.deepcopy(raw)
        try:
            fn(m)
            text = dump(m)
        except Exception as e:  # a site that cannot be applied
            out.append({"op": op, "desc": desc, "status": "invalid", "why": f"site not applicable: {e}"})
            continue
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        rec = {"op": op, "desc": desc, "hash": h, "text": text}
        if text == base:
            rec["status"] = "equivalent"
            rec["why"] = "the mutated policy is identical to the original"
        elif h in seen:
            rec["status"] = "duplicate"
            rec["why"] = f"same policy as {seen[h]}"
        else:
            seen[h] = desc
            rec["status"] = "pending"
        out.append(rec)
    return out


def load_mutant(text: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mutant.yml"
        p.write_text(text, encoding="utf-8")
        return Machine.load(p)


def stratified_sample(work: list[dict], k: int, seed: int) -> list[dict]:
    """k mutants, spread over the operators in proportion to their number (largest remainder), but at least one from every operator while k allows; drawn with `seed`."""
    rng = random.Random(seed)
    by_op: dict[str, list[dict]] = defaultdict(list)
    for m in work:
        by_op[m["op"]].append(m)
    ops = sorted(by_op)
    quota = {o: len(by_op[o]) * k / len(work) for o in ops}
    take = {o: int(quota[o]) for o in ops}
    for o in sorted(ops, key=lambda o: (-(quota[o] - take[o]), o))[: k - sum(take.values())]:
        take[o] += 1
    if k >= len(ops):  # the floor of one: moved from the operators with the most surplus over their share
        for o in [o for o in ops if take[o] == 0]:
            donor = max((d for d in ops if take[d] > 1), key=lambda d: (take[d] - quota[d], d))
            take[donor] -= 1
            take[o] = 1
    out = []
    for o in ops:
        out += rng.sample(by_op[o], take[o])
    return out


def analyse(workflow: str, seed: int = 0, limit: int | None = None, cache: Path | None = None, progress=None) -> dict:
    t0 = time.time()
    base_run = Run(workflow, http=False).run()
    suite = {key_of(r) for r in base_run.records}
    ps = base_run.ps
    mutants = generate_mutants(base_run.spec.workflow_path)
    for m in mutants:  # a mutant the engine will not load is invalid; found before the sample is drawn, so the sample holds only mutants that can be judged
        if m["status"] == "pending":
            try:
                load_mutant(m["text"])
            except Exception as e:
                m.update(status="invalid", why=str(e)[:200])
    work = [m for m in mutants if m["status"] not in ("invalid", "equivalent", "duplicate")]
    if limit and len(work) > limit:
        work = stratified_sample(work, limit, seed)
    cache = Path(cache) if cache else None
    cfile = (cache / f"{workflow}.json") if cache else None
    done = json.loads(cfile.read_text()) if cfile and cfile.exists() else {}
    salt = hashlib.md5(json.dumps([p.to_json() for p in ps.paths], sort_keys=True, default=str).encode()).hexdigest()[:10]  # a verdict is for these paths: other paths, other verdicts
    for n, m in enumerate(work, 1):
        ck = f"{m['hash']}:{salt}"
        if ck in done:
            m.update(done[ck])
            continue
        try:
            machine = load_mutant(m["text"])
        except Exception as e:
            m.update(status="invalid", why=str(e)[:200])
            done[ck] = {k: m[k] for k in ("status", "why")}
            continue
        try:
            run = Run(workflow, http=False, ps=ps, machine=machine, everyone=True).run()
            diff = sorted({key_of(r) for r in run.records if differs(r)})
            steps = run.step_findings
            res = {"status": "killed" if (diff or steps) else "survived", "differing": [list(k) for k in diff], "step_findings": steps, "requests": len(run.records)}
        except Exception as e:
            res = {"status": "killed", "differing": [], "step_findings": [{"what": f"the run failed on the mutant: {type(e).__name__}: {e}"[:200]}], "requests": 0}
        m.update(res)
        done[ck] = res
        if progress:
            progress(n, len(work), m)
    if cfile:
        cfile.parent.mkdir(parents=True, exist_ok=True)
        cfile.write_text(json.dumps(done))
    # the pool of requests, for the baseline
    pool = {key_of(r) for r in Run(workflow, http=False, ps=ps, everyone=True).run().records}
    pool_list = sorted(pool)
    suite_keys = suite & pool
    judged = [m for m in work if m["status"] in ("killed", "survived")]
    killed_by_suite = {}
    differing = {}
    for m in judged:
        d = {tuple(k) for k in m.get("differing", [])}
        for f in m.get("step_findings", []):  # a situation the mutant cannot reach differs in every request from it onward
            if "path" in f:
                d |= {k for k in pool if k[0] == f["path"] and k[1] >= f["index"]}
        differing[m["hash"]] = d
        killed_by_suite[m["hash"]] = bool(d & suite_keys) or (m["status"] == "killed" and not d)
    size = len(suite_keys)
    reps = random_baseline.repetitions(pool_list, size, {m["hash"]: differing[m["hash"]] for m in judged}, seed, REPS)
    ops = defaultdict(lambda: {"generated": 0, "invalid": 0, "equivalent": 0, "judged": 0, "killed": 0, "killed_by_suite": 0, "survivors": []})
    for m in mutants:
        o = ops[m["op"]]
        o["generated"] += 1
        if m["status"] in ("invalid",):
            o["invalid"] += 1
        elif m["status"] in ("equivalent", "duplicate"):
            o["equivalent"] += 1
    for m in judged:
        o = ops[m["op"]]
        o["judged"] += 1
        o["killed"] += m["status"] == "killed"
        o["killed_by_suite"] += killed_by_suite[m["hash"]]
        if m["status"] == "survived":
            o["survivors"].append(m["desc"])
    for op, o in ops.items():
        hs = [m["hash"] for m in judged if m["op"] == op]
        scores = [sum(r[h] for h in hs) / len(hs) for r in reps] if hs else []
        o["score"] = round(o["killed_by_suite"] / o["judged"], 3) if o["judged"] else None
        o["baseline_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        o["baseline_range"] = [round(min(scores), 3), round(max(scores), 3)] if scores else None
    total = {"generated": len(mutants), "invalid": sum(o["invalid"] for o in ops.values()), "equivalent": sum(o["equivalent"] for o in ops.values()),
             "judged": len(judged), "killed_by_suite": sum(killed_by_suite.values())}
    total["score"] = round(total["killed_by_suite"] / total["judged"], 3) if judged else None
    allscores = [sum(r.values()) / len(r) for r in reps] if judged else []
    total["baseline_mean"] = round(sum(allscores) / len(allscores), 3) if allscores else None
    total["baseline_range"] = [round(min(allscores), 3), round(max(allscores), 3)] if allscores else None
    records = [{"hash": m["hash"], "op": m["op"], "desc": m["desc"], "status": m["status"], "killed_by_suite": killed_by_suite[m["hash"]]} for m in judged]
    return {"records": records, "baseline_reps": [[h for h, v in r.items() if v] for r in reps], "sample_size": len(work), "total_mutants": len(mutants),
            "workflow": workflow, "seed": seed, "limit": limit, "seconds": round(time.time() - t0, 1), "suite_requests": size, "pool_requests": len(pool_list),
            "by_operator": dict(ops), "total": total, "not_expressible": NOT_EXPRESSIBLE,
            "survivors": [{"op": m["op"], "desc": m["desc"], "hash": m["hash"]} for m in judged if m["status"] == "survived"],
            "mutants_judged": len(judged), "sampled": bool(limit)}


def to_markdown(r: dict) -> str:
    t = r["total"]
    out = [f"### {r['workflow']}" + (f" (a sample of {r['sample_size']} of {r['total_mutants']} mutants, seed {r['seed']})" if r["sampled"] else ""), "",
           f"seed {r['seed']}, limit {r['limit']}, {r['seconds']} s; suite {r['suite_requests']} requests (one person per class), pool {r['pool_requests']} requests (every person of the cast).", "",
           "| operator | generated | invalid | equivalent | judged | killed by the suite | mutation score | random baseline mean | baseline range |", "|---|---|---|---|---|---|---|---|---|"]
    for op, o in sorted(r["by_operator"].items()):
        out.append(f"| {op} | {o['generated']} | {o['invalid']} | {o['equivalent']} | {o['judged']} | {o['killed_by_suite']} | {o['score']} | {o['baseline_mean']} | {o['baseline_range']} |")
    out.append(f"| **all** | {t['generated']} | {t['invalid']} | {t['equivalent']} | {t['judged']} | {t['killed_by_suite']} | {t['score']} | {t['baseline_mean']} | {t['baseline_range']} |")
    out += ["", "Mutation score = killed / judged, where judged = valid mutants not identical to the original; a survivor is counted as non-equivalent until a person says otherwise (so the score is a lower bound). Equivalent here = the mutated YAML is identical to the original or to an earlier mutant.", "",
            "Survivors (survived the full pool; each needs a human look):", ""]
    out += [f"- {s['op']}: {s['desc']}" for s in r["survivors"]] or ["- none"]
    out += ["", "Not expressible: " + "; ".join(r["not_expressible"])]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    args = list(argv)
    opts = {"--seed": "0", "--limit": None, "--cache": None, "--json": None, "--md": None}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = args[i + 1]
            del args[i : i + 2]
    res = analyse(args[0], int(opts["--seed"]), int(opts["--limit"]) if opts["--limit"] else None, opts["--cache"],
                  progress=lambda n, N, m: print(f"  {n}/{N} {m['op']} {m['status']}", file=sys.stderr))
    text = to_markdown(res)
    print(text)
    if opts["--json"]:
        Path(opts["--json"]).write_text(json.dumps(res, indent=1))
    if opts["--md"]:
        Path(opts["--md"]).write_text(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
