"""Turns run logs into the report tables (Markdown and JSON).

Inputs, in one directory (`--logs`): `<workflow>.json` (a run log of `evalkit.run --log`), `<workflow>.baseline.json` (the same run with the filter
off, inside the harness), `<workflow>.regression.json` (the iteration-1 workflow run again on the current code) and `mut-<workflow>.json` (a
`evalkit.mutate --json` result). Missing files leave their cells empty. Every cell shows n. Zero failures are reported with the Wilson upper bound.

    python -m evalkit.report --logs work/smoke --md work/report-smoke.md --json work/report-smoke.json
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

ITERATION_1 = "card1_policy_analysis_simple"
ITERATION_2 = ["card2_policy_analysis_individual", "policy_analysis_team_single", "policy_analysis_other_unit_single", "card5_policy_analysis_sensitive"]
Z = 1.959964


def wilson_upper(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + Z * Z / n
    return min(1.0, (p + Z * Z / (2 * n) + Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / d)


def rate(k: int, n: int) -> str:
    """`k/n`, and for k = 0 the 95% Wilson upper bound, never a bare zero."""
    return f"{k}/{n}" + (f" (upper {wilson_upper(0, n):.2%})" if k == 0 and n else "")


def load(d: Path, name: str) -> dict | None:
    """A log of the directory; the repository keeps the large grid logs only as `.json.gz`."""
    f = d / name
    if f.exists():
        return json.loads(f.read_text())
    z = d / (name + ".gz")
    return json.loads(gzip.open(z, "rt", encoding="utf-8").read()) if z.exists() else None


def tally(rows: list[dict]) -> dict:
    return {"n": len(rows), "leak": sum(r["leak"] for r in rows), "denial": sum(r["denial"] for r in rows), "wrong_version": sum(r["wrong_version"] for r in rows),
            "existence_leak": sum(bool(r.get("existence_leak")) for r in rows), "existence_checked": sum(bool(r.get("existence_checked")) for r in rows),
            "guidance_checked": sum(r.get("guidance_ok") is not None for r in rows), "guidance_wrong": sum(r.get("guidance_ok") is False for r in rows)}


def by(rows: list[dict], key) -> dict[str, dict]:
    keys = sorted({key(r) for r in rows})
    return {k: tally([r for r in rows if key(r) == k]) for k in keys}


def cross_unit(rows: list[dict]) -> dict[str, dict]:
    return {"another unit of the same organization": tally([r for r in rows if "same-org" in r["class"] and "other-unit" in r["class"]]),
            "another organization": tally([r for r in rows if "other-org" in r["class"]])}


def slice_table(title: str, groups: dict[str, dict]) -> list[str]:
    out = [f"**{title}**", "", f"| {title.split(' ')[-1]} | requests (n) | wrongful allow | wrongful denial |", "|---|---|---|---|"]
    out += [f"| {k} | {t['n']} | {rate(t['leak'], t['n'])} | {rate(t['denial'], t['n'])} |" for k, t in groups.items()]
    return out + [""]


VERDICTS = Path(__file__).resolve().parents[2] / "eval" / "results" / "survivor_verdicts.json"


def verdicts() -> dict:
    """Survivors judged by a person: {workflow: {mutant hash: {"desc", "verdict": "equivalent" | "cast_limit" | "gap", "reason"}}}."""
    return json.loads(VERDICTS.read_text()) if VERDICTS.exists() else {}


def adjusted(mut: dict, wf: str) -> dict:
    """The mutation score with and without the survivors a person judged equivalent, and the random baseline computed the same way (same sample, same repetitions)."""
    v = verdicts().get(wf, {})
    recs = mut["records"]
    equiv = {r["hash"] for r in recs if not r["killed_by_suite"] and v.get(r["hash"], {}).get("verdict") == "equivalent"}
    kept = [r for r in recs if r["hash"] not in equiv]
    killed = sum(r["killed_by_suite"] for r in kept)
    base = [sum(h in set(rep) for h in (r["hash"] for r in kept)) / len(kept) for rep in mut["baseline_reps"]] if kept else []
    base_all = [sum(h in set(rep) for h in (r["hash"] for r in recs)) / len(recs) for rep in mut["baseline_reps"]] if recs else []
    return {"total_mutants": mut["total_mutants"], "sample": mut["sample_size"], "judged": len(recs), "killed": sum(r["killed_by_suite"] for r in recs),
            "equivalent": len(equiv), "judged_without_equivalent": len(kept),
            "score": round(sum(r["killed_by_suite"] for r in recs) / len(recs), 3) if recs else None,
            "score_without_equivalent": round(killed / len(kept), 3) if kept else None,
            "baseline_mean": round(sum(base_all) / len(base_all), 3) if base_all else None,
            "baseline_mean_without_equivalent": round(sum(base) / len(base), 3) if base else None,
            "baseline_range_without_equivalent": [round(min(base), 3), round(max(base), 3)] if base else None,
            "survivors": [{"hash": r["hash"], "op": r["op"], "desc": r["desc"], **{k: v.get(r["hash"], {}).get(k, "NO VERDICT" if k == "verdict" else "") for k in ("verdict", "reason")}} for r in recs if not r["killed_by_suite"]]}


def who_asks(cls: str) -> str:
    if "no-org" in cls:
        return "an outsider"
    if "other-org" in cls:
        return "another organization"
    if "other-unit" in cls:
        return "another unit of the same organization"
    if "owning-unit" in cls:
        return "the same unit"
    return "unclassified: " + cls


def by_who(rows: list[dict]) -> dict[str, dict]:
    return by(rows, lambda r: who_asks(r["class"]))


def coverage_line(wf: str) -> str:
    f = Path(__file__).resolve().parents[2] / "eval" / "results" / "paths.json"
    row = next((r for r in json.loads(f.read_text())["workflows"] if r["workflow"] == wf), None)
    if row is None:
        return ""
    st, ed, pr, op = row["states"], row["edges"], row["edge_pairs"], row["options"]
    return f"Coverage reached: states {st[0]}/{st[1]}, edges {ed[0]}/{ed[1]}, edge pairs {pr[0]}/{pr[1]}, options {op[0]}/{op[1]}; {row['paths']} paths, {row['situations']} situations."


def per_workflow_row(d: Path, wf: str) -> dict:
    log, mut = load(d, f"{wf}.json"), load(d, f"mut-{wf}.json")
    if log is None:
        return {"workflow": wf}
    recs, s = log["records"], log["summary"]
    t = tally(recs)
    paths = s.get("paths")
    row = {"workflow": wf, "situations": len({(r["path"], r["index"]) for r in recs}), "requests": t["n"], "wrongful_allow": t["leak"], "wrongful_denial": t["denial"]}
    if mut:
        tot = mut["total"]
        row.update(mutants=tot["judged"], killed=tot["killed_by_suite"], mutation_score=tot["score"], mutation_sampled=mut["sampled"])
    return row


def iteration1(d: Path) -> list[str]:
    log, base, reg = load(d, f"{ITERATION_1}.json"), load(d, f"{ITERATION_1}.baseline.json"), load(d, f"{ITERATION_1}.regression.json")
    if log is None:
        return ["(no run log for the simple variant)"]
    chans = by(log["records"], lambda r: r["channel"])
    bchans = by(base["records"], lambda r: r["channel"]) if base else {}
    out = ["| channel | requests (n) | wrongful allow | wrongful denial | wrong version | existence leaks | leaks with no access control (baseline) |", "|---|---|---|---|---|---|---|"]
    for c, t in chans.items():
        b = bchans.get(c)
        out.append(f"| {c} | {t['n']} | {rate(t['leak'], t['n'])} | {rate(t['denial'], t['n'])} | {t['wrong_version']}/{t['n']} | {t['existence_leak']}/{t['existence_checked']} | "
                   + (f"{b['leak']}/{b['n']}" if b else "") + " |")
    a = tally(log["records"])
    out.append(f"| **all** | {a['n']} | {rate(a['leak'], a['n'])} | {rate(a['denial'], a['n'])} | {a['wrong_version']}/{a['n']} | {a['existence_leak']}/{a['existence_checked']} | "
               + (f"{tally(base['records'])['leak']}/{tally(base['records'])['n']}" if base else "") + " |")
    out.append(f"| correct guidance (DR3) | {a['guidance_checked']} | | | | wrong: {a['guidance_wrong']}/{a['guidance_checked']} | |")
    out += ["", coverage_line(ITERATION_1)]
    mut = load(d, f"mut-{ITERATION_1}.json")
    if mut and "records" in mut:
        j = adjusted(mut, ITERATION_1)
        out += ["", f"Mutation (all {j['total_mutants']} mutants generated, {j['judged']} judged): {j['killed']}/{j['judged']} killed, score {j['score']}; random baseline mean {j['baseline_mean']}. "
                f"Without the {j['equivalent']} survivors judged equivalent: {j['killed']}/{j['judged_without_equivalent']} = {j['score_without_equivalent']}; random baseline mean {j['baseline_mean_without_equivalent']} (range {j['baseline_range_without_equivalent']})."]
    if reg:
        r = tally(reg["records"])
        out += ["", f"Regression (the iteration-1 run again on the current code): {r['n']} requests, wrongful allow {rate(r['leak'], r['n'])}, wrongful denial {rate(r['denial'], r['n'])}."]
    return out


def iteration2(d: Path) -> list[str]:
    out = []
    for wf in ITERATION_2:
        log = load(d, f"{wf}.json")
        if log is None:
            continue
        out += [f"#### {wf}", "", "| person | requests (n) | wrongful allow | wrongful denial |", "|---|---|---|---|"]
        out += [f"| {k} | {t['n']} | {rate(t['leak'], t['n'])} | {rate(t['denial'], t['n'])} |" for k, t in cross_unit(log["records"]).items()]
        out.append("")
    return out or ["(no run logs for the iteration-2 workflows)"]


def build(d: Path) -> tuple[str, dict]:
    workflows = [ITERATION_1, *ITERATION_2, "knowledge_asset", "card7_official_document", "regulation_single"]
    rows = [per_workflow_row(d, wf) for wf in workflows]
    out = ["## Per workflow", "", "| workflow | situations | requests (n) | mutants | killed | mutation score | wrongful allow | wrongful denial |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "requests" not in r:
            continue
        n = r["requests"]
        out.append(f"| {r['workflow']} | {r['situations']} | {n} | {r.get('mutants', '')} | {r.get('killed', '')} | {r.get('mutation_score', '')}{' (sampled)' if r.get('mutation_sampled') else ''} | {rate(r['wrongful_allow'], n)} | {rate(r['wrongful_denial'], n)} |")
    out += ["", "## Iteration 1: the simple variant", ""] + iteration1(d)
    out += ["", "## Iteration 2: the four Ministry workflows, across units", ""] + iteration2(d)
    log = load(d, f"{ITERATION_1}.json")
    if log:
        out += ["", "## Slices of the simple variant", ""]
        out += slice_table("per person class", by(log["records"], lambda r: r["class"]))
        out += slice_table("per state", by(log["records"], lambda r: r["state"]))
    return "\n".join(out), {"rows": rows}


def main(argv: list[str]) -> None:
    args = dict(zip(argv[::2], argv[1::2]))
    d = Path(args.get("--logs", "work/smoke"))
    if "--e2" in args:  # the per-workflow table and the splits of E3, from the run logs of E2
        t8, data = table_viii(d)
        text = "## Request grid and mutation analysis, per workflow\n\n" + "\n".join(t8) + "\n\n## E3: errors split by who asks\n\n" + "\n".join(e3(d))
        Path(args["--md"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--md"]).write_text("# E2 and E3 (built from run logs)\n\nSource logs: `" + str(d) + "`. Cells show n; zero failures carry the 95% Wilson upper bound.\n\n" + text + "\n")
        Path(args["--json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--json"]).write_text(json.dumps(data, indent=1))
        print(text)
        return
    text, data = build(d)
    head = "# Report (built from run logs)\n\nSource logs: `" + str(d) + "`. Cells show n; zero failures carry the 95% Wilson upper bound.\n\n"
    if "--md" in args:
        Path(args["--md"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--md"]).write_text(head + text + "\n")
    if "--json" in args:
        Path(args["--json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--json"]).write_text(json.dumps(data, indent=1))
    print(head + text)



def table_viii(d: Path) -> tuple[list[str], dict]:
    """E2: one row per workflow. Columns: situations, requests, wrongful allow, wrongful denial (each with its upper bound at zero), wrong version, existence leaks,
    wrong guidance, wrongful allows with no access control, mutants (sample of total), judged, killed, score, score without equivalent survivors, random baseline
    (both ways), coverage (states, edges, edge pairs)."""
    wfs = [ITERATION_1, *ITERATION_2, "knowledge_asset", "card7_official_document", "regulation_single"]
    head = ["workflow", "situations", "requests (n)", "wrongful allow", "wrongful denial", "wrong version", "existence leaks", "wrong guidance", "wrongful allows, no access control",
            "mutants: sample / total", "judged", "killed", "score", "score without equivalent", "random baseline", "random baseline without equivalent", "states", "edges", "edge pairs"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    data, surv = {}, []
    for wf in wfs:
        log, base, mut = load(d, f"{wf}.json"), load(d, f"{wf}.baseline.json"), load(d, f"mut-{wf}.json")
        if log is None:
            continue
        t = tally(log["records"])
        bl = tally(base["records"]) if base else None
        cov = next((r for r in json.loads((Path(__file__).resolve().parents[2] / "eval" / "results" / "paths.json").read_text())["workflows"] if r["workflow"] == wf), None)
        j = adjusted(mut, wf) if mut and "records" in mut else None
        cells = [wf, str(len({(r["path"], r["index"]) for r in log["records"]})), str(t["n"]), rate(t["leak"], t["n"]), rate(t["denial"], t["n"]), f"{t['wrong_version']}/{t['n']}",
                 f"{t['existence_leak']}/{t['existence_checked']}", f"{t['guidance_wrong']}/{t['guidance_checked']}", f"{bl['leak']}/{bl['n']}" if bl else ""]
        cells += ([f"{j['sample']} / {j['total_mutants']}", str(j["judged"]), str(j["killed"]), str(j["score"]), f"{j['killed']}/{j['judged_without_equivalent']} = {j['score_without_equivalent']}",
                   str(j["baseline_mean"]), str(j["baseline_mean_without_equivalent"])] if j else [""] * 7)
        cells += ([f"{cov['states'][0]}/{cov['states'][1]}", f"{cov['edges'][0]}/{cov['edges'][1]}", f"{cov['edge_pairs'][0]}/{cov['edge_pairs'][1]}"] if cov else [""] * 3)
        out.append("| " + " | ".join(cells) + " |")
        data[wf] = {"requests": t, "baseline_leaks": bl["leak"] if bl else None, "mutation": j}
        if j:
            surv += [f"- **{wf}** {s['op']}: {s['desc']} — {s['verdict']}: {s['reason']}" for s in j["survivors"]]
    rows = [v for v in data.values()]
    tot = {k: sum(v["requests"][k] for v in rows) for k in ("n", "leak", "denial", "wrong_version", "existence_leak", "existence_checked", "guidance_wrong", "guidance_checked")}
    judged = sum(v["mutation"]["judged"] for v in rows if v["mutation"])
    killed = sum(v["mutation"]["killed"] for v in rows if v["mutation"])
    allows = sum(v["baseline_leaks"] or 0 for v in rows)
    out.append("| All workflows | | " + " | ".join([str(tot["n"]), rate(tot["leak"], tot["n"]), rate(tot["denial"], tot["n"]), f"{tot['wrong_version']}/{tot['n']}",
                f"{tot['existence_leak']}/{tot['existence_checked']}", f"{tot['guidance_wrong']}/{tot['guidance_checked']}", f"{allows}/{tot['n']}", "", str(judged), str(killed)]) + " | | | | | | | |")
    data["all"] = {"requests": tot, "baseline_leaks": allows, "mutants_judged": judged, "mutants_killed": killed}
    reg, first = load(d, f"{ITERATION_1}.regression.json"), load(d, f"{ITERATION_1}.json")
    if reg and first:
        r, f = tally(reg["records"]), tally(first["records"])
        key = lambda x: (x["path"], x["index"], x["person"], x["channel"], str(x["target"]))
        a, b = {key(x): (x["want"], x["got"]) for x in first["records"]}, {key(x): (x["want"], x["got"]) for x in reg["records"]}
        out += ["", f"Regression: the iteration-1 workflow ({ITERATION_1}) run again on the current code: {r['n']} requests, wrongful allow {rate(r['leak'], r['n'])}, wrongful denial {rate(r['denial'], r['n'])}, "
                f"wrong version {r['wrong_version']}/{r['n']}, existence leaks {r['existence_leak']}/{r['existence_checked']}, wrong guidance {r['guidance_wrong']}/{r['guidance_checked']}; "
                f"its records are {'identical to' if a == b else 'DIFFERENT from'} the E1 records (same {len(a)} requests, same wanted and returned results)."]
        data["card1_regression"] = {"requests": r, "identical_to_e1": a == b}
    return out + ["", "### Every survivor, with its reason", ""] + (surv or ["- none"]), data


def e3(d: Path) -> list[str]:
    """E3, from the grid logs: the two errors split by who asks, per workflow and over all workflows."""
    wfs = [ITERATION_1, *ITERATION_2, "knowledge_asset", "card7_official_document", "regulation_single"]
    allrows, out = [], []
    for wf in wfs:
        log = load(d, f"{wf}.json")
        if log is None:
            continue
        allrows += log["records"]
        out += [f"#### {wf}", "", "| who asks | requests (n) | wrongful allow | wrongful denial |", "|---|---|---|---|"]
        out += [f"| {k} | {t['n']} | {rate(t['leak'], t['n'])} | {rate(t['denial'], t['n'])} |" for k, t in by_who(log["records"]).items()]
        out.append("")
    out += ["#### all workflows together", "", "| who asks | requests (n) | wrongful allow | wrongful denial |", "|---|---|---|---|"]
    out += [f"| {k} | {t['n']} | {rate(t['leak'], t['n'])} | {rate(t['denial'], t['n'])} |" for k, t in by_who(allrows).items()]
    return out

if __name__ == "__main__":
    main(sys.argv[1:])
