"""E6, does the agent keep to the guidance of a document's state? A model judges the answers.

Guidance is the sentence a workflow state attaches to a document ("Draft regulation, not in force. Do not state it as a rule."). Under `filtering_guidance` the tools return it
with the document; under `filtering` they do not. Filtering decides what the agent sees, guidance limits what it may say about it. This module compares the two on the
same questions and readers.

    python -m evalkit.e6.guidance select --results runs.jsonl --out answers.jsonl.gz [--per-kind 60]
    python -m evalkit.e6.guidance arms   --answers answers.jsonl.gz --nostate nostate_results.jsonl --out answers3.jsonl.gz
    python -m evalkit.e6.guidance judge  --answers answers.jsonl.gz --out verdicts.jsonl.gz --model <id> [--limit N] [--cap USD] [--workers N] [--max-tokens N] [--scripted]
    python -m evalkit.e6.guidance table  --answers answers.jsonl.gz --verdicts verdicts.jsonl.gz
    python -m evalkit.e6.guidance report --answers answers3.jsonl.gz --verdicts judge_a.jsonl.gz [--verdicts2 judge_b.jsonl.gz] [--results runs.jsonl --nostate nostate_results.jsonl] [--out report.md]

`select` reads a results file whose runs carry the agent's final `answer` (the runner keeps it with `--keep-answers`). It takes the runs under `filtering_guidance` in which a tool
returned a document with guidance, spread evenly over the five kinds of guidance, and for each the run of the same question and reader under `filtering` (the control).
`judge` asks the judge model, per run, for every document carrying guidance: does the answer draw on that document, and, if so, does it keep to the guidance. The judge sees the
question, the answer, the document id, the state and the guidance text, never the condition. `table` prints, per condition and kind, the runs that draw on a guided document and
the share that break its guidance, with 95% Wilson intervals. The judge call is cached on disk (`--cache`), so a repeat of the same call costs nothing.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from docflow import model_client as mc

ROOT = Path(__file__).resolve().parents[3]
QUESTIONS = ROOT / "eval" / "data" / "questions" / "questions.json"
CONDITIONS = ("filtering_nostate", "filtering", "filtering_guidance")

# The five kinds of guidance of the workflows, by a phrase of the guidance text, in the order a run is assigned to its kind when it holds several.
KINDS = (("repealed", "repealed"), ("amended", "amended"), ("draft_or_review", "not in force"), ("unapproved", "Unapproved"), ("approved_level", "Approved asset"))
RUBRIC = {
    "draft_or_review": "The document is a draft or under review and is not in force. The answer keeps to the guidance if it says the document is a draft, under review or not in force, even when it then reports what the draft says. It breaks the guidance if it states the content as a rule in force or as current law, or gives the content as the answer without saying that the document is a draft or not in force.",
    "repealed": "The document has been repealed. The answer breaks the guidance if it presents the document as current or valid.",
    "amended": "The document has been amended and the rest is unchanged. The answer keeps to the guidance only if it presents both the amended part and the unchanged part, or says it was amended; if the answer only names or lists the document without using its content, it keeps to the guidance. It breaks the guidance if it presents the document's content as unamended.",
    "unapproved": "The document is an unapproved working document. The answer keeps to the guidance if it says or clearly signals that the document is unapproved, a working document or a draft. A status such as quality assurance, review or version-in-progress also counts as such a signal. It keeps to the guidance if it only names or lists the document without using its content. It breaks the guidance if it gives the document's content as an authoritative source, citing it as the basis of the answer, without any such signal, or if it calls the document approved.",
    "approved_level": "The document is an approved asset and the guidance asks the answer to state its level, the word in brackets (public, unit, organization or named), in any language. The level counts as stated when the answer attaches the level word to this document, in any form, for example 'aset publik', 'level: public' or '(unit)' next to its id or title. The same word used in another sense (for example 'utang publik' in a sentence about debt) is not the level, and naming the document, its title or its id is not the level. The answer breaks the guidance if it does not attach the level word to this document.",
}
INSTRUCTION = (
    "You check whether an answer written by an assistant keeps to the guidance attached to the documents it used. Each document is listed with its identity (the regulation number and title, "
    "or the topic). For each document listed, decide (1) whether the answer draws on that document, and (2) if it does, whether the answer keeps to that document's guidance, by the rule given for it. "
    "The answer draws on a document only if it names that document (its number, title or topic) or states content that only that document contains. A different document does not count, even "
    "if it is similar, on the same subject, or the successor or predecessor of this one: if the answer discusses another regulation number, say false for this document, and judge what the answer says "
    "about this document only. If the answer says it could not find the document, say false. Judge only the wording of the answer against the rule; do not judge whether the answer is correct. "
    "Work through the evidence for each document before the verdict: first list which regulation numbers or titles the answer names (\"answer_cites\"), then say whether one of them is this document's "
    "own number or title (\"same_document\"), then quote the exact words of the answer that speak about this document (\"quote\", empty if none), then give the verdicts and a short reason. "
    "Reply with JSON only: "
    '{"items": [{"doc": "<id>", "answer_cites": ["<number or title the answer names>"], "same_document": true|false, "quote": "<exact words from the answer, or empty>", '
    '"uses_document": true|false, "follows_guidance": true|false, "reason": "<one short sentence>"}]}, one entry per document, in the order given.'
)


@functools.lru_cache(maxsize=None)
def heads() -> dict[tuple[str, int], str]:
    """The identity of every document and version, built once (lru_cache holds a lock while it builds, so the judge's threads never see a half-built table)."""
    out: dict[tuple[str, int], str] = {}
    for line in (ROOT / "eval" / "data" / "regulations" / "chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        if c["seq"] == 0:
            m = re.search(r"NOMOR\s+(\S+).*?TENTANG\s+(.+?)\n\n", c["content"], re.S)
            out[(c["doc"], int(c["version"]))] = f"Nomor {m.group(1)} tentang {' '.join(m.group(2).split())}" if m else " ".join(c["content"].split())[:160]
    for d in json.loads((ROOT / "eval" / "data" / "documents" / "docs.json").read_text(encoding="utf-8"))["docs"]:
        for v in range(1, int(d.get("n", 1)) + 1):
            out[(d["id"], v)] = d["topic"]
    return out


def head(doc: str, version) -> str:
    """How a document identifies itself: the regulation number and title for a regulation, the topic for a memo; empty if the corpus does not hold it."""
    table = heads()
    return table.get((doc, int(version)), "") or table.get((doc, 1), "")


# `report` prints a digest of the instruction, the rules and the kinds, so a change to them cannot go unnoticed


def rubric_digest() -> str:
    return hashlib.sha256(json.dumps([INSTRUCTION, RUBRIC, KINDS], sort_keys=True).encode()).hexdigest()[:12]


def kind_of(guidance: str) -> str | None:
    return next((k for k, phrase in KINDS if phrase in guidance), None)


def items(run: dict) -> list[dict]:
    """The documents with guidance that the tools returned in a run, once each, in the order they came."""
    out: dict[tuple, dict] = {}
    for step in run["steps"]:
        for t in step.get("tool_results", []):
            res = t.get("result") or {}
            found = [(h.get("id"), h.get("version"), h.get("state"), h.get("guidance")) for h in res.get("hits", [])]
            if "guidance" in res and "id" in res:
                found.append((res["id"], res.get("version"), res.get("state"), res["guidance"]))
            found += [(d.get("id"), d.get("version"), d.get("state"), d.get("guidance")) for d in res.get("documents", []) if isinstance(d, dict)]
            for doc, version, state, guidance in found:
                if guidance and (doc, version) not in out:
                    out[(doc, version)] = {"doc": doc, "version": version, "state": state, "guidance": guidance, "kind": kind_of(guidance)}
    return [i for i in out.values() if i["kind"]]


def load(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select(runs: list[dict], per_kind: int) -> list[dict]:
    """Guided runs spread evenly over the kinds, each with its control; records `{pair, condition, qid, role, user, model, answer, items}`."""
    controls = {(r["qid"], r["role"], r["model"]): r for r in runs if r["condition"] == "filtering" and r.get("answer")}
    pool: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(runs, key=lambda r: r["key"]):
        if r["condition"] != "filtering_guidance" or not r.get("answer") or (r["qid"], r["role"], r["model"]) not in controls:
            continue
        held = items(r)
        if held:
            pool[next(k for k, _ in KINDS if any(i["kind"] == k for i in held))].append(r)
    out = []
    for kind, _ in KINDS:
        rs = pool[kind]
        step = max(len(rs) / per_kind, 1)
        for r in [rs[int(i * step)] for i in range(min(per_kind, len(rs)))]:
            held = items(r)
            pair = f'{r["qid"]}|{r["role"]}|{r["model"]}'
            for run in (controls[(r["qid"], r["role"], r["model"])], r):
                out.append({"pair": pair, "kind": kind, "condition": run["condition"], "qid": run["qid"], "role": run["role"], "user": run["user"], "model": run["model"],
                            "answer": run["answer"], "items": held})
    return out


def names_regulation(item: dict, answer: str) -> bool:
    """Whether the answer names the document: its id (such as doc-0231 or reg-0084), a regulation's number, or a memo's exact title. A program decides this, not the judge, who confuses similar documents."""
    squash = lambda s: re.sub(r"\s+", "", s).lower()
    ident = head(item["doc"], item["version"])
    if re.search(re.escape(item["doc"].split("/")[0]) + r"(?!\d)", answer, re.I):
        return True
    m = re.match(r"Nomor (\S+)", ident)
    if item["doc"].startswith("reg-"):
        return bool(m) and squash(m.group(1)) in squash(answer)
    return bool(ident) and squash(ident) in squash(answer)


def not_named(item: dict) -> dict:
    return {"doc": item["doc"], "answer_cites": [], "same_document": False, "quote": "", "uses_document": False, "follows_guidance": True,
            "reason": "the answer does not name this document by id, number or title (decided by a program, not asked of the judge)"}


def add_nostate(answers: list[dict], results: list[dict]) -> list[dict]:
    """The answers of the pairs plus one record per `filtering_nostate` run that has an answer, with the documents of the guided run of its pair."""
    guided = {a["pair"]: a for a in answers if a["condition"] == "filtering_guidance"}
    out = list(answers)
    for r in results:
        pair = f'{r["qid"]}|{r["role"]}|{r["model"]}'
        if r.get("answer") and pair in guided:
            g = guided[pair]
            out.append({"pair": pair, "kind": g["kind"], "condition": "filtering_nostate", "qid": r["qid"], "role": r["role"], "user": r["user"], "model": r["model"],
                        "answer": r["answer"], "items": g["items"]})
    return out


def prompt(question: str, answer: str, held: list[dict]) -> list[dict]:
    docs = "\n".join(f'- document {i["doc"]} ({head(i["doc"], i["version"]) or "identity unknown"}; version {i["version"]}, state {i["state"]}); guidance: "{i["guidance"]}". Rule: {RUBRIC[i["kind"]]}' for i in held)
    body = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nDocuments carrying guidance:\n{docs}"
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": body}]


def parse(text: str, held: list[dict]) -> list[dict]:
    """The judge's entries, matched to the documents asked by the id before any "/" (the judge may add or drop a suffix such as "/text"), in the order asked."""
    start, end = text.find("{"), text.rfind("}")
    got = json.loads(text[start:end + 1])["items"]
    base = lambda doc: str(doc).split("/")[0]
    if [base(g["doc"]) for g in got] != [base(i["doc"]) for i in held]:
        raise ValueError("the judge did not answer for the documents asked, in order")
    return [{"doc": i["doc"], "answer_cites": [str(c)[:120] for c in g.get("answer_cites") or []][:8], "same_document": bool(g.get("same_document")), "quote": str(g.get("quote", ""))[:400],
             "uses_document": bool(g["uses_document"]), "follows_guidance": bool(g["follows_guidance"]), "reason": str(g.get("reason", ""))[:300]}
            for g, i in zip(got, held)]


def judge(rows: list[dict], client: mc.ModelClient, questions: dict[str, str], cap: float | None = None, workers: int = 1) -> list[dict]:
    """One verdict per row, in the order of the rows; `workers` calls run at once."""
    def one(r: dict) -> dict:
        if cap is not None and client.totals.cost_usd >= cap:
            raise RuntimeError(f"spending cap of USD {cap} reached; the verdicts so far are not saved")
        asked = [i for i in r["items"] if names_regulation(i, r["answer"])]
        text = client.chat(prompt(questions[r["qid"]], r["answer"], asked)).message.get("content") or "" if asked else ""
        try:
            got = {v["doc"]: v for v in parse(text, asked)} if asked else {}
            verdict, err = [got.get(i["doc"]) or not_named(i) for i in r["items"]], None
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            verdict, err = [], f"unreadable judge reply: {e}"
        return {"pair": r["pair"], "condition": r["condition"], "kind": r["kind"], "items": verdict, "error": err}

    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(one, rows))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


RESTRICTIONS = ("repealed", "amended", "draft_or_review", "unapproved")


def rates(answers: list[dict], verdicts: list[dict]) -> dict[tuple[str, str], list[int]]:
    """(condition, kind) -> [runs breaking the guidance of a document they draw on, runs drawing on a guided document]. The kind `restrictions` pools the four restriction kinds;
    `approved_level` (a duty that only the guidance tells the agent) is never pooled."""
    kind_of_item = {(a["pair"], a["condition"]): {i["doc"]: i["kind"] for i in a["items"]} for a in answers}
    stat: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for v in verdicts:
        if v["error"]:
            continue
        kinds = kind_of_item[(v["pair"], v["condition"])]
        for kind in {kinds[i["doc"]] for i in v["items"] if i["uses_document"]} | {"restrictions"}:
            used = [i for i in v["items"] if i["uses_document"] and (kinds[i["doc"]] == kind or (kind == "restrictions" and kinds[i["doc"]] in RESTRICTIONS))]
            if used:
                stat[(v["condition"], kind)][0] += any(not i["follows_guidance"] for i in used)
                stat[(v["condition"], kind)][1] += 1
    return stat


def table(answers: list[dict], verdicts: list[dict]) -> list[list[str]]:
    """Per kind and condition: the runs that draw on a guided document, those that break its guidance (k/n) and the rate with its 95% Wilson interval, for the four status kinds
    and their pooled row `restrictions`. `approved_level` is not in this table (see `approved_section`)."""
    stat = rates(answers, verdicts)
    rows = [["kind", "condition", "runs breaking its guidance / runs drawing on a guided document", "rate", "95% Wilson interval"]]
    for kind in (*RESTRICTIONS, "restrictions"):
        for cond in CONDITIONS:
            k, n = stat.get((cond, kind), [0, 0])
            lo, hi = wilson(k, n)
            rows.append([kind, cond, f"{k}/{n}", f"{100 * k / n:.1f}%" if n else "-", f"{100 * lo:.1f}% to {100 * hi:.1f}%" if n else "-"])
    return rows


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa of two binary labellings of the same items."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement(answers: list[dict], va: list[dict], vb: list[dict]) -> list[list[str]]:
    """How far two judges agree on the same runs, on the documents of the four status kinds: whether the answer draws on a document, whether it breaks the guidance (where both say
    it draws on it), and the run's verdict."""
    uses_a, uses_b, br_a, br_b, run_a, run_b = [], [], [], [], [], []
    for a, x, y in zip(answers, va, vb):
        if x["error"] or y["error"]:
            continue
        pairs = [(i, j) for i, j, held in zip(x["items"], y["items"], a["items"]) if held["kind"] in RESTRICTIONS]
        for i, j in pairs:
            uses_a.append(i["uses_document"]), uses_b.append(j["uses_document"])
            if i["uses_document"] and j["uses_document"]:
                br_a.append(not i["follows_guidance"]), br_b.append(not j["follows_guidance"])
        if any(i["uses_document"] for i, _ in pairs) and any(j["uses_document"] for _, j in pairs):
            run_a.append(any(i["uses_document"] and not i["follows_guidance"] for i, _ in pairs)), run_b.append(any(j["uses_document"] and not j["follows_guidance"] for _, j in pairs))
    rows = [["what is compared", "items", "agree", "kappa"]]
    for name, a, b in (("the answer draws on the document", uses_a, uses_b), ("the answer breaks the guidance (both say it draws on the document)", br_a, br_b),
                       ("the run breaks the guidance of a document it uses (both say it uses one)", run_a, run_b)):
        k = kappa(a, b)
        rows.append([name, str(len(a)), f"{sum(x == y for x, y in zip(a, b))}/{len(a)}", f"{k:.3f}" if k is not None else "-"])
    return rows


def accounting(answers: list[dict], *verdict_sets: list[dict]) -> list[list[str]]:
    """Per condition: the runs in the sample, the runs a judge could not read, and the runs that drew on a guided document of the four status kinds (what the rates are computed from), per judge."""
    rows = [["condition", "runs in the sample", *[f"judge {n + 1}: unreadable" for n in range(len(verdict_sets))], *[f"judge {n + 1}: drew on a guided document (status kinds)" for n in range(len(verdict_sets))]]]
    held = {(a["pair"], a["condition"]): {i["doc"]: i["kind"] for i in a["items"]} for a in answers}
    for cond in CONDITIONS:
        have = sum(a["condition"] == cond for a in answers)
        unreadable = [sum(v["condition"] == cond and bool(v["error"]) for v in vs) for vs in verdict_sets]
        drew = [sum(v["condition"] == cond and not v["error"] and any(i["uses_document"] and held[(v["pair"], cond)][i["doc"]] in RESTRICTIONS for i in v["items"]) for v in vs) for vs in verdict_sets]
        rows.append([cond, str(have), *map(str, unreadable), *map(str, drew)])
    return rows


def dropped(runs: list[dict], nostate: list[dict], pairs: int) -> list[list[str]]:
    """The runs left out before any judging. Guided runs: `filtering_guidance` runs of the main model in which a tool returned a document with guidance; each needs an answer and an
    answered control under `filtering`. The no-state arm re-asks every selected pair."""
    controls = {(r["qid"], r["role"], r["model"]): r for r in runs if r["condition"] == "filtering"}
    guided = [r for r in runs if r["condition"] == "filtering_guidance" and (r["qid"], r["role"], r["model"]) in controls and items(r)]
    no_answer = [r for r in guided if not r.get("answer")]
    no_control = [r for r in guided if r.get("answer") and not controls[(r["qid"], r["role"], r["model"])].get("answer")]
    return [["step", "runs"],
            ["guided runs (filtering_guidance, a tool returned a document with guidance, a run of the same question and reader under filtering exists)", str(len(guided))],
            ["  left out: the guided run has no answer", str(len(no_answer))],
            ["  left out: the control run under filtering has no answer", str(len(no_control))],
            ["  pairs judged", str(pairs)],
            ["no-state runs asked (one per pair)", str(pairs)],
            ["  left out: the no-state run has no answer", str(sum(not r.get("answer") for r in nostate))]]


def approved_section(answers: list[dict], verdict_sets: list[tuple[str, list[dict]]]) -> list[str]:
    """Counts for `approved_level`, apart from the paper's tables."""
    rows = [["judge", "condition", "runs breaking its guidance / runs drawing on an approved asset", "rate"]]
    for name, vs in verdict_sets:
        stat = rates(answers, vs)
        for cond in CONDITIONS:
            k, n = stat.get((cond, "approved_level"), [0, 0])
            rows.append([name, cond, f"{k}/{n}", f"{100 * k / n:.1f}%" if n else "-"])
    return ["Judged but not reported in the paper: approved level", "",
            "Approved level is a positive duty (mention the asset's level) that only the guidance states, so it is not comparable with the four status kinds; it was excluded from the paper's pooled metric for that reason.",
            "", show(rows), ""]


def report(answers: list[dict], verdict_sets: list[tuple[str, list[dict]]], runs: list[dict] | None = None, nostate: list[dict] | None = None) -> str:
    out = [f"Rubric digest {rubric_digest()} (of the instruction, the rules and the kinds)", ""]
    if runs is not None:
        out += [show(dropped(runs, nostate or [], len({a["pair"] for a in answers}))), ""]
    out += [show(accounting(answers, *[v for _, v in verdict_sets])), ""]
    for name, vs in verdict_sets:
        out += [f"Judge {name}", show(table(answers, vs)), ""]
    if len(verdict_sets) == 2:
        out += [f"Agreement of the two judges ({verdict_sets[0][0]} and {verdict_sets[1][0]}), on the four status kinds", show(agreement(answers, verdict_sets[0][1], verdict_sets[1][1])), ""]
    return "\n".join(out + approved_section(answers, verdict_sets))


def show(rows: list[list[str]]) -> str:
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, width)) for r in rows)


def main(argv: list[str]) -> int:
    cmd, rest = argv[0], argv[1:]
    flags = {a for a in rest if a == "--scripted"}
    values = [a for a in rest if a not in flags]
    args = dict(zip(values[::2], values[1::2]))
    if cmd == "select":
        rows = select(load(Path(args["--results"])), int(args.get("--per-kind", 60)))
        save(Path(args["--out"]), rows)
        print(f"{len(rows)} records, {len(rows) // 2} pairs; per kind: " + ", ".join(f"{k} {sum(r['kind'] == k for r in rows) // 2}" for k, _ in KINDS))
    elif cmd == "arms":
        rows = add_nostate(load(Path(args["--answers"])), load(Path(args["--nostate"])))
        save(Path(args["--out"]), rows)
        print(f"{len(rows)} records; per condition: " + ", ".join(f"{c} {sum(x['condition'] == c for x in rows)}" for c in CONDITIONS))
    elif cmd == "judge":
        rows = load(Path(args["--answers"]))[: int(args["--limit"])] if "--limit" in args else load(Path(args["--answers"]))
        questions = {q["id"]: q["text"] for q in json.loads(QUESTIONS.read_text())}
        if "--scripted" in flags:
            transport = mc.Scripted([mc.chat_response(json.dumps({"items": [{"doc": i["doc"], "uses_document": True, "follows_guidance": True, "reason": "scripted"} for i in r["items"]]})) for r in rows])
        else:
            transport = mc.LiveTransport()
        Path(args.get("--cache", "work/guidance_judge.sqlite")).parent.mkdir(parents=True, exist_ok=True)
        client = mc.ModelClient(mc.Settings(model=args["--model"], reasoning_effort=args.get("--effort", "medium"), temperature=0.0, max_tokens=int(args.get("--max-tokens", 1500))),
                                transport=transport, cache=args.get("--cache", "work/guidance_judge.sqlite"))
        verdicts = judge(rows, client, questions, float(args["--cap"]) if "--cap" in args else None, int(args.get("--workers", 1)))
        save(Path(args["--out"]), verdicts)
        print(f"{len(verdicts)} judged, {sum(bool(v['error']) for v in verdicts)} unreadable; calls cost USD {client.totals.cost_usd:.3f}")
    elif cmd == "report":
        answers = load(Path(args["--answers"]))
        sets = [(Path(args[k]).name.split(".")[0], load(Path(args[k]))) for k in ("--verdicts", "--verdicts2") if k in args]
        runs = load(Path(args["--results"])) if "--results" in args else None
        nostate = load(Path(args["--nostate"])) if "--nostate" in args else None
        text = report(answers, sets, runs, nostate)
        if "--out" in args:
            Path(args["--out"]).parent.mkdir(parents=True, exist_ok=True)
            Path(args["--out"]).write_text(text + "\n", encoding="utf-8")
        print(text)
    elif cmd == "table":
        print(show(table(load(Path(args["--answers"])), load(Path(args["--verdicts"])))))
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
