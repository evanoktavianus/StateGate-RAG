"""The run list of E6: one run is a question, a condition, an asker (one who may read the document, one who may not) and a model.

    python -m evalkit.e6.plan --out runs/e6/plan.json [--second qwen/qwen3.7-flash]

Conditions: `no_access_control` (every person is served the newest version of everything, no guidance), `filtering` (the service as it is, guidance left out of every
tool result), `filtering_guidance` (the service as it is). Main model: all 1,076 questions, three conditions. Second model: the stratified 200, conditions
`no_access_control` and `filtering_guidance`. The reader is the first person who may read the document, in the order same unit, another unit of the same organization,
another organization, outsider (regulations: lead, team, staff, investor, employee, citizen, other organization); the non-reader is a person who may not read it
(or open it), by preference of the owning organization, drawn by a checksum of the question id; where everyone may read the document there is no non-reader run.
Runs are ordered by question (in the order of a checksum), all the runs of a question together, so that a run stopped half way keeps whole pairs.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

from docflow.support import regulations as R

from .. import qplan, retrieval_utility as U
from .sample import SEED

MAIN_MODEL = "z-ai/glm-5.3-flash"
CONDITIONS = ("no_access_control", "filtering", "filtering_guidance")
SECOND_CONDITIONS = ("no_access_control", "filtering_guidance")
ROOT = Path(__file__).resolve().parents[3]


def tag(model: str) -> str:
    return model.split("/")[-1]


def _cat(user, doc_unit: str, org: str | None) -> str:
    if user.unit == doc_unit:
        return "same unit"
    return "another unit of the same organization" if user.org == org else "another organization" if user.org else "outsider"


def pick_users(q: dict) -> tuple[dict | None, dict | None]:
    """(reader, non_reader), each {"category", "user"} (the reader also "serve", the version served), or None where there is nobody."""
    if q["workflow"] == "regulation_single":
        world, corpus, regs = U.G["reg"]
        reg = regs[q["doc"].split("/")[0]]
        users = [(p, R.persona_user(reg, p)) for p in R.PERSONAS]
        reader = next(((p, u, world.readable_version(q["doc"], u, for_search=False)) for p, u in users
                       if u and world.readable_version(q["doc"], u, for_search=False) is not None and world.readable_version(q["doc"], u, for_search=True) is not None), None)
        non = [(p, u) for p, u in users if u and world.readable_version(q["doc"], u, for_search=False) is None and world.readable_version(q["doc"], u, for_search=True) is None]
        n = min(non, key=lambda x: zlib.crc32(f"{q['id']}{x[1]}".encode())) if non else None
        return ({"category": "same unit:" + reader[0] if reader[0] in ("lead", "team", "staff") else "outside the unit:" + reader[0], "user": reader[1], "serve": reader[2]} if reader else None,
                {"category": "outside the unit:" + n[0], "user": n[1]} if n else None)
    worlds, corpus, assignment = U.G["syn"]
    world, a = worlds[q["workflow"]], assignment[q["doc"]]
    order = ["same unit", "another unit of the same organization", "another organization", "outsider"]
    readers, non = [], []
    for u in sorted(world.users):
        cat = _cat(world.users[u], a["unit"], a["organization"])
        serve = world.readable_version(q["doc"], u, for_search=False)
        find = world.readable_version(q["doc"], u, for_search=True)
        if serve is not None and find is not None:
            readers.append((order.index(cat), cat, u, serve))
        elif serve is None and find is None:
            non.append((order.index(cat) if cat != "another unit of the same organization" else -1, cat, u))
    r = min(readers) if readers else None
    n = min(non, key=lambda x: (x[0], zlib.crc32(f"{q['id']}{x[2]}".encode()))) if non else None
    return ({"category": r[1], "user": r[2], "serve": r[3]} if r else None, {"category": n[1], "user": n[2]} if n else None)


def askers_for(q: dict) -> dict:
    """The state of the question's document, and per role (reader, non_reader) the asker; a reader is asked only where the question has a known answer for the version they are served."""
    reader, non = pick_users(q)
    world = (U.G["reg"][0] if q["workflow"] == "regulation_single" else U.G["syn"][0][q["workflow"]])
    state = world.state_of(q["doc"])
    out: dict = {"state": state}
    if reader:
        if q["kind"] == "status":
            known = qplan.STATUS.get(q["workflow"], {}).get(state)
        elif q["kind"] == "content":
            known = q["answer"]
        else:
            known = q["answer_a"] if reader["serve"] == q["pair"]["a"] else q["answer_b"] if reader["serve"] == q["pair"]["b"] else None
        if known:
            out["reader"] = {"user": reader["user"], "category": reader["category"]}
    if non:
        out["non_reader"] = {"user": non["user"], "category": non["category"]}
    return out


def build(second_model: str | None = None) -> list[dict]:
    qs = U.G["questions"]
    stratified = set(json.loads((ROOT / "eval/data/questions" / "stratified200.json").read_text()))
    runs = []
    for q in sorted(qs, key=lambda q: zlib.crc32(f"{SEED}\0run\0{q['id']}".encode())):
        sp = askers_for(q)
        jobs = [(MAIN_MODEL, c) for c in CONDITIONS]
        if second_model and q["id"] in stratified:
            jobs += [(second_model, c) for c in SECOND_CONDITIONS]
        for model, cond in jobs:
            for role in ("reader", "non_reader"):
                if role not in sp:
                    continue
                runs.append({"key": f"{q['id']}|{cond}|{role}|{tag(model)}", "qid": q["id"], "kind": q["kind"], "workflow": q["workflow"], "state": sp["state"],
                             "regulation": q["workflow"] == "regulation_single", "condition": cond, "role": role, "model": model, "question": q["text"],
                             "user": sp[role]["user"], "category": sp[role]["category"]})
    return runs


def main(argv: list[str]) -> None:
    args = dict(zip(argv[::2], argv[1::2]))
    U.setup()
    runs = build(args.get("--second"))
    out = Path(args.get("--out", "runs/e6/plan.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, ensure_ascii=False))
    from collections import Counter
    print(len(runs), "runs;", dict(Counter((r["model"], r["condition"], r["role"]) for r in runs)))
    miss = Counter(r["qid"] for r in runs if r["role"] == "reader")
    print("questions with a reader:", len(miss), "of", len(U.G["questions"]))


if __name__ == "__main__":
    main(sys.argv[1:])
