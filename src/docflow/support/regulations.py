"""Replays the generated regulation set (`eval/data/regulations/plan.json` and `docs.jsonl`) into a `World`.

Each regulation is one document under the `regulation_single` workflow: its text (the supporting records are not
part of it). The drafters submit it, and a promulgation event brings it into force. An amendment or a repeal is another
regulation; when that one comes into force the host fires the event on the regulation it names (`amend` or
`repeal`). Regulations come into force in an order that lets every amendment arrive before the repeal of what it
amends, so the final state of each is the one the plan records.

Two organizations: O1 (the ministry, one unit per directorate) and O2 (a fictional regulator, unit BPP).
The same seven personas exist for every regulation; which user plays which depends on the regulation's
unit and organization (`persona_user`).
"""

from __future__ import annotations

import json
from graphlib import TopologicalSorter
from pathlib import Path

from ..stores.world import World

PERSONAS = ("lead", "team", "staff", "inv", "emp", "cit", "xorg")


def load(directory: str | Path) -> tuple[dict, list[dict]]:
    d = Path(directory)
    plan = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    docs = [json.loads(line) for line in (d / "docs.jsonl").open(encoding="utf-8")]
    return plan, docs


def persona_user(reg: dict, persona: str) -> str | None:
    """The user who plays `persona` for this regulation. `xorg` is someone of the other organization."""
    o2 = reg["org"] == "O2"
    return {"lead": f"lead_{reg['unit']}", "team": f"team_{reg['unit']}", "staff": f"staff_{reg['unit']}",
            "inv": "inv2" if o2 else "inv", "emp": "emp2" if o2 else "emp", "cit": "cit", "xorg": "emp" if o2 else "ag"}[persona]


def add_people(world: World, plan: dict) -> None:
    world.add_org("O1"), world.add_org("O2")
    for reg in plan["regulations"]:
        u = reg["unit"]
        if f"lead_{u}" not in world.users:
            world.add_user(f"lead_{u}", reg["org"], unit=u, positions=("leadership",))
            world.add_user(f"team_{u}", reg["org"], unit=u)
            world.add_user(f"staff_{u}", reg["org"], unit=u)
    for user, org, unit in (("inv", "O1", "INV"), ("emp", "O1", "OTH"), ("ag", "O2", "AGY"), ("inv2", "O2", "INV2"), ("emp2", "O2", "OTH2")):
        world.add_user(user, org, unit=unit)
    world.add_user("cit")


def promulgation_order(plan: dict) -> list[str]:
    """Targets come into force before what amends them, and amendments before the regulation that repeals the target."""
    regs = {r["id"]: r for r in plan["regulations"]}
    ts = TopologicalSorter()
    for r in regs.values():
        if r["state"] not in ("in_force", "amended", "repealed"):
            continue
        ts.add(r["id"])
        for a in r["amending"]:
            if not a.get("pending") and regs[a["target"]]["state"] in ("in_force", "amended", "repealed"):
                ts.add(r["id"], a["target"])  # r after its target
        if r["repeals"]:
            ts.add(r["repeals"], r["id"])  # the repealer after the target...
            for a in r["amends"]:
                if a["by"] != r["repeals"]:  # one regulation that amends and repeals does the amendment first (its links fire in that order)
                    ts.add(r["repeals"], a["by"])  # ...and after every amendment of it
    return list(ts.static_order())


def text_id(reg: dict) -> str:
    """The id of a regulation's one document, its text."""
    return reg["id"] + "/text"


def build_world(workflows, plan: dict, docs: list[dict]) -> World:
    world = World(workflows)
    add_people(world, plan)
    for org in ("O1", "O2"):
        world.add_workspace(f"ws-{org}", org)
    versions: dict[str, int] = {}
    for d in docs:
        if d["kind"] == "text":
            versions[d["doc"]] = max(versions.get(d["doc"], 0), d["version"])
    regs = {r["id"]: r for r in plan["regulations"]}
    for reg in plan["regulations"]:
        team, lead = persona_user(reg, "team"), persona_user(reg, "lead")
        doc = text_id(reg)
        world.add_document(doc, f"ws-{reg['org']}", team, "regulation_single", creator=lead,
                           assign={"drafting_team": [team], "invited": [persona_user(reg, "inv")]})
        for _ in range(versions[doc] - 1):
            world.act(doc, team, "upload")
        if reg["state"] != "working_draft":
            world.act(doc, team, "submit_for_review")
    for rid in promulgation_order(plan):  # the host fires what the coming into force of this regulation does to others
        world.fire_event(text_id(regs[rid]), "promulgate", "promulgated")
        for a in regs[rid]["amending"]:
            world.fire_event(text_id(regs[a["target"]]), "amend", "amendment_in_force")
        for target in (r for r in plan["regulations"] if r["repeals"] == rid):
            world.fire_event(text_id(target), "repeal", "repeal_in_force")
    return world
