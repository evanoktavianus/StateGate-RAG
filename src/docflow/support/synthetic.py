"""Replays the paper's synthetic dataset (`eval/data/documents/docs.json`) into a `World`.

Each manifest document is put under `policy_analysis` and driven through the moves that lead to the
state the manifest records: uploads for its versions, a submit to a QA reviewer, a publish at the
release's audience (unit / organization / public), in order.

The dataset has one organization. `split_orgs` (default) makes two of it, so an `organization`
release means something: units U1 and U2 stay in O1, units U3 and U4 become O2. Users follow their
unit; the external users belong to none.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..stores.world import World

QA_GROUP = "QA team"
UNIT_ORG = {"U1": "O1", "U2": "O1", "U3": "O2", "U4": "O2"}


def load(path: str | Path) -> tuple[list[dict], dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["docs"], data["identity"]


def build_world(workflows, docs: list[dict], identity: dict, *, split_orgs: bool = True, workflow: str = "policy_analysis") -> World:
    world = World(workflows)

    def org_of(unit):
        if unit is None:
            return None
        return UNIT_ORG[unit] if split_orgs else identity["orgs"][0]["id"]

    for unit in {u for u in UNIT_ORG}:
        world.add_org(org_of(unit))

    for u in identity["users"]:
        qa = "qa_member" in u["roles"]
        world.add_user(u["id"], org=org_of(u["unit"]), groups=(QA_GROUP,) if qa else (), unit=u["unit"])

    qa_roles = {QA_GROUP: "editor"}
    for org in identity["orgs"]:
        for unit in org["units"]:
            members = {u["id"]: "contributor" for u in identity["users"] if u["unit"] == unit["id"] and "analyst" in u["roles"]}
            world.add_workspace(unit["workspace"], org_of(unit["id"]), members, qa_roles)
    for pw in identity["personal_workspaces"]:
        owner = next(u for u in identity["users"] if u["id"] == pw["owner"])
        world.add_workspace(pw["id"], org_of(owner["unit"]), {owner["id"]: "contributor"}, qa_roles)

    qa_users = [u["id"] for u in identity["users"] if "qa_member" in u["roles"]]
    for doc in docs:
        replay(world, doc, qa_users, workflow)
    return world


def replay(world: World, doc: dict, qa_users: list[str], workflow: str = "policy_analysis") -> None:
    """Adds one manifest document and drives it to the state the manifest records."""
    owner, n = doc["owner"], int(doc["n"])
    world.add_document(doc["id"], doc["workspace"], owner, workflow, invited=doc["collaborators"])
    reviewer = next(q for q in qa_users if q != owner)

    def upload_to(version: int) -> None:
        while world.documents[doc["id"]].timeline.version < version:
            if world.documents[doc["id"]].timeline.state == "released":
                world.act(doc["id"], owner, "start_update")  # a new version is uploaded in `working`, not in `released`
            world.act(doc["id"], owner, "upload")

    def submit() -> None:
        world.act(doc["id"], owner, "submit", input={"reviewer": [reviewer]})

    for release in doc["releases"]:
        upload_to(release["file_version"])
        if world.documents[doc["id"]].timeline.state != "under_review":
            submit()
        world.act(doc["id"], reviewer, "publish", input={"audience": release["level"]})

    if doc["state"] == "in_qa":
        upload_to(int(doc["qa_version"] or n))
        if world.documents[doc["id"]].timeline.state != "under_review":
            submit()
    upload_to(n)
