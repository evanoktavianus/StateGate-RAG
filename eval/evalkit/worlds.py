"""The fixtures of a workflow: who exists, where, and what a new document of the workflow is given.

A world file (`eval/evalkit/worlds/<workflow>.yml`) names the scenario file whose people and workspaces it reuses, the document, and
the values the creator gives. A standard cast is added to every world: a person of another unit of the same organization, a person of
another organization, and a signed-in outsider with no organization. Everything in a world is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from docflow import Machine
from docflow.stores.world import World
from docflow.stores.workflow_definitions import _Loader

ROOT = Path(__file__).resolve().parents[2]
WORLDS = Path(__file__).resolve().parent / "worlds"

# Added to every world. Units and organizations are made up here, so they clash with nothing in a scenario.
STANDARD_UNIT = "ZU"
STANDARD_OTHER_ORG = {"org": "OZ", "unit": "ZO"}


@dataclass
class Spec:
    workflow: str
    people: dict[str, dict]
    workspaces: dict[str, dict]
    workspace: str
    doc_id: str
    owner: str
    creator: str | None
    invited: list[str]
    assign: dict[str, list[str]]  # given when the document is created, and again whenever an action asks for the name
    attrs: dict[str, list[str]]  # the values to try for each attribute set at creation; the first is the base
    notes: str = ""
    file: Path = field(default=None)  # type: ignore[assignment]
    workflow_file: Path | None = None  # a workflow outside workflows/ (a test's own)

    @property
    def workflow_path(self) -> Path:
        return self.workflow_file or ROOT / "workflows" / f"{self.workflow}.yml"

    def machine(self) -> Machine:
        return Machine.load(self.workflow_path)

    def variants(self) -> list[dict[str, str]]:
        """The attribute sets a document is created with: the base, then each other value of each attribute, one at a time."""
        base = {k: v[0] for k, v in self.attrs.items()}
        out = [base]
        for k, values in self.attrs.items():
            out += [{**base, k: v} for v in values[1:]]
        return out


def load_spec(name: str) -> Spec:
    path = WORLDS / f"{name}.yml"
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    scenario = yaml.load((ROOT / raw["scenario"]).read_text(encoding="utf-8"), Loader=_Loader)
    people = {**scenario["people"], **raw.get("people", {})}
    workspaces = {**scenario.get("workspaces", {}), **raw.get("workspaces", {})}
    doc = raw["document"]
    org = people[doc["owner"]].get("org")
    for pid, p in {
        "z_same_org_other_unit": {"org": org, "unit": STANDARD_UNIT},
        "z_other_org": dict(STANDARD_OTHER_ORG),
        "z_outsider": {},
    }.items():
        people.setdefault(pid, p)
    attrs = {k: v if isinstance(v, list) else [v] for k, v in (doc.get("attrs") or {}).items()}
    spec = Spec(
        workflow=raw["workflow"], people=people, workspaces=workspaces, workspace=doc.get("workspace", next(iter(workspaces))),
        doc_id=doc["id"], owner=doc["owner"], creator=doc.get("creator"), invited=list(doc.get("invited", [])),
        assign={k: list(v) for k, v in (doc.get("assign") or {}).items()}, attrs=attrs, notes=raw.get("notes", ""), file=path,
    )
    from .people import extend, policy_terms  # the people each person class needs, derived from the workflow's own who-terms

    extend(spec, policy_terms(spec.workflow_path))
    return spec


def build_cast(spec: Spec, machine: Machine | None = None, store=None) -> World:
    """A fresh world with the organizations, people and workspaces, and no document."""
    world = World({spec.workflow: machine or spec.machine()}, store)
    for org in sorted({p.get("org") for p in spec.people.values() if p.get("org")}):
        world.add_org(org)
    for pid, p in spec.people.items():
        world.add_user(pid, p.get("org"), p.get("groups", ()), p.get("org_admin", False), p.get("unit"), p.get("positions", ()), p.get("attrs"))
    for wid, w in spec.workspaces.items():
        world.add_workspace(wid, w.get("org"), w.get("members"), w.get("group_roles"))
    return world


def create_document(world: World, spec: Spec, attrs: dict[str, str], doc_id: str | None = None) -> None:
    """Creates the spec's document as its creator would create it."""
    machine = world.workflows[spec.workflow]
    assign = {k: v for k, v in spec.assign.items() if machine.create and k in machine.create.assign}
    world.add_document(doc_id or spec.doc_id, spec.workspace, spec.owner, spec.workflow, invited=spec.invited,
                       creator=spec.creator, assign=assign or None, attrs=attrs or None)


def build_world(spec: Spec, attrs: dict[str, str], machine: Machine | None = None, store=None) -> World:
    """A fresh world with the cast, and the document created as its creator would create it."""
    world = build_cast(spec, machine, store)
    create_document(world, spec, attrs)
    return world
