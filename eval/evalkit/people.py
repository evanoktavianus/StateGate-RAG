"""Person classes: equivalence classes of a person's relation to a document, derived from the who-terms a workflow uses.

`used_terms` reads the workflow file (plain YAML) and lists the groups, positions, roles, assignment names and attribute tests it speaks of.
`extend` adds to a fixture the people that make each class exist: one person per class in the owning unit, another unit of the same
organization, another organization, an outsider, and per group, position and role used, plus the boundary cases a human writes in
`eval/evalkit/worlds/<workflow>.boundary.yml` (and a few made here: in two groups, a position and a role together).
`classify` puts every person of a cast into a class for one situation, from the oracle's own record of the document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .oracle import doc as odoc
from .oracle import who as owho
from .oracle.policy import Loader

WORLDS = Path(__file__).resolve().parent / "worlds"
ROLES = ("admin", "editor", "contributor", "viewer")


@dataclass
class Terms:
    groups: set[str] = field(default_factory=set)
    positions: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    assigned: set[str] = field(default_factory=set)
    attr_tests: set[tuple] = field(default_factory=set)  # ("attr", name) | ("same", person attr, document attr)
    words: set[str] = field(default_factory=set)


def used_terms(raw: Any) -> Terms:
    t = Terms()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if len(x) == 1:
                ((k, v),) = x.items()
                vals = v if isinstance(v, list) else [v]
                if k == "group" and all(isinstance(i, str) for i in vals):
                    t.groups |= {i.lower() for i in vals}
                elif k == "position" and all(isinstance(i, str) for i in vals):
                    t.positions |= {i.lower() for i in vals}
                elif k == "role" and all(isinstance(i, str) for i in vals):
                    t.roles |= set(vals)
                elif k == "assigned" and isinstance(v, str):
                    t.assigned.add(v)
                elif k == "attr" and isinstance(v, dict):
                    t.attr_tests |= {("attr", n) for n in v}
                elif k == "same" and isinstance(v, dict):
                    t.attr_tests |= {("same", p, d) for p, d in v.items()}
            for k, v in x.items():
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, str) and x in ("public", "everyone", "organization", "unit", "invited", "org_admin", "anyone", "author"):
            t.words.add(x)

    walk(raw)
    return t


def policy_terms(workflow_path: Path) -> Terms:
    return used_terms(yaml.load(Path(workflow_path).read_text(encoding="utf-8"), Loader=Loader))


def extend(spec, terms: Terms) -> None:
    """Adds the people each class needs to the spec's cast, and their workspace roles. Ids start with `z_`; nothing already there is replaced."""
    owner = spec.people[spec.owner]
    org, unit = owner.get("org"), owner.get("unit")
    ws = spec.workspaces.setdefault(spec.workspace, {"org": org})
    members = ws.setdefault("members", {})
    other_unit = "ZU"

    def add(pid: str, **kw) -> None:
        spec.people.setdefault(pid, kw)

    add("z_unit_no_role", org=org, unit=unit)
    add("z_unit_with_role", org=org, unit=unit)
    members.setdefault("z_unit_with_role", "contributor")
    add("z_org_with_role_other_unit", org=org, unit=other_unit)
    members.setdefault("z_org_with_role_other_unit", "viewer")
    for r in sorted(terms.roles & set(ROLES)):
        add(f"z_role_{r}", org=org, unit=unit)
        members.setdefault(f"z_role_{r}", r)
    for p in sorted(terms.positions):
        add(f"z_pos_{p}", org=org, unit=unit, positions=[p])
        add(f"z_pos_{p}_other_unit", org=org, unit=other_unit, positions=[p])
    for g in sorted(terms.groups):
        add(f"z_group_{g.replace(' ', '_')}", org=org, unit=other_unit, groups=[g])
    gs = sorted(terms.groups)
    if len(gs) >= 2:  # a member of two groups
        add("z_two_groups", org=org, unit=other_unit, groups=gs[:2])
    if terms.positions:  # holds a position and a role
        add("z_position_and_role", org=org, unit=unit, positions=[sorted(terms.positions)[0]])
        members.setdefault("z_position_and_role", "viewer")
    boundary = WORLDS / f"{spec.workflow}.boundary.yml"
    if boundary.exists():  # people and their workspace roles
        raw = yaml.load(boundary.read_text(encoding="utf-8"), Loader=Loader)
        for pid, p in (raw.get("people") or {}).items():
            add(pid, **p)
        for pid, role in (raw.get("roles") or {}).items():
            members.setdefault(pid, role)


# ── classes, one situation at a time ──
def classify(cast: odoc.Cast, doc: odoc.Doc, terms: Terms) -> dict[str, str]:
    """person id -> class name, from the oracle's record of the document (never from the system)."""
    out = {}
    for pid, p in cast.items():
        parts = []
        if pid == doc.owner:
            parts.append("author")
        parts.append("same-org" if owho.in_org(p, doc) else ("no-org" if p.org is None else "other-org"))
        if p.unit is None:
            parts.append("no-unit")
        elif p.unit == doc.unit:
            parts.append("owning-unit")
        else:
            parts.append("other-unit")
        role = owho.person_role(p, doc)
        parts.append(f"role:{role}" if role else "no-role")
        parts += [f"group:{g}" for g in sorted(set(p.groups) & terms.groups)]
        parts += [f"pos:{x}" for x in sorted(set(p.positions) & terms.positions)]
        parts += [f"assigned:{n}" for n in sorted(terms.assigned) if pid in doc.assigned.get(n, ())]
        if pid in doc.invited:
            parts.append("invited")
        for test in sorted(terms.attr_tests):
            if test[0] == "same":
                hit = owho.covers({"same": {test[1]: test[2]}}, p, doc, "access")
                parts.append(f"same[{test[1]}={test[2]}]:{'yes' if hit else 'no'}")
        out[pid] = " ".join(parts)
    return out


def representatives(classes: dict[str, str]) -> dict[str, str]:
    """class name -> the first person of the cast in it."""
    reps: dict[str, str] = {}
    for pid, c in classes.items():
        reps.setdefault(c, pid)
    return reps
