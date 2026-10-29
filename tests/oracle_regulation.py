"""The expected access to a regulation's text, from the regulation workflow's table and not from the engine.

Keeps the rows of the oracle of the group version for the text and drops the rows for the records.
For a person and a regulation there is one relation; for a relation and a state there is a row of the table. Nothing here
calls the engine's access code.

    relation   who
    lead       leadership of the regulation's drafting unit
    team       the drafting team named for the regulation
    staff      the drafting unit's staff, not on the team
    inv        a person of another unit, invited to review this regulation
    emp        anyone else in the regulation's organization
    cit        signed in, no organization
    xorg       a member of the other organization

    table:             draft         review        in force / amended / repealed
    text               team, lead    team, lead,   everyone
                                     inv
"""

from __future__ import annotations

TABLE = {
    "working_draft": {"team", "lead"},
    "review": {"team", "lead", "inv"},
}
for _s in ("in_force", "amended", "repealed"):
    TABLE[_s] = {"lead", "team", "staff", "inv", "emp", "cit", "xorg"}


def users(plan: dict) -> dict[str, tuple]:
    """Every person the host knows: id -> (organization, unit, kind)."""
    out = {"cit": (None, None, "cit"), "inv": ("O1", "INV", "inv"), "emp": ("O1", "OTH", "emp"), "ag": ("O2", "AGY", "emp"),
           "inv2": ("O2", "INV2", "inv"), "emp2": ("O2", "OTH2", "emp")}
    for r in plan["regulations"]:
        for kind in ("lead", "team", "staff"):
            out[f"{kind}_{r['unit']}"] = (r["org"], r["unit"], kind)
    return out


def relation(user: tuple, reg: dict) -> str:
    org, unit, kind = user
    if kind == "cit":
        return "cit"
    if org != reg["org"]:
        return "xorg"
    if unit == reg["unit"]:
        return kind  # lead, team or staff of this very unit
    invited = "inv2" if reg["org"] == "O2" else "inv"
    return "inv" if kind == "inv" and unit == {"inv": "INV", "inv2": "INV2"}[invited] else "emp"


def visible(plan: dict, versions: dict[str, int]) -> dict[str, dict[str, int]]:
    """{user: {document: version served}}, for the text of every regulation. `versions` is how many versions each
    text has (the replay uploads every version before the regulation moves on)."""
    people, out = users(plan), {}
    for uid, u in people.items():
        seen = {}
        for reg in plan["regulations"]:
            doc = reg["id"] + "/text"
            if relation(u, reg) in TABLE[reg["state"]]:
                seen[doc] = versions[doc]  # draft and review serve the latest; the rest the version in force: the same, all were uploaded before
        out[uid] = seen
    return out
