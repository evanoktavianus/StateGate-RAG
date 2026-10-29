"""Who a term covers, from the written reference of the workflow language (the "Who" section).

A term is a word (`author`), a one-key mapping (`{role: editor}`, `{assigned: team}`, `{all: [..]}`), or a list of terms (any of them).
`mode` is "act" (for `by`, `from`, `except`) or "access" (for the `who` of an access rule). A person is a `Person` of oracle/doc.py.
"""

from __future__ import annotations

from typing import Any


def covers(term: Any, person, doc, mode: str, default_access: str = "deny") -> bool:
    if term is None:
        return True
    if isinstance(term, list):
        return any(covers(t, person, doc, mode, default_access) for t in term)
    if isinstance(term, str):
        return _word(term, person, doc, mode, default_access)
    ((kind, value),) = term.items()
    values = value if isinstance(value, list) else [value]
    if kind == "role":
        return person_role(person, doc) in values
    if kind == "user":
        return f"{person.id}@example.test" in values or person.id in values
    if kind == "group":
        return bool({str(v).lower() for v in values} & set(person.groups))
    if kind == "position":
        return bool({str(v).lower() for v in values} & set(person.positions))
    if kind == "assigned":
        return person.id in doc.assigned.get(value, ())
    if kind == "all":
        return all(covers(t, person, doc, mode, default_access) for t in value)
    if kind == "attr":
        return all(_overlap(_person_values(person, name), want) for name, want in value.items())
    if kind == "same":
        for person_attr, doc_attr in value.items():
            mine, theirs = _person_values(person, person_attr), _doc_value(doc, doc_attr)
            if theirs is None or theirs not in mine:
                return False
        return True
    raise ValueError(f"unknown who term {term!r}")


def person_role(person, doc) -> str | None:
    """The strongest role the person has in the workspace: their own, or through a group the workspace gives a role."""
    order = ("admin", "editor", "contributor", "viewer")
    roles = [r for r in (doc.workspace.get("members", {}).get(person.id), *(doc.workspace.get("group_roles", {}).get(g) for g in person.groups)) if r]
    return min(roles, key=order.index) if roles else None


def in_org(person, doc) -> bool:
    return person.org is not None and person.org == doc.workspace_org


def _word(word: str, person, doc, mode: str, default_access: str) -> bool:
    role = person_role(person, doc)
    match word:
        case "anyone":
            return True if (mode == "access" or default_access == "deny") else role is not None
        case "author":
            return person.id == doc.owner
        case "everyone":
            return role is not None
        case "public":
            return role is None  # signed in with no role: of this organization or another (the scenario tables say so)
        case "invited":
            return person.id in doc.invited
        case "organization":
            return in_org(person, doc) and (mode == "act" or role is None)
        case "unit":
            return person.unit is not None and person.unit == doc.unit
        case "org_admin":
            return person.org_admin and in_org(person, doc)
    raise ValueError(f"unknown who word {word!r}")


def _person_values(person, name: str) -> set[str]:
    if name == "unit":
        return {person.unit} if person.unit else set()
    if name == "group":
        return set(person.groups)
    v = person.attrs.get(name)
    return set() if v is None else {str(x) for x in (v if isinstance(v, list) else [v])}


def _overlap(have: set[str], want: Any) -> bool:
    return bool(have & {str(x) for x in (want if isinstance(want, list) else [want])})


def _doc_value(doc, name: str) -> str | None:
    if name == "unit":
        return doc.unit
    v = doc.choices.get(name)
    return None if v is None else str(v)
