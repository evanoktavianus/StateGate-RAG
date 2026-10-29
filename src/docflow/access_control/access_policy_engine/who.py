"""`who`: the language for naming people, used by `by`, `from`, `except` and access rules."""

from __future__ import annotations

from dataclasses import dataclass

from ...support.model import ROLES, Context, Person

_SIMPLE = ("anyone", "author", "org_admin", "everyone", "public", "invited", "organization", "unit")
_KEYED = ("role", "user", "group", "assigned")

EXPECTED = (
    "`anyone`, `author`, `everyone`, `public`, `invited`, `organization`, `unit`, `org_admin`, "
    "`role: …`, `user: …`, `group: …`, `position: …`, `assigned: …`, `attr: …`, `same: …`, `all: [ … ]` or a list of those"
)


@dataclass(frozen=True)
class Term:
    kind: str
    values: tuple[str, ...] = ()
    parts: tuple["Term", ...] = ()  # for `all`: every part must cover the person
    pairs: tuple = ()  # for `attr`: (person attribute, accepted values); for `same`: (person attribute, (subject attribute,))


Who = tuple[Term, ...] | None  # None: not restricted


def parse(raw, where: str, problems: list[str]) -> Who:
    """Reads a `who` from a workflow file, adding anything wrong with it to `problems`."""
    if raw is None:
        return None
    terms: list[Term] = []
    for item in raw if isinstance(raw, list) else [raw]:
        term = _term(item, where, problems)
        if term is not None:
            terms.append(term)
    return tuple(terms)


def _term(item, where: str, problems: list[str]) -> Term | None:
    if isinstance(item, str) and item in _SIMPLE:
        return Term(item)
    if isinstance(item, dict) and len(item) == 1:
        (kind, value), = item.items()
        if kind == "all" and isinstance(value, list) and value:
            parts = tuple(t for t in (_term(v, where, problems) for v in value) if t is not None)
            return Term("all", parts=parts)
        if kind in ("attr", "same") and isinstance(value, dict) and value:
            pairs = tuple((str(k), tuple(str(x) for x in (v if isinstance(v, list) else [v]))) for k, v in value.items())
            return Term(kind, pairs=pairs)
        if kind == "position":
            return Term("position", tuple(str(v).lower() for v in (value if isinstance(value, list) else [value])))
        if kind == "assigned" and isinstance(value, str):
            return Term("assigned", (value,))
        if kind in ("role", "user", "group"):
            values = tuple(str(v) for v in (value if isinstance(value, list) else [value]))
            if kind == "role":
                for role in values:
                    if role not in ROLES:
                        problems.append(f"{where}: there is no role {role!r}")
                return Term("role", values)
            return Term(kind, tuple(v.lower() for v in values))
    problems.append(f"{where} must be {EXPECTED}, got {item!r}")
    return None


def covers(who: Who, person: Person, ctx: Context, mode: str = "act") -> bool:
    """Whether `who` covers `person` (an unrestricted `who` covers everyone).

    `mode` is "act" (who may do an action) or "access" (who a document's
    access rule is for). They differ for three terms: `everyone` is anyone
    with a role on the document in access, anyone at all when acting;
    `public` is for people with no role and only means something in access;
    `organization` is anyone of the organization when acting or being chosen, and those of it with no role in access.
    """
    return who is None or any(_matches(t, person, ctx, mode) for t in who)


def _matches(term: Term, person: Person, ctx: Context, mode: str) -> bool:
    match term.kind:
        case "anyone":
            return True
        case "everyone":
            return mode == "act" or person.role is not None
        case "public":
            return mode == "access" and person.role is None
        case "organization":
            # acting or being chosen: anyone of the organization; in access: those of it with no role here
            return person.in_org and (mode == "act" or person.role is None)
        case "all":
            return all(_matches(p, person, ctx, mode) for p in term.parts)
        case "attr":  # the person has this attribute, with one of these values
            return all(set(values) & _person_values(person, name) for name, values in term.pairs)
        case "same":  # the person's attribute is the subject's: their unit is the document's requesting unit
            return all((v := _subject_value(ctx, subject)) is not None and v in _person_values(person, name) for name, (subject,) in term.pairs)
        case "position":
            return any(p in person.positions for p in term.values)
        case "unit":
            return person.unit is not None and person.unit == ctx.unit
        case "author":
            return person.id is not None and person.id == ctx.author_id
        case "role":
            return person.role is not None and str(person.role) in term.values
        case "user":
            return person.email is not None and person.email.lower() in term.values
        case "group":
            return any(g in term.values for g in person.groups)
        case "org_admin":
            return person.org_admin
        case "invited":
            return person.id in ctx.invited
        case "assigned":
            return person.id in ctx.assigned.get(term.values[0], ())
    return False


def _person_values(person: Person, name: str) -> set[str]:
    """A person's values for an attribute: the ones the language names, else `person.attrs`."""
    match name:
        case "unit":
            v = person.unit
        case "position":
            v = person.positions
        case "group":
            v = person.groups
        case "role":
            v = person.role
        case "id":
            v = person.id
        case _:
            v = person.attrs.get(name)
    return {str(x) for x in (v if isinstance(v, (list, tuple, set)) else [v]) if x is not None}


def _subject_value(ctx: Context, name: str) -> str | None:
    """A document's attribute: its unit, or any attribute declared by the workflow (set at creation or chosen)."""
    return ctx.unit if name == "unit" else ctx.choices.get(name)
