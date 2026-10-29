"""The workflow language: one definition of every word, used to check files, generate the JSON Schema and the reference.

A workflow file may use only these words. Anything else is refused, naming the word, so a typo is an error and
never a silently ignored setting. The language grows only by adding a word here (and what it means in the engine);
a requirement type is the one open extension point (`requirements.register`) and is outside this vocabulary.

    python -m docflow.access_control.access_policy_engine.dsl schema      > workflow.schema.json
    python -m docflow.access_control.access_policy_engine.dsl reference   > workflows/language_reference.txt
"""

from __future__ import annotations

import json
import sys

# Each section: the words allowed there, with what they mean.
VOCAB: dict[str, dict[str, str]] = {
    "workflow": {
        "name": "what people see and pick",
        "initial": "the state a document starts in",
        "default_access": "`own` (a person no rule covers keeps their own role and the latest version) or `deny` (gets nothing)",
        "states": "the states, by name",
        "create": "who may create a document and which people they must name",
        "attributes": "the attributes a document carries, with their allowed values (left empty: any value); rules test them with `when` and `same`",
    },
    "state": {
        "guidance": "what an agent is told about a document here",
        "on": "the actions this state accepts; any other action is refused",
        "access": "who reaches the document while it is here, and which version",
    },
    "transition": {
        "to": "the state it leads to (the same state means stay)",
        "by": "who may do it (left out: anyone who can act)",
        "requires": "something that must be satisfied first",
        "assign": "people the actor must name",
        "choose": "options the actor must pick, which access rules can depend on",
        "event": "fired only by this outside event, never by a person",
    },
    "assign": {
        "from": "who can be chosen",
        "min": "at least this many (default 1)",
        "max": "at most this many",
        "except": "who can not be chosen",
        "odd": "an odd number of people",
    },
    "access": {
        "who": "who the rule is for; the first rule that covers a person is theirs",
        "role": "the role they have here: editor, contributor, viewer or none",
        "serve": "the version they open: latest, nothing, or { from: <state> }",
        "find": "the version a search finds for them (left out: the same as serve)",
        "when": "only while the choice or attribute under that name has this value (or one of a list)",
        "guidance": "what an agent is told about the document for a person this rule covers (left out: the state's guidance)",
    },
    "create": {"by": "who may create it", "assign": "the roles the creator must fill, as for `assign`", "set": "the attributes the creator must give"},
    "attribute": {"values": "the allowed values (left out: any)"},
}

WHO_TERMS = {
    "anyone": "any member (when acting); everyone (in access rules)",
    "author": "whoever created the document",
    "everyone": "anyone with a role on the document",
    "public": "anyone signed in with no role (access rules)",
    "invited": "anyone the document itself was opened to",
    "organization": "anyone of the organization (acting); those with no role here (access rules)",
    "unit": "people of the document's unit",
    "org_admin": "an admin of the organization",
    "role: <name>": "members with that role (admin, editor, contributor, viewer); a list works",
    "user: <email>": "that person; a list works",
    "group: <name>": "people in that group; a list works",
    "position: <name>": "people holding that position (leadership, staff …); a list works",
    "attr: { <name>: <value> }": "people whose attribute has that value (or one of a list); every pair must hold",
    "same: { <person attr>: <document attr> }": "people whose attribute equals the document's (their unit is its requesting unit, their group its group)",
    "assigned: <name>": "the people chosen under that name",
    "all: [ … ]": "people every one of the listed terms covers",
}

REQUIREMENTS = {
    "approval": {"settings": ["pass", "choices"], "means": "enough people have approved the current version; pass: any | all | <number>"},
    "signature": {"settings": ["pass", "choices"], "means": "enough people have signed the current version, with evidence bound to it; pass: any | all | <number>"},
}


def check_keys(where: str, mapping: dict, section: str, problems: list[str]) -> None:
    """Adds a problem for every word in `mapping` that the language does not have in `section`."""
    for key in mapping:
        if key not in VOCAB[section]:
            problems.append(f"{where}: there is no {key!r} here (the words are {', '.join(VOCAB[section])})")


# ── JSON Schema ────────────────────────────────────────────────────────────

def _props(section: str, **overrides) -> dict:
    return {k: overrides.get(k, {"description": d}) for k, d in VOCAB[section].items()}


def schema() -> dict:
    who = {
        "anyOf": [
            {"enum": ["anyone", "author", "everyone", "public", "invited", "organization", "unit", "org_admin"]},
            {"type": "object", "minProperties": 1, "maxProperties": 1,
             "properties": {k: {} for k in ("role", "user", "group", "position", "assigned", "all", "attr", "same")}, "additionalProperties": False},
            {"type": "array", "items": {"$ref": "#/$defs/who"}},
        ]
    }
    serve = {"anyOf": [{"enum": ["latest", "nothing"]}, {"type": "object", "properties": {"from": {"type": "string"}}, "required": ["from"], "additionalProperties": False}]}
    obj = lambda props, **kw: {"type": "object", "properties": props, "additionalProperties": False, **kw}
    defs = {
        "who": who,
        "assign": obj(_props("assign", **{"from": {"$ref": "#/$defs/who"}, "except": {"$ref": "#/$defs/who"}}), required=["from"]),
        "transition": obj(
            _props("transition", by={"$ref": "#/$defs/who"},
                   requires={"type": "object", "required": ["type"], "properties": {"type": {"enum": list(REQUIREMENTS)}}},
                   assign={"type": "object", "additionalProperties": {"$ref": "#/$defs/assign"}},
                   choose={"type": "object", "additionalProperties": {"type": "array", "items": {"type": ["string", "integer"]}, "minItems": 1}}),
            required=["to"]),
        "accessRule": obj(_props("access", who={"$ref": "#/$defs/who"}, role={"enum": ["editor", "contributor", "viewer", "none"]},
                                 serve=serve, find=serve, guidance={"type": "string"},
                                 when={"type": "object", "additionalProperties": {"anyOf": [{"type": ["string", "integer"]}, {"type": "array"}]}}), required=["who"]),
    }
    state = obj(_props(
        "state",
        guidance={"type": "string"},
        on={"type": "object", "additionalProperties": {"$ref": "#/$defs/transition"}},
        access={"type": "array", "items": {"$ref": "#/$defs/accessRule"}},
    ))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "docflow workflow",
        "type": "object",
        "required": ["name", "initial", "states"],
        "properties": _props(
            "workflow",
            default_access={"enum": ["own", "deny"]},
            states={"type": "object", "additionalProperties": state, "minProperties": 1},
            create=obj(_props("create", by={"$ref": "#/$defs/who"}, assign={"type": "object", "additionalProperties": {"$ref": "#/$defs/assign"}},
                              set={"type": "array", "items": {"type": "string"}})),
            attributes={"type": "object", "additionalProperties": {"anyOf": [{"type": "null"}, obj(_props("attribute", values={"type": "array", "items": {"type": ["string", "integer"]}}))]}},
        ),
        "additionalProperties": False,
        "$defs": defs,
    }


def reference() -> str:
    out = ["# The workflow language", "", "Generated from `docflow.access_control.access_policy_engine.dsl`. A word not listed here is refused.", ""]
    for section, words in VOCAB.items():
        out += [f"## {section}", "", "| word | means |", "|---|---|", *[f"| `{k}` | {v} |" for k, v in words.items()], ""]
    out += ["## who", "", "| term | covers |", "|---|---|", *[f"| `{k}` | {v} |" for k, v in WHO_TERMS.items()], ""]
    out += ["## requirement types", "", "| type | settings | means |", "|---|---|---|",
            *[f"| `{k}` | {', '.join(v['settings'])} | {v['means']} |" for k, v in REQUIREMENTS.items()], ""]
    return "\n".join(out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "reference"
    print(json.dumps(schema(), indent=2) if what == "schema" else reference())
