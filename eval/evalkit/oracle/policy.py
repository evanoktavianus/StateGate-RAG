"""A workflow file read with `yaml.safe_load`, into plain dictionaries. PyYAML reads the key `on` as the boolean True; it is handled here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import re

import yaml


class Loader(yaml.SafeLoader):
    """`yaml.safe_load`, except that only true and false are booleans: `on` and `off` stay words (a state's `on:`, a person called `off`)."""


Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"] for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def _on(state: dict | None) -> dict:
    if not state:
        return {}
    return state.get("on", state.get(True)) or {}


@dataclass
class Policy:
    name: str
    initial: str
    default_access: str  # "own" (the default of the language) or "deny"
    states: dict[str, dict]
    create: dict | None
    attributes: dict

    # ── reading ──
    def actions(self, state: str) -> dict[str, dict]:
        return _on(self.states.get(state))

    def access_rules(self, state: str) -> list[dict]:
        return (self.states.get(state) or {}).get("access") or []

    def guidance(self, state: str) -> str | None:
        return (self.states.get(state) or {}).get("guidance")


def load_policy(path: str | Path) -> Policy:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=Loader)
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> Policy:
    return Policy(
        name=raw.get("name"), initial=raw["initial"], default_access=raw.get("default_access", "own"),
        states={k: (v or {}) for k, v in raw["states"].items()}, create=raw.get("create"), attributes=raw.get("attributes") or {},
    )
