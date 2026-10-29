"""Workflow definitions: how a workflow file is read.

The YAML files in `workflows/` are the source of every workflow. `Machine.load` reads one with this loader and checks it;
the checked workflows are held by the world.
"""

from __future__ import annotations

import re

import yaml


class _Loader(yaml.SafeLoader):
    """YAML 1.1 reads `on` and `yes` as booleans, and `on:` is the key every state uses. Only true/false are booleans here."""


_Loader.yaml_implicit_resolvers = {
    key: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
