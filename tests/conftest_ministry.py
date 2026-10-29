"""A ministry (O1, units U1 and U2) and an outside organization (O2), for the knowledge-asset tests."""

import pytest

from docflow import Machine
from docflow.stores.world import World
from conftest import WORKFLOWS

NAMES = ("knowledge_asset",)


@pytest.fixture
def world():
    w = World({n: Machine.load(WORKFLOWS / f"{n}.yml") for n in NAMES})
    w.add_org("O1"), w.add_org("O2")
    for u, unit in (("ann", "U1"), ("bob", "U1"), ("cat", "U2")):  # three authors
        w.add_user(u, "O1", unit=unit)
    w.add_user("u1b", "O1", unit="U1")  # a viewer in the workspace, unit U1
    w.add_user("u1x", "O1", unit="U1")  # unit U1, no role in the workspace
    w.add_user("m2", "O1", unit="U2")  # the ministry, another unit, no role
    w.add_user("km1", "O1", groups=("Knowledge management",), unit="U1")
    for i in range(1, 6):
        w.add_user(f"c{i}", "O1", groups=("Committee",), unit="U2")
    w.add_user("ext", "O2")
    w.add_workspace(
        "ws", "O1",
        {"ann": "contributor", "bob": "contributor", "cat": "contributor", "u1b": "viewer"},
        {"Knowledge management": "editor", "Committee": "viewer"},
    )
    return w
