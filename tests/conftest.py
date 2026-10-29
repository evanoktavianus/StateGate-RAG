from pathlib import Path

import pytest

from docflow import Context, Machine, Person

WORKFLOWS = Path(__file__).parent.parent / "workflows"  # the workflows of the paper
FIXTURES = Path(__file__).parent / "fixtures"  # small workflows only the unit tests use (`policy_analysis`, `report`)


def load(name: str) -> Machine:
    path = WORKFLOWS / f"{name}.yml"
    return Machine.load(path if path.exists() else FIXTURES / f"{name}.yml")


def person(id: str, role: str | None = "viewer", **kw) -> Person:
    return Person(id=id, email=f"{id}@acme.test", role=role, **kw)


@pytest.fixture
def people():
    return {
        "ann": person("ann", "contributor"),  # the author
        "bob": person("bob", "editor"),
        "cat": person("cat", "editor"),
        "dan": person("dan", "viewer"),
    }


@pytest.fixture
def ctx(people):
    """Builds a Context for `who` acting; `members` is everyone."""

    def make(who: str, **kw) -> Context:
        return Context(actor=people[who], author_id="ann", members=tuple(people.values()), **kw)

    return make
