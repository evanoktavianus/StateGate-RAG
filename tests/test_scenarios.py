"""Every scenario file is a test: a workflow's expected outcomes, written as data, checked on every channel."""

from pathlib import Path

import pytest

from docflow.support import scenario

ROOT = Path(__file__).parent.parent
FILES = sorted((ROOT / "scenarios").glob("*.yml"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_scenario(path):
    result = scenario.run(path, ROOT / "workflows")
    assert result.ok, "\n".join(result.failures)
    assert result.checks > 0


def test_a_wrong_expectation_fails_so_a_pass_means_something():
    """The runner must be able to fail: change one cell of a passing scenario and it has to say so."""
    import yaml
    from docflow.stores.workflow_definitions import _Loader

    sc = yaml.load((ROOT / "scenarios" / "regulation_single.yml").read_text(), Loader=_Loader)
    first_table = next(s["expect_access"] for s in sc["steps"] if "expect_access" in s)
    first_table["table"]["emp"] = "T"  # an employee of another unit may not see a working draft
    first_table["versions"] = {"T": 1}  # and the text is version 2, not 1
    result = scenario.Runner(sc, ROOT / "workflows").run()
    assert not result.ok
    assert any("emp" in f for f in result.failures) and any("version" in f for f in result.failures)
