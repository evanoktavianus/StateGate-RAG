"""The independent oracle agrees with every scenario table (and a wrong table is caught)."""

import ast
from pathlib import Path

import pytest
import yaml

from evalkit.oracle import scenario as oracle_scenario

ROOT = Path(__file__).parent.parent
FILES = sorted((ROOT / "scenarios").glob("*.yml"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_the_oracle_agrees_with_the_hand_written_table(path):
    assert oracle_scenario.run(path, ROOT / "workflows") == []


def test_a_wrong_table_is_caught(tmp_path):
    sc = yaml.safe_load((ROOT / "scenarios" / "card1_policy_analysis_simple.yml").read_text())
    first = next(s["expect_access"] for s in sc["steps"] if "expect_access" in s)
    first["table"]["staff"] = "D"  # staff may not open a working analysis
    first["versions"] = {"D": 1}
    p = tmp_path / "wrong.yml"
    p.write_text(yaml.dump(sc))
    bad = oracle_scenario.run(p, ROOT / "workflows")
    assert any("staff opens d" in b for b in bad) and any("version" in b for b in bad)


def test_the_oracle_imports_nothing_of_the_system():
    """It must be independent of the Access policy engine, the who-term evaluator and the history code."""
    for f in (ROOT / "eval" / "evalkit" / "oracle").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] == "docflow" for n in names), (f.name, names)
