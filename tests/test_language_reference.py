"""The stored language reference and JSON Schema are what the one vocabulary definition generates."""

import json
from pathlib import Path

from docflow.access_control.access_policy_engine import dsl

ROOT = Path(__file__).resolve().parent.parent


def test_the_stored_language_reference_is_the_generated_one():
    assert (ROOT / "workflows" / "language_reference.txt").read_text().rstrip("\n") == dsl.reference().rstrip("\n")


def test_the_stored_json_schema_is_the_generated_one():
    assert json.loads((ROOT / "workflow.schema.json").read_text()) == dsl.schema()
