"""A second, independent evaluator of workflow policy.

It reads a workflow file with `yaml.safe_load` (only `true` and `false` are booleans, so `on` stays a word), and implements who-terms, the first-matching access rule, version selectors, and a small
interpreter of a document's history, from the written language reference (the workflow language of the paper). It imports nothing from
`docflow`: not the Access policy engine, not the who-term evaluator, not the history code. It never asks the system what state a document is in.

It derives from the same YAML as the policy under test: it can detect faults in the system and policy mutants, but it cannot tell whether
the policy as written matches the regulation. The workflows and scenario tables cover that.
"""

from .policy import Policy, load_policy
from .doc import Doc, Decision, Person, Cast

__all__ = ["Policy", "load_policy", "Doc", "Decision", "Person", "Cast"]
