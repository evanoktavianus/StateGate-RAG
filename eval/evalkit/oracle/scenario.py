"""The scenario tables, run through the oracle instead of the system. The oracle is checked against them first: where it
disagrees with a table, the oracle is wrong and is fixed. Uses only the oracle and `yaml.safe_load`."""

from __future__ import annotations

from pathlib import Path

import yaml

from .doc import Cast, Doc, Decision
from .policy import Loader, load_policy


def run(path: str | Path, workflows_dir: str | Path) -> list[str]:
    """Returns the disagreements between the scenario's expectations and the oracle's answers (empty: they agree)."""
    sc = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=Loader)
    policies = {n: load_policy(Path(workflows_dir) / f"{n}.yml") for n in sc["workflows"]}
    cast = Cast.from_people(sc["people"])
    workspaces = sc.get("workspaces", {})
    docs: dict[str, Doc] = {}
    bad: list[str] = []
    checks = 0

    def create(a: dict):
        ws = workspaces.get(a.get("workspace", ""), {})
        return Doc.create(policies[a["workflow"]], cast, ws, ws.get("org"), a["id"], a["owner"], a.get("creator"), a.get("assign"), a.get("attrs"), a.get("invited", ()))

    def do(step: dict) -> Decision:
        (kind, a), = step.items()
        if kind == "add_document":
            made = create(a)
            if isinstance(made, Decision):
                return made
            docs[a["id"]] = made
            return Decision(True)
        d = docs[a["doc"]]
        if kind == "upload":
            return d.apply(a["by"], "upload")
        if kind == "act":
            return d.apply(a["by"], a["action"], a.get("input") or {})
        if kind == "event":
            return d.apply(None, a["action"], event=a["event"])
        raise ValueError(kind)

    for i, step in enumerate(sc["steps"], 1):
        (kind, a), = step.items()
        if kind == "expect_refused":
            checks += 1
            got = do(a["step"])
            if got.allowed:
                bad.append(f"step {i}: the oracle accepts {a['step']}, the table says it is refused")
            elif a.get("reason") not in (None, got.reason):
                bad.append(f"step {i}: refused with {got.reason}, the table says {a['reason']}")
        elif kind == "expect_state":
            checks += 1
            if docs[a["doc"]].state != a["state"]:
                bad.append(f"step {i}: {a['doc']} is in {docs[a['doc']].state}, the table says {a['state']}")
        elif kind == "expect_log":
            pass  # what the system logged is not the oracle's to say
        elif kind == "expect_access":
            letters_of = lambda v: set(v) if isinstance(v, (list, str)) else set()
            find = a.get("find", a["table"])
            for person, letters in a["table"].items():
                for letter, doc_id in a["docs"].items():
                    d = docs[doc_id]
                    opened, found = d.access(person), d.access(person, find=True)
                    checks += 2
                    if (opened is not None) != (letter in letters_of(letters)):
                        bad.append(f"step {i}: {person} opens {doc_id}: {opened}, the table says {letter in letters_of(letters)}")
                    if (found is not None) != (letter in letters_of(find.get(person, letters))):
                        bad.append(f"step {i}: {person} finds {doc_id}: {found}, the table says {letter in letters_of(find.get(person, letters))}")
                    want = a.get("versions", {}).get(letter)
                    if want is not None and opened is not None and opened != want:
                        bad.append(f"step {i}: {person} opens {doc_id} at version {opened}, the table says {want}")
                    if want is not None and found is not None and found != want:
                        bad.append(f"step {i}: {person} finds {doc_id} at version {found}, the table says {want}")
                    if opened is not None and letter in a.get("guidance", {}) and d.guidance(person) != a["guidance"][letter]:
                        bad.append(f"step {i}: guidance of {doc_id} for {person} is {d.guidance(person)!r}, the table says {a['guidance'][letter]!r}")
        elif kind == "event":
            do(step)  # an event the state does not leave through is rejected and logged, not an error
        else:
            got = do(step)
            if not got.allowed:
                bad.append(f"step {i}: the oracle refuses {step} ({got.reason}), the scenario does it")
    return bad
