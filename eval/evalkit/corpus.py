"""The synthetic set replayed under the workflows of the paper, over units of two organizations, with real actions.

Each document of the set (`eval/data/documents/docs.json`, with its text in the corpus) keeps the unit it has in the set. The units are spread over two
organizations: U1, U2 and U3 in O1, U4 in O2 (U5 and U6 of O2 hold people and no documents, so that every cast has a unit besides its own). For each
workflow and each unit the fixture cast is copied into that unit (`unit_spec`): the owner's unit becomes the unit, the other units of the owner's
organization become its sibling units, other organizations become the other organization, group names that only label an attribute value are suffixed
with the unit (in lower case), and attribute values that name units follow the units.

The documents are put in every *cell* of every workflow: a state together with the attribute values, the choices (taken from the paths
of `evalkit.paths`), once in every unit, and the rest of the documents are spread over the cells. A document is driven there with the steps of
the path, through `World.act` and `World.fire_event`. The text of version k of a document is its own version k in the corpus.

    docs = json.load(open("eval/data/documents/docs.json"))["docs"]; corpus = Corpus.load("eval/data/documents", embeddings=False)
    built = build(docs, corpus)          # {workflow: World}, the assignment, the coverage
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from docflow.stores.search_index import Corpus
from docflow.stores.world import World

from . import paths as paths_mod
from .worlds import Spec, build_cast, create_document, load_spec

UNITS = {"O1": ["U1", "U2", "U3"], "O2": ["U4", "U5", "U6"]}
ORG_OF = {u: o for o, us in UNITS.items() for u in us}
PLACEMENT = Path(__file__).resolve().parents[1] / "data" / "placement.json"  # where each document of the set sits in the runs of the paper
DOC_UNITS = ["U1", "U2", "U3", "U4"]  # the units the set puts documents in


def unit_spec(spec: Spec, unit: str) -> Spec:
    """The fixture of a workflow, moved into `unit`: every person's id gets the unit as a prefix (`U2.analyst`)."""
    owner = spec.people[spec.owner]
    o0, u0 = owner.get("org"), owner.get("unit")
    org = ORG_OF[unit]
    other_org = "O2" if org == "O1" else "O1"
    siblings = [u for u in UNITS[org] if u != unit]
    foreign = UNITS[other_org]
    umap, omap = {u0: unit}, {o0: org}
    i = j = 0
    for p in spec.people.values():
        u, o = p.get("unit"), p.get("org")
        if o is not None and o not in omap:
            omap[o] = other_org
        if u is None or u in umap:
            continue
        if o == o0:
            umap[u] = siblings[i % len(siblings)]
            i += 1
        else:
            umap[u] = foreign[j % len(foreign)]
            j += 1
    labels = {v for vals in spec.attrs.values() for v in vals} - set(umap)  # free labels such as a group g1
    group_labels = {g for p in spec.people.values() for g in p.get("groups", ()) if g in labels}
    pid = lambda x: f"{unit}.{x}"
    gname = lambda g: f"{g}-{unit.lower()}" if g in group_labels else g  # lower case: the system lower-cases a person's groups, not an attribute value
    people = {pid(k): {**v, "org": omap.get(v.get("org")), "unit": umap.get(v.get("unit")), "groups": [gname(g) for g in v.get("groups", ())]}
              for k, v in spec.people.items()}
    workspaces = {f"{unit}.{k}": {**w, "org": omap.get(w.get("org")), "members": {pid(m): r for m, r in (w.get("members") or {}).items()}}
                  for k, w in spec.workspaces.items()}
    attrs = {k: [umap.get(v, gname(v)) for v in vals] for k, vals in spec.attrs.items()}
    return replace(spec, people=people, workspaces=workspaces, workspace=f"{unit}.{spec.workspace}", owner=pid(spec.owner),
                   creator=pid(spec.creator) if spec.creator else None, invited=[pid(x) for x in spec.invited],
                   assign={k: [pid(x) for x in v] for k, v in spec.assign.items()}, attrs=attrs)


@dataclass
class Cell:
    workflow: str
    unit: str
    path: object  # paths_mod.Path_
    index: int  # steps of the path to take
    state: str
    attrs: dict
    choices: tuple
    version: int  # versions the document has when it gets there

    @property
    def key(self) -> tuple:
        return (self.state, json.dumps(self.attrs, sort_keys=True), self.choices)


def cells_of(ps: paths_mod.PathSet, unit: str) -> list[Cell]:
    """Every distinct (state, attributes, choices) the paths reach, by the shortest way."""
    best: dict[tuple, Cell] = {}
    rich: dict[tuple, Cell] = {}  # the same cell reached with the most versions a path gives it (the shortest such path)
    for p in ps.paths:
        for sit, _ in paths_mod.situations(ps, p):
            c = Cell(ps.workflow, unit, p, sit.index, sit.state, p.attrs, sit.choices, sit.version)
            if c.key not in best or c.index < best[c.key].index:
                best[c.key] = c
            if c.key not in rich or (c.version, -c.index) > (rich[c.key].version, -rich[c.key].index):
                rich[c.key] = c
    extra = [c for k, c in rich.items() if c.version > best[k].version]
    # one richer cell for each state (the one with the most versions), so that a document with a history of versions is in every state that can have one
    top: dict[str, Cell] = {}
    for c in extra:
        if c.state not in top or (c.version, -c.index) > (top[c.state].version, -top[c.state].index):
            top[c.state] = c
    return sorted([*best.values(), *top.values()], key=lambda c: (c.state, json.dumps(c.attrs, sort_keys=True), c.choices, c.version))


@dataclass
class Built:
    worlds: dict[str, World]
    assignment: list[dict]
    specs: dict[tuple, Spec]  # (workflow, unit) -> the fixture moved into the unit
    cells: dict[tuple, list[Cell]]  # (workflow, unit) -> its cells
    placed: dict[str, tuple] = field(default_factory=dict)  # document -> (spec of its unit, the cell it was driven to)

    def coverage(self) -> dict:
        """What the replay holds, by workflow: documents in each state, each attribute value, each unit."""
        out: dict = {"units": {u: ORG_OF[u] for u in sorted({a["unit"] for a in self.assignment})},
                     "units_per_organization": {}, "documents": len(self.assignment), "workflows": {}}
        for u, o in out["units"].items():
            out["units_per_organization"].setdefault(o, []).append(u)
        for wf in sorted({a["workflow"] for a in self.assignment}):
            rows = [a for a in self.assignment if a["workflow"] == wf]
            states = defaultdict(lambda: defaultdict(int))
            attr_values: dict = defaultdict(lambda: defaultdict(int))
            for a in rows:
                states[a["state"]][a["unit"]] += 1
                for k, v in a["attrs"].items():
                    attr_values[k][v] += 1
            out["workflows"][wf] = {"documents": len(rows), "states": {s: dict(u) for s, u in sorted(states.items())},
                                    "attribute_values": {k: dict(v) for k, v in sorted(attr_values.items())}}
        return out


def build(docs: list[dict], corpus: Corpus, workflows: list[str] | None = None, units: list[str] | None = None, need: dict[str, int] | None = None, placement: list[dict] | None = None) -> Built:
    """Replays `docs` (manifest rows with `id`, `unit` and `n`, the number of versions in the corpus) into one world per workflow.

    A row with `"kind": "topic"` (and `topic_index`) is a same-topic document of the paid set; it is placed after the others, by `_place_topics`.
    `need` maps a document to the newest version a known-answer question asks of it: a document is put, where it can be, in a cell that has that version.

    `placement` is the assignment of the documents to workflows (rows as `Built.assignment`: document, workflow, unit, path, step). With the default workflows
    and units and the documents of the paper's set, the rows of `eval/data/placement.json` are used, and a document without a row of a workflow in use is left
    out. `placement=[]` spreads the documents over the workflows, whatever the file says."""
    if placement is None and workflows is None and units is None and PLACEMENT.exists():
        recorded = json.loads(PLACEMENT.read_text())
        if {r["doc"] for r in recorded} <= {d["id"] for d in docs}:
            placement = recorded
    workflows = workflows or paths_mod.WORKFLOWS
    units = units or DOC_UNITS
    need = need or {}
    topics = [d for d in docs if d.get("kind") == "topic"]
    docs = [d for d in docs if d.get("kind") != "topic"]
    by_unit = {u: sorted((d for d in docs if d["unit"] == u), key=lambda d: d["id"]) for u in units}
    sets = {w: paths_mod.generate(w, upload_rule=False) for w in workflows}
    specs: dict[tuple, Spec] = {}
    cells: dict[tuple, list[Cell]] = {}
    pathsets: dict[tuple, paths_mod.PathSet] = {}
    for w in workflows:
        for u in units:
            specs[(w, u)] = unit_spec(sets[w].spec, u)
            pathsets[(w, u)] = paths_mod.generate(w, specs[(w, u)], upload_rule=False)
            cells[(w, u)] = cells_of(pathsets[(w, u)], u)
    worlds: dict[str, World] = {}
    for w in workflows:
        merged = replace(specs[(w, units[0])], people={}, workspaces={})
        for u in units:
            merged.people.update(specs[(w, u)].people)
            merged.workspaces.update(specs[(w, u)].workspaces)
        worlds[w] = build_cast(merged, sets[w].spec.machine())
    assignment: list[dict] = []
    placed: dict[str, tuple] = {}
    if placement:
        rows = {r["doc"]: r for r in placement}
        for u in units:
            for d in by_unit[u]:
                _place_as_recorded(worlds, specs, cells, rows.get(d["id"]), u, d, assignment, placed)
        for d in sorted(topics, key=lambda d: d["id"]):
            _place_as_recorded(worlds, specs, cells, rows.get(d["id"]), d["unit"], d, assignment, placed)
            if assignment and assignment[-1]["doc"] == d["id"]:
                assignment[-1]["kind"] = "topic"
        for w in workflows:
            _check_versions(worlds[w], corpus)
        return Built(worlds, assignment, specs, cells, placed)
    for u in units:
        todo = [(w, c) for w in workflows for c in cells[(w, u)]]  # one document for every cell, longest history first
        todo.sort(key=lambda t: -t[1].version)
        pool = sorted(by_unit[u], key=lambda d: (int(d["n"]), d["id"]))
        used: set[str] = set()
        picks: list[tuple[str, Cell, dict]] = []
        for w, c in todo:
            d = next((d for d in pool if d["id"] not in used and int(d["n"]) >= c.version and need.get(d["id"], 0) <= c.version), None) \
                or next((d for d in pool if d["id"] not in used and int(d["n"]) >= c.version), None)
            if d is None:
                raise ValueError(f"no document of {u} has {c.version} versions for the cell {c.state} of {w}")
            used.add(d["id"])
            picks.append((w, c, d))
        rest = [d for d in pool if d["id"] not in used]  # the others are spread over the cells, each where it has enough versions
        for k, d in enumerate(rest):
            fits = [(w, c) for w, c in todo if int(d["n"]) >= c.version]
            fits = [(w, c) for w, c in fits if c.version >= need.get(d["id"], 0)] or fits  # where it can, a cell that has the version asked of it
            w, c = fits[k % len(fits)]
            picks.append((w, c, d))
        for w, c, d in sorted(picks, key=lambda t: t[2]["id"]):
            _place(worlds[w], specs[(w, u)], c, d, assignment)
            placed[d["id"]] = (specs[(w, u)], c)
    _place_topics(worlds, specs, cells, topics, sets, assignment, placed)
    for w in workflows:
        _check_versions(worlds[w], corpus)
    return Built(worlds, assignment, specs, cells, placed)


def _place_as_recorded(worlds, specs, cells, row: dict | None, unit: str, doc: dict, assignment: list[dict], placed: dict) -> None:
    """Put `doc` where `row` says (the same path, the same step), if its workflow is in use."""
    if row is None or row["workflow"] not in worlds:
        return
    w = row["workflow"]
    cell = next((c for c in cells[(w, unit)] if c.path.id == row["path"] and c.index == row["steps"]), None)
    if cell is None:
        raise ValueError(f"{doc['id']}: no cell of {w} in {unit} at step {row['steps']} of path {row['path']}")
    _place(worlds[w], specs[(w, unit)], cell, doc, assignment)
    placed[doc["id"]] = (specs[(w, unit)], cell)


def _place_topics(worlds, specs, cells, topics: list[dict], sets: dict, assignment: list[dict], placed: dict) -> None:
    """The same-topic documents of the paid set: a topic's four documents (one per unit) are in one workflow, in the unit they were written for, and
    in different states (the state index moves with the unit), so that a search for the topic meets the same subject in several units and several states."""
    wfs = [w for w in sets if len(sets[w].model.states) >= 4]
    for d in sorted(topics, key=lambda d: d["id"]):
        t, u = int(d["topic_index"]), d["unit"]
        w = wfs[t % len(wfs)]
        eligible = [c for c in cells[(w, u)] if c.version <= int(d["n"])]
        order = list(sets[w].model.states)
        states = sorted({c.state for c in eligible}, key=order.index)
        state = states[(t + DOC_UNITS.index(u)) % len(states)]
        c = next(c for c in eligible if c.state == state)
        _place(worlds[w], specs[(w, u)], c, d, assignment)
        assignment[-1]["kind"] = "topic"
        placed[d["id"]] = (specs[(w, u)], c)


def rebind(inp: dict, doc_id: str, version: int) -> dict:
    """A recorded input with its signature bound to this document's current version (the path was recorded on another document)."""
    if "signature" not in inp:
        return inp
    return {**inp, "signature": {**inp["signature"], "digest": f"{doc_id}@{version}"}}


def _place(world: World, spec: Spec, cell: Cell, doc: dict, assignment: list[dict]) -> None:
    create_document(world, spec, cell.attrs, doc["id"])
    for step in cell.path.steps[: cell.index]:
        if step.event is not None:
            if world.fire_event(doc["id"], step.action, step.event) is None:
                raise ValueError(f"{doc['id']}: the event {step.action} was rejected")
        else:
            world.act(doc["id"], step.actor, step.action, input=rebind(step.input, doc["id"], world.documents[doc["id"]].timeline.version))
    tl = world.documents[doc["id"]].timeline
    assert (tl.state, tl.version) == (cell.state, cell.version), (doc["id"], tl.state, tl.version, cell.state, cell.version)
    assignment.append({"doc": doc["id"], "workflow": cell.workflow, "unit": cell.unit, "organization": ORG_OF[cell.unit], "state": cell.state,
                       "attrs": cell.attrs, "choices": dict(cell.choices), "versions": cell.version, "path": cell.path.id, "steps": cell.index})


def _check_versions(world: World, corpus: Corpus) -> None:
    """Every version a document has in the world has its text in the corpus."""
    for doc in world.documents.values():
        for v in range(1, doc.timeline.version + 1):
            if not corpus.by_version.get((doc.id, v)):
                raise ValueError(f"{doc.id} version {v} has no text in the corpus")


def report(b: Built) -> str:
    cov = b.coverage()
    out = ["| workflow | documents | states with documents (documents per unit U1/U2/U3/U4) |", "|---|---|---|"]
    for wf, c in cov["workflows"].items():
        cells = "; ".join(f"{s}: " + "/".join(str(u.get(x, 0)) for x in DOC_UNITS) for s, u in c["states"].items())
        out.append(f"| {wf} | {c['documents']} | {cells} |")
    out += ["", "Attribute values, with the documents that carry each:", ""]
    for wf, c in cov["workflows"].items():
        for k, vals in c["attribute_values"].items():
            out.append(f"- {wf}, `{k}`: " + ", ".join(f"{v} ({n})" for v, n in sorted(vals.items())))
    levels = defaultdict(int)
    for a in b.assignment:
        if a["workflow"] == "knowledge_asset" and "level" in a["choices"]:
            levels[a["choices"]["level"]] += 1
    out.append("- knowledge_asset, choice `level`: " + ", ".join(f"{k} ({n})" for k, n in sorted(levels.items())))
    out += ["", "Units and organizations: " + "; ".join(f"{o}: {', '.join(us)}" for o, us in cov["units_per_organization"].items()) + ".",
            f"Documents: {cov['documents']}."]
    return "\n".join(out)


if __name__ == "__main__":
    import datetime, os, platform, sys, time
    from pathlib import Path

    args = sys.argv[1:]
    data = Path(args[args.index("--data") + 1]) if "--data" in args else Path("eval/data/documents")
    t0 = time.time()
    docs = json.loads((data / "docs.json").read_text())["docs"]
    built = build(docs, Corpus.load(data, embeddings=False))
    text = report(built)
    print(text)
    if "--out" in args:
        base = Path(args[args.index("--out") + 1])
        Path(str(base) + ".md").write_text(
            "# The synthetic set replayed under the workflows of the paper\n\n"
            f"Commit `{os.environ.get('EVAL_COMMIT', 'unknown')}`; {datetime.date.today()}; machine {platform.node()}; command `python -m evalkit.corpus --data eval/data/documents --out eval/results/05-corpus-replay`; "
            f"run time {time.time() - t0:.0f} s; no seed (deterministic). Retrieval-side only: no model call.\n\n" + text + "\n")
        Path(str(base) + ".json").write_text(json.dumps({"coverage": built.coverage(), "assignment": built.assignment}, indent=0))
