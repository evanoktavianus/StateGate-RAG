"""The world of E7: for each workflow, the 1,024 documents of the synthetic set replayed into that workflow
alone (so each workflow has its own states for the same document ids), the corpus text of `eval/data/documents`, and the planted passages loaded into copies of the corpus.

Corpus variants, per workflow: `main` (the passages planted for the workflow's planted prompts) and `alt` (main, with the second prompt that shares a carrier passage with an
earlier one: its text replaces the first's). A planted passage replaces the whole passage (the carrier's own text with the injection in it), keyword statistics included.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from docflow.stores.search_index import Corpus, words

from .. import corpus as C
from ..oracle import Cast, Doc
from ..oracle.policy import load_policy

ROOT = Path(__file__).resolve().parents[3]


def fork(corpus: Corpus) -> Corpus:
    c = copy.copy(corpus)
    c.passages, c._tf, c._len, c._df = list(corpus.passages), list(corpus._tf), list(corpus._len), Counter(corpus._df)
    c._matrix = c._norms = None
    return c


def plant(c: Corpus, items: list[dict]) -> Corpus:
    """Replaces the passage (doc, version, seq) of each item by its text, in the index too."""
    for it in items:
        rows = c.by_version[(it["doc"], it["version"])]
        i = next(r for r in rows if c.passages[r].seq == it["seq"])
        old = c.passages[i]
        for w in c._tf[i]:
            c._df[w] -= 1
            if c._df[w] <= 0:
                del c._df[w]
        new = replace(old, content=it["text"])
        tf = Counter(words(new.breadcrumb + " " + new.content))
        c._df.update(tf.keys())
        c._tf[i], c._len[i], c.passages[i] = tf, sum(tf.values()), new
    c._avg = sum(c._len) / max(len(c._len), 1)
    return c


class WfOracle:
    """What each person may read of each document of one workflow's world, by the independent oracle (serve = opens, find = is shown by a search)."""

    def __init__(self, built, workflow: str) -> None:
        self.b = built
        people = {}
        for sp in built.specs.values():
            people.update(sp.people)
        self.cast = Cast.from_people(people)
        self.people = people
        first = next(iter(built.specs.values()))
        self.policy = load_policy(first.workflow_path)
        self._doc: dict = {}

    def exists(self, doc: str) -> bool:
        return doc in self.b.placed

    def _od(self, doc: str):
        if doc not in self._doc:
            spec, cell = self.b.placed[doc]
            ws = spec.workspaces[spec.workspace]
            assign = {k: v for k, v in spec.assign.items() if self.policy.create and k in (self.policy.create.get("assign") or {})}
            od = Doc.create(self.policy, self.cast, ws, ws.get("org"), doc, spec.owner, spec.creator, assign or None, cell.attrs or None, spec.invited)
            for s in cell.path.steps[: cell.index]:
                inp = C.rebind(s.input, doc, od.version) if s.input else s.input
                if not od.apply(s.actor, s.action, inp, s.event).allowed:
                    raise ValueError(f"the oracle refuses a step of {doc}")
            self._doc[doc] = od
        return self._doc[doc]

    def access(self, doc: str, person: str, find: bool) -> int | None:
        return self._od(doc).access(person, find=find) if self.exists(doc) else None


class Shared:
    def __init__(self, prompts_path: Path | None = None, data: Path = ROOT / "eval/data/documents", only: list[str] | None = None) -> None:
        self.prompts = json.loads((prompts_path or ROOT / "eval" / "data" / "injections" / "prompts.json").read_text())["prompts"]
        wfs = sorted({p["workflow"] for p in self.prompts})
        if only:
            wfs = [w for w in wfs if w in only]
        docs = json.loads((Path(data) / "docs.json").read_text())["docs"]
        base = Corpus.load(data, embeddings=False)
        C._check_versions = lambda w, c: None  # that check reads the corpus text; here the corpus is loaded separately
        self.worlds, self.oracle, self.corpus = {}, {}, {}
        self.alt_ids: set[str] = set()
        for wf in wfs:
            b = C.build(docs, None, workflows=[wf])
            self.worlds[wf], self.oracle[wf] = b.worlds[wf], WfOracle(b, wf)
            main, alt, seen = [], [], set()
            for p in sorted((p for p in self.prompts if p["workflow"] == wf and p["placement"] == "planted"), key=lambda p: p["id"]):
                pl = p["planted"]
                key = (pl["doc"], pl["version"], pl["seq"])
                if key in seen:
                    alt.append(pl)
                    self.alt_ids.add(p["id"])
                else:
                    main.append(pl)
                    seen.add(key)
            m = plant(fork(base), main)
            self.corpus[(wf, "main")] = m
            self.corpus[(wf, "alt")] = plant(fork(m), alt)
        self.base = base
