"""The same-topic documents (`eval/data/same_topic_memos/`) joined to the synthetic set, and the plan of the question set (`evalkit.qplan`).

    python -m evalkit.known --data eval/data/documents --topics eval/data/same_topic_memos --out work/corpus --plan eval/data/questions/plan.json

`load(...)` gives the corpus (the synthetic set with the same-topic documents added), the manifest
that `evalkit.corpus.build` replays, and `need`, the newest version each known-answer question asks of a document. `placement` says, for every question,
where its document is (workflow, unit, state), whether the version it asks for exists in the replayed world, and who opens exactly that version
according to the system and according to the independent oracle. Retrieval side only: no model call, no network.
"""

from __future__ import annotations

import array
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from docflow.stores.search_index import DIMS, Corpus, Passage

from . import corpus as C
from . import paths as paths_mod
from .oracle import Cast, Doc
from .oracle.policy import load_policy


def _rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load(data: Path, topics: Path, embeddings: bool = False) -> tuple[Corpus, list[dict]]:
    """(corpus, manifest). The corpus is the synthetic set with the same-topic documents added after it.
    With `embeddings` the vectors are loaded too (the topic vectors only if `eval/data/same_topic_memos/embeddings.f32` exists)."""
    data, topics = Path(data), Path(topics)
    plain = data / "chunks.jsonl"
    passages = [Passage(r["doc"], r["version"], r["seq"], tuple(r["pages"]), r["breadcrumb"], r["content"], r["row"])
                for r in _rows(plain if plain.exists() else data / "chunks.jsonl.gz")]
    vecs = None
    if embeddings:
        vecs = array.array("f")
        vecs.frombytes((data / "embeddings.f32").read_bytes())
    tdocs = list(_rows(topics / "docs.jsonl"))
    for r in _rows(topics / "chunks.jsonl.gz"):
        passages.append(Passage(r["doc"], r["version"], r["seq"], tuple(r["pages"]), r["breadcrumb"], r["content"], len(passages)))
    if vecs is not None and (topics / "embeddings.f32").exists():
        vecs.frombytes((topics / "embeddings.f32").read_bytes())
    corpus = Corpus(passages, vecs)
    base = json.loads((data / "docs.json").read_text())["docs"]
    manifest = [{"id": d["id"], "unit": d["unit"], "n": d["n"]} for d in base]
    versions = defaultdict(set)
    for d in tdocs:
        versions[d["id"]].add(d["version"])
    for doc_id, vs in sorted(versions.items()):
        t = next(d for d in tdocs if d["id"] == doc_id)
        manifest.append({"id": doc_id, "unit": t["unit"], "n": max(vs), "kind": "topic", "topic_index": int(doc_id[2:4]), "topic": t["topic"]})
    return corpus, manifest


def _oracle_doc(b: C.Built, wf: str, doc_id: str):
    world = b.worlds[wf]
    people = {pid: {"org": u.org, "unit": u.unit, "groups": list(u.groups), "positions": list(u.positions), "attrs": dict(u.attrs), "org_admin": u.org_admin}
              for pid, u in world.users.items()}
    cast = Cast.from_people(people)
    policy = load_policy(paths_mod.generate(wf).spec.workflow_path)
    spec, cell = b.placed[doc_id]
    ws = spec.workspaces[spec.workspace]
    create = {k: v for k, v in spec.assign.items() if policy.create and k in (policy.create.get("assign") or {})}
    od = Doc.create(policy, cast, ws, ws.get("org"), doc_id, spec.owner, spec.creator, create or None, cell.attrs or None, spec.invited)
    for s in cell.path.steps[: cell.index]:
        if not od.apply(s.actor, s.action, C.rebind(s.input, doc_id, od.version), s.event).allowed:
            raise ValueError(f"the oracle refuses a step of {doc_id}: {s}")
    return od, list(cast)


def placement(b: C.Built, questions: list[dict]) -> list[dict]:
    """For each known-answer question: its document, the version asked, where the document is, whether that version exists in the world,
    and who opens exactly that version (system and oracle; `agree` when the two lists are the same)."""
    by_doc = {a["doc"]: a for a in b.assignment}
    cache: dict[str, tuple] = {}
    rows = []
    for q in questions:
        a = by_doc[q["doc"]]
        world = b.worlds[a["workflow"]]
        have = world.documents[q["doc"]].timeline.version
        if q["doc"] not in cache:
            od, cast = _oracle_doc(b, a["workflow"], q["doc"])
            cache[q["doc"]] = (od, cast)
        od, cast = cache[q["doc"]]
        system = sorted(p for p in cast if world.readable_version(q["doc"], p, for_search=False) == q["version"])
        oracle = sorted(p for p in cast if od.access(p) == q["version"])
        rows.append({"question": q["id"], "doc": q["doc"], "version": q["version"], "workflow": a["workflow"], "unit": a["unit"], "state": a["state"],
                     "versions_in_world": have, "version_exists": q["version"] <= have, "readers_system": system, "readers_oracle": oracle,
                     "agree": system == oracle})
    return rows


def report_topics(b: C.Built) -> str:
    topic = [a for a in b.assignment if a.get("kind") == "topic"]
    by_topic = defaultdict(list)
    for a in topic:
        by_topic[a["doc"][:4]].append(a)
    states_per_topic = Counter(len({a["state"] for a in v}) for v in by_topic.values())
    wf = Counter(a["workflow"] for a in topic)
    out = ["## Same-topic documents (`eval/data/same_topic_memos/`)", "",
           f"{len(topic)} documents (60 topics x 4 units), placed in the unit they were written for; a topic's four documents are in one workflow and in different states.", "",
           "| distinct states among a topic's four documents | topics |", "|---|---|"] + [f"| {k} | {n} |" for k, n in sorted(states_per_topic.items())]
    out += ["", "| workflow | topic documents |", "|---|---|"] + [f"| {w} | {n} |" for w, n in sorted(wf.items())]
    return "\n".join(out)


if __name__ == "__main__":
    import datetime, os, platform, sys, time

    from . import qplan

    args = sys.argv[1:]
    opt = lambda k, d: Path(args[args.index(k) + 1]) if k in args else Path(d)
    t0 = time.time()
    data, topics = opt("--data", "eval/data/documents"), opt("--topics", "eval/data/same_topic_memos")
    docs = json.loads((data / "docs.json").read_text())["docs"]
    chunks = qplan.chunks_by_version(data)
    pairs = qplan.version_pairs(chunks, docs)
    if "--assignment" in args:  # the replay was made already: plan from its assignment
        built, manifest = None, []
        assignment = json.loads(opt("--assignment", "").read_text())["assignment"]
    else:
        corpus, manifest = load(data, topics)
        built = C.build(manifest, corpus, need=qplan.need_of(pairs))
        assignment = built.assignment
        text = C.report(built) + "\n\n" + report_topics(built)
        print(text)
    if "--out" in args and built is not None:
        base = opt("--out", "work/corpus")
        Path(str(base) + ".md").write_text(
            "# The synthetic set and the same-topic documents in the corpus replay\n\n"
            f"Commit `{os.environ.get('EVAL_COMMIT', 'unknown')}`; {datetime.date.today()}; machine {platform.node()}; command `python -m evalkit.known`; run time {time.time() - t0:.0f} s; "
            "no seed (deterministic). Retrieval side only: no model call, no network. These are checks of placement, not evaluation results.\n\n" + text + "\n")
        Path(str(base) + ".json").write_text(json.dumps({"coverage": built.coverage(), "assignment": built.assignment}, indent=0))
    if "--plan" in args:
        docinfo = {d["id"]: {"lang": d["lang"], "topic": d["topic"]} for d in docs}
        slots, notes = qplan.allocate(assignment, pairs, chunks, docinfo)
        out = opt("--plan", "eval/data/questions/plan.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"slots": slots, "notes": notes, "version_pairs": len(pairs)}, ensure_ascii=False, indent=0))
        print(f"\n{len(slots)} slots for the synthetic-set workflows; version pairs found: {len(pairs)}; notes: {notes}")
