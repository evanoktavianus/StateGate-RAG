"""E5, retrieval utility with the question set, from the stored vectors only (no model call, no network).

For each content and version question and for each of up to two askers who may read the target, the three searches (words, meaning, both) run on the
asker's own view, as the service runs them (`world.visible(user, for_search=True)`, then the corpus). What is scored, for k = 5 and 10:

  document recall@k     1 if a passage of the target document is among the top k passages (the target is the only relevant document and the asker may read it:
                        that is what "authorized" means here; a question whose target the asker cannot read is not asked of that asker)
  evidence recall@k     1 if a top-k passage of the target document holds the expected answer text (asked only where that text is in some passage of the version the asker reads)
  context precision@k   the share of the k returned passages that are (document) from the target document, (evidence) hold the expected answer text

The expected answer of a content question is its answer; of a version question, the answer of the version the asker reads. Status questions are answered from the
document's state, not from passages, so they are not scored here (counted). The regulation questions are reported on their own: the regulation number helps words.

    python -m evalkit.retrieval_utility --json eval/results/runs/e5/utility.json --md eval/results/runs/e5/utility.md [--workers 4] [--limit N] [--no-filter]
    python -m evalkit.retrieval_utility --show eval/results/runs/e5/utility.json     # the tables of a stored result, nothing is run
"""

from __future__ import annotations

import array
import json
import re
import sys
import time
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

from docflow import Machine
from docflow.document_service import retrieve
from docflow.stores.search_index import DIMS, Corpus
from docflow.support import regulations as R

from . import corpus as C
from . import known, qplan
from .report import wilson_upper

ROOT = Path(__file__).resolve().parents[2]
KS = (5, 10)
MODES = ("keyword", "semantic", "hybrid")
G: dict = {}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def question_vectors() -> dict[tuple, list[float]]:
    index = json.loads((ROOT / "eval/data/questions" / "embeddings_index.json").read_text())
    raw = array.array("f")
    raw.frombytes((ROOT / "eval/data/questions" / "embeddings.f32").read_bytes())
    return {(r["id"], r["role"]): list(raw[i * DIMS:(i + 1) * DIMS]) for i, r in enumerate(index)}


def setup(data: Path = ROOT / "eval/data/documents", topics: Path = ROOT / "eval/data/same_topic_memos", reg: Path = ROOT / "eval/data/regulations") -> None:
    questions = json.loads((ROOT / "eval/data/questions" / "questions.json").read_text())
    G["questions"] = questions
    G["vec"] = question_vectors()
    docs = json.loads((data / "docs.json").read_text())["docs"]
    corpus, manifest = known.load(data, topics, embeddings=True)
    if len(corpus._embeddings) != len(corpus.passages) * DIMS:
        raise ValueError("the vector files do not hold one vector for every passage: join them again with scripts/join_vectors.py")
    G["alignment"] = {"topic_passages": len(corpus.passages) - (data / "embeddings.f32").stat().st_size // (DIMS * 4), "vectors_one_per_chunk": True}
    pairs = qplan.version_pairs(qplan.chunks_by_version(data), docs)
    built = C.build(manifest, corpus, need=qplan.need_of(pairs))
    G["syn"] = (built.worlds, corpus, {a["doc"]: a for a in built.assignment})
    G["built"] = built
    rplan, rdocs = R.load(reg)
    rworld = R.build_world({"regulation_single": Machine.load(ROOT / "workflows" / "regulation_single.yml")}, rplan, rdocs)
    G["reg"] = (rworld, Corpus.load(reg, embeddings=True), {r["id"]: r for r in rplan["regulations"]})
    G["reg_plan"], G["reg_docs"] = rplan, rdocs


def served_texts(corpus: Corpus, doc: str, version: int) -> list[str]:
    return [norm(p.content) for p in corpus.passages_of(doc, version)]


def askers(q: dict) -> list[tuple[str, str, object, dict]]:
    """[(category, user, world, corpus)] — up to two askers who may read the target document."""
    if q["workflow"] == "regulation_single":
        world, corpus, regs = G["reg"]
        reg = regs[q["doc"].split("/")[0]]
        found = []
        for cat, group in (("same unit", ("lead", "team", "staff")), ("outside the unit", ("inv", "emp", "cit", "xorg"))):
            for persona in group:
                u = R.persona_user(reg, persona)
                if u and q["doc"] in world.visible(u, for_search=True):
                    found.append((cat + ":" + persona, u, world, corpus))
                    break
        return found
    worlds, corpus, assignment = G["syn"]
    world = worlds[q["workflow"]]
    a = assignment[q["doc"]]
    org = a["organization"]
    found, seen = [], set()
    for u in sorted(world.users):
        usr = world.users[u]
        cat = "same unit" if usr.unit == a["unit"] else "another unit of the same organization" if usr.org == org else "another organization" if usr.org else "outsider"
        if cat in seen or q["doc"] not in world.visible(u, for_search=True):
            continue
        seen.add(cat)
        found.append((cat, u, world, corpus))
    order = ["same unit", "another unit of the same organization", "another organization", "outsider"]
    return sorted(found, key=lambda f: order.index(f[0]))[:2]


def expected(q: dict, corpus: Corpus, served: int) -> str | None:
    if q["kind"] == "content":
        return q["answer"]
    return q["answer_a"] if served == q["pair"]["a"] else q["answer_b"] if served == q["pair"]["b"] else None


NO_FILTER = False  # set by `--no-filter`: every search reads every version of every document of the corpus, what an unguarded index would return


class _Everything:
    """In place of the dict of versions a person may read: every (document, version) of the corpus. The search reads only `items()`."""

    def __init__(self, corpus) -> None:
        self._items = [(d, v) for (d, v) in corpus.by_version]

    def items(self):
        return self._items


_ALL: dict = {}


def _everything(corpus) -> _Everything:
    if id(corpus) not in _ALL:
        _ALL[id(corpus)] = _Everything(corpus)
    return _ALL[id(corpus)]


def one(i: int) -> list[dict]:
    q = G["questions"][i]
    out = []
    if q["kind"] == "status":
        return out
    forms = [("original", q["text"], "question")] + [("paraphrase", t, f"paraphrase.{k}") for k, t in enumerate(q["paraphrases"])]
    for cat, user, world, corpus in askers(q):
        vis = world.visible(user, for_search=True)
        corpus = _index(q, world, corpus)  # PostgreSQL when DATABASE_URL is set, else the in-memory corpus
        served = vis[q["doc"]]
        want = expected(q, corpus, served)
        has_text = bool(want) and any(norm(want) in t for t in served_texts(corpus, q["doc"], served))
        for form, text, role in forms:
            vec = G["vec"][(q["id"], role)]
            scope = _everything(corpus) if NO_FILTER else vis
            words = corpus.search(text, scope, retrieve.HYBRID_DEPTH)
            meaning = corpus.similar(vec, scope, retrieve.HYBRID_DEPTH)
            ranked = {"keyword": words, "semantic": meaning, "hybrid": retrieve.fuse(words, meaning)}
            for mode in MODES:
                hits = ranked[mode]
                row = {"id": q["id"], "kind": q["kind"], "regulation": q["workflow"] == "regulation_single", "workflow": q["workflow"], "asker": cat, "form": form, "role": role, "mode": mode, "evidence_scored": has_text}
                for k in KS:
                    top = hits[:k]
                    doc_hits = [p for _, p in top if p.doc == q["doc"]]
                    ev_hits = [p for p in doc_hits if has_text and norm(want) in norm(p.content)]
                    if NO_FILTER and k == KS[-1]:  # a document or version in the top k that the asker may not read
                        row[f"hidden_in_top{k}"] = any(vis.get(p.doc) != p.version for _, p in top)
                    row[f"doc_recall@{k}"] = bool(doc_hits)
                    row[f"doc_precision@{k}"] = len(doc_hits) / k
                    if has_text:
                        row[f"evidence_recall@{k}"] = bool(ev_hits)
                        row[f"evidence_precision@{k}"] = len(ev_hits) / k
                out.append(row)
    return out


_INDEX: dict = {}


def _index(q: dict, world, corpus):
    """The index the searches of this question read: the PostgreSQL index of the world (one per workflow, loaded once per process) or the corpus."""
    from . import pgstore
    if not pgstore.enabled():
        return corpus
    key = q["workflow"]
    if (key, id(world)) not in _INDEX:
        _INDEX[(key, id(world))] = pgstore.LazyIndex(world, corpus, key)
    return _INDEX[(key, id(world))]


def run(workers: int = 4, limit: int | None = None) -> dict:
    t0 = time.time()
    setup()
    n = min(len(G["questions"]), limit) if limit else len(G["questions"])
    with mp.get_context("fork").Pool(workers) as pool:  # forked after the setup: the worlds and vectors are shared, not copied (macOS starts processes by `spawn`, which would not have them)
        parts = pool.map(one, range(n), chunksize=8)
    rows = [r for p in parts for r in p]
    status = sum(q["kind"] == "status" for q in G["questions"])
    return {"alignment": G["alignment"], "rows": rows, "questions": n, "status_not_scored": status, "seconds": round(time.time() - t0, 1)}


def _pg_on() -> bool:
    from . import pgstore
    return pgstore.enabled()


def aggregate(rows: list[dict], keyf, where=lambda r: True) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if not where(r):
            continue
        a = agg[keyf(r)]
        a["pairs"] += 1
        for k in KS:
            a[f"doc_recall@{k}"] += r[f"doc_recall@{k}"]
            a[f"doc_precision@{k}"] += r[f"doc_precision@{k}"]
            if f"hidden_in_top{k}" in r:
                a[f"hidden_in_top{k}"] += r[f"hidden_in_top{k}"]
            if f"evidence_recall@{k}" in r:
                a[f"evidence_n"] += k == KS[0]
                a[f"evidence_recall@{k}"] += r[f"evidence_recall@{k}"]
                a[f"evidence_precision@{k}"] += r[f"evidence_precision@{k}"]
    return agg


def table(rows: list[dict], title: str, keyf, where=lambda r: True) -> list[str]:
    agg = aggregate(rows, keyf, where)
    hidden = any(f"hidden_in_top{KS[-1]}" in r for r in rows)
    head = ["group", "queries (n)", "doc recall@5", "doc recall@10", "doc precision@5", "doc precision@10", "queries with evidence text", "evidence recall@5", "evidence recall@10", "evidence precision@5", "evidence precision@10"] + (["hidden in top 10"] if hidden else [])
    out = [f"**{title}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, a in sorted(agg.items()):
        n, e = a["pairs"], a["evidence_n"]
        f = lambda x, d: f"{x / d:.3f}" if d else "-"
        out.append("| " + " | ".join([" / ".join(key) if isinstance(key, tuple) else str(key), str(int(n)), f(a["doc_recall@5"], n), f(a["doc_recall@10"], n), f(a["doc_precision@5"], n),
                                      f(a["doc_precision@10"], n), str(int(e)), f(a["evidence_recall@5"], e), f(a["evidence_recall@10"], e), f(a["evidence_precision@5"], e), f(a["evidence_precision@10"], e)] + ([f(a[f"hidden_in_top{KS[-1]}"], n)] if hidden else [])) + " |")
    return out + [""]


def to_markdown(res: dict) -> str:
    rows = res["rows"]
    pairs = len({(r["id"], r["asker"]) for r in rows})
    out = ["# E5: retrieval utility with the question set (stored vectors, no model call)", "",
           f"{res['questions']} questions; {res['status_not_scored']} status questions are answered from the document's state and are not scored here; {len({r['id'] for r in rows})} content and version questions were asked of "
           f"{pairs} (question, asker) pairs, each as the original question and as its three paraphrases (reported separately), each with three searches. {res['seconds']} s. "
           f"Topic-passage vectors: {res['alignment']}.", "",
           "Context precision is in every row: *doc precision@k* is the share of the k returned passages that come from the target document, *evidence precision@k* the share that hold the expected answer text.", ""]
    syn, reg = (lambda r: not r["regulation"]), (lambda r: r["regulation"])
    out += table(rows, "Main table, synthetic-set questions: query form and search mode", lambda r: (r["form"], r["mode"]), syn)
    out += table(rows, "Regulation questions, reported separately: query form and search mode", lambda r: (r["form"], r["mode"]), reg)
    out += table(rows, "Synthetic-set questions, by kind, form and mode", lambda r: (r["kind"], r["form"], r["mode"]), syn)
    out += table(rows, "Regulation questions, by kind, form and mode", lambda r: (r["kind"], r["form"], r["mode"]), reg)
    out += table(rows, "Synthetic-set questions, by workflow and mode (original questions)", lambda r: (r["workflow"], r["mode"]), lambda r: syn(r) and r["form"] == "original")
    out += table(rows, "By who asks and mode (original questions, all)", lambda r: (r["asker"].split(":")[0], r["mode"]), lambda r: r["form"] == "original")
    return "\n".join(out)


def main(argv: list[str]) -> None:
    global NO_FILTER
    if "--show" in argv:
        print(to_markdown(json.loads(Path(argv[argv.index("--show") + 1]).read_text())))
        return
    NO_FILTER = "--no-filter" in argv
    argv = [a for a in argv if a != "--no-filter"]
    if NO_FILTER and _pg_on():
        sys.exit("--no-filter reads the in-memory corpus: unset DATABASE_URL or set DOCFLOW_STORE=memory")
    args = dict(zip(argv[::2], argv[1::2]))
    res = run(int(args.get("--workers", 4)), int(args["--limit"]) if "--limit" in args else None)
    if "--json" in args:
        Path(args["--json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(args["--json"]).write_text(json.dumps(res))
    text = to_markdown(res)
    if "--md" in args:
        Path(args["--md"]).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
