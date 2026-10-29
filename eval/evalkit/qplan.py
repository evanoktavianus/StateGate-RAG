"""The plan of the question set: 1,200 questions of three kinds, placed over the workflows and their states. Code only: no model, no network.

    content  660  a question about something the document says; the answer is an exact span of one passage
    version  300  a fact that differs between two versions of one document; the question is the same, the expected answer depends on the version a person may open
    status   240  the document's status (final, approved, disseminated, in force, amended, repealed ...), from the workflow state by a table written here

Per workflow bucket (simple variant 150, the other policy analysis variants 300, knowledge asset 250, official documents 200, regulations 300) the
questions are spread over the states of each workflow in turn. Half of the questions are Indonesian and half English. A *slot* is a question to
be written: its kind, workflow, state, document, version(s), passage(s) and language; the text is in `eval/data/questions/questions.json`.

The status answer is a label the state maps to (STATUS below); it is not a span of a passage, so the rule "the answer is in its passage" is checked for
content and version questions and the status answer is checked against this table (the oracle for status).
"""

from __future__ import annotations

import gzip
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

KINDS = {"content": 660, "version": 300, "status": 240}
BUCKETS = {  # workflow bucket -> (questions, workflows)
    "simple": (150, ["card1_policy_analysis_simple"]),
    "policy": (300, ["card2_policy_analysis_individual", "policy_analysis_team_single", "policy_analysis_other_unit_single", "card5_policy_analysis_sensitive"]),
    "knowledge": (250, ["knowledge_asset"]),
    "official": (200, ["card7_official_document"]),
    "regulation": (300, ["regulation_single"]),
}

# The stages a status question can have as its answer, and the stage each state of each workflow means. One table, from the
# workflows; a test checks that every state of every workflow is in it. (Decisions: `finalized` of the simple variant is "final", `stored`
# of the official document is "diarsipkan".)
STAGES = ["draf", "dalam reviu", "diajukan", "diserahkan", "dalam persetujuan", "disetujui", "ditandatangani", "disebarluaskan", "berlaku", "diubah", "dicabut",
          "dirilis ke unit", "dibagikan ke unit peminta", "final", "diarsipkan"]
STATUS = {
    "card1_policy_analysis_simple": {"working_document": "draf", "finalized": "final"},
    "card2_policy_analysis_individual": {"working_document": "draf", "group_head_review": "dalam reviu", "submitted": "diajukan", "handed_over": "diserahkan"},
    "policy_analysis_team_single": {"working_document": "draf", "group_head_review": "dalam reviu", "submitted": "diajukan", "handed_over": "diserahkan"},
    "policy_analysis_other_unit_single": {"working_document": "draf", "group_head_review": "dalam reviu", "submitted": "diajukan", "handed_over": "diserahkan",
                                          "shared_with_requesting_unit": "dibagikan ke unit peminta"},
    "card5_policy_analysis_sensitive": {"working_document": "draf", "group_head_review": "dalam reviu", "submitted": "diajukan", "decision_taken": "disetujui",
                                        "released_to_unit": "dirilis ke unit"},
    "knowledge_asset": {"working": "draf", "quality_assurance": "dalam reviu", "disseminated": "disebarluaskan"},
    "card7_official_document": {"draft": "draf", "under_approval": "dalam persetujuan", "approved": "disetujui", "signed": "ditandatangani", "stored": "diarsipkan"},
    "regulation_single": {"working_draft": "draf", "review": "dalam reviu", "in_force": "berlaku", "amended": "diubah", "repealed": "dicabut"},
}
# Eight wordings; none contains a stage word, and each refers to the document in its own way.
STATUS_Q = [
    "Apa status terkini {doc}?",
    "Pada tahap apa {doc} saat ini?",
    "Sampai di tahap mana {doc}?",
    "Bagaimana kedudukan {doc} dalam alur kerjanya sekarang?",
    "{Doc} sedang berada pada tahap apa?",
    "Apa tahapan saat ini dari {doc}?",
    "Di posisi mana {doc} dalam proses penyelesaiannya?",
    "Apa keadaan terakhir {doc} dari sisi tahapnya?",
]


def crc(*parts) -> int:
    return zlib.crc32("\0".join(map(str, parts)).encode())


def status_of(workflow: str, state: str, lang: str = "id") -> str | None:
    """The stage a state means (Indonesian only)."""
    return STATUS.get(workflow, {}).get(state)


# ── version pairs ───────────────────────────────────────────────────────────

MONTHS = ("januari februari maret april mei juni juli agustus september oktober november desember january february march may june july august october december").split()
DATE = (r"(?:\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}"
        r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})")
NUM = re.compile(DATE + r"|(?:ARD\s)?\d[\d.,]*\s?(?:%|persen|miliar|triliun|juta|hari|bulan|tahun|percent|billion|trillion|million|days|months|years)?")
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def chunks_by_version(data: Path) -> dict[tuple[str, int], list[tuple[int, str]]]:
    plain, zipped = data / "chunks.jsonl", data / "chunks.jsonl.gz"
    opener = (lambda: open(plain, encoding="utf-8")) if plain.exists() else (lambda: gzip.open(zipped, "rt", encoding="utf-8"))
    out: dict = defaultdict(list)
    with opener() as f:
        for line in f:
            r = json.loads(line)
            out[(r["doc"], r["version"])].append((r["seq"], r["content"]))
    return out


def _words(s: str) -> set[str]:
    return set(re.findall(r"[^\W\d_]+", s.lower()))


def related(x: str, y: str) -> bool:
    """Whether one value is a truncation or a part of the other ("27 Mei" in "27 Mei 2027", "12" in "12,5%"): such a pair differs by a missing piece, not by a value."""
    tok = lambda v: re.findall(r"\w+", v.lower())
    a, b = x.lower().strip(), y.lower().strip()
    return a in b or b in a or set(tok(a)) <= set(tok(b)) or set(tok(b)) <= set(tok(a))


def sig(figure: str) -> tuple:
    """The words and signs of a figure apart from its digits and the names of months: 'ARD 8,06 miliar' and 'ARD 1,30 miliar' have the same, and so have two dates."""
    return tuple(sorted(w for w in re.findall(r"[^\W\d_]+|%", figure.lower()) if w not in MONTHS))


def _prep(chunk_list):
    """The sentences of a version that carry a figure, each with its word set and figures, computed once."""
    out = []
    for seq, text in chunk_list:
        for sent in SENT.split(text):
            sent = sent.strip()
            if len(sent) > 40:
                figs = [x.strip().rstrip(".,") for x in NUM.findall(sent)]
                if figs:
                    out.append((seq, sent, frozenset(_words(sent)), figs))
    return out


def version_pairs(chunks, docs: list[dict]) -> list[dict]:
    """For each document and each consecutive pair of versions (a, b): sentences that say the same thing with a different figure.
    {doc, a, b, seq_a, seq_b, sent_a, sent_b, x, y}: x is the figure in version a, y the one in version b (different)."""
    out = []
    for d in docs:
        n = int(d["n"])
        prepared = {v: _prep(chunks.get((d["id"], v), [])) for v in range(1, n + 1)}
        for a in range(1, n):
            b = a + 1
            best = {}
            for seq_b, sb, wb, nb in prepared[b]:
                for seq_a, sa, wa, na in prepared[a]:
                    if len(wa & wb) < 0.6 * len(wa | wb):
                        continue
                    # the figures must stand in the same places and be of the same kind (same unit words); the first that differs is the pair
                    if len(na) != len(nb):
                        continue
                    idx = [i for i in range(len(na)) if na[i] != nb[i] and sig(na[i]) == sig(nb[i]) and not related(na[i], nb[i])]
                    if idx:
                        best[sb] = {"doc": d["id"], "a": a, "b": b, "seq_a": seq_a, "seq_b": seq_b, "sent_a": sa, "sent_b": sb, "x": na[idx[0]], "y": nb[idx[0]]}
                        break
            out += sorted(best.values(), key=lambda r: crc(r["doc"], r["sent_b"]))[:3]
    return out


def need_of(pairs: list[dict]) -> dict[str, int]:
    """The version a document should have in the replayed world so that a version question can be asked of it."""
    need: dict[str, int] = {}
    for p in pairs:
        need[p["doc"]] = min(need.get(p["doc"], 99), p["b"])
    return need


# ── allocation ──────────────────────────────────────────────────────────────

def counts(total: int, parts: dict) -> dict:
    """Largest-remainder split of `total` over `parts` (name -> weight)."""
    w = sum(parts.values())
    raw = {k: total * v / w for k, v in parts.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


# kinds per bucket: content / version / status (660 / 300 / 240 in all)
BUCKET_KINDS = {"simple": (83, 38, 29), "policy": (165, 75, 60), "knowledge": (138, 62, 50), "official": (110, 50, 40), "regulation": (164, 75, 61)}


def targets() -> dict[tuple[str, str], int]:
    """(workflow, kind) -> the number of questions, from the bucket sizes of the evaluation plan."""
    out: dict = {}
    for b, (n, wfs) in BUCKETS.items():
        assert sum(BUCKET_KINDS[b]) == n, b
        for k, total in zip(KINDS, BUCKET_KINDS[b]):
            out.update({(w, k): c for w, c in counts(total, {w: 1 for w in wfs}).items()})
    assert {k: sum(v for (w, kk), v in out.items() if kk == k) for k in KINDS} == KINDS
    return out


def spread(rows: list[dict], n: int, key) -> list[dict]:
    """n rows, taken in turn from the groups of `key` (states), each group in a fixed order (first question of each document first); fewer if the rows run out."""
    groups: dict = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r.get("k", 0), crc(r["doc"], "spread"))):
        groups[key(r)].append(r)
    order = sorted(groups)
    out, i = [], 0
    while len(out) < n and any(groups.values()):
        g = groups[order[i % len(order)]]
        if g:
            out.append(g.pop(0))
        i += 1
    return out


# ── the type gate ───────────────────────────────────────────────────────────
# What a question asks for, read from its question word, and whether an answer is of that kind. Code decides when it can; what it cannot classify
# ("siapa", "unit mana", "apa", "mengapa") goes to the model.

PERCENT = r"\d[\d.,]*\s?(?:%|persen|persentase)"
MONEY = r"(?:ARD|Rp)\s?\d[\d.,]*\s?(?:miliar|triliun|juta)?|\d[\d.,]*\s?(?:miliar|triliun|juta)(?:\s?ARD)?"
DURATION = r"\d[\d.,]*\s?(?:hari kerja|hari|bulan|tahun|minggu|pekan|jam)"
MONTH_INDEX = r"bulan ke-?\s?\d+"
_TYPE_RE = {"date": DATE, "percent": PERCENT, "duration": DURATION, "month_index": MONTH_INDEX, "figure": r"\d"}


# Real names that must not appear in a question (the sets are fictional: Arunika, ARD)
REAL_TERMS = re.compile(r"\b(Indonesia\w*|Jakarta|rupiah|Rp\.?|APBN|RAPBN|APBD|Bank Indonesia|Kemenkeu|DJP|DJBC|OJK|BUMN|BUMD|Pertamina|Pancasila|Surabaya|Bandung|Medan)\b")
DANGLING = {"setelah", "sebelum", "dan", "atau", "yang", "dengan", "untuk", "pada", "dari", "di", "ke", "oleh", "dalam", "serta", "selain", "terhadap", "bagi", "antara", "sebesar", "tentang", "mengenai"}


def expected_type(question: str) -> str | None:
    """date / month_index / duration / percent / figure, or None when the question word does not say."""
    q = question.lower()
    if re.search(r"bulan ke\s?-?\s?berapa|bulan keberapa", q):
        return "month_index"
    if re.search(r"\bkapan\b|tanggal berapa|sejak kapan|mulai kapan|pada tanggal berapa", q):
        return "date"
    if re.search(r"(?:se)?berapa(?:kah)? lama|(?:se)?berapa(?:kah)? (?:hari|bulan|tahun|minggu)|jangka waktu berapa|paling lama berapa", q):
        return "duration"
    if re.search(r"(?:se)?berapa(?:kah)?\s+(?:besar\s+)?(?:persen|persentase)|(?:se)?berapa(?:kah)?\s+\w+\s+persen|persentase\s+berapa", q):
        return "percent"
    if re.search(r"\b(?:se)?berapa(?:kah)?\b", q):
        return "figure"
    return None


def type_ok(kind: str, answer: str) -> bool:
    return bool(re.search(_TYPE_RE[kind], answer, re.I))


def span_type(span: str) -> str | None:
    """What an answer-first span is: date, percent, duration or money (a figure)."""
    for kind, rx in (("date", DATE), ("percent", PERCENT), ("duration", DURATION), ("figure", MONEY)):
        if re.fullmatch(rx, span.strip(), re.I):
            return kind
    return None


def span_candidates(passage: str, doc: str) -> list[dict]:
    """Figures, dates and durations that occur exactly once in the passage, each with its sentence: {span, type, sentence}, in a fixed order by hash."""
    out = []
    for kind, rx in (("date", DATE), ("percent", PERCENT), ("duration", DURATION), ("figure", MONEY)):
        for m in re.finditer(rx, passage, re.I):
            span = m.group(0).strip().rstrip(".,")
            if passage.count(span) != 1 or not span_type(span):
                continue
            sent = next((x.strip() for x in SENT.split(passage) if span in x and len(x.strip()) >= 40), None)
            if sent:
                out.append({"span": span, "type": span_type(span), "sentence": sent})
    return sorted({c["span"]: c for c in out}.values(), key=lambda c: crc(doc, c["span"]))


# ── slots ───────────────────────────────────────────────────────────────────
# Every question is Indonesian and about an Indonesian-language document. English documents stay in the corpus as distractors and are never a target.
# More slots than the targets are made and marked `spare`: they stand in for the slots that cannot be made valid, in their own workflow and kind.
SPARE = 0.15  # the first ten percent of a group are spares (ids v####); the next five percent are `extra`, with their own ids (x####)
FINDING_SHARE = 0.45  # of the model-written content questions of the synthetic workflows the answer is a finding, a reason or a recommendation chosen by the model (more than a third of all content questions)


def _passages_for_content(chunks, doc: str, version: int) -> list:
    """The passages of the version with figures in them, in a fixed order by hash."""
    cands = [(seq, t) for seq, t in chunks.get((doc, version), []) if len(t) >= 300 and len(re.findall(r"\d", t)) >= 4]
    return sorted(cands, key=lambda c: crc(doc, version, c[0]))


def _spares(n: int) -> int:
    return -(-n * 15 // 100)  # ceil(15%)


def _reserve(n: int) -> int:
    return -(-n * 10 // 100)  # a further 10%: the reserve (y####), used only when the spares of a workflow and kind are gone


def _first_spares(n: int) -> int:
    return -(-n * 10 // 100)  # ceil(10%): the spares that batch 1 already knew


def allocate(assignment: list[dict], pairs: list[dict], chunks, docinfo: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Slots for the seven synthetic-set workflows, from Indonesian documents only (plus about 10% spares). `assignment` is the replay's
    (doc, workflow, unit, state, versions). A target that the Indonesian documents cannot fill is reported in the notes and left short."""
    t, slots, notes = targets(), [], []
    pairs_by_doc: dict = defaultdict(list)
    for p in pairs:
        pairs_by_doc[p["doc"]].append(p)
    for b, (_, wfs) in BUCKETS.items():
        for w in (w for w in wfs if w != "regulation_single"):
            rows = [r for r in assignment if r["workflow"] == w and r.get("kind") != "topic" and docinfo[r["doc"]]["lang"] == "id"]
            taken: set = set()
            for kind in ("version", "content", "status"):
                n = t[(w, kind)]
                want = n + _spares(n) + _reserve(n)

                def room(r, kind=kind) -> int:  # how many different questions of this kind the document can carry
                    if kind == "version":
                        return len([p for p in pairs_by_doc.get(r["doc"], ()) if p["b"] <= r["versions"]])
                    if kind == "content":
                        return min(len(_passages_for_content(chunks, r["doc"], r["versions"])), 6)
                    return 2
                expanded = [{**r, "k": k} for r in rows for k in range(room(r))]
                chosen = spread([r for r in expanded if r["doc"] not in taken], want, lambda r: r["state"])
                if len(chosen) < want:  # not enough documents that no other kind uses: reuse them
                    left = [r for r in expanded if (r["doc"], r["k"]) not in {(c["doc"], c["k"]) for c in chosen}]
                    chosen += spread(left, want - len(chosen), lambda r: r["state"])
                if len(chosen) < n:
                    notes.append(f"{w}: {kind}: {len(chosen)} of {n} (the Indonesian documents of this workflow have no more room)")
                taken |= {r["doc"] for r in chosen[: n + _first_spares(n)]}  # the extra spares do not change what the other kinds may take
                made = []
                for i, r in enumerate(chosen):
                    s = {"kind": kind, "workflow": w, "state": r["state"], "unit": r["unit"], "doc": r["doc"], "lang": "id", "doc_lang": "id",
                         "title": docinfo[r["doc"]]["topic"], "world_versions": r["versions"], "spare": i >= n}
                    if kind == "content":
                        s.update(version=r["versions"], k=r["k"])
                    elif kind == "version":
                        p = sorted((p for p in pairs_by_doc[r["doc"]] if p["b"] <= r["versions"]), key=lambda p: crc(r["doc"], p["sent_b"]))[r["k"]]
                        s.update(pair=p, answer_a=p["x"], answer_b=p["y"])
                    else:
                        s.update(answer=status_of(w, r["state"]))
                    s["extra"] = n + _first_spares(n) <= i < n + _spares(n)
                    s["reserve"] = i >= n + _spares(n)
                    made.append(s)
                if kind == "content":
                    # a share of the questions have an answer the model chooses (a finding, a reason, a recommendation); the others an answer chosen by code (a figure or a date).
                    # The extra spares are decided on their own, apart from the other slots.
                    cut, cut2 = n + _first_spares(n), n + _spares(n)
                    for part in (made[:cut], made[cut:cut2], made[cut2:]):
                        order = sorted(range(len(part)), key=lambda i: crc(part[i]["doc"], part[i]["k"], "mode"))
                        finding = set(order[: -(-len(part) * 45 // 100)])
                        for i, s in enumerate(part):
                            plist = _passages_for_content(chunks, s["doc"], s["version"])
                            mode, pick = "finding", plist[s["k"]]
                            if i not in finding:
                                for j in range(len(plist)):
                                    seq, text = plist[(s["k"] + j) % len(plist)]
                                    cands = span_candidates(text, s["doc"])
                                    if cands:
                                        mode, pick = "span", (seq, text)
                                        s.update(answer=cands[0]["span"], span_type=cands[0]["type"], sentence=cands[0]["sentence"])
                                        break
                            s.update(mode=mode, seq=pick[0], passage=pick[1])
                slots += made
            have = {r["state"] for r in rows}
            every = {r["state"] for r in assignment if r["workflow"] == w and r.get("kind") != "topic"}
            if every - have:
                notes.append(f"{w}: no Indonesian document in the state(s) {sorted(every - have)}")
    return slots, notes


def regulation_slots(plan: dict, chunks, fmt, labels: dict, articles: dict) -> tuple[list[dict], list[str]]:
    """Slots for the regulation workflow, from the Indonesian regulations of the plan (facts per version) and their passages (plus about 10% spares)."""
    t, notes, slots = targets(), [], []
    regs = [r for r in plan["regulations"] if r["lang"] == "id"]
    facts = list(articles)

    def passage(reg, v, answer):
        for seq, text in chunks.get((f"{reg['id']}/text", v), []):
            if answer in text:
                return seq, text
        return None

    def make(kind, rows, n):
        chosen = spread(rows, n + _spares(n) + _reserve(n), lambda r: r["state"])
        if len(chosen) < n:
            notes.append(f"regulation_single: {kind}: {len(chosen)} of {n} (the Indonesian regulations have no more room)")
        return [(r, i >= n, (i >= n + _first_spares(n)) and i < n + _spares(n), i >= n + _spares(n)) for i, r in enumerate(chosen)]

    rows = []
    for r in regs:
        f = facts[crc(r["id"], "fact") % 5]
        v = len(r["versions"])
        ans = fmt(f, r["versions"][-1]["facts"][f], "id", r)
        p = passage(r, v, ans)
        if p:
            rows.append({**r, "doc": r["id"], "fact": f, "answer": ans, "seq": p[0], "passage": p[1], "version": v})
    for r, spare, extra, reserve in make("content", rows, t[("regulation_single", "content")]):
        slots.append({"kind": "content", "workflow": "regulation_single", "state": r["state"], "unit": r["unit"], "doc": f"{r['id']}/text", "lang": "id", "doc_lang": "id",
                      "title": r["title"], "number": r["number"], "world_versions": r["version"], "version": r["version"], "seq": r["seq"], "passage": r["passage"], "fact": r["fact"], "mode": "span",
                      "article": articles[r["fact"]], "label": labels["id"][r["fact"]], "answer": r["answer"], "spare": spare, "extra": extra, "reserve": reserve})
    vrows = []
    for r in regs:
        for a in range(1, len(r["versions"])):
            ch = [f for f in r["versions"][a]["changed"] if f in articles]
            if not ch:
                continue
            f = ch[crc(r["id"], a) % len(ch)]
            xa, xb = fmt(f, r["versions"][a - 1]["facts"][f], "id", r), fmt(f, r["versions"][a]["facts"][f], "id", r)
            pa, pb = passage(r, a, xa), passage(r, a + 1, xb)
            if pa and pb and xa != xb and not related(xa, xb):
                vrows.append({**r, "doc": r["id"], "fact": f, "a": a, "b": a + 1, "xa": xa, "xb": xb, "pa": pa, "pb": pb})
                break
    for r, spare, extra, reserve in make("version", vrows, t[("regulation_single", "version")]):
        slots.append({"kind": "version", "workflow": "regulation_single", "state": r["state"], "unit": r["unit"], "doc": f"{r['id']}/text", "lang": "id", "doc_lang": "id",
                      "title": r["title"], "number": r["number"], "world_versions": len(r["versions"]), "fact": r["fact"], "article": articles[r["fact"]], "label": labels["id"][r["fact"]],
                      "answer_a": r["xa"], "answer_b": r["xb"], "spare": spare, "extra": extra, "reserve": reserve,
                      "pair": {"a": r["a"], "b": r["b"], "seq_a": r["pa"][0], "seq_b": r["pb"][0], "sent_a": r["pa"][1], "sent_b": r["pb"][1], "x": r["xa"], "y": r["xb"]}})
    for r, spare, extra, reserve in make("status", [{**r, "doc": r["id"]} for r in regs], t[("regulation_single", "status")]):
        slots.append({"kind": "status", "workflow": "regulation_single", "state": r["state"], "unit": r["unit"], "doc": f"{r['id']}/text", "lang": "id", "doc_lang": "id",
                      "title": r["title"], "number": r["number"], "world_versions": len(r["versions"]), "answer": status_of("regulation_single", r["state"]), "spare": spare, "extra": extra, "reserve": reserve})
    return slots, notes


def number(slots: list[dict]) -> list[dict]:
    """v#### for the slots and the first spares, x#### for the extra spares."""
    key = lambda s: (s["workflow"], s["kind"], s.get("spare", False), s["state"], s["doc"], s.get("seq", 0), s.get("k", 0))
    for prefix, pick in (("v", lambda s: not s.get("extra") and not s.get("reserve")), ("x", lambda s: s.get("extra")), ("y", lambda s: s.get("reserve"))):
        for i, s in enumerate(sorted((s for s in slots if pick(s)), key=key), 1):
            s["id"] = f"{prefix}{i:04d}"
    return sorted(slots, key=lambda s: s["id"])


# ── batches ─────────────────────────────────────────────────────────────────
BATCHES = {1: (0.0, 0.1), 2: (0.1, 0.4), 3: (0.4, 0.7), 4: (0.7, 1.0)}  # 120, 360, 360, 360 of 1,200: every kind and workflow in the same proportion


def batch_of(slots: list[dict]) -> dict[str, int]:
    """Batch 1 is a tenth of every (workflow, kind) group, batches 2 to 4 three tenths each, chosen by a fixed hash. Spares belong to no batch (0)."""
    out: dict[str, int] = {s["id"]: 0 for s in slots if s.get("spare")}
    groups: dict = defaultdict(list)
    for s in slots:
        if not s.get("spare"):
            groups[(s["workflow"], s["kind"])].append(s)
    sizes = {1: 1, 2: 3, 3: 3, 4: 3}
    for g in groups.values():
        g.sort(key=lambda s: crc(s["id"], "batch"))
        take = counts(len(g), sizes)  # largest remainder: a tenth, then three tenths three times
        i = 0
        for b in (1, 2, 3, 4):
            for s in g[i:i + take[b]]:
                out[s["id"]] = b
            i += take[b]
    return out
