"""Passages of documents and a search that only looks where a person may.

`Corpus.search(query, visible)` takes `visible`, the `{document: version}` map a `World` gives for one
person, and ranks only the passages of those versions. Nothing else is ever scored, so a passage a
person may not read can't come back, whatever the query.
"""

from __future__ import annotations

import array
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DIMS = 1536
try:
    import numpy as _np
except ImportError:  # the pure-Python search below does the same, only slower
    _np = None
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Passage:
    doc: str
    version: int
    seq: int
    pages: tuple[int, int]
    breadcrumb: str
    content: str
    row: int


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


class Corpus:
    def __init__(self, passages: list[Passage], embeddings: array.array | None = None) -> None:
        self.passages = passages
        self._embeddings = embeddings
        self._matrix = self._norms = None
        self.by_version: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, p in enumerate(passages):
            self.by_version[(p.doc, p.version)].append(i)
        self._tf = [Counter(words(p.breadcrumb + " " + p.content)) for p in passages]
        self._df: Counter = Counter(w for tf in self._tf for w in tf)
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)

    @classmethod
    def load(cls, directory: str | Path, embeddings: bool = True) -> Corpus:
        d = Path(directory)
        passages = []
        for line in (d / "chunks.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            passages.append(Passage(r["doc"], r["version"], r["seq"], tuple(r["pages"]), r["breadcrumb"], r["content"], r["row"]))
        vecs = None
        if embeddings and (d / "embeddings.f32").exists():
            vecs = array.array("f")
            with (d / "embeddings.f32").open("rb") as f:
                vecs.frombytes(f.read())
        return cls(passages, vecs)

    def add(self, doc: str, version: int, texts: list[str], vectors: list[list[float]] | None = None) -> None:
        """Adds a document version's passages (for documents created during a test)."""
        self.add_passages(doc, version, [(seq, (0, 0), "", text) for seq, text in enumerate(texts)], vectors)

    def add_passages(self, doc: str, version: int, items: list[tuple[int, tuple[int, int], str, str]], vectors: list[list[float]] | None = None) -> None:
        """Adds a version's passages as (seq, pages, breadcrumb, content), with their vectors if the index holds vectors."""
        for (seq, pages, crumb, text), n in zip(items, range(len(items))):
            row = len(self.passages)
            p = Passage(doc, version, seq, tuple(pages), crumb, text, row)
            self.passages.append(p)
            self.by_version[(doc, version)].append(row)
            tf = Counter(words(crumb + " " + text))
            self._tf.append(tf)
            self._df.update(tf.keys())
            self._len.append(sum(tf.values()))
            if vectors is not None and self._embeddings is not None:
                self._matrix = self._norms = None  # drop the numpy view first: it holds the array, and is rebuilt at the next search
                self._embeddings.extend(vectors[n])
        self._avg = sum(self._len) / max(len(self._len), 1)

    def passages_of(self, doc: str, version: int) -> list[Passage]:
        """The passages of one version, in order."""
        return [self.passages[i] for i in self.by_version.get((doc, version), ())]

    def copy_version(self, src: str, dst: str, version: int | None = None) -> None:
        """Gives `dst` version 1 with the passages of `src` (its newest version, or the one given)."""
        version = version or max(v for d, v in self.by_version if d == src)
        rows = self.by_version[(src, version)]
        self.add(dst, 1, [self.passages[i].content for i in rows],
                 [self.vector_of(self.passages[i]) for i in rows] if self._embeddings is not None else None)

    # ── Searching ──────────────────────────────────────────────────────

    def candidates(self, visible: dict[str, int]) -> list[int]:
        """The rows of the versions in `visible`: the only passages a search may score."""
        return [i for doc, version in visible.items() for i in self.by_version.get((doc, version), ())]

    def search(self, query: str, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        """Keyword search (BM25) over the passages `visible` allows."""
        terms = list(dict.fromkeys(words(query)))
        rows = self.candidates(visible)
        # Statistics come from the passages the person may read and no others: a hidden document must not
        # move a score, an order or a count, whatever it says.
        n, k1, b = len(rows), 1.5, 0.75
        df = {t: sum(1 for i in rows if t in self._tf[i]) for t in terms}
        avg = sum(self._len[i] for i in rows) / max(n, 1)
        scored = []
        for i in rows:
            tf, score = self._tf[i], 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / avg))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: -s[0])
        return [(s, self.passages[i]) for s, i in scored[:limit]]

    def similar(self, vector, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        """Search by meaning: cosine similarity to `vector`, over the passages `visible` allows."""
        if self._embeddings is None:
            raise RuntimeError("no embeddings loaded")
        q = list(vector)
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        if _np is not None:  # the same cosine, on a matrix; identical order, much faster
            if self._matrix is None:
                self._matrix = _np.frombuffer(self._embeddings, dtype=_np.float32).reshape(-1, DIMS)
                self._norms = _np.maximum(_np.linalg.norm(self._matrix, axis=1), 1e-12)
            rows = self.candidates(visible)
            sims = (self._matrix[rows] @ _np.asarray(q, dtype=_np.float32)) / (self._norms[rows] * qn)
            order = _np.argsort(-sims, kind="stable")[:limit]
            return [(float(sims[j]), self.passages[rows[j]]) for j in order]
        scored = []
        for i in self.candidates(visible):
            v = self._embeddings[i * DIMS : (i + 1) * DIMS]
            dot = sum(a * b for a, b in zip(q, v))
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            scored.append((dot / (qn * vn), i))
        scored.sort(key=lambda s: -s[0])
        return [(s, self.passages[i]) for s, i in scored[:limit]]

    def vector_of(self, passage: Passage) -> list[float]:
        return list(self._embeddings[passage.row * DIMS : (passage.row + 1) * DIMS])
