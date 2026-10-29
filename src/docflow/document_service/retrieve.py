"""Retrieve: passages of the versions a search finds for a person, by words, by meaning, or both.

The visible versions are chosen first; the search ranks only those passages.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..stores.search_index import Passage

RRF_K = 60  # reciprocal rank fusion: a passage ranked r in a list adds 1 / (RRF_K + r)
HYBRID_DEPTH = 50  # how many passages of each list are fused


@dataclass(frozen=True)
class Hit:
    doc: str
    version: int
    score: float
    passage: Passage
    guidance: str | None
    state: str


def keyword(access, corpus, user: str, query: str, limit: int = 5) -> list[Hit]:
    visible = access.visible(user, for_search=True)
    return [Hit(p.doc, p.version, score, p, access.guidance(p.doc, user), access.state_of(p.doc)) for score, p in corpus.search(query, visible, limit)]


def semantic(access, corpus, user: str, vector, limit: int = 5) -> list[Hit]:
    """Search by meaning, over what a search finds for `user`."""
    visible = access.visible(user, for_search=True)
    return [Hit(p.doc, p.version, s, p, access.guidance(p.doc, user), access.state_of(p.doc)) for s, p in corpus.similar(vector, visible, limit)]


def fuse(*ranked: list, limit: int | None = None) -> list[tuple[float, Passage]]:
    """Reciprocal rank fusion of ranked lists of (score, passage), at passage level: a passage ranked r from the top of a
    list adds 1 / (RRF_K + r + 1). Best first; ties keep the order in which the passages first appeared."""
    key = lambda p: (p.doc, p.version, p.seq)
    score: dict = defaultdict(float)
    passages: dict = {}
    for ranking in ranked:
        for rank, (_, p) in enumerate(ranking):
            score[key(p)] += 1 / (RRF_K + rank + 1)
            passages[key(p)] = p
    order = sorted(score, key=lambda k: -score[k])
    return [(score[k], passages[k]) for k in (order if limit is None else order[:limit])]


def hybrid(access, corpus, user: str, query: str, vector, limit: int = 5) -> list[Hit]:
    """Words and meaning together: both rank the same visible passages, and reciprocal rank fusion combines the two lists."""
    visible = access.visible(user, for_search=True)
    fused = fuse(corpus.search(query, visible, HYBRID_DEPTH), corpus.similar(vector, visible, HYBRID_DEPTH), limit=limit)
    return [Hit(p.doc, p.version, score, p, access.guidance(p.doc, user), access.state_of(p.doc)) for score, p in fused]


class PartitionedIndex:
    """The search index of a repository, one index per unit.

    A person's search asks only the units that hold something they may read (`visible` already says which). If that is one
    unit, its ranking is the answer. If it is several, each unit ranks its own passages (a keyword score is computed from
    the passages of that unit only) and the lists are merged by reciprocal rank fusion, because scores from different
    indexes are not comparable; a tie goes to the unit that sorts first.
    """

    def __init__(self, repository) -> None:
        self.repository = repository

    def _by_unit(self, visible: dict[str, int]) -> dict[str, dict[str, int]]:
        parts: dict[str, dict[str, int]] = {}
        for doc, version in visible.items():
            parts.setdefault(self.repository.unit_of_document(doc), {})[doc] = version
        return dict(sorted(parts.items()))

    def _merged(self, ask, visible: dict[str, int], limit: int) -> list[tuple[float, Passage]]:
        lists = [ask(self.repository.index(unit), part) for unit, part in self._by_unit(visible).items()]
        lists = [ranked for ranked in lists if ranked]
        if len(lists) <= 1:
            return lists[0][:limit] if lists else []
        return fuse(*lists, limit=limit)

    def search(self, query: str, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        return self._merged(lambda index, part: index.search(query, part, limit), visible, limit)

    def similar(self, vector, visible: dict[str, int], limit: int = 5) -> list[tuple[float, Passage]]:
        return self._merged(lambda index, part: index.similar(vector, part, limit), visible, limit)

    def passages_of(self, doc: str, version: int) -> list[Passage]:
        return self.repository.index(self.repository.unit_of_document(doc)).passages_of(doc, version)
