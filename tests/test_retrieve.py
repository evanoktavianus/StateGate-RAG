"""Retrieve: the fusion of two rankings. Expected values are worked out in the tests."""

import random

import pytest

from docflow.document_service import retrieve
from docflow.stores.search_index import Passage


def p(doc, seq=0):
    return Passage(doc, 1, seq, (0, 0), "", f"{doc}{seq}", 0)


A, B, C = p("A"), p("B"), p("C")


def test_fusing_two_rankings_by_hand():
    """words: A, B. meaning: B, C, A. With k=60 a passage at rank r from the top adds 1 / (60 + r + 1):
    A = 1/61 + 1/63, B = 1/62 + 1/61, C = 1/62. B beats A (1/62 > 1/63), A beats C."""
    fused = retrieve.fuse([(3.0, A), (2.0, B)], [(0.9, B), (0.8, C), (0.7, A)])
    assert [x.doc for _, x in fused] == ["B", "A", "C"]
    assert [s for s, _ in fused] == pytest.approx([1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 62])


def test_the_limit_cuts_after_fusing_and_a_single_ranking_keeps_its_order():
    two = retrieve.fuse([(3.0, A), (2.0, B)], [(0.9, B), (0.8, C), (0.7, A)], limit=2)
    assert [x.doc for _, x in two] == ["B", "A"]
    assert [x.doc for _, x in retrieve.fuse([(1, C), (1, A), (1, B)])] == ["C", "A", "B"]
    assert retrieve.fuse() == [] and retrieve.fuse([]) == []


def test_a_tie_keeps_the_order_the_passages_first_appeared_in():
    fused = retrieve.fuse([(1.0, A)], [(1.0, B)])  # each 1/61
    assert [x.doc for _, x in fused] == ["A", "B"]


def test_passages_of_one_document_are_fused_apart():
    """Fusion is by passage: two passages of the same document are two entries, not one."""
    a0, a1 = p("A", 0), p("A", 1)
    fused = retrieve.fuse([(2.0, a0), (1.0, a1)], [(0.5, a1)])
    assert [(x.doc, x.seq) for _, x in fused] == [("A", 1), ("A", 0)]  # a1: 1/62 + 1/61; a0: 1/61


def old_rrf(*lists, k=60):
    """A plain reciprocal-rank fusion, kept here as the reference for `retrieve.fuse`."""
    from collections import defaultdict
    score = defaultdict(float)
    for ranked in lists:
        for rank, key in enumerate(ranked):
            score[key] += 1 / (k + rank + 1)
    return [key for key, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def test_it_orders_exactly_as_the_script_did():
    rng = random.Random(5)
    pool = [p(f"d{i}", s) for i in range(6) for s in range(3)]
    for _ in range(200):
        words = [(1.0, x) for x in rng.sample(pool, rng.randint(0, 12))]
        meaning = [(1.0, x) for x in rng.sample(pool, rng.randint(0, 12))]
        key = lambda x: (x.doc, x.version, x.seq)
        look = {key(x): x for _, x in words + meaning}
        expected = [look[k] for k in old_rrf([key(x) for _, x in words], [key(x) for _, x in meaning])]
        assert [x for _, x in retrieve.fuse(words, meaning)] == expected
