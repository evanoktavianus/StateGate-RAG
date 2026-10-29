"""evalkit.pgstore: a world and its corpus written into PostgreSQL answer a search like the in-memory corpus does (needs DATABASE_URL, else skipped)."""

import array
import os
import secrets

import pytest

from docflow.stores.search_index import Corpus
from conftest_ministry import world  # noqa: F401

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is not set: PostgreSQL is not tested")

DIMS = 1536


def vector(seed: float) -> list[float]:
    """Vectors that point in different directions, so no two similarities tie (on an exact tie the two stores may order the passages differently)."""
    return [seed, 1.0 - seed] + [0.0] * (DIMS - 2)


def build(world):
    world.add_document("d1", "ws", "ann", "knowledge_asset")
    world.add_document("d2", "ws", "bob", "knowledge_asset")
    corpus = Corpus([], array.array("f"))
    corpus.add("d1", 1, ["alpha beta gamma", "delta epsilon alpha"], [vector(1.0), vector(0.5)])
    corpus.add("d2", 1, ["beta gamma", "zeta alpha"], [vector(0.2), vector(0.9)])
    return corpus


def test_the_index_in_postgresql_answers_like_the_corpus(world):
    from evalkit import pgstore

    corpus = build(world)
    index = pgstore.LazyIndex(world, corpus, "unit-test-" + secrets.token_hex(4))
    try:
        for user in ("ann", "bob", "u1b", "ext"):
            visible = world.visible(user, for_search=True)
            for query in ("alpha", "beta gamma", "nothing here"):
                a, b = index.search(query, visible, 5), corpus.search(query, visible, 5)
                assert [(p.doc, p.version, p.seq) for _, p in a] == [(p.doc, p.version, p.seq) for _, p in b], (user, query)
            a, b = index.similar(vector(1.0), visible, 5), corpus.similar(vector(1.0), visible, 5)
            assert [(p.doc, p.version, p.seq) for _, p in a] == [(p.doc, p.version, p.seq) for _, p in b], user
        assert world.visible("ext", for_search=True) == {}  # the outsider sees nothing, here and in PostgreSQL
    finally:
        index.drop()


def test_a_second_index_for_the_same_world_reuses_what_was_loaded(world):
    from evalkit import pgstore

    corpus = build(world)
    key = "unit-test-" + secrets.token_hex(4)
    first = pgstore.LazyIndex(world, corpus, key)
    second = pgstore.LazyIndex(world, corpus, key)
    try:
        first.search("alpha", world.visible("ann", for_search=True), 5)
        # the second one finds the marker and loads nothing: making it fail on any load proves it
        second._load = lambda repo: (_ for _ in ()).throw(AssertionError("loaded twice"))
        assert second.search("alpha", world.visible("ann", for_search=True), 5)
    finally:
        first.drop()
