"""Reads from PostgreSQL (pgvector) instead of memory, for E5, E6 and E7.

The worlds and the corpus of these experiments are built in memory. When `DATABASE_URL` is set (and `DOCFLOW_STORE` is not `memory`) each
world is written once into PostgreSQL with the engine's own writer (`stores.world_store.Writer`: the documents, their histories, every version with its
passages and vectors), and every search and every passage the service reads then comes from the repository's per-unit index (`PartitionedIndex`: keyword
ranking in the application over the visible passages, ranking by meaning in SQL with pgvector). What a person may see is still decided by the world and
the service; PostgreSQL holds the passages.

The load happens the first time a process needs the index, under an advisory lock, so many worker processes can share one load; a marker table says it is
done, and a later run reuses it (the namespace is named after the world, the corpus size and the key). `python -m evalkit.pgstore --drop` removes
everything this module made.
"""

from __future__ import annotations

import os
import sys
import threading
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ROOT / "work" / "pgfiles"
PREFIX = "evp"


def enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL")) and os.environ.get("DOCFLOW_STORE", "postgres") != "memory"


def wrap(world, corpus, key: str):
    """The index to give a `DocumentService` in place of `corpus`: the PostgreSQL index of this world when enabled, else the corpus itself."""
    return LazyIndex(world, corpus, key) if enabled() else corpus


class LazyIndex:
    """Loads the world into PostgreSQL on first use and answers `search`, `similar` and `passages_of` from there; opens a connection of its own in each process."""

    def __init__(self, world, corpus, key: str) -> None:
        self._world, self._corpus, self._key = world, corpus, key
        size = f"{key}|{len(corpus.passages)}|{len(world.documents)}"
        self._ns = PREFIX + format(zlib.crc32(size.encode()), "08x")
        self._pid, self._index = None, None
        self._mutex = threading.Lock()

    def _get(self):
        if self._pid != os.getpid():
            with self._mutex:  # the threads of one process open the index once
                if self._pid != os.getpid():
                    self._index = self._open()
                    self._pid = os.getpid()
        return self._index

    def _open(self):
        import psycopg
        from docflow.document_service.retrieve import PartitionedIndex
        from docflow.stores.postgres_repository import PostgresRepository

        FILES.mkdir(parents=True, exist_ok=True)
        lock = zlib.crc32(self._ns.encode())
        guard = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)  # holds the advisory lock while this process creates and loads the schema
        guard.execute("SELECT pg_advisory_lock(%s)", (lock,))
        try:
            repo = PostgresRepository(os.environ["DATABASE_URL"], FILES / self._ns, namespace=self._ns)
            with repo._lock:
                repo._conn.execute("CREATE TABLE IF NOT EXISTS public.evp_loaded (ns text PRIMARY KEY, passages integer)")
                if repo._conn.execute("SELECT 1 FROM public.evp_loaded WHERE ns = %s", (self._ns,)).fetchone() is None:
                    self._load(repo)
                    repo._conn.execute("INSERT INTO public.evp_loaded VALUES (%s, %s)", (self._ns, len(self._corpus.passages)))
        finally:
            guard.execute("SELECT pg_advisory_unlock(%s)", (lock,))
            guard.close()
        return PartitionedIndex(repo)

    def _load(self, repo) -> None:
        from docflow.stores.repository import PassageIn
        from docflow.stores.world_store import Writer

        corpus = self._corpus
        vectors = getattr(corpus, "_embeddings", None) is not None

        def passages_for(doc_id: str, version: int):
            found = corpus.passages_of(doc_id, version)
            if not found:
                return None
            return [PassageIn(p.seq, tuple(p.pages), p.breadcrumb, p.content, tuple(corpus.vector_of(p)) if vectors else None) for p in found]

        files = {n: ROOT / "workflows" / f"{n}.yml" for n in self._world.workflows}
        Writer(repo, files, passages_for).sync(self._world)

    def drop(self) -> None:
        """Removes this world's schemas and its marker (for tests and for a clean start)."""
        import psycopg

        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
            for (name,) in conn.execute("SELECT nspname FROM pg_namespace WHERE nspname = %s OR starts_with(nspname, %s)", (self._ns, self._ns + "_u_")).fetchall():
                conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            if conn.execute("SELECT to_regclass('public.evp_loaded')").fetchone()[0]:
                conn.execute("DELETE FROM public.evp_loaded WHERE ns = %s", (self._ns,))
        self._pid, self._index = None, None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get(), name)


def drop_all() -> int:
    """Removes every namespace this module made and its marker table. Returns how many namespaces were dropped."""
    from docflow.stores.postgres_repository import PostgresRepository
    import tempfile

    repo = PostgresRepository(os.environ["DATABASE_URL"], Path(tempfile.mkdtemp()), namespace=PREFIX + "00000000")
    n = 0
    with repo._lock:
        names = [r[0] for r in repo._conn.execute("SELECT nspname FROM pg_namespace WHERE nspname LIKE %s", (PREFIX + "%",))]
        for name in names:
            repo._conn.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            n += 1
        repo._conn.execute("DROP TABLE IF EXISTS public.evp_loaded")
    return n


if __name__ == "__main__":
    if "--drop" in sys.argv:
        print(f"dropped {drop_all()} schemas")
    else:
        print(__doc__)
