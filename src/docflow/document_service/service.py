"""The document service: what an agent's tools, an API and a UI call to read and search.

Each call asks enforcement first (which documents, which version, what to say about them), then fetches the content for that
from the stores. Enforcement is the only door to the stores and keeps the decision log.
"""

from __future__ import annotations

from ..access_control.enforcement import Enforcement, NotFound  # noqa: F401  (NotFound is raised through here)
from ..stores.search_index import Corpus
from ..stores.world import World
from . import read, retrieve
from .read import Listed
from .retrieve import Hit


class DocumentService:
    def __init__(self, world: World, corpus: Corpus) -> None:
        self.world, self.corpus = world, corpus
        self.enforcement = Enforcement(world)

    @property
    def log(self) -> list[dict]:
        return self.enforcement.log

    def list_documents(self, user: str, workspace: str | None = None) -> list[Listed]:
        return read.list_documents(self.enforcement, user, workspace)

    def get_document(self, user: str, doc_id: str, version: int | None = None) -> dict:
        """The document's passages at the version `user` opens. Naming any other version is refused too."""
        return read.get_document(self.enforcement, self.corpus, user, doc_id, version)

    def search(self, user: str, query: str, limit: int = 5) -> list[Hit]:
        limit = self.enforcement.limit(limit)
        hits = retrieve.keyword(self.enforcement, self.corpus, user, query, limit)
        self.enforcement.record("search", user, query, [(h.doc, h.version) for h in hits], {"query": query, "limit": limit})
        return hits

    def similar(self, user: str, vector, limit: int = 5) -> list[Hit]:
        """Search by meaning, over what a search finds for `user`."""
        limit = self.enforcement.limit(limit)
        hits = retrieve.semantic(self.enforcement, self.corpus, user, vector, limit)
        self.enforcement.record("similar", user, "<vector>", [(h.doc, h.version) for h in hits], {"limit": limit})
        return hits

    def hybrid(self, user: str, query: str, vector, limit: int = 5) -> list[Hit]:
        """Search by words and meaning together, over what a search finds for `user`."""
        limit = self.enforcement.limit(limit)
        hits = retrieve.hybrid(self.enforcement, self.corpus, user, query, vector, limit)
        self.enforcement.record("hybrid", user, query, [(h.doc, h.version) for h in hits], {"query": query, "limit": limit})
        return hits
