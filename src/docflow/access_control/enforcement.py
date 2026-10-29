"""Enforcement: the decision on every way to a document, and the only door from the document service to the stores.

The document service (`document_service/`) asks here before it fetches anything:

    list_visible  the documents the person may open, each at the version they open
    open          one document by id, at the version they open (never another)
    visible       the versions a search finds for the person
    limit         how many passages one search may return

A document the person may not open is refused exactly as one that does not exist (`NotFound`), so the answer
does not reveal that it is there. The decision log keeps the reason; the caller never gets it.

The engine is reached only through `world.readable_version`, `world.visible` and `world.guidance`.
"""

from __future__ import annotations

from ..stores.world import World

# One call cannot dump a corpus. A search never returns more than MAX_SEARCH_LIMIT passages. The agent's tools
# (`agent_harness/tool_controller.py`) cut a list at MAX_LIST documents and a document at MAX_GET_PASSAGES
# passages of MAX_PASSAGE_CHARS characters each.
MAX_SEARCH_LIMIT = 50
MAX_LIST = 200
MAX_GET_PASSAGES = 100
MAX_PASSAGE_CHARS = 4000


class NotFound(Exception):
    """No such document, for this person."""


class Enforcement:
    def __init__(self, world: World) -> None:
        self.world = world
        # Every decision: channel, person, what was asked, the arguments, what was returned and in which state.
        # A refusal also keeps its reason here; the reason is never returned to the caller.
        self.log: list[dict] = []

    def record(self, channel: str, user: str, asked: str, docs: list[tuple[str, int]], arguments: dict, reason: str | None = None) -> None:
        entry = {"channel": channel, "user": user, "asked": asked, "arguments": arguments,
                 "returned": [{"doc": d, "version": v, "state": self.world.state_of(d)} for d, v in docs]}
        if reason is not None:
            entry["reason"] = reason
        self.log.append(entry)

    @staticmethod
    def limit(limit: int) -> int:
        return max(1, min(int(limit), MAX_SEARCH_LIMIT))

    def visible(self, user: str, *, for_search: bool = True) -> dict[str, int]:
        return self.world.visible(user, for_search=for_search)

    def guidance(self, doc_id: str, user: str | None = None):
        return self.world.guidance(doc_id, user)

    def state_of(self, doc_id: str) -> str:
        return self.world.state_of(doc_id)

    def list_visible(self, user: str, workspace: str | None = None) -> list[tuple[str, int, str, str, str | None]]:
        """(id, version, state, workflow, guidance) of each document the person may open."""
        out = []
        for doc_id, version in self.world.visible(user, for_search=False).items():
            doc = self.world.documents[doc_id]
            if workspace is None or doc.workspace == workspace:
                out.append((doc_id, version, self.world.state_of(doc_id), doc.workflow, self.world.guidance(doc_id, user)))
        self.record("list", user, workspace or "*", [(r[0], r[1]) for r in out], {"workspace": workspace})
        return out

    def open(self, user: str, doc_id: str, version: int | None = None) -> tuple[int, str, str | None]:
        """(version, state, guidance) of the document at the version `user` opens. Naming any other version is refused too."""
        doc = self.world.documents.get(doc_id)
        opened = self.world.readable_version(doc_id, user, for_search=False) if doc else None
        asked, arguments = f"{doc_id}@{version}", {"doc_id": doc_id, "version": version}
        if opened is None or (version is not None and version != opened):
            if doc_id not in self.world.documents:
                reason = "no_such_document"
            elif self.world.readable_version(doc_id, user, for_search=False) is None:
                reason = "not_opened"
            else:
                reason = "version_not_opened"
            self.record("get", user, asked, [], arguments, reason)
            raise NotFound(doc_id)
        self.record("get", user, asked, [(doc_id, opened)], arguments)
        return opened, self.world.state_of(doc_id), self.world.guidance(doc_id, user)
