"""Read: list documents and get one by id, each at the version the person opens.

The access decision is enforcement's: it says which documents and which version, and refuses what the person may not open.
This module fetches the content for what it was told.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Listed:
    id: str
    version: int
    state: str
    workflow: str
    guidance: str | None


def list_documents(access, user: str, workspace: str | None = None) -> list[Listed]:
    return [Listed(*row) for row in access.list_visible(user, workspace)]


def get_document(access, corpus, user: str, doc_id: str, version: int | None = None) -> dict:
    """The document's passages at the version `user` opens; `access.open` raises `NotFound` for any other."""
    opened, state, guidance = access.open(user, doc_id, version)
    return {"id": doc_id, "version": opened, "state": state, "guidance": guidance, "passages": corpus.passages_of(doc_id, opened)}
