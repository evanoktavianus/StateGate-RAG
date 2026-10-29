"""Documents: each document's metadata. Where it stands is read off its history (`state_history`)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state_history import Timeline


@dataclass
class Document:
    id: str
    workspace: str
    owner: str
    workflow: str
    timeline: Timeline
    invited: set[str] = field(default_factory=set)
