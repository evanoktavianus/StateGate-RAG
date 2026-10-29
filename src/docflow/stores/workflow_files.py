"""The workflows of a folder of YAML files, read again whenever a file is added, removed or saved.

    files = WorkflowFiles("workflows")
    world = World(workflows=files)         # `files` is a read-only mapping of name -> Machine; every lookup sees the files as they are now

The workflows are named by their file (`knowledge_asset.yml` is `knowledge_asset`), as `World` and the evaluation code name them. A workflow can be changed without restarting. A file that does not make sense (it cannot be read, or `Machine.load` lists problems) is left out with a warning saying why,
and the workflow last read from that same file stays in use; a file that has never been read correctly gives no workflow. A file that is gone takes its workflow with it.
`rejected` holds, for each file that is wrong now, its problems. `register(name, machine)` adds a workflow that is not a file (one built in code, or in a test).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

from ..access_control.access_policy_engine.machine import Machine
from ..support.model import InvalidWorkflow

log = logging.getLogger(__name__)


class WorkflowFiles(Mapping):
    def __init__(self, directory: str | Path, registered: dict[str, Machine] | None = None) -> None:  # `registered`: name -> Machine, workflows that are not files
        self.directory = Path(directory)
        self.rejected: dict[str, list[str]] = {}
        self._registered: dict[str, Machine] = dict(registered or {})
        self._seen: tuple | None = None  # (path, modification time, size) of every file at the last read
        self._by_path: dict[str, Machine] = {}
        self._machines: dict[str, Machine] = {}
        self._lock = threading.RLock()

    def machines(self) -> dict[str, Machine]:
        """Every workflow by name: the files as they are now, plus the registered ones."""
        with self._lock:
            files = self._listing()
            if files != self._seen:
                self._read(files)
            return {**self._machines, **self._registered}

    def _listing(self) -> tuple:
        out = []
        for path in self.directory.glob("*.yml"):
            try:
                stat = path.stat()
            except OSError:  # gone since it was listed
                continue
            out.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(sorted(out))

    def _read(self, files: tuple) -> None:
        by_path: dict[str, Machine] = {}
        rejected: dict[str, list[str]] = {}
        for path, _, _ in files:
            try:
                by_path[path] = Machine.load(path)
            except InvalidWorkflow as error:
                rejected[path] = error.problems
                log.warning("workflow file %s was not loaded: %s", path, "; ".join(error.problems))
                if path in self._by_path:  # keep the workflow last read from this file rather than lose the one documents are under
                    by_path[path] = self._by_path[path]
        self._seen, self._by_path, self.rejected = files, by_path, rejected
        self._machines = {Path(path).stem: machine for path, machine in by_path.items()}

    def register(self, name: str, machine: Machine) -> None:
        with self._lock:
            self._registered[name] = machine

    def unregister(self, name: str) -> None:
        with self._lock:
            self._registered.pop(name, None)

    def __getitem__(self, name: str) -> Machine:
        return self.machines()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.machines())

    def __len__(self) -> int:
        return len(self.machines())
