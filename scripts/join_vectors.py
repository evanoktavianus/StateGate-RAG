"""Joins the parts of the two vector files, and checks each part and the joined file against the sha256 in eval/data/release_files.json.

    python3 scripts/join_vectors.py                       # both files (510 MB)
    python3 scripts/join_vectors.py eval/data/documents/embeddings.f32

Each file is kept in the repository as parts of at most 48 MB beside the place it belongs (`embeddings.f32.part-01`, ...), because GitHub takes no file over 100 MB. A file already there with the right hash is left alone; a joined file is kept only if its hash matches.

    python3 scripts/join_vectors.py --split eval/data/documents/embeddings.f32   # cut a whole file into parts and write its entry to the manifest
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "eval" / "data" / "release_files.json"
PART_BYTES = 48_000_000


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(text: str | None = None) -> dict[str, dict]:
    """path -> {size, sha256, parts: [{name, size, sha256}]}, from `eval/data/release_files.json`."""
    return json.loads(text if text is not None else MANIFEST.read_text(encoding="utf-8"))["files"]


def join(paths: list[str], root: Path = ROOT, wanted: dict | None = None) -> int:
    wanted = wanted if wanted is not None else manifest()
    chosen = paths or list(wanted)
    unknown = [p for p in chosen if p not in wanted]
    if unknown:
        sys.exit(f"not in eval/data/release_files.json: {unknown}")
    bad = 0
    for p in chosen:
        entry = wanted[p]
        dest = root / p
        if dest.exists() and dest.stat().st_size == entry["size"] and sha256_of(dest) == entry["sha256"]:
            print(f"ok (already there)  {p}")
            continue
        tmp = dest.with_name(dest.name + ".joining")
        problem = ""
        with tmp.open("wb") as out:
            for part in entry["parts"]:
                src = dest.with_name(part["name"])
                if not src.exists():
                    problem = f"part missing: {src.relative_to(root)}"
                    break
                if sha256_of(src) != part["sha256"]:
                    problem = f"part differs from the manifest: {src.relative_to(root)}"
                    break
                with src.open("rb") as f:
                    while block := f.read(1 << 20):
                        out.write(block)
        if not problem and sha256_of(tmp) != entry["sha256"]:
            problem = "joined file differs from the manifest"
        if problem:
            tmp.unlink(missing_ok=True)
            print(f"FAILED  {p}: {problem}")
            bad += 1
        else:
            tmp.replace(dest)
            print(f"ok  {p}")
    return bad


def split(path: Path, root: Path = ROOT, part_bytes: int = PART_BYTES) -> dict:
    """Cut a whole file into parts beside it and return its manifest entry."""
    parts = []
    with path.open("rb") as f:
        number = 0
        while block := f.read(part_bytes):
            number += 1
            name = f"{path.name}.part-{number:02d}"
            path.with_name(name).write_bytes(block)
            parts.append({"name": name, "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    return {"size": path.stat().st_size, "sha256": sha256_of(path), "parts": parts}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--split"]:
        data = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"files": {}}
        for arg in sys.argv[2:]:
            data["files"][str(Path(arg).resolve().relative_to(ROOT))] = split(Path(arg).resolve())
        MANIFEST.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        sys.exit(0)
    sys.exit(1 if join(sys.argv[1:]) else 0)
