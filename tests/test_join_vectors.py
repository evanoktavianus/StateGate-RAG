"""The script that joins the parts of the vector files: it reads eval/data/release_files.json and keeps only a joined file whose sha256 matches."""

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("join_vectors", ROOT / "scripts" / "join_vectors.py")
J = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J)


def test_the_manifest_lists_the_two_vector_files_with_their_parts():
    m = J.manifest()
    assert sorted(m) == ["eval/data/documents/embeddings.f32", "eval/data/regulations/embeddings.f32"]
    for entry in m.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert sum(p["size"] for p in entry["parts"]) == entry["size"]
        assert all(p["size"] <= 48_000_000 for p in entry["parts"])  # GitHub takes no file over 100 MB, and warns above 50 MB


def test_the_parts_in_the_repository_are_the_ones_the_manifest_lists():
    for path, entry in J.manifest().items():
        for part in entry["parts"]:
            f = ROOT / path
            f = f.with_name(part["name"])
            if not f.exists():
                pytest.skip(f"{f.name} is not here")
            assert f.stat().st_size == part["size"] and J.sha256_of(f) == part["sha256"], part["name"]


def _wanted(tmp_path, data=b"vectors", size=3):
    folder = tmp_path / "x"
    folder.mkdir()
    parts = []
    for i in range(0, len(data), size):
        block = data[i:i + size]
        name = f"a.bin.part-{i // size + 1:02d}"
        (folder / name).write_bytes(block)
        parts.append({"name": name, "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    return {"x/a.bin": {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}}


def test_the_parts_are_joined_into_the_file(tmp_path):
    wanted = _wanted(tmp_path)
    assert J.join([], tmp_path, wanted) == 0
    assert (tmp_path / "x" / "a.bin").read_bytes() == b"vectors"
    (tmp_path / "x" / "a.bin.part-01").unlink()
    assert J.join([], tmp_path, wanted) == 0  # already there with the right hash: left alone


def test_a_damaged_part_is_reported_and_leaves_no_file(tmp_path, capsys):
    wanted = _wanted(tmp_path)
    (tmp_path / "x" / "a.bin.part-02").write_bytes(b"XXX")
    assert J.join([], tmp_path, wanted) == 1
    assert not (tmp_path / "x" / "a.bin").exists() and not list((tmp_path / "x").glob("*.joining")) and "part differs" in capsys.readouterr().out


def test_a_missing_part_is_reported(tmp_path, capsys):
    wanted = _wanted(tmp_path)
    (tmp_path / "x" / "a.bin.part-03").unlink()
    assert J.join([], tmp_path, wanted) == 1 and "part missing" in capsys.readouterr().out


def test_a_file_the_manifest_does_not_list_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        J.join(["x/other.bin"], tmp_path, _wanted(tmp_path))
