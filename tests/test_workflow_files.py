"""A folder of workflow files read again whenever a file is added, removed or saved; a broken file keeps the workflow last read from it."""

import logging
import os
import shutil
from pathlib import Path

from docflow import Machine
from docflow.stores.workflow_files import WorkflowFiles

WORKFLOWS = Path(__file__).resolve().parent.parent / "workflows"


def copy(tmp_path, name, to=None):
    target = tmp_path / (to or name)
    shutil.copy(WORKFLOWS / name, target)
    return target


def touch(path):
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))  # a later modification time, whatever the clock resolution


def test_every_file_is_a_workflow_by_name_and_the_folder_is_a_mapping(tmp_path):
    copy(tmp_path, "knowledge_asset.yml")
    copy(tmp_path, "card1_policy_analysis_simple.yml")
    files = WorkflowFiles(tmp_path)
    assert sorted(files) == ["card1_policy_analysis_simple", "knowledge_asset"]
    assert len(files) == 2 and files["knowledge_asset"].initial == Machine.load(WORKFLOWS / "knowledge_asset.yml").initial and "nothing" not in files


def test_a_file_added_later_is_a_workflow_without_a_restart(tmp_path):
    copy(tmp_path, "knowledge_asset.yml")
    files = WorkflowFiles(tmp_path)
    assert len(files) == 1
    copy(tmp_path, "regulation_single.yml")
    assert "regulation_single" in files and len(files) == 2


def test_a_file_saved_again_changes_its_workflow(tmp_path):
    path = copy(tmp_path, "knowledge_asset.yml")
    files = WorkflowFiles(tmp_path)
    before = files["knowledge_asset"].guidance
    path.write_text(path.read_text().replace("Unapproved working document; do not present it as a knowledge asset.", "Working document, not approved."))
    touch(path)
    assert files["knowledge_asset"].guidance != before and "Working document, not approved." in files["knowledge_asset"].guidance.values()


def test_a_broken_file_keeps_the_workflow_last_read_from_it_and_says_why(tmp_path, caplog):
    path = copy(tmp_path, "knowledge_asset.yml")
    files = WorkflowFiles(tmp_path)
    good = files["knowledge_asset"]
    path.write_text("name: knowledge_asset\nstates: {}\n")
    touch(path)
    with caplog.at_level(logging.WARNING):
        assert files["knowledge_asset"] is good  # the workflow last read from this file stays in use
    assert any("was not loaded" in r.message for r in caplog.records) and list(files.rejected) == [str(path)] and files.rejected[str(path)]
    path.write_text((WORKFLOWS / "knowledge_asset.yml").read_text())  # saved correctly again
    touch(path)
    touch(path)
    assert files["knowledge_asset"] is not good and files.rejected == {}


def test_a_file_that_was_never_read_correctly_gives_no_workflow(tmp_path):
    (tmp_path / "broken.yml").write_text("name: broken\nstates: not a mapping\n")
    files = WorkflowFiles(tmp_path)
    assert len(files) == 0 and len(files.rejected) == 1


def test_a_file_that_is_gone_takes_its_workflow_with_it(tmp_path):
    path = copy(tmp_path, "knowledge_asset.yml")
    copy(tmp_path, "regulation_single.yml")
    files = WorkflowFiles(tmp_path)
    assert len(files) == 2
    path.unlink()
    assert list(files) == ["regulation_single"]


def test_a_workflow_that_is_not_a_file_can_be_registered_and_removed(tmp_path):
    copy(tmp_path, "knowledge_asset.yml")
    files = WorkflowFiles(tmp_path)
    built = Machine.load(WORKFLOWS / "regulation_single.yml")
    files.register("regulation_single", built)
    assert files["regulation_single"] is built and len(files) == 2
    files.unregister("regulation_single")
    assert list(files) == ["knowledge_asset"]


def test_a_world_uses_the_files_as_they_are_now(tmp_path):
    from docflow.stores.world import World

    path = copy(tmp_path, "card1_policy_analysis_simple.yml", "simple.yml")
    files = WorkflowFiles(tmp_path)
    world = World(workflows=files)
    world.add_org("O1"), world.add_user("u1", "O1"), world.add_workspace("W1", "O1", members={"u1": "editor"})
    world.add_document("d1", "W1", "u1", "simple")
    before = world.workflows["simple"]
    path.write_text(path.read_text() + "\n# saved again\n")
    touch(path)
    assert world.workflows["simple"] is not before and world.state_of("d1") == before.initial
