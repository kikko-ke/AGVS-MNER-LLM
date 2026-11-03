"""Storage safety checks; these tests never touch real experiment outputs."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import relocate_old_checkpoints as relocation


def checkpoint(tmp_path, content=b"checkpoint bytes"):
    project = tmp_path / "project"
    source = project / "outputs_llm" / "old_run" / "best_model.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    archive = tmp_path / "archive"
    destination = archive / source.relative_to(project)
    return project, archive, relocation.Checkpoint(source, destination, len(content), len(content))


def test_same_filesystem_plan_is_read_only(tmp_path):
    project, archive, item = checkpoint(tmp_path)
    free = relocation.shutil.disk_usage(project).free
    plan = relocation.make_plan(project, archive, target_free=free + relocation.GIB)
    assert "SAME filesystem" in plan.blockers[0]
    assert not archive.exists()
    assert not item.source.is_symlink()


def test_no_migration_when_target_already_met(tmp_path):
    project, archive, _ = checkpoint(tmp_path)
    plan = relocation.make_plan(project, archive, target_free=1)
    assert not plan.checkpoints and not plan.blockers
    assert not archive.exists()


def different_filesystem(monkeypatch, project):
    """Simulate two mounts for planning only, without bypassing copy verification."""
    device = project.stat().st_dev
    monkeypatch.setattr(relocation, "_device", lambda path: device if path == project else device + 1)


def test_plan_targets_only_explicit_legacy_weights_and_checks_archive_space(tmp_path, monkeypatch):
    project, archive, item = checkpoint(tmp_path, b"x" * 4096)
    for relative in ("outputs_v61/run/best_model.pt", "dataset/best_model.pt",
                     "outputs_llm/run/config.json"):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve")
    different_filesystem(monkeypatch, project)
    monkeypatch.setattr(relocation.shutil, "disk_usage", lambda _: SimpleNamespace(free=4096))
    plan = relocation.make_plan(project, archive, target_free=8192, headroom=1)
    assert [row.source for row in plan.checkpoints] == [item.source]
    assert any("lacks space" in reason for reason in plan.blockers)
    assert not archive.exists()


def test_rejects_nested_archive(tmp_path):
    project, _, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="separate directory trees"):
        relocation.make_plan(project, project / "archive")


def test_blocked_apply_never_starts_a_copy(tmp_path, monkeypatch):
    project, archive, item = checkpoint(tmp_path)
    monkeypatch.setattr(relocation.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    monkeypatch.setattr(relocation.sys, "argv", ["relocate", "--project", str(project),
                                               "--archive", str(archive), "--apply"])
    assert relocation.main() == 1
    assert not archive.exists() and not item.source.is_symlink()


def test_conflicting_archive_never_overwritten(tmp_path):
    project, archive, item = checkpoint(tmp_path)
    item.destination.parent.mkdir(parents=True)
    item.destination.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Conflicting archive"):
        relocation.relocate_one(item, project, archive)
    assert item.source.read_bytes() == b"checkpoint bytes"
    assert item.destination.read_bytes() == b"different checkpoint"
    assert not item.source.is_symlink()


def test_link_failure_keeps_source_and_verified_archive(tmp_path, monkeypatch):
    project, archive, item = checkpoint(tmp_path)

    def fail_link(*args, **kwargs):
        raise OSError("simulated symlink permission failure")

    monkeypatch.setattr(Path, "symlink_to", fail_link)
    with pytest.raises(OSError, match="symlink permission"):
        relocation.relocate_one(item, project, archive)
    assert item.source.read_bytes() == item.destination.read_bytes() == b"checkpoint bytes"
    assert not item.source.is_symlink()
    assert not list(project.rglob(".checkpoint-*"))
    assert not list(archive.rglob(".checkpoint-copy-*"))


def test_copy_corruption_never_replaces_source(tmp_path, monkeypatch):
    project, archive, item = checkpoint(tmp_path)
    monkeypatch.setattr(relocation.shutil, "copyfileobj", lambda source, output, **_: output.write(b"bad copy"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        relocation.relocate_one(item, project, archive)
    assert item.source.read_bytes() == b"checkpoint bytes"
    assert not item.source.is_symlink()
    assert not item.destination.exists()
    assert not list(archive.rglob(".checkpoint-copy-*"))


def test_concurrent_source_update_never_replaced_by_link(tmp_path, monkeypatch):
    project, archive, item = checkpoint(tmp_path)
    copy = relocation.shutil.copyfileobj

    def changed_copy(source, output, **kwargs):
        copy(source, output, **kwargs)
        item.source.write_bytes(b"new weights written by another process")

    monkeypatch.setattr(relocation.shutil, "copyfileobj", changed_copy)
    with pytest.raises(ValueError, match="Source changed"):
        relocation.relocate_one(item, project, archive)
    assert item.source.read_bytes() == b"new weights written by another process"
    assert not item.source.is_symlink() and not item.destination.exists()


def require_symlinks(tmp_path):
    target = tmp_path / "symlink_target"
    link = tmp_path / "symlink_probe"
    target.write_text("probe")
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"OS does not permit symlinks: {error}")
    link.unlink()
    target.unlink()


def test_verified_relocation_preserves_bytes_and_reuses_interrupted_copy(tmp_path):
    require_symlinks(tmp_path)
    project, archive, item = checkpoint(tmp_path)
    item.destination.parent.mkdir(parents=True)
    item.destination.write_bytes(item.source.read_bytes())
    digest = relocation.relocate_one(item, project, archive)
    assert item.source.is_symlink()
    assert item.source.resolve() == item.destination
    assert item.source.read_bytes() == b"checkpoint bytes"
    assert digest == relocation.sha256(item.destination)
    assert '"status": "linked"' in (archive / "relocation_journal.jsonl").read_text()


def test_successful_new_copy_and_archive_symlink_guard(tmp_path):
    require_symlinks(tmp_path)
    project, archive, item = checkpoint(tmp_path)
    digest = relocation.relocate_one(item, project, archive)
    assert item.source.is_symlink() and relocation.sha256(item.source) == digest
    alias = tmp_path / "archive_alias"
    alias.symlink_to(archive, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink/junction"):
        relocation.make_plan(project, alias)
