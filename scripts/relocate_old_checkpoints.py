"""Plan (default) or verify-and-relocate old checkpoints to another filesystem.

Only the explicitly listed legacy experiments are eligible. The archive is not
an external backup: keep it with the instance and include it in future backups.
No torch, CUDA, network access, or third-party dependencies are needed.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile


GIB = 1024 ** 3
LEGACY_ROOTS = (
    "outputs_baseline",
    "outputs_llm",
    "outputs_atomic",
    "outputs_context_internal/llm_context_seed1234",
    "outputs_dropout_selection_internal",
    "outputs_component_v1",
)


@dataclass(frozen=True)
class Checkpoint:
    source: Path
    destination: Path
    size: int
    allocated: int


@dataclass
class Plan:
    project: Path
    archive: Path
    checkpoints: list
    data_free: int
    archive_free: int
    target_free: int
    headroom: int
    blockers: list

    @property
    def copy_bytes(self):
        return sum(item.size for item in self.checkpoints)

    @property
    def reclaim_bytes(self):
        return sum(item.allocated for item in self.checkpoints)


def _no_symlink_components(path):
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError(f"Refusing symlink/junction path component: {item}")


def _nearest_existing(path):
    while not path.exists():
        if path == path.parent:
            raise ValueError(f"No existing ancestor: {path}")
        path = path.parent
    if not path.is_dir():
        raise ValueError(f"Expected archive directory/ancestor: {path}")
    return path


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _device(path):
    return path.stat().st_dev


def _paths(project, archive):
    project = Path(os.path.abspath(project))
    archive = Path(os.path.abspath(archive))
    _no_symlink_components(project)
    _no_symlink_components(archive)
    project = project.resolve(strict=True)
    if not project.is_dir():
        raise ValueError(f"Not a project directory: {project}")
    if _inside(archive, project) or _inside(project, archive):
        raise ValueError("Archive and project must be separate directory trees")
    return project, archive


def make_plan(project, archive, target_free=20.5 * GIB, headroom=2 * GIB):
    """Read-only preflight: never create, copy, remove, or link anything here."""
    if not all(math.isfinite(v) and v > 0 for v in (target_free, headroom)):
        raise ValueError("Space targets must be finite and positive")
    project, archive = _paths(project, archive)
    ancestor = _nearest_existing(archive)
    data_free = shutil.disk_usage(project).free
    archive_free = shutil.disk_usage(ancestor).free
    plan = Plan(project, archive, [], data_free, archive_free,
                int(target_free), int(headroom), [])
    if data_free >= target_free:
        return plan
    if _device(project) == _device(ancestor):
        plan.blockers.append("Archive and project are on the SAME filesystem; relocation cannot free space")
        return plan
    for relative in LEGACY_ROOTS:
        root = project / relative
        _no_symlink_components(root)
        if not root.is_dir():
            continue
        for source in sorted(root.rglob("best_model.pt")):
            # Links have already been relocated; hard-linked files may reclaim
            # no blocks. Only ordinary files on the data filesystem qualify.
            info = source.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                continue
            _no_symlink_components(source)
            if info.st_dev != _device(project) or info.st_size <= 0:
                continue
            allocated = getattr(info, "st_blocks", None)
            allocated = allocated * 512 if allocated is not None else info.st_size
            if allocated <= 0:
                continue
            destination = archive / source.relative_to(project)
            _no_symlink_components(destination)
            if destination.exists() and not destination.is_file():
                raise ValueError(f"Archive destination is not a file: {destination}")
            plan.checkpoints.append(Checkpoint(source, destination, info.st_size, allocated))
            if data_free + plan.reclaim_bytes >= target_free:
                break
        if data_free + plan.reclaim_bytes >= target_free:
            break
    if data_free + plan.reclaim_bytes < target_free:
        plan.blockers.append("Eligible old checkpoints cannot reclaim enough space; no files will be changed")
    # Conservative even when verified copies from an interrupted run exist.
    if archive_free < plan.copy_bytes + headroom:
        plan.blockers.append("Archive filesystem lacks space for all selected copies plus headroom; no files will be changed")
    return plan


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Source must be an ordinary, unshared file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _journal(archive, record):
    path = archive / "relocation_journal.jsonl"
    _no_symlink_components(path)
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError(f"Refusing shared/non-file journal: {path}")
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def relocate_one(item, project, archive):
    """Verify a copy before atomically replacing the original with a link.

    A failed copy, checksum, or link creation leaves the source file intact.
    A verified archive copy may remain after interruption and is reused only if
    byte-identical. Existing different files are never overwritten.
    """
    project, archive = _paths(project, archive)
    source, destination = item.source, item.destination
    if not _inside(source, project) or destination != archive / source.relative_to(project):
        raise ValueError("Checkpoint paths do not match the bounded relocation plan")
    _no_symlink_components(source)
    _no_symlink_components(destination)
    original = _snapshot(source)
    if original[2] != item.size:
        raise ValueError(f"Source size changed since planning: {source}")
    digest = sha256(source)
    if _snapshot(source) != original:
        raise ValueError(f"Source changed while hashing: {source}")
    if destination.exists() and (not destination.is_file() or sha256(destination) != digest):
        raise ValueError(f"Conflicting archive file; refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_copy = None
    temp_link = None
    try:
        if not destination.exists():
            with tempfile.NamedTemporaryFile(prefix=".checkpoint-copy-", dir=destination.parent,
                                             delete=False) as output:
                temp_copy = Path(output.name)
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if sha256(temp_copy) != digest or _snapshot(source) != original:
                raise ValueError(f"Source changed or copied checksum mismatch: {source}")
            shutil.copystat(source, temp_copy)
            # Exclusive publication: cannot overwrite a file created meanwhile.
            os.link(temp_copy, destination)
            temp_copy.unlink()
            temp_copy = None
        _sync_directory(destination.parent)
        if sha256(destination) != digest or _snapshot(source) != original:
            raise ValueError(f"Verification failed before replacing source: {source}")
        record = {"source": str(source), "archive": str(destination),
                  "sha256": digest, "bytes": item.size}
        _journal(archive, {**record, "status": "copied_verified"})
        # Reserve an unpredictable temporary name in the original directory.
        fd, name = tempfile.mkstemp(prefix=".checkpoint-link-", dir=source.parent)
        os.close(fd)
        temp_link = Path(name)
        temp_link.unlink()
        temp_link.symlink_to(destination)
        if _snapshot(source) != original or temp_link.resolve(strict=True) != destination:
            raise ValueError(f"Source/link changed before atomic replacement: {source}")
        os.replace(temp_link, source)
        temp_link = None
        _sync_directory(source.parent)
        _journal(archive, {**record, "status": "linked"})
        return digest
    finally:
        for temporary in (temp_copy, temp_link):
            if temporary is not None and (temporary.exists() or temporary.is_symlink()):
                temporary.unlink()


def show_plan(plan):
    print(f"Project: {plan.project}")
    print(f"Archive: {plan.archive}")
    print(f"Data filesystem free: {plan.data_free / GIB:.2f} GiB; target: {plan.target_free / GIB:.2f} GiB")
    print(f"Archive filesystem free: {plan.archive_free / GIB:.2f} GiB")
    for item in plan.checkpoints:
        print(f"  {item.size / GIB:.3f} GiB  {item.source}\n             -> {item.destination}")
    print(f"Selected: {len(plan.checkpoints)} checkpoints; copy: {plan.copy_bytes / GIB:.2f} GiB")
    print(f"Expected data free: {(plan.data_free + plan.reclaim_bytes) / GIB:.2f} GiB")
    print(f"Required archive free (including headroom): {(plan.copy_bytes + plan.headroom) / GIB:.2f} GiB")
    for reason in plan.blockers:
        print(f"BLOCKED: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, default=Path("/root/mner_checkpoint_archive"))
    parser.add_argument("--target-free-gib", type=float, default=20.5)
    parser.add_argument("--headroom-gib", type=float, default=2.0)
    parser.add_argument("--apply", action="store_true", help="Copy, verify SHA256, then replace sources with symlinks")
    args = parser.parse_args()
    try:
        plan = make_plan(args.project, args.archive, args.target_free_gib * GIB, args.headroom_gib * GIB)
        show_plan(plan)
        if plan.blockers:
            return 1
        if not args.apply:
            print("PLAN ONLY: nothing changed. Review the paths, then rerun with --apply.")
            return 0
        if not plan.checkpoints:
            print("Target already met; nothing to relocate.")
            return 0
        for index, item in enumerate(plan.checkpoints, 1):
            if shutil.disk_usage(plan.project).free >= plan.target_free:
                break
            # Check available space again in case another process consumed it.
            ancestor = _nearest_existing(item.destination.parent)
            if _device(ancestor) == _device(plan.project):
                raise ValueError("Archive filesystem changed since preflight")
            if shutil.disk_usage(ancestor).free < item.size + plan.headroom:
                raise ValueError("Archive free space fell below the per-file safety limit")
            print(f"[{index}/{len(plan.checkpoints)}] Copying {item.source}", flush=True)
            digest = relocate_one(item, plan.project, plan.archive)
            print(f"Verified and linked: SHA256={digest}", flush=True)
        final_free = shutil.disk_usage(plan.project).free
        print(f"Data filesystem free now: {final_free / GIB:.2f} GiB")
        print(f"KEEP {plan.archive}; original checkpoint paths now depend on this archive.")
        if final_free < plan.target_free:
            print("Target not reached. Do not bypass the environment installer space check.")
            return 1
        return 0
    except (OSError, ValueError) as error:
        print(f"STOPPED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
