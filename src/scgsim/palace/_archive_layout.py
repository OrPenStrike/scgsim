"""Sibling `.tar.gz` layout for a prepared Palace run folder."""

from __future__ import annotations

from pathlib import Path

_LEGACY_INNER_ARCHIVE = "palace_handoff.tar.gz"


def sibling_run_archive_path(run_dir: Path) -> Path:
    """Return `{run_dir.name}.tar.gz` beside the run folder."""
    run_dir = Path(run_dir)
    return run_dir.parent / f"{run_dir.name}.tar.gz"


def sibling_run_archive_relative(run_dir: Path) -> str:
    """Return the run-folder-relative path of the sibling archive."""
    return f"../{Path(run_dir).name}.tar.gz"


def run_archive_arcname(run_dir: Path, path: Path) -> str:
    """Name a tar member so extracting the sibling archive recreates the folder."""
    return f"{Path(run_dir).name}/{path.relative_to(run_dir).as_posix()}"


def logical_tar_member_name(member_name: str, folder: str) -> str | None:
    """Map a tar member name onto a path inside the run folder.

    New archives prefix members with the run folder name. Legacy archives store
    run-folder-relative paths at the tar root. The top-level folder member has
    no logical path.
    """
    name = member_name.replace("\\", "/").rstrip("/")
    prefix = f"{folder}/"
    if name == folder:
        return None
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def resolve_run_archive_path(root: Path, relative: str) -> Path:
    """Resolve a resource-record archive path without leaving the parent folder."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact paths must be non-empty relative paths.")
    root = Path(root).resolve()
    expected = sibling_run_archive_relative(root)
    if relative == expected:
        candidate = sibling_run_archive_path(root).resolve()
        if candidate.parent != root.parent:
            raise ValueError("artifact path escapes the run directory.")
        return candidate
    if relative == _LEGACY_INNER_ARCHIVE:
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("artifact path escapes the run directory.")
        return candidate
    raise ValueError(
        "handoff archive must be named like the run folder and sit beside it "
        f"(expected {expected!r})."
    )
