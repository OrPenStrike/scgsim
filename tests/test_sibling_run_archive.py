"""Sibling handoff archive layout."""

from __future__ import annotations

import tarfile
import tempfile
import unittest
from pathlib import Path

from scgsim.palace._archive_layout import (
    logical_tar_member_name,
    resolve_run_archive_path,
    run_archive_arcname,
    sibling_run_archive_path,
    sibling_run_archive_relative,
)


class SiblingRunArchiveTests(unittest.TestCase):
    def test_archive_sits_beside_run_folder_with_the_same_name(self) -> None:
        run_dir = Path("/tmp/runs/2026-08-19-scgsim-f1-ct448-Eigenmode01")
        archive = sibling_run_archive_path(run_dir)
        self.assertEqual(archive.parent, run_dir.parent)
        self.assertEqual(archive.name, f"{run_dir.name}.tar.gz")
        self.assertEqual(
            sibling_run_archive_relative(run_dir),
            f"../{run_dir.name}.tar.gz",
        )

    def test_tar_members_are_prefixed_with_the_folder_name(self) -> None:
        run_dir = Path("/tmp/runs/Run01")
        path = run_dir / "config.json"
        self.assertEqual(run_archive_arcname(run_dir, path), "Run01/config.json")
        self.assertEqual(logical_tar_member_name("Run01/config.json", "Run01"), "config.json")
        self.assertIsNone(logical_tar_member_name("Run01", "Run01"))
        self.assertEqual(logical_tar_member_name("config.json", "Run01"), "config.json")

    def test_resolve_accepts_sibling_path_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            run_dir = parent / "Run01"
            run_dir.mkdir()
            archive = sibling_run_archive_path(run_dir)
            archive.write_bytes(b"unused")
            resolved = resolve_run_archive_path(run_dir, "../Run01.tar.gz")
            self.assertEqual(resolved, archive.resolve())
            with self.assertRaises(ValueError):
                resolve_run_archive_path(run_dir, "../other.tar.gz")

    def test_round_trip_tar_extracts_to_the_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            run_dir = parent / "Run01"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}\n", encoding="utf-8")
            archive_path = sibling_run_archive_path(run_dir)
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(
                    run_dir / "config.json",
                    arcname=run_archive_arcname(run_dir, run_dir / "config.json"),
                    recursive=False,
                )
            self.assertEqual(archive_path.name, "Run01.tar.gz")
            self.assertEqual(archive_path.parent, run_dir.parent)
            extract_root = parent / "extracted"
            extract_root.mkdir()
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_root, filter="data")
            self.assertTrue((extract_root / "Run01" / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
