"""Source-bound preview object and explicit rendered/unavailable artifacts."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .scene import (
    _FIXED_COLORS,
    _MODES,
    _MODE_TITLES,
    _VIEWS,
    _Mode,
    _with_scene_palette,
)
from .rendering import (
    _add_parts,
    _configure_headless_rendering,
    _interactive_legend,
    _legend,
    _render_mode,
)
from .loaders._common import (
    _artifact_matches,
    _atomic_json,
    _confined,
    _json,
    _receipt_authority_sha256,
    _sha256,
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class GeometryPreviewArtifact:
    """One rendered contact sheet, or one explicit unavailable preview mode."""

    mode: str
    status: Literal["available", "unavailable"]
    contact_sheet: Path | None = None
    reason: str | None = None

    def _ipython_display_(self) -> None:
        try:
            from IPython.display import HTML, Image, display
        except ImportError:
            print(self)
            return
        if self.contact_sheet is not None:
            display(Image(filename=str(self.contact_sheet)))
        else:
            display(
                HTML(
                    "<div style='padding:0.8rem;border:1px solid #bbb'>"
                    f"<strong>{_MODE_TITLES[self.mode]}: Unavailable</strong><br>"
                    f"{self.reason}</div>"
                )
            )


class GeometryPreview:
    """A bound geometry source with four explicit preview modes."""

    def __init__(
        self,
        *,
        root: Path,
        backend: Literal["palace", "aedt"],
        source_hashes: Mapping[str, str],
        source_identity: Mapping[str, Any] | None = None,
        bind_aedt_receipt: bool = False,
        modes: Mapping[str, _Mode],
    ) -> None:
        self.root = root.resolve()
        self.backend = backend
        self._source_hashes = dict(source_hashes)
        self._source_identity = dict(source_identity or {})
        self._bind_receipt = bind_aedt_receipt
        self._modes = {name: _with_scene_palette(mode) for name, mode in modes.items()}
        if set(self._modes) != set(_MODES):
            raise ValueError("geometry preview requires exactly four named modes")
        self._manifest_path = self.root / "metadata/geometry_preview_manifest.json"

    def show_materials(self) -> GeometryPreviewArtifact:
        return self._show("materials")

    def show_boundaries(self) -> GeometryPreviewArtifact:
        return self._show("boundaries")

    def show_surface_epr(self) -> GeometryPreviewArtifact:
        return self._show("surface_epr")

    def show_mesh(self) -> GeometryPreviewArtifact:
        return self._show("mesh")

    def explore(
        self,
        mode: Literal["materials", "boundaries", "surface_epr", "mesh"],
    ) -> Any:
        """Display one inline interactive semantic scene in a notebook."""
        if mode not in _MODES:
            raise ValueError(f"unsupported geometry preview mode: {mode!r}")
        selected = self._modes[mode]
        if selected.unavailable_reason is not None:
            return GeometryPreviewArtifact(
                mode=mode,
                status="unavailable",
                reason=selected.unavailable_reason,
            )
        self._verify_sources()
        try:
            import pyvista as pv
            from IPython.display import HTML, display
        except ImportError as exc:
            raise RuntimeError(
                "interactive geometry preview requires scgsim[visualization] "
                "inside a Jupyter notebook"
            ) from exc
        display(HTML(_interactive_legend(mode, selected.parts)))
        _configure_headless_rendering()
        plotter = pv.Plotter(notebook=True, off_screen=True)
        plotter.set_background("white")
        _add_parts(plotter, selected.parts)
        plotter.add_axes(line_width=3, labels_off=False)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
        return plotter.show(jupyter_backend="html", return_viewer=True)

    def show_all_previews(self) -> None:
        results = (
            self.show_materials(),
            self.show_boundaries(),
            self.show_surface_epr(),
            self.show_mesh(),
        )
        try:
            from IPython.display import display
        except ImportError:
            for result in results:
                print(result)
        else:
            for result in results:
                display(result)

    def _show(self, mode_name: str) -> GeometryPreviewArtifact:
        mode = self._modes[mode_name]
        if mode.unavailable_reason is not None:
            return GeometryPreviewArtifact(
                mode=mode_name,
                status="unavailable",
                reason=mode.unavailable_reason,
            )
        self._verify_sources()
        try:
            contact_sheet, artifacts = _render_mode(
                root=self.root, mode_name=mode_name, parts=mode.parts
            )
        except Exception as exc:
            self._write_manifest(failed_mode=mode_name, failure=str(exc))
            raise
        self._write_manifest(rendered_mode=mode_name, artifacts=artifacts)
        return GeometryPreviewArtifact(
            mode=mode_name, status="available", contact_sheet=contact_sheet
        )

    def _verify_sources(self) -> None:
        for relative, expected in self._source_hashes.items():
            path = _confined(self.root, relative)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"geometry preview source changed: {relative}")
        expected_receipt = self._source_identity.get("receipt_authority_sha256")
        if expected_receipt is not None:
            receipt_path = self.root / "metadata/aedt_run_receipt.json"
            if (
                not receipt_path.is_file()
                or _receipt_authority_sha256(receipt_path) != expected_receipt
            ):
                raise ValueError("AEDT preview receipt authority changed")

    def _write_manifest(
        self,
        *,
        rendered_mode: str | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        failed_mode: str | None = None,
        failure: str | None = None,
    ) -> None:
        previous: dict[str, Any] = {}
        if self._manifest_path.is_file():
            previous = _json(self._manifest_path)
        renderer = {
            "name": "PyVista off-screen",
            "scgsim": _version("scgsim"),
            "pyvista": _version("pyvista"),
            "vtk": _version("vtk"),
            "meshio": _version("meshio") if self.backend == "palace" else None,
            "pillow": _version("pillow"),
        }
        views = [item[0] for item in _VIEWS]
        tile_size = [1600, 1200]
        contact_sheet_size = [4800, 4200]
        same_source = (
            previous.get("schema_version") == "scgsim.geometry-preview.v1"
            and previous.get("backend") == self.backend
            and previous.get("source_hashes") == self._source_hashes
            and previous.get("source_identity") == self._source_identity
            and previous.get("renderer") == renderer
            and previous.get("views") == views
            and previous.get("palette") == _FIXED_COLORS
            and previous.get("tile_size_px") == tile_size
            and previous.get("contact_sheet_size_px") == contact_sheet_size
        )
        mode_records = dict(previous.get("modes", {})) if same_source else {}
        for name in _MODES:
            item = self._modes[name]
            if name == rendered_mode:
                mode_records[name] = {
                    "status": "available",
                    "legend": _legend(item.parts),
                    "artifacts": artifacts,
                }
            elif name == failed_mode:
                mode_records[name] = {"status": "failed", "reason": failure}
            elif item.unavailable_reason:
                mode_records[name] = {
                    "status": "unavailable",
                    "reason": item.unavailable_reason,
                }
            else:
                existing = mode_records.get(name, {})
                existing_artifacts = existing.get("artifacts", [])
                if any(
                    not _artifact_matches(self.root, artifact)
                    for artifact in existing_artifacts
                ):
                    existing_artifacts = []
                mode_records[name] = {
                    "status": "available",
                    "legend": _legend(item.parts),
                    "artifacts": existing_artifacts,
                }
        payload = {
            "schema_version": "scgsim.geometry-preview.v1",
            "backend": self.backend,
            "source_hashes": self._source_hashes,
            "source_identity": self._source_identity,
            "renderer": renderer,
            "views": views,
            "palette": _FIXED_COLORS,
            "tile_size_px": tile_size,
            "contact_sheet_size_px": contact_sheet_size,
            "modes": mode_records,
            "simulation_status_affected": False,
            "identity_role": "preview receipt; excluded from solver and handoff identity",
        }
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self._manifest_path, payload)
        if self._bind_receipt:
            self._bind_aedt_receipt(payload)

    def _bind_aedt_receipt(self, manifest: Mapping[str, Any]) -> None:
        receipt_path = self.root / "metadata/aedt_run_receipt.json"
        if not receipt_path.is_file():
            return
        receipt = _json(receipt_path)
        artifacts = {
            item["path"]: item["sha256"]
            for mode in manifest["modes"].values()
            for item in mode.get("artifacts", ())
        }
        receipt["geometry_preview"] = {
            "status": "available" if artifacts else "not_rendered",
            "manifest": {
                "path": self._manifest_path.relative_to(self.root).as_posix(),
                "sha256": _sha256(self._manifest_path),
            },
            "artifacts": artifacts,
            "simulation_status_affected": False,
        }
        _atomic_json(receipt_path, receipt)
