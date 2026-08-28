"""Shared headless renderer and Palace geometry-preview loader."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import math
import shutil
import tempfile
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_MODES = ("materials", "boundaries", "surface_epr", "mesh")
_MODE_TITLES = {
    "materials": "Materials",
    "boundaries": "Boundaries",
    "surface_epr": "Surface EPR",
    "mesh": "Mesh",
}
_VIEWS = (
    ("plus-x", "+X", (1.0, 0.0, 0.0), None),
    ("minus-x", "-X", (-1.0, 0.0, 0.0), None),
    ("plus-y", "+Y", (0.0, 1.0, 0.0), None),
    ("minus-y", "-Y", (0.0, -1.0, 0.0), None),
    ("plus-z", "+Z", (0.0, 0.0, 1.0), None),
    ("minus-z", "-Z", (0.0, 0.0, -1.0), None),
    ("above-ne", "Above +X +Y", (1.0, 1.0, 1.0), None),
    ("above-sw", "Above -X -Y", (-1.0, -1.0, 1.0), None),
    ("below-se", "Below +X -Y", (1.0, -1.0, -1.0), None),
    ("below-nw", "Below -X +Y", (-1.0, 1.0, -1.0), None),
    (
        "x-center-clip",
        "X-center visualization clip — not solver boundary",
        (1.0, 1.0, 1.0),
        ("x", 1.0),
    ),
    (
        "y-center-clip",
        "Y-center visualization clip — not solver boundary",
        (1.0, 1.0, 1.0),
        ("y", 1.0),
    ),
)
_FIXED_COLORS = {
    "Ground": "#3A3A3A",
    "PEC": "#3A3A3A",
    "LumpedPort": "#CC3377",
    "MA": "#E69F00",
    "MS": "#009E73",
    "SA": "#0072B2",
    "MS_MA": "#7B2CBF",
    "unassigned": "#D3D3D3",
}
_OTHER_COLORS = (
    "#56B4E9",
    "#F0E442",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#E69F00",
    "#000000",
)
_DIMENSIONS = {
    "vertex": 0,
    "line": 1,
    "line3": 1,
    "triangle": 2,
    "triangle6": 2,
    "quad": 2,
    "quad8": 2,
    "quad9": 2,
    "tetra": 3,
    "tetra10": 3,
    "hexahedron": 3,
    "hexahedron20": 3,
    "hexahedron27": 3,
    "wedge": 3,
    "pyramid": 3,
}


@dataclass(frozen=True)
class _Part:
    dataset: Any
    semantic_id: str
    label: str
    role: str
    color: str
    opacity: float
    count: int
    show_edges: bool = False


@dataclass(frozen=True)
class _Mode:
    parts: tuple[_Part, ...] = ()
    unavailable_reason: str | None = None


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
        self._modes = dict(modes)
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
        """Display one interactive semantic scene in a connected notebook."""
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
        plotter = pv.Plotter(notebook=True, off_screen=True)
        plotter.set_background("white")
        _add_parts(plotter, selected.parts)
        plotter.add_axes(line_width=3, labels_off=False)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
        return plotter.show(
            jupyter_backend="server",
            return_viewer=True,
            jupyter_kwargs={"server_proxy_enabled": True},
        )

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


def inspect_palace_geometry(run_dir: str | Path) -> GeometryPreview:
    """Bind one Palace mesh to its exact semantic/config metadata."""
    root = Path(run_dir).expanduser().resolve()
    paths = {
        "palace.msh": root / "palace.msh",
        "config.json": root / "config.json",
        "metadata/mesh_manifest.json": root / "metadata/mesh_manifest.json",
        "metadata/palace_index_map.json": root / "metadata/palace_index_map.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Palace preview inputs are missing: {missing!r}")

    try:
        import meshio
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "Palace geometry preview requires scgsim[visualization]"
        ) from exc

    mesh_manifest = _json(paths["metadata/mesh_manifest.json"])
    index_map = _json(paths["metadata/palace_index_map.json"])
    config = _json(paths["config.json"])
    mesh = meshio.read(paths["palace.msh"])
    grid = pv.from_meshio(mesh)
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None or len(physical_blocks) != len(mesh.cells):
        raise ValueError("Palace mesh has no complete gmsh:physical cell tags")
    dimensions: list[int] = []
    cell_types: list[str] = []
    physical: list[int] = []
    for block, tags in zip(mesh.cells, physical_blocks, strict=True):
        if block.type not in _DIMENSIONS:
            raise ValueError(f"unsupported Palace mesh cell type: {block.type}")
        dimensions.extend([_DIMENSIONS[block.type]] * len(block.data))
        cell_types.extend([block.type] * len(block.data))
        physical.extend(int(value) for value in tags)
    if len(physical) != grid.n_cells:
        raise ValueError("Palace mesh cell-tag cardinality mismatch")
    dimensions_array = np.asarray(dimensions)
    physical_array = np.asarray(physical)
    cell_types_array = np.asarray(cell_types)

    grouped_records: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for record in mesh_manifest.get("groups", ()):
        for attribute in record["attributes"]:
            key = (int(record["dimension"]), int(attribute))
            grouped_records.setdefault(key, []).append(record)
    actual = {
        (int(dim), int(attribute))
        for dim, attribute in zip(dimensions, physical, strict=True)
    }
    if actual != set(grouped_records):
        raise ValueError("Palace mesh physical tags do not match mesh manifest")
    field_names: dict[tuple[int, int], list[str]] = {}
    for name, value in mesh.field_data.items():
        key = (int(value[1]), int(value[0]))
        field_names.setdefault(key, []).append(str(name))
    if actual != set(field_names):
        raise ValueError("Palace mesh field names do not cover every physical tag")
    records = {
        key: next(
            (item for item in items if item.get("solver_use") == "solver_active"),
            items[0],
        )
        for key, items in grouped_records.items()
    }

    def cells(dimension: int, attribute: int) -> Any:
        ids = np.flatnonzero(
            (dimensions_array == dimension) & (physical_array == attribute)
        )
        if not len(ids):
            raise ValueError(
                f"Palace physical group {(dimension, attribute)} has no cells"
            )
        return grid.extract_cells(ids)

    materials = _palace_materials(config, records, cells)
    boundaries = _palace_boundaries(config, index_map, records, cells)
    surface_epr = _palace_surface_epr(config, index_map, records, cells)
    mesh_parts: list[_Part] = []
    for dimension, attribute in sorted(actual):
        types = sorted(
            set(
                cell_types_array[
                    (dimensions_array == dimension) & (physical_array == attribute)
                ]
            )
        )
        name = " / ".join(sorted(field_names[(dimension, attribute)]))
        mesh_parts.append(
            _Part(
                dataset=cells(dimension, attribute),
                semantic_id=f"mesh:{dimension}:{attribute}",
                label=f"{name} · dim {dimension} · {'/'.join(types)}",
                role="solution physical group",
                color=_color(name),
                opacity=0.35 if dimension == 3 else 0.85,
                count=int(
                    (
                        (dimensions_array == dimension) & (physical_array == attribute)
                    ).sum()
                ),
                show_edges=True,
            )
        )
    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    return GeometryPreview(
        root=root,
        backend="palace",
        source_hashes=source_hashes,
        modes={
            "materials": _Mode(tuple(materials)),
            "boundaries": _Mode(tuple(boundaries)),
            "surface_epr": (
                _Mode(tuple(surface_epr))
                if surface_epr
                else _Mode(
                    unavailable_reason="No structured Surface-EPR assignment is present."
                )
            ),
            "mesh": _Mode(tuple(mesh_parts)),
        },
    )


def _palace_materials(
    config: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[_Part]:
    assignments: set[int] = set()
    for material in config.get("Domains", {}).get("Materials", ()):
        for attribute in material.get("Attributes", ()):
            attribute = int(attribute)
            if attribute in assignments:
                raise ValueError(f"Palace volume {attribute} has multiple materials")
            assignments.add(attribute)
    parts: list[_Part] = []
    volume_attributes = {
        attribute for dimension, attribute in records if dimension == 3
    }
    if volume_attributes != assignments:
        raise ValueError("Palace volume groups and config materials do not match")
    for attribute in sorted(assignments):
        record = records[(3, attribute)]
        physical = record.get("physical_attribute", {})
        if not isinstance(physical, Mapping):
            raise TypeError("Palace physical material identity must be a mapping")
        kinds = tuple(physical.get("material_kinds", ()))
        ids = tuple(physical.get("material_ids", ()))
        if len(kinds) != 1 or len(ids) != 1:
            raise ValueError("Palace volume requires one exact material identity")
        material_id = str(ids[0])
        kind = str(kinds[0])
        dataset = cells(3, attribute)
        parts.append(
            _Part(
                dataset,
                material_id,
                material_id,
                kind,
                _color(material_id),
                0.08 if kind == "vacuum" else 0.42,
                dataset.n_cells,
            )
        )
    return parts


def _palace_boundaries(
    config: Mapping[str, Any],
    index_map: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[_Part]:
    boundary_root = config.get("Boundaries", {})
    assignments: dict[int, tuple[str, str, str]] = {}
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for item in index_map.get("entries", ()):
        if item.get("index") is None:
            continue
        key = (str(item.get("section")), int(item["index"]))
        if key in indexed:
            raise ValueError(f"Palace index map has duplicate identity {key!r}")
        indexed[key] = item

    def add(attribute: int, semantic_id: str, label: str, role: str) -> None:
        attribute = int(attribute)
        previous = assignments.get(attribute)
        value = (semantic_id, label, role)
        if previous is not None and previous != value:
            raise ValueError(f"Palace boundary {attribute} has conflicting assignments")
        assignments[attribute] = value

    for section, role in (("Ground", "Ground"), ("PEC", "PEC")):
        for attribute in boundary_root.get(section, {}).get("Attributes", ()):
            add(int(attribute), role, role, role)
    for section, role in (("Terminal", "Terminal"), ("LumpedPort", "LumpedPort")):
        configured: set[int] = set()
        for item in boundary_root.get(section, ()):
            index = int(item["Index"])
            if index in configured:
                raise ValueError(f"Palace {section} index {index} is duplicated")
            configured.add(index)
            record = indexed.get((f"Boundaries.{section}", index))
            if record is None:
                raise ValueError(
                    f"Palace {section} index {index} lacks index-map identity"
                )
            label = str(
                record.get("entry_name")
                or record.get("terminal_name")
                or record.get("net_id")
                or f"{section} {index}"
            )
            attributes = tuple(int(value) for value in item.get("Attributes", ()))
            if attributes != tuple(
                int(value) for value in record.get("attributes", ())
            ):
                raise ValueError(f"Palace {section} config/index-map mismatch")
            for attribute in attributes:
                add(int(attribute), f"{section}:{index}", label, role)
        mapped = {
            index
            for (mapped_section, index) in indexed
            if mapped_section == f"Boundaries.{section}"
        }
        if mapped != configured:
            raise ValueError(f"Palace {section} config/index-map identities differ")
    parts: list[_Part] = []
    surface_attributes = {
        attribute for dimension, attribute in records if dimension == 2
    }
    missing = sorted(set(assignments) - surface_attributes)
    if missing:
        raise ValueError(
            f"Palace configured boundary attributes are absent: {missing!r}"
        )
    for attribute in sorted(surface_attributes):
        semantic_id, label, role = assignments.get(
            attribute,
            (
                f"unassigned:{attribute}",
                "No explicit boundary assignment",
                "unassigned",
            ),
        )
        dataset = cells(2, attribute)
        parts.append(
            _Part(
                dataset,
                semantic_id,
                label,
                role,
                _color(role if role in _FIXED_COLORS else semantic_id),
                1.0 if role != "unassigned" else 0.5,
                dataset.n_cells,
            )
        )
    return parts


def _palace_surface_epr(
    config: Mapping[str, Any],
    index_map: Mapping[str, Any],
    records: Mapping[tuple[int, int], Mapping[str, Any]],
    cells: Any,
) -> list[_Part]:
    entries: dict[int, Mapping[str, Any]] = {}
    for item in index_map.get("entries", ()):
        if item.get("section") != "Boundaries.Postprocessing.Dielectric":
            continue
        index = int(item["index"])
        if index in entries:
            raise ValueError(f"Palace Surface-EPR index {index} is duplicated")
        entries[index] = item
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    configured: set[int] = set()
    for item in (
        config.get("Boundaries", {}).get("Postprocessing", {}).get("Dielectric", ())
    ):
        index = int(item["Index"])
        if index in configured:
            raise ValueError(f"Palace Surface-EPR index {index} is duplicated")
        configured.add(index)
        record = entries.get(index)
        if record is None or tuple(item.get("Attributes", ())) != tuple(
            record.get("attributes", ())
        ):
            raise ValueError("Palace Surface-EPR config/index-map mismatch")
        for attribute in item.get("Attributes", ()):
            grouped.setdefault(int(attribute), []).append(record)
    if set(entries) != configured:
        raise ValueError("Palace Surface-EPR config/index-map identities differ")
    parts: list[_Part] = []
    for attribute, mappings in sorted(grouped.items()):
        if (2, attribute) not in records:
            raise ValueError(f"Surface-EPR attribute {attribute} is absent from mesh")
        kinds = {
            str(
                item.get("interface_type")
                or (
                    item.get("metadata", {}).get("interface_type")
                    if isinstance(item.get("metadata"), Mapping)
                    else ""
                )
            )
            for item in mappings
        }
        role = (
            "MS_MA"
            if kinds == {"MA", "MS"}
            else next(iter(kinds))
            if len(kinds) == 1 and next(iter(kinds)) in {"MA", "MS", "SA"}
            else ""
        )
        if not role:
            raise ValueError(f"unsupported Surface-EPR mapping: {sorted(kinds)!r}")
        labels = sorted(
            {
                str(
                    item.get("surface_id")
                    or (
                        item.get("metadata", {}).get("surface_id")
                        if isinstance(item.get("metadata"), Mapping)
                        else None
                    )
                    or item.get("entry_name")
                    or item["index"]
                )
                for item in mappings
            }
        )
        dataset = cells(2, attribute)
        parts.append(
            _Part(
                dataset,
                f"surface-epr:{attribute}:{role}",
                " / ".join(labels),
                role,
                _FIXED_COLORS[role],
                0.95,
                dataset.n_cells,
            )
        )
    return parts


def _render_mode(
    *, root: Path, mode_name: str, parts: tuple[_Part, ...]
) -> tuple[Path, list[dict[str, Any]]]:
    if not parts:
        raise ValueError(f"available preview mode {mode_name} has no geometry")
    try:
        import pyvista as pv
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("geometry rendering requires scgsim[visualization]") from exc

    final_dir = root / "previews/geometry" / mode_name
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{mode_name}-", dir=final_dir.parent
    ) as raw:
        temporary = Path(raw)
        bounds = _combined_bounds(parts)
        center = (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        for filename, label, direction, clip in _VIEWS:
            plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
            plotter.set_background("white")
            _add_parts(plotter, parts, clip=clip, center=center)
            plotter.add_axes(line_width=3, labels_off=False)
            plotter.add_text(label, position="upper_left", font_size=12, color="black")
            vector = _unit(direction)
            distance = max(span * 2.5, 1.0)
            plotter.camera.position = tuple(
                center[i] + vector[i] * distance for i in range(3)
            )
            plotter.camera.focal_point = center
            plotter.camera.up = _camera_up(vector)
            plotter.camera.parallel_projection = True
            plotter.camera.parallel_scale = max(span * 0.62, 1.0)
            plotter.reset_camera_clipping_range()
            plotter.screenshot(temporary / f"{filename}.png")
            plotter.close()

        sheet = Image.new("RGB", (4800, 4200), "white")
        draw = ImageDraw.Draw(sheet)
        title_font = _font(ImageFont, 74)
        header_font = _font(ImageFont, 42)
        body_font = _font(ImageFont, 31)
        draw.text(
            (120, 70),
            f"{_MODE_TITLES[mode_name]} — Geometry Preview",
            fill="black",
            font=title_font,
        )
        draw.text(
            (120, 170),
            "Color is presentation; semantic authority is solver metadata.",
            fill="#444444",
            font=body_font,
        )
        legend = _legend(parts)
        columns = 4
        column_width = 1140
        rows = max(math.ceil(len(legend) / columns), 1)
        row_height = min(220, 1080 // rows)
        for index, item in enumerate(legend):
            column = index % columns
            row = index // columns
            x = 120 + column * column_width
            y = 270 + row * row_height
            draw.rectangle(
                (x, y + 8, x + 54, y + 62),
                fill=item["color"],
                outline="#333333",
            )
            wrapped = textwrap.wrap(item["label"], width=42) or [item["label"]]
            semantic = textwrap.wrap(item["semantic_id"], width=42) or [
                item["semantic_id"]
            ]
            text = "\n".join([*wrapped, *semantic, f"{item['role']} · {item['count']}"])
            draw.multiline_text(
                (x + 72, y), text, fill="black", font=body_font, spacing=2
            )
        draw.text(
            (120, 1360),
            "CURATED12 — cutaways are visualization clips, not solver boundaries",
            fill="#333333",
            font=header_font,
        )
        for index, (filename, _, _, _) in enumerate(_VIEWS):
            tile = Image.open(temporary / f"{filename}.png").convert("RGB")
            tile.thumbnail((1200, 900))
            x = (index % 4) * 1200
            y = 1500 + (index // 4) * 900
            sheet.paste(tile, (x, y))
        sheet.save(temporary / "contact-sheet.png")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(temporary, final_dir)

    artifacts: list[dict[str, Any]] = []
    for path in sorted(final_dir.glob("*.png")):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return final_dir / "contact-sheet.png", artifacts


def _add_parts(
    plotter: Any,
    parts: tuple[_Part, ...],
    *,
    clip: tuple[str, float] | None = None,
    center: tuple[float, float, float] | None = None,
) -> None:
    for part in parts:
        dataset = part.dataset
        if clip is not None:
            if center is None:
                raise ValueError("visualization clip requires a scene center")
            normal = (1.0, 0.0, 0.0) if clip[0] == "x" else (0.0, 1.0, 0.0)
            dataset = dataset.clip(normal=normal, origin=center, invert=False)
        if dataset.n_cells:
            plotter.add_mesh(
                dataset,
                color=part.color,
                opacity=part.opacity,
                show_edges=part.show_edges,
                edge_color="#555555",
                line_width=0.5,
            )


def _interactive_legend(mode_name: str, parts: tuple[_Part, ...]) -> str:
    rows = "".join(
        "<tr>"
        f"<td><span style='display:inline-block;width:1.1rem;height:1.1rem;"
        f"background:{html.escape(item['color'])};border:1px solid #555'></span></td>"
        f"<td>{html.escape(item['label'])}</td>"
        f"<td>{html.escape(item['semantic_id'])}</td>"
        f"<td>{html.escape(item['role'])}</td>"
        f"<td style='text-align:right'>{item['count']}</td>"
        "</tr>"
        for item in _legend(parts)
    )
    return (
        "<div style='margin:0.4rem 0 0.8rem'>"
        f"<strong>{html.escape(_MODE_TITLES[mode_name])} — Interactive Geometry</strong>"
        "<div style='color:#555;margin:0.2rem 0 0.5rem'>"
        "Color is presentation; semantic authority is solver metadata."
        "</div>"
        "<table style='border-collapse:collapse'>"
        "<thead><tr><th></th><th>Label</th><th>Semantic ID</th><th>Semantic role</th>"
        "<th>Entities</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _legend(parts: tuple[_Part, ...] | list[_Part]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str, str, str], int] = {}
    for part in parts:
        key = (part.semantic_id, part.label, part.role, part.color)
        aggregated[key] = aggregated.get(key, 0) + part.count
    return [
        {
            "semantic_id": semantic_id,
            "label": label,
            "role": role,
            "color": color,
            "count": count,
        }
        for (semantic_id, label, role, color), count in aggregated.items()
    ]


def _combined_bounds(parts: tuple[_Part, ...]) -> tuple[float, ...]:
    bounds = [part.dataset.bounds for part in parts if part.dataset.n_points]
    if not bounds:
        raise ValueError("preview geometry has no points")
    result = (
        min(item[0] for item in bounds),
        max(item[1] for item in bounds),
        min(item[2] for item in bounds),
        max(item[3] for item in bounds),
        min(item[4] for item in bounds),
        max(item[5] for item in bounds),
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("preview geometry bounds are non-finite")
    return result


def _color(identity: str) -> str:
    if identity in _FIXED_COLORS:
        return _FIXED_COLORS[identity]
    digest = hashlib.sha256(identity.encode()).digest()
    return _OTHER_COLORS[int.from_bytes(digest[:2], "big") % len(_OTHER_COLORS)]


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(item * item for item in vector))
    return tuple(item / length for item in vector)  # type: ignore[return-value]


def _camera_up(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    return (0.0, 1.0, 0.0) if abs(direction[2]) > 0.9 else (0.0, 0.0, 1.0)


def _font(image_font: Any, size: int) -> Any:
    try:
        return image_font.truetype("DejaVuSans.ttf", size)
    except OSError:
        return image_font.load_default()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _confined(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("preview artifact identity must be relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("preview artifact identity escapes the run directory")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_authority_sha256(path: Path) -> str:
    payload = _json(path)
    payload.pop("geometry_preview", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact_matches(root: Path, artifact: Any) -> bool:
    if not isinstance(artifact, Mapping):
        return False
    relative = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        return False
    path = _confined(root, relative)
    return path.is_file() and _sha256(path) == expected
