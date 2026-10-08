"""Load Palace mesh and metadata into detached semantic preview modes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._common import _json, _sha256
from ..preview import GeometryPreview
from ..scene import _FIXED_COLORS, _Mode, _Part, _color


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
    grid = _high_order_grid(mesh, mesh_manifest, pv)
    physical_blocks = mesh.cell_data.get("gmsh:physical")
    if physical_blocks is None or len(physical_blocks) != len(mesh.cells):
        raise ValueError("Palace mesh has no complete gmsh:physical cell tags")
    dimensions: list[int] = []
    cell_types: list[str] = []
    physical: list[int] = []
    for block, tags in zip(mesh.cells, physical_blocks, strict=True):
        dimensions.extend([block.dim] * len(block.data))
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


def _high_order_grid(mesh: Any, manifest: Mapping[str, Any], pv: Any) -> Any:
    """Retain complete Gmsh p3+ nodes in native VTK Lagrange cell order.

    meshio already maps quadratic cells to VTK order. Complete higher-order
    simplices retain Gmsh order, so native reference lattices supply the exact
    permutation instead of dropping their non-corner nodes.
    """
    import numpy as np
    from meshio._vtk_common import meshio_to_vtk_type
    from vtkmodules.vtkCommonDataModel import (
        vtkLagrangeCurve,
        vtkLagrangeTriangle,
        vtkLagrangeTetra,
    )

    records = manifest.get("meshing", {}).get("statistics", {}).get("element_types", ())
    families = {
        "line": (vtkLagrangeCurve, "VTK_LAGRANGE_CURVE"),
        "triangle": (vtkLagrangeTriangle, "VTK_LAGRANGE_TRIANGLE"),
        "tetra": (vtkLagrangeTetra, "VTK_LAGRANGE_TETRAHEDRON"),
    }
    cells = []
    for block in mesh.cells:
        record = next(
            (
                r
                for r in records
                if r["dimension"] == block.dim
                and r["nodes_per_element"] == block.data.shape[1]
            ),
            None,
        )
        family = next((name for name in families if block.type.startswith(name)), None)
        if record is None or record["order"] <= 2 or family is None:
            cells.append((block.type, block.data))
            continue
        cls, vtk_type = families[family]
        order, count, dimension = (
            record["order"],
            record["nodes_per_element"],
            block.dim,
        )
        native = np.asarray(record["reference_coordinates"]).reshape(count, dimension)
        if dimension == 1:
            native = (native + 1.0) / 2.0
        source_lattice = {
            tuple(round(float(x) * order) for x in point): i
            for i, point in enumerate(native)
        }
        cell = cls()
        cell.GetPointIds().SetNumberOfIds(count)
        cell.GetPoints().SetNumberOfPoints(count)
        cell.Initialize()
        target = np.asarray(cell.GetParametricCoords()).reshape(count, 3)[:, :dimension]
        permutation = [
            source_lattice[tuple(round(float(x) * order) for x in point)]
            for point in target
        ]
        cells.append((vtk_type, block.data[:, permutation]))
    # Supply each cell's complete connectivity length directly: the fixed-node
    # lookup in PyVista's meshio reader does not cover native Lagrange types.
    connectivity = [
        np.column_stack((np.full(len(data), data.shape[1]), data)).ravel()
        for _, data in cells
    ]
    cell_types = np.concatenate(
        [np.full(len(data), meshio_to_vtk_type[kind]) for kind, data in cells]
    )
    grid = pv.UnstructuredGrid(
        np.concatenate(connectivity).astype(np.int64, copy=False),
        cell_types,
        np.asarray(mesh.points, dtype=np.float64),
    )
    grid.point_data.update(
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in mesh.point_data.items()
        }
    )
    grid.cell_data.update(
        {key: np.concatenate(values) for key, values in mesh.cell_data.items()}
    )
    return grid
