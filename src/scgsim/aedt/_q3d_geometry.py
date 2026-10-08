"""Body-backed Q3D geometry from an immutable normalized source snapshot.

Each source polygon-with-holes remains one body. No union, lattice conversion,
fragmentation, or native geometry operation belongs to this lowering step.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from scgsim.semantics.route_a import geometry_z_range
from scgsim.sgb import GeometryPlanSnapshot

from ._epr_models import canonical_sha256, detached
from .spec import (
    MatrixRunControl,
    PdkMaterial,
    Q3dBodySpec,
    Q3dNetSpec,
    Q3dSpec,
)
from .util import file_sha256, write_json


def _entity_polygons(entity, polygons):
    if entity.polygon_ids:
        return [
            (pid, polygons[pid].exterior, polygons[pid].holes)
            for pid in sorted(entity.polygon_ids)
        ]
    if entity.material_kind == "conductor":
        raise ValueError(f"Q3D conductor {entity.semantic_id!r} has no source polygons")
    geometry = entity.geometry
    if "outer_loop" in geometry:
        return [(None, geometry["outer_loop"], geometry.get("hole_loops", ()))]
    bounds = geometry["domain_bounds_um"]
    xmin, ymin = bounds["x_min_um"], bounds["y_min_um"]
    xmax, ymax = bounds["x_max_um"], bounds["y_max_um"]
    return [(None, ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), ())]


def lower_q3d_geometry(
    snapshot: GeometryPlanSnapshot,
    *,
    directory: Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    run_control: MatrixRunControl,
    region_padding_um: Sequence[float],
) -> Q3dSpec:
    """Write canonical source provenance and body declarations; never open AEDT."""
    if not isinstance(snapshot, GeometryPlanSnapshot):
        raise TypeError("snapshot must be GeometryPlanSnapshot")
    source = snapshot.geometry_input
    if source.boundary_curves or source.boundary_reconstruction:
        raise NotImplementedError(
            "AEDT finite Q3D lowering does not support active source curve intent"
        )
    stack = snapshot.stack
    conductors = [
        entity for entity in source.entities if entity.material_kind == "conductor"
    ]
    if any(entity.net_id is None for entity in conductors):
        raise ValueError(
            "Q3D conductors require explicit final GeometryPlan Net membership"
        )
    net_ids = {entity.net_id for entity in conductors}
    if set(net_types) != net_ids:
        raise ValueError("net_types must cover exactly the final conductor Nets")
    ground_nets = tuple(physical_ground_nets)
    if isinstance(physical_ground_nets, str | bytes) or not set(ground_nets) <= net_ids:
        raise ValueError("physical_ground_nets must select final conductor Nets")
    imported = [
        entity
        for entity in source.entities
        if entity.material_kind in {"conductor", "dielectric"}
    ]
    source_materials = stack["materials"]
    for entity in source.entities:
        material = materials[entity.material_id]
        record = source_materials[entity.material_id]
        expected_kind = (
            "superconductor"
            if entity.material_kind == "conductor"
            else entity.material_kind
        )
        if (
            material.material_id != entity.material_id
            or material.kind != expected_kind
            or record["kind"] != entity.material_kind
        ):
            raise ValueError(
                f"Q3D explicit material binding differs from source {entity.material_id!r}"
            )
    vacuum_ids = {
        entity.material_id
        for entity in source.entities
        if entity.material_kind == "vacuum"
    }
    if not vacuum_ids:
        vacuum_ids = {
            key for key, value in source_materials.items() if value["kind"] == "vacuum"
        }
    if len(vacuum_ids) != 1:
        raise ValueError("Q3D source requires exactly one vacuum material identity")
    vacuum_id = next(iter(vacuum_ids))
    if (
        materials[vacuum_id].kind != "vacuum"
        or materials[vacuum_id].material_id != vacuum_id
    ):
        raise ValueError("Q3D vacuum binding differs from source")

    dbu_um = source.metadata["source_dbu_um"]
    polygons = {polygon.polygon_id: polygon for polygon in source.polygons}
    bodies = []
    members = {net_id: [] for net_id in sorted(net_ids)}
    for entity in sorted(imported, key=lambda value: value.semantic_id):
        zmin, zmax = geometry_z_range(entity.geometry, entity.semantic_id)
        if zmax <= zmin:
            raise ValueError(
                f"Q3D entity {entity.semantic_id!r} requires positive finite thickness"
            )
        role = (
            "substrate"
            if entity.material_kind == "dielectric"
            else "ground"
            if entity.net_id in ground_nets
            else "signal"
        )
        for polygon_id, exterior, holes in _entity_polygons(entity, polygons):
            body_id = f"Q3D_BODY_{len(bodies) + 1:06d}"
            source_pair = (
                (entity.geometry["gds_layer"], entity.geometry["gds_datatype"])
                if "gds_layer" in entity.geometry
                else None
            )
            bodies.append(
                Q3dBodySpec(
                    body_id=body_id,
                    exterior_um=exterior,
                    holes_um=holes,
                    z_min_um=zmin,
                    z_max_um=zmax,
                    material_id=entity.material_id,
                    physical_role=role,
                    net_id=entity.net_id
                    if entity.material_kind == "conductor"
                    else None,
                    source_entity_id=entity.semantic_id,
                    source_polygon_id=polygon_id,
                    source_occurrence_path=entity.metadata.get(
                        "source_occurrence_path"
                    ),
                    source_local_entity_id=entity.metadata.get(
                        "source_local_entity_id"
                    ),
                    source_level=entity.metadata.get(
                        "logical_layer_id", entity.metadata.get("pdk_level_id")
                    ),
                    source_layer_datatype=source_pair,
                )
            )
            if entity.material_kind == "conductor":
                members[entity.net_id].append(body_id)
    directory.mkdir(parents=True, exist_ok=True)
    original_path = directory / "source.gds"
    original_path.write_bytes(snapshot.gds_bytes)
    stack_path = directory / "geometry_stack.json"
    write_json(stack_path, stack)
    source_payload = detached(asdict(source))
    # These adapter paths identify temporary snapshot transport, not source
    # semantics. Keep the detached trace portable without changing the snapshot.
    source_payload["metadata"]["gds_file"] = "geometry/source.gds"
    source_payload["metadata"]["stack_file"] = "metadata/geometry_stack.json"
    geometry_source = {
        "schema_version": "scgsim.aedt.q3d-geometry-source.v3",
        "source_gds_sha256": hashlib.sha256(snapshot.gds_bytes).hexdigest(),
        "source_geometry_sha256": canonical_sha256(source_payload),
        "source_stack_sha256": file_sha256(stack_path),
        "source_dbu_um": dbu_um,
        "physical_ground_nets": list(ground_nets),
        "bodies": [body.to_payload() for body in bodies],
    }
    trace_path = directory / "geometry_trace.json"
    write_json(
        trace_path,
        {
            "geometry_source": geometry_source,
            "source_geometry": source_payload,
            "source_occurrences": detached(snapshot.source_occurrences),
            "body_geometry": "normalized source exterior/holes without fragmentation; no locator sheets",
        },
    )
    geometry_source["files"] = {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in (
            ("canonical_gds", original_path),
            ("stack", stack_path),
            ("trace", trace_path),
        )
    }
    return Q3dSpec(
        project_name=project_name,
        design_name=design_name,
        materials=materials,
        vacuum_material_id=vacuum_id,
        bodies=tuple(bodies),
        nets=tuple(
            Q3dNetSpec(net_id, net_types[net_id], tuple(members[net_id]))
            for net_id in sorted(net_ids)
        ),
        run_control=run_control,
        region_padding_um=region_padding_um,
        solve_ac_rl=False,
        geometry_source=geometry_source,
    )
