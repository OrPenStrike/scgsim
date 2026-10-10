"""HFSS native EPR geometry construction, binding, and readback."""

from __future__ import annotations


import json

import math

import time

from collections.abc import Mapping, Sequence

from decimal import Decimal

from fractions import Fraction

from importlib.metadata import version

from itertools import pairwise

from pathlib import Path

from typing import Any

from scgsim.semantics.route_a import geometry_z_range

from scgsim.geometry._primitives.spatial import _geometry_ref_surface_z_um

from scgsim.aedt._io import file_sha256

from scgsim.aedt.epr.geometry import (
    _INSET_CIRCLE_POINTS,
    _INSET_METHOD,
    _base_sheet_plan,
    _entity_z_range,
    _junction_polygon,
    _junction_terminal_line,
    _native_entity_name,
    _native_name,
    _plain,
    _route_a_sheet_z,
)

from scgsim.aedt.epr.junction_partition import (
    _closed_contact,
    _cross,
    _inside_ring,
    _section,
)

from scgsim.aedt.epr.models import (
    PlanarJunction,
    PreparedPlanarGeometry,
    canonical_sha256,
    detached,
)

from scgsim.aedt.epr.fields import create_verified_face_list

from scgsim.aedt.runtime.native.common import (
    _native_boundary_type,
    _native_object_boolean_property,
    _native_object_evidence,
    _resolve_native_assignment,
    native_object_property,
)


def _analysis_surface_sheet(
    app: Any, geometry_ref: Mapping[str, Any], *, name: str
) -> tuple[Any, list[list[float]]]:
    local_region = geometry_ref.get("local_region")
    plane_basis = geometry_ref.get("plane_basis")
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(local_region, Mapping) and isinstance(plane_basis, Mapping):
        points = _lift_inset_ring(plane_basis, local_region["exterior"])
        holes = [
            _lift_inset_ring(plane_basis, ring)
            for ring in local_region.get("holes", ())
        ]
    elif isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in {"top", "bottom"}:
            raise ValueError("analysis surface loop requires an explicit Z plane")
        z_um = _geometry_ref_surface_z_um(geometry_ref)
        points = [[float(x), float(y), z_um] for x, y in outer]
        holes = [
            [[float(x), float(y), z_um] for x, y in ring]
            for ring in geometry_ref.get("hole_loops", ())
        ]
    else:
        raise ValueError("horizontal analysis surface requires a planar loop")
    if len(points) < 3 or any(
        len(point) != 3 or not all(math.isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError("analysis surface points are invalid")
    sheet = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=True,
    )
    if sheet is False or sheet is None:
        raise RuntimeError(f"failed to create analysis surface {name!r}")
    for index, ring in enumerate(holes):
        hole_name = _native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            ring,
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=True,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create analysis hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract analysis hole {hole_name!r}")
    return app.modeler[name], points


def _native_sheet_side_binding(
    native_normal: Sequence[float],
    field_side: str,
    modeling: str | None,
    *,
    unknown_side_error: str,
    ambiguous_side_error: str,
) -> tuple[tuple[float, float, float], bool]:
    """Map a source side while limiting the empirical rule to ThinFilm."""

    if field_side == "top":
        desired = (0.0, 0.0, 1.0)
    elif field_side == "bottom":
        desired = (0.0, 0.0, -1.0)
    else:
        raise RuntimeError(unknown_side_error)
    orientation = sum(
        float(normal) * target for normal, target in zip(native_normal, desired)
    )
    if abs(orientation) <= 1e-12:
        raise RuntimeError(ambiguous_side_error)
    if modeling == "thin_film":
        # Human-selected for pinned PyAEDT 1.3.0 / AEDT 2024.2: FacePrimitive.normal
        # points toward Adjacent. This is not a direct native arrow-vector accessor.
        adjacent_side = orientation > 0.0
    else:
        # Preserve the pre-existing Solid/historical convention; its physical
        # face-side correspondence has not been established by this change.
        adjacent_side = orientation < 0.0
    return desired, adjacent_side


def _lift_inset_ring(
    plane: Mapping[str, Any], ring: Sequence[Sequence[float]]
) -> list[list[float]]:
    origin = tuple(float(value) for value in plane["origin_um"])
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    return [
        [
            origin[axis] + float(point[0]) * u[axis] + float(point[1]) * v[axis]
            for axis in range(3)
        ]
        for point in ring
    ]


def bind_inset_surface_selections(
    app: Any,
    binding: Mapping[str, Any],
    base_selection: Mapping[str, Any],
    margin_um: float,
    inset_plan: Mapping[tuple[str, float], Mapping[str, Any]],
    sheet_facts: dict[str, dict[str, Any]],
    sheet_names: Mapping[str, Mapping[str, Any]],
    *,
    modeling: str | None,
) -> list[dict[str, Any]]:
    """Create or rebind non-model inset sheets without changing solver CAD."""

    plane = binding["mask_plane"]
    # Keep the existing positional caller contract; direction comes from this
    # source contribution, never from the base Sheet's orientation.
    field_side = str(binding["contribution"]["side"])
    planned = inset_plan[(str(binding["binding_id"]), margin_um)]
    regions = planned["regions"]
    dbu_um = float(binding["mask_support"]["source_dbu_um"])
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        contour_sha256 = canonical_sha256(
            {
                "plane": plane,
                "region": region,
                "method": _INSET_METHOD,
                "source_dbu_um": dbu_um,
                "klayout_version": version("klayout"),
            }
        )
        planned_name = sheet_names[contour_sha256]
        name = str(planned_name["name"])
        facts = sheet_facts.get(name)
        if facts is None:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                sheet, _ = _analysis_surface_sheet(
                    app,
                    {"plane_basis": plane, "local_region": region},
                    name=name,
                )
            native = _native_object_evidence(app, name)
            faces = sheet.faces
            if native["native_object_type"] != "Sheet" or len(faces) != 1:
                raise RuntimeError(f"inset EPR selection {name!r} is not one sheet")
            if _native_object_boolean_property(sheet, "Model"):
                raise RuntimeError(
                    f"inset EPR selection {name!r} became model geometry"
                )
            normal = faces[0].normal
            if (
                normal is None
                or len(normal) != 3
                or not all(math.isfinite(float(value)) for value in normal)
            ):
                raise RuntimeError(f"inset EPR selection {name!r} has no native normal")
            facts = {
                "contour_sha256": contour_sha256,
                "shared_sheet_members": planned_name["members"],
                "native": native,
                "native_normal": [float(value) for value in normal],
            }
            sheet_facts[name] = facts
        if facts["contour_sha256"] != contour_sha256:
            raise RuntimeError(f"inset EPR selection name collision: {name!r}")
        native = facts["native"]
        native_normal = facts["native_normal"]
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            field_side,
            modeling,
            unknown_side_error=f"inset EPR selection {name!r} has unknown field side",
            ambiguous_side_error=(
                f"inset EPR selection {name!r} has ambiguous field side"
            ),
        )
        result.append(
            {
                "selection_name": name,
                "component_index": index,
                "contour_sha256": contour_sha256,
                "contour": region,
                "adjacent_side": adjacent_side,
                "native_normal": native_normal,
                "polygon_approximation": {
                    "method": _INSET_METHOD,
                    "source_dbu_um": dbu_um,
                    "points_per_circle": _INSET_CIRCLE_POINTS,
                    "requested_margin_um": margin_um,
                    "integer_margin_dbu": planned["integer_margin_dbu"],
                    "effective_margin_um": planned["effective_margin_um"],
                    "klayout_version": version("klayout"),
                },
                "reused_source_sheet": False,
                **native,
            }
        )
    return result


def _polygon_sheet(
    app: Any, polygon: Mapping[str, Any], *, name: str, z_um: float
) -> Any:
    points = [[float(x), float(y), z_um] for x, y in polygon["exterior"]]
    value = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=False,
    )
    if value is False or value is None:
        raise RuntimeError(f"failed to create planar source polygon {name!r}")
    for index, ring in enumerate(polygon["holes"]):
        hole_name = _native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            [[float(x), float(y), z_um] for x, y in ring],
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=False,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create planar hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract planar hole {hole_name!r}")
    return app.modeler[name]


def _swept_z_solid(
    app: Any,
    polygon: Mapping[str, Any],
    *,
    name: str,
    z_min_um: float,
    z_max_um: float,
) -> Any:
    """Sweep a planar loop along explicit global +Z, independent of its normal."""

    if z_max_um <= z_min_um:
        raise ValueError(f"solid {name!r} has empty Z extent")
    sheet = _polygon_sheet(app, polygon, name=name, z_um=z_min_um)
    body = app.modeler.sweep_along_vector(
        sheet.name, ["0um", "0um", f"{z_max_um - z_min_um:.17g}um"]
    )
    if body is False or body is None:
        raise RuntimeError(f"failed to sweep solid {name!r} along positive Z")
    bounds = body.bounding_box
    scale = max(1.0, abs(z_min_um), abs(z_max_um))
    if (
        len(bounds) != 6
        or not math.isclose(
            float(bounds[2]), z_min_um, rel_tol=0.0, abs_tol=1e-9 * scale
        )
        or not math.isclose(
            float(bounds[5]), z_max_um, rel_tol=0.0, abs_tol=1e-9 * scale
        )
    ):
        raise RuntimeError(f"solid {name!r} native Z range differs from source")
    return body


def _bounds_box(app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]) -> Any:
    geometry = entity["geometry"]
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks domain_bounds_um"
        )
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in bounds for key in required):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} bounds are incomplete"
        )
    z_min = geometry.get("z_min_um", geometry.get("z_um"))
    z_max = geometry.get("z_max_um")
    if z_max is None and z_min is not None and geometry.get("thickness_um") is not None:
        z_max = float(z_min) + float(geometry["thickness_um"])
    if z_min is None or z_max is None:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks a finite z range"
        )
    origin = [float(bounds["x_min_um"]), float(bounds["y_min_um"]), float(z_min)]
    size = [
        float(bounds["x_max_um"]) - origin[0],
        float(bounds["y_max_um"]) - origin[1],
        float(z_max) - origin[2],
    ]
    if any(value <= 0.0 for value in size):
        raise ValueError(f"solution domain {entity['semantic_id']!r} has empty bounds")
    native_material = str(entity["material_id"])
    value = app.modeler.create_box(
        origin,
        size,
        name=_native_entity_name(source, "domain", entity["semantic_id"]),
        material=native_material,
    )
    if value is False or value is None:
        raise RuntimeError(
            f"failed to create solution domain {entity['semantic_id']!r}"
        )
    return value


def _solution_body(
    app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]
) -> Any:
    geometry = entity["geometry"]
    if "outer_loop" not in geometry:
        return _bounds_box(app, entity, source)
    z_min, z_max = geometry_z_range(geometry, entity["semantic_id"])
    if z_max <= z_min:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} has empty Z extent"
        )
    polygon = {
        "exterior": geometry["outer_loop"],
        "holes": geometry.get("hole_loops", ()),
    }
    name = _native_entity_name(source, "domain", entity["semantic_id"])
    body = _swept_z_solid(app, polygon, name=name, z_min_um=z_min, z_max_um=z_max)
    body.material_name = str(entity["material_id"])
    return body


def _prepared_modeling_identity(prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Keep legacy route labels on historical rebinds; new records name modeling."""
    if prepared.modeling is not None:
        return {"modeling": prepared.modeling}
    return {"route": prepared.route}


def _validate_native_modeling_source(
    prepared: PreparedPlanarGeometry, source: Mapping[str, Any]
) -> None:
    """Keep current effective-layer provenance separate from legacy route reads."""
    if prepared.modeling is None:
        if "modeling" in source:
            raise ValueError("historical EPR geometry has current modeling provenance")
        return
    if not isinstance(source.get("modeling"), Mapping):
        raise ValueError("prepared EPR effective-layer provenance is unavailable")


def _native_conductor_geometry(
    source: Mapping[str, Any], entity: Mapping[str, Any]
) -> tuple[str, float, float]:
    """Read the producer's effective native shape and Z without remapping it."""
    if "modeling" in source:
        modeling = source["modeling"]
        if not isinstance(modeling, Mapping):
            raise TypeError("prepared EPR modeling provenance is unavailable")
        physical_layer = entity.get("physical_layer")
        if not isinstance(physical_layer, Mapping):
            raise ValueError(
                f"EPR conductor {entity.get('semantic_id')!r} lacks "
                "effective-layer provenance"
            )
        representation = physical_layer.get("representation")
        if representation not in {"solid", "sheet"}:
            raise ValueError("prepared EPR conductor representation is invalid")
        if entity.get("representation") != representation:
            raise ValueError(
                f"EPR conductor {entity.get('semantic_id')!r} representation "
                "differs from its effective layer"
            )
        return (
            str(representation),
            float(physical_layer["effective_z_min_um"]),
            float(physical_layer["effective_z_max_um"]),
        )

    # Historical saved-project reads retain the recorded route basis. New native
    # construction is rejected before reaching this reader-only interpretation.
    z_min_um, z_max_um = _entity_z_range(entity)
    if entity["representation"] == "surface_sheet":
        z_min_um = z_max_um = _route_a_sheet_z(source, entity)
        return "sheet", z_min_um, z_max_um
    return "solid", z_min_um, z_max_um


def bind_native_surface_references(
    app: Any,
    prepared: PreparedPlanarGeometry,
    native_geometry: Mapping[str, Any],
    request: Any,
) -> list[dict[str, Any]]:
    """Bind whole Solid caps as references, separate from masked Sheet supports.

    Source ownership and constant-section Z construction select the cap. Native
    normals describe projection only; they do not establish an incident trace.
    Saved reuse changes selection metadata, never CAD or solved field topology.
    """
    if prepared.modeling != "solid":
        return []
    selected = (
        {item.contribution_id for item in prepared.contributions}
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    source = prepared.source
    specs = {item.contribution_id: item for item in prepared.contributions}
    objects = native_geometry["objects"]
    references: dict[str, dict[str, Any]] = {}
    for binding in prepared.surface_bindings:
        contribution = binding["contribution"]
        cid = contribution["contribution_id"]
        kind = contribution["classification"]
        if cid not in selected or kind not in {"MA", "MS", "SA"}:
            continue
        z_um = _geometry_ref_surface_z_um(binding["geometry_ref"])
        if kind == "SA":
            owner = binding["substrate_domain_id"]
            entities = [
                item for item in source["solution_regions"]
                if item["semantic_id"] == owner
            ]
            candidates = [
                item for item in objects
                if item["kind"] == "solution_domain" and item["semantic_id"] == owner
            ]
        else:
            owner = binding["owner_semantic_id"]
            entities = [
                item for item in source["conductors"]
                if item["semantic_id"] == owner
            ]
            source_ids = {
                specs[cid].source_polygon_id,
                *(contribution.get("outer_source_ids") or ()),
                *(contribution.get("hole_source_ids") or ()),
            }
            candidates = [
                item for item in objects
                if item["kind"] == "conductor" and item["semantic_id"] == owner
                and item["source_polygon_id"] in source_ids
            ]
        if len(entities) != 1 or not candidates:
            raise RuntimeError(f"native reference {cid!r} lacks source-owned Solid {owner!r}")
        entity = entities[0]
        if kind == "SA":
            z_min, z_max = _entity_z_range(entity)
        else:
            representation, z_min, z_max = _native_conductor_geometry(source, entity)
            if representation != "solid":
                raise RuntimeError(f"native reference {cid!r} owner is not a Solid")
        if z_um == z_max and z_um != z_min:
            cap = "top"
        elif z_um == z_min and z_um != z_max:
            cap = "bottom"
        else:
            raise RuntimeError(f"native reference {cid!r} is not a source cap of {owner!r}")
        for native in candidates:
            name = native["object_name"]
            obj = app.modeler.get_object_from_name(name)
            if obj is None or native["native_object_type"] != "Solid":
                raise RuntimeError(f"native reference Solid {name!r} is unavailable")
            centers = []
            for face in obj.faces:
                center = face.center_from_aedt
                if center is False or center is None or len(center) != 3:
                    raise RuntimeError(f"native reference {name!r} face {face.id} center unavailable")
                centers.append((float(center[2]), face, list(center)))
            if not centers:
                raise RuntimeError(f"native reference {name!r} has no faces")
            extreme = (max if cap == "top" else min)(item[0] for item in centers)
            matches = [item for item in centers if item[0] == extreme]
            if len(matches) != 1:
                raise RuntimeError(f"native reference {name!r} {cap} cap is ambiguous")
            _, face, center = matches[0]
            normal = face.normal
            if normal is None or len(normal) != 3:
                raise RuntimeError(f"native reference {name!r} cap projection unavailable")
            normal = [float(value) for value in normal]
            identity = {"object_name": name, "cap": cap, "projection_normal": normal}
            reference_id = "native_cap_" + canonical_sha256(identity)[:24]
            record = references.get(reference_id)
            if record is None:
                edges = []
                for edge in face.edges:
                    vertices = []
                    samples = []
                    for vertex in edge.vertices:
                        position = vertex.position
                        if int(vertex.id) <= 0:
                            # PyAEDT can synthesize an edge-parameter sample when
                            # the native edge has no vertex IDs (e.g. a closed edge).
                            samples.append({
                                "basis": "PyAEDT synthesized edge-parameter sample",
                                "position_um": list(position) if position is not None else None,
                                "native_vertex_id": None,
                            })
                            continue
                        if position is None or len(position) != 3:
                            raise RuntimeError(f"native reference {name!r} vertex {vertex.id} unavailable")
                        vertices.append({"vertex_id": int(vertex.id), "position_um": list(position)})
                    edges.append({
                        "edge_id": int(edge.id), "vertices": vertices,
                        "native_vertex_identity_available": bool(vertices) and not samples,
                        "parametric_samples": samples,
                    })
                area = face.area
                if area is None or area is False:
                    raise RuntimeError(f"native reference {name!r} cap area unavailable")
                face_ids = [int(face.id)]
                face_list = create_verified_face_list(
                    app, name=_native_name("reference", reference_id),
                    face_ids=face_ids, reuse_existing=True,
                )
                record = {
                    "reference_id": reference_id,
                    "contribution_ids": [], "binding_ids": [], "interface_kinds": [],
                    "selection_name": face_list["name"], "face_ids": face_ids,
                    "face_list": face_list, "owner_semantic_id": owner,
                    "source_polygon_id": native.get("source_polygon_id"),
                    "object_name": name, "native_object_id": native["native_object_id"],
                    "source_cap": cap, "source_z_um": z_um,
                    "source_supports": [], "incident_domains": [],
                    "native_normal": normal, "projection_normal": normal,
                    "native_normal_basis": "PyAEDT FacePrimitive.normal geometric projection",
                    "native_area_m2": float(area) * 1e-12,
                    "native_faces": [{"face_id": int(face.id), "center_um": center,
                                      "edges": edges, "area_um2": float(area)}],
                    "support_kind": "whole_native_cap",
                    "sampling_basis": "native_solid_owner_face; incident_trace_unproven",
                }
                references[reference_id] = record
            for key, value in (("contribution_ids", cid), ("binding_ids", binding["binding_id"]),
                               ("interface_kinds", kind)):
                if value not in record[key]:
                    record[key].append(value)
            record["source_supports"].append({"binding_id": binding["binding_id"],
                                               "geometry_ref": binding["geometry_ref"]})
            incident = {"effective_domain_id": binding["effective_domain_id"],
                        "effective_material_id": binding["effective_material_id"],
                        "substrate_domain_id": binding["substrate_domain_id"]}
            if incident not in record["incident_domains"]:
                record["incident_domains"].append(incident)
    return detached([references[key] for key in sorted(references)])


def _material_readback(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for material_id, record in materials.items():
        if not isinstance(record, Mapping) or record.get("kind") not in {
            "vacuum",
            "dielectric",
        }:
            continue
        material = app.materials.exists_material(material_id)
        if material is False or material is None:
            raise RuntimeError(f"native material {material_id!r} is unavailable")
        observed[material_id] = {
            "relative_permittivity": float(material.permittivity.value),
            "relative_permeability": float(material.permeability.value),
        }
    return observed


def _native_solution_materials(source: Mapping[str, Any]) -> dict[str, Any]:
    """Install only materials carried by native solution-domain bodies."""

    catalog = source["materials"]
    material_ids = {str(item["material_id"]) for item in source["solution_regions"]}
    material_ids.add(str(source["native_region"]["material_id"]))
    missing = material_ids - set(catalog)
    if missing:
        raise ValueError(f"native solution materials are missing: {sorted(missing)!r}")
    return {material_id: catalog[material_id] for material_id in sorted(material_ids)}


def _install_material_catalog(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    for material_id, record in materials.items():
        if not isinstance(material_id, str) or not isinstance(record, Mapping):
            raise TypeError("prepared material catalog must be string-keyed mappings")
        kind = record.get("kind")
        if kind == "conductor":
            continue
        if kind not in {"vacuum", "dielectric"}:
            raise ValueError(f"unsupported prepared material kind {kind!r}")
        permittivity = record.get("permittivity", 1.0 if kind == "vacuum" else None)
        if (
            isinstance(permittivity, bool)
            or not isinstance(permittivity, (int, float))
            or not math.isfinite(float(permittivity))
            or float(permittivity) <= 0.0
        ):
            raise ValueError(
                f"prepared material {material_id!r} requires positive permittivity"
            )
        properties: dict[str, float] = {"permittivity": float(permittivity)}
        loss = record.get("loss_tangent")
        if loss is not None:
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
                or float(loss) < 0.0
            ):
                raise ValueError(
                    f"prepared material {material_id!r} has invalid loss tangent"
                )
            properties["dielectric_loss_tangent"] = float(loss)
        observed = app.materials.exists_material(material_id)
        material = observed or app.materials.add_material(material_id, properties)
        if material is False or material is None:
            raise RuntimeError(f"failed to define native material {material_id!r}")
        native_permittivity = float(material.permittivity.value)
        if not math.isclose(
            native_permittivity, float(permittivity), rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError(
                f"native material {material_id!r} permittivity readback differs"
            )
    return _material_readback(app, materials)


_NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM = 1e-5


def _native_coordinates_close(left, right) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and abs(float(a) - float(b)) <= _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM
        for a, b in zip(left, right)
    )


def _native_straight_ring(ring):
    """Comparison copy with whole monotone straight chains, never moved nodes."""
    points = [tuple(Fraction(str(float(v))) for v in point) for point in ring]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or any(a == b for a, b in zip(points, (*points[1:], points[0]))):
        raise RuntimeError(
            "native junction boundary is unresolved at readback resolution"
        )
    chains = [[a, b] for a, b in zip(points, (*points[1:], points[0]))]
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            chain = chains[index - 1] + chains[index][1:]
            parameters = [
                _native_edge_projection(p, chain[0], chain[-1])[0] for p in chain
            ]
            if parameters == sorted(parameters) and all(
                _native_on_segment(p, chain[0], chain[-1]) for p in chain
            ):
                chains[index - 1] = chain
                del chains[index]
                del points[index]
                changed = True
                break
    if any(
        _native_coordinates_close(a, b)
        for a, b in zip(points, (*points[1:], points[0]))
    ):
        raise RuntimeError(
            "native junction boundary is unresolved at readback resolution"
        )
    return points, chains


def _native_polygon_correspondence(
    loops, polygon, name, deviations, *, subdivisions=True
):
    """Match boundary corner cycles, retaining raw native loops for contacts.

    Both raw chains must cover each other at the existing coordinate policy;
    independent straight-chain reduction alone could hide opposing deviations.
    The central rectangle disables reduction and retains its four-edge contract.
    """
    source_rings = (polygon["exterior"], *polygon["holes"])
    actual_rings = [ring for rings in loops for ring in rings]
    if len(loops) != 1 or len(actual_rings) != len(source_rings):
        raise RuntimeError(
            f"native junction conductor shape/holes differ from source: {name!r}"
        )

    def comparison(rings):
        corners, chains = [], {}
        for ring in rings:
            points = [tuple(Fraction(str(float(v))) for v in p) for p in ring]
            if subdivisions:
                points, raw_chains = _native_straight_ring(points)
            else:
                raw_chains = [[a, b] for a, b in zip(points, (*points[1:], points[0]))]
            corners.extend(points)
            chains.update({tuple(sorted((c[0], c[-1]))): c for c in raw_chains})
        return corners, chains

    expected, expected_chains = comparison(source_rings)
    actual, actual_chains = comparison(actual_rings)
    matches = {}
    for point in actual:
        candidates = [p for p in expected if _native_coordinates_close(point, p)]
        if len(candidates) != 1 or candidates[0] in matches:
            raise RuntimeError(
                f"native junction vertex correspondence is missing or ambiguous: {name!r}"
            )
        matches[candidates[0]] = point
        deviations.append(max(abs(float(a - b)) for a, b in zip(point, candidates[0])))
    if len(actual) != len(expected) or set(matches) != set(expected):
        raise RuntimeError(f"native junction vertex topology differs: {name!r}")
    if len(actual_chains) != len(actual) or len(expected_chains) != len(expected):
        raise RuntimeError(f"native junction boundary has repeated edges: {name!r}")
    reverse = {point: wanted for wanted, point in matches.items()}
    for edge, chain in actual_chains.items():
        source_chain = expected_chains.get(tuple(sorted(reverse[p] for p in edge)))
        if source_chain is None:
            raise RuntimeError(
                f"native junction conductor shape/holes differ from source: {name!r}"
            )
        for nodes, other in ((chain, source_chain), (source_chain, chain)):
            for point in nodes:
                segments = list(pairwise(other))
                if not any(_native_on_segment(point, a, b) for a, b in segments):
                    raise RuntimeError(
                        f"native junction boundary chain differs from source: {name!r}"
                    )
                residuals = []
                for a, b in segments:
                    parameter, projected = _native_edge_projection(point, a, b)
                    nearest = a if parameter < 0 else b if parameter > 1 else projected
                    residuals.append(
                        max(abs(float(p - q)) for p, q in zip(point, nearest))
                    )
                deviations.append(min(residuals))
    return matches


def _native_edge_projection(point, start, end):
    delta = tuple(b - a for a, b in zip(start, end))
    norm = sum(v * v for v in delta)
    if not norm:
        raise RuntimeError("native junction boundary has a collapsed edge")
    parameter = sum((p - a) * v for p, a, v in zip(point, start, delta)) / norm
    projected = tuple(a + parameter * v for a, v in zip(start, delta))
    return parameter, projected


def _native_on_segment(point, start, end) -> bool:
    parameter, projected = _native_edge_projection(point, start, end)
    nearest = start if parameter < 0 else end if parameter > 1 else projected
    return _native_coordinates_close(point, nearest)


def _native_edge_section(bodies, start, end, *, boundary_only=False):
    """Sections of actual footprints, with only bounded boundary comparisons."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = dx * dx + dy * dy
    if not norm or _native_coordinates_close(start, end):
        raise RuntimeError("native junction edge is unresolved at readback resolution")
    intervals, contacts = [], set()

    def clamp(value):
        point = (start[0] + value * dx, start[1] + value * dy)
        if _native_coordinates_close(point, start):
            return Fraction(0)
        if _native_coordinates_close(point, end):
            return Fraction(1)
        return value

    for rings in bodies:
        if not boundary_only:
            transformed = tuple(
                tuple(
                    (
                        ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                        dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
                    )
                    for p in ring
                )
                for ring in rings
            )
            spans, points = _section(transformed, 1, 0, 0, 1)
            intervals.extend((clamp(a), clamp(b)) for a, b in spans)
            contacts.update(clamp(v) for v in points)
        for ring in rings:
            for a, b in zip(ring, (*ring[1:], ring[0])):
                ta, pa = _native_edge_projection(a, start, end)
                tb, pb = _native_edge_projection(b, start, end)
                near_a = _native_coordinates_close(a, pa)
                near_b = _native_coordinates_close(b, pb)
                if near_a and 0 <= clamp(ta) <= 1:
                    contacts.add(clamp(ta))
                if near_b and 0 <= clamp(tb) <= 1:
                    contacts.add(clamp(tb))
                if near_a and near_b:
                    lo, hi = (
                        max(Fraction(0), clamp(min(ta, tb))),
                        min(Fraction(1), clamp(max(ta, tb))),
                    )
                    if lo < hi:
                        intervals.append((lo, hi))
    spans = [(a, b) for a, b in intervals if a < b]
    contacts.update(a for a, b in intervals if a == b)
    return spans, contacts


def _native_edge_covered(bodies, start, end) -> bool:
    intervals, _ = _native_edge_section(bodies, start, end, boundary_only=True)
    reached = Fraction(0)
    for lo, hi in sorted(intervals):
        left = tuple(a + lo * (b - a) for a, b in zip(start, end))
        right = tuple(a + reached * (b - a) for a, b in zip(start, end))
        if lo > reached and not _native_coordinates_close(left, right):
            return False
        reached = max(reached, hi)
    return reached == 1


def _native_boundary_contact(left, right) -> bool:
    """Require a resolved positive span; tiny raw subdivisions cannot prove one."""
    for rings in left:
        for ring in rings:
            for start, end in zip(ring, (*ring[1:], ring[0])):
                if start != end and _native_coordinates_close(start, end):
                    continue
                spans, _ = _native_edge_section(right, start, end, boundary_only=True)
                for lo, hi in spans:
                    a = tuple(p + lo * (q - p) for p, q in zip(start, end))
                    b = tuple(p + hi * (q - p) for p, q in zip(start, end))
                    if not _native_coordinates_close(a, b):
                        return True
    return False


def _native_closed_contact(left, right) -> bool:
    return _closed_contact(left, right) or any(
        _native_on_segment(point, a, b)
        for regions, others in ((left, right), (right, left))
        for rings in regions
        for ring in rings
        for point in ring
        for other in others
        for boundary in other
        for a, b in zip(boundary, (*boundary[1:], boundary[0]))
    )


def _native_strict_inside(point, rings) -> bool:
    return (
        _inside_ring(point, rings[0]) == 1
        and all(_inside_ring(point, hole) == -1 for hole in rings[1:])
        and not any(
            _native_on_segment(point, a, b)
            for ring in rings
            for a, b in zip(ring, (*ring[1:], ring[0]))
        )
    )


def _native_cross_sign(a, b, point):
    _, projected = _native_edge_projection(point, a, b)
    value = _cross(a, b, point)
    return (
        0 if _native_coordinates_close(point, projected) else (1 if value > 0 else -1)
    )


def _native_z_may_contact(left, right) -> bool:
    """Exclude forbidden contact only when actual Z ranges are separated."""
    low, high = max(left[0], right[0]), min(left[1], right[1])
    return low <= high or _native_coordinates_close((low,), (high,))


def _native_z_contact(left, left_solid, right, right_solid) -> bool:
    """Required contact compatibility; sheets need full consistent planes."""
    if not left_solid and not right_solid:
        return all(_native_coordinates_close((a,), (b,)) for a in left for b in right)
    if left_solid and right_solid:
        return _native_z_may_contact(left, right)
    solid, sheet = (left, right) if left_solid else (right, left)
    return all(
        (z >= solid[0] or _native_coordinates_close((z,), (solid[0],)))
        and (z <= solid[1] or _native_coordinates_close((z,), (solid[1],)))
        for z in sheet
    )


def _native_face_edges(face, name):
    """Read straight edges once; failed observations stay in the private receipt."""
    edges = []
    edge_observations = []
    for edge in face.edges:
        vertices = [vertex.position for vertex in edge.vertices]
        if len(vertices) != 2 or any(p is None or len(p) != 3 for p in vertices):
            raise RuntimeError(f"native junction boundary is not polygonal: {name!r}")
        midpoint = edge.midpoint
        expected_midpoint = [(float(a) + float(b)) / 2 for a, b in zip(*vertices)]
        length = edge.length
        chord = math.dist(*vertices)
        try:
            midpoint_values = [float(v) for v in midpoint]
        except (TypeError, ValueError, OverflowError):
            midpoint_values = []
        midpoint_valid = len(midpoint_values) == 3 and all(
            math.isfinite(v) for v in midpoint_values
        )
        midpoint_matches = midpoint_valid and _native_coordinates_close(
            midpoint_values, expected_midpoint
        )
        try:
            length_valid = (
                not isinstance(length, bool)
                and isinstance(length, (int, float))
                and math.isfinite(float(length))
            )
        except OverflowError:
            length_valid = False
        length_matches = length_valid and _native_coordinates_close((length,), (chord,))
        if not midpoint_matches or not length_matches:

            def observed(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                if isinstance(value, (list, tuple)):
                    return [observed(v) for v in value]
                if value is None or isinstance(value, (str, bool, int, float)):
                    return value
                return repr(value)

            residuals = (
                [a - b for a, b in zip(midpoint_values, expected_midpoint)]
                if midpoint_valid
                else None
            )
            diagnostic = {
                "object_name": name,
                "edge_id": getattr(edge, "__dict__", {}).get("id"),
                "face_id": getattr(face, "__dict__", {}).get(
                    "_id", getattr(face, "__dict__", {}).get("id")
                ),
                "model_units": "um",
                "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
                "relative_tolerance": 0.0,
                "endpoints_um": vertices,
                "observed_midpoint_um": midpoint,
                "midpoint_type": type(midpoint).__name__,
                "expected_midpoint_um": expected_midpoint,
                "native_length_um": length,
                "native_length_type": type(length).__name__,
                "chord_length_um": chord,
                "midpoint_residuals_um": residuals,
                "length_residual_um": float(length) - chord if length_valid else None,
                "failed_conditions": [
                    key
                    for key, failed in (
                        ("midpoint_invalid", not midpoint_valid),
                        ("midpoint_mismatch", midpoint_valid and not midpoint_matches),
                        ("native_length_invalid", not length_valid),
                        ("native_length_mismatch", length_valid and not length_matches),
                    )
                    if failed
                ],
            }
            raise RuntimeError(
                f"native junction edge is curved or its linear readback differs: {name!r}"
                + "; edge_diagnostic="
                + json.dumps(
                    {key: observed(value) for key, value in diagnostic.items()},
                    allow_nan=False,
                )
            )
        a, b = [tuple(Fraction(str(float(v))) for v in p) for p in vertices]
        edges.append((a, b))
        edge_observations.append(
            {
                "edge_id": getattr(edge, "__dict__", {}).get("id"),
                "vertices_um": [list(map(float, point)) for point in vertices],
                "midpoint_um": list(map(float, midpoint)),
                "native_length_um": float(length),
                "chord_length_um": chord,
                "maximum_midpoint_coordinate_deviation_um": max(
                    abs(float(a) - float(b))
                    for a, b in zip(midpoint, expected_midpoint)
                ),
            }
        )
    return edges, edge_observations


def _native_planar_loops(obj: Any, z_um: float, observations=None) -> list[tuple]:
    """Read actual planar boundaries at absolute 1e-5 um coordinate resolution.

    Independent vertex/midpoint/length getters need not serialize identically.
    Midpoint and actual native length jointly check straight edges; ambiguity
    below this comparison resolution is not an exact CAD contact guarantee.
    """
    from scgsim.geometry._primitives.loops import (
        _cancel_reversed_planar_edges,
        _simple_planar_loops_from_edges,
    )

    loops = []
    for face in obj.faces:
        positions = [vertex.position for vertex in face.vertices]
        if not positions or any(p is None or len(p) != 3 for p in positions):
            raise RuntimeError(
                f"native junction face vertices unavailable: {obj.name!r}"
            )
        if any(not _native_coordinates_close((p[2],), (z_um,)) for p in positions):
            continue
        if not _native_coordinates_close(
            (min(float(p[2]) for p in positions),),
            (max(float(p[2]) for p in positions),),
        ):
            raise RuntimeError(
                "native junction face has inconsistent actual Z coordinates"
            )
        xyz_edges, edge_observations = _native_face_edges(face, obj.name)
        edges = [(a[:2], b[:2]) for a, b in xyz_edges]
        if observations is not None:
            observations.append(
                {
                    "object_name": obj.name,
                    "face_id": getattr(face, "__dict__", {}).get(
                        "_id", getattr(face, "__dict__", {}).get("id")
                    ),
                    "requested_plane_z_um": z_um,
                    "maximum_plane_coordinate_deviation_um": max(
                        abs(float(point[2]) - z_um) for point in positions
                    ),
                    "face_vertices_um": [
                        list(map(float, point)) for point in positions
                    ],
                    "edges": edge_observations,
                }
            )
        # Edge direction is not a native face-loop ordering contract.
        face_loops = _simple_planar_loops_from_edges(
            _cancel_reversed_planar_edges(edges)
        )

        def area(ring):
            return abs(
                sum(
                    a[0] * b[1] - b[0] * a[1]
                    for a, b in zip(ring, (*ring[1:], ring[0]))
                )
            )

        outer = max(face_loops, key=area)
        loops.append((outer, *(ring for ring in face_loops if ring is not outer)))
    if not loops:
        raise RuntimeError(
            f"native junction conductor has no planar face at source Z: {obj.name!r}"
        )
    return loops


def _native_extrusion_sides(obj, bottom, top, z_range, thickness, observations):
    """Verify straight prism sides against actual caps, independent of subdivision.

    Each actual cap edge must belong to exactly one side face by native ID;
    only associated endpoints are compared under the coordinate policy. IDs are
    scoped to this live object, never source geometry or another session. Raw
    subdivision edges retain their own incidence. Side faces remain simple
    vertical quadrilaterals after comparison-only straight-chain reduction;
    general face tessellation is unsupported.
    """
    from scgsim.geometry._primitives.loops import (
        _cancel_reversed_planar_edges,
        _simple_planar_loops_from_edges,
    )

    cap_observations = observations[-2:]
    cap_edges = ({}, {})
    side_owners = ({}, {})
    cap_z = [
        [p[2] for p in observation["face_vertices_um"]]
        for observation in cap_observations
    ]
    boundary_edges = [
        tuple(tuple(Fraction(str(v)) for v in p) for p in edge["vertices_um"])
        for observation in cap_observations
        for edge in observation["edges"]
    ]
    if any(
        not _native_coordinates_close((height,), (thickness,))
        for height in (max(cap_z[1]) - min(cap_z[0]), min(cap_z[1]) - max(cap_z[0]))
    ):
        raise RuntimeError("native junction actual top/bottom extrusion differs")

    def layer(point):
        found = [
            n
            for n, z in enumerate(z_range)
            if _native_coordinates_close((point[2],), (z,))
        ]
        if len(found) != 1 or any(
            not _native_coordinates_close((point[2],), (z,)) for z in cap_z[found[0]]
        ):
            raise RuntimeError("native junction actual extrusion vertex Z differs")
        return found[0]

    def incidence_error(
        condition,
        n,
        cap_edge=None,
        side_edge=None,
        side_face_id=None,
        prior_side_face_id=None,
    ):
        diagnostic = {
            "object_name": obj.name,
            "condition": condition,
            "cap_index": n,
            "cap_face_id": cap_observations[n].get("face_id"),
            "edge_id": (cap_edge or side_edge or {}).get("edge_id"),
            "cap_edge_endpoints_um": (cap_edge or {}).get("vertices_um"),
            "side_edge_endpoints_um": (side_edge or {}).get("vertices_um"),
            "side_face_id": side_face_id,
            "prior_side_face_id": prior_side_face_id,
            "model_units": "um",
            "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
        }
        return RuntimeError(
            "native junction extrusion cap-side incidence differs"
            + "; extrusion_incidence_diagnostic="
            + json.dumps(diagnostic, allow_nan=False)
        )

    for n, observation in enumerate(cap_observations):
        for edge in observation["edges"]:
            edge_id = edge.get("edge_id")
            if isinstance(edge_id, bool) or not isinstance(edge_id, int):
                raise incidence_error("missing_cap_edge_id", n, cap_edge=edge)
            if edge_id in cap_edges[0] or edge_id in cap_edges[1]:
                raise incidence_error("duplicate_cap_edge_id", n, cap_edge=edge)
            cap_edges[n][edge_id] = edge

    for face in obj.faces:
        positions = [vertex.position for vertex in face.vertices]
        if positions and any(
            all(_native_coordinates_close((p[2],), (z,)) for p in positions)
            for z in z_range
        ):
            continue  # Cap shape and straight edges were already read in full.
        edges, edge_observations = _native_face_edges(face, obj.name)
        rings = _simple_planar_loops_from_edges(_cancel_reversed_planar_edges(edges))
        if len(rings) != 1:
            raise RuntimeError("native junction extrusion side has unsupported loops")
        corners, chains = _native_straight_ring(rings[0])
        sides = [[p for p in corners if layer(p) == n] for n in (0, 1)]
        if len(corners) != 4 or any(len(side) != 2 for side in sides):
            raise RuntimeError(
                "native junction extrusion side is not a vertical quadrilateral"
            )
        if any(
            layer(a) != layer(b) and not _native_coordinates_close(a[:2], b[:2])
            for a, b in zip(corners, (*corners[1:], corners[0]))
        ):
            raise RuntimeError("native junction extrusion side has nonvertical edges")
        for point in sides[0]:
            mates = [p for p in sides[1] if _native_coordinates_close(point[:2], p[:2])]
            if len(mates) != 1 or not _native_coordinates_close(
                (float(mates[0][2] - point[2]),), (thickness,)
            ):
                raise RuntimeError(
                    "native junction actual top/bottom extrusion differs"
                )
        edge_records = {
            tuple(sorted(edge)): observation
            for edge, observation in zip(edges, edge_observations)
        }
        if len(edge_records) != len(edges):
            raise RuntimeError(
                "native junction extrusion side has repeated boundary edges"
            )
        face_state = getattr(face, "__dict__", {})
        face_id = face_state.get("_id", face_state.get("id"))
        for chain in chains:
            n = layer(chain[0])
            if layer(chain[-1]) != n:
                continue  # Existing vertical-chain checks retain intermediate nodes.
            for a, b in pairwise(chain):
                side_edge = edge_records[tuple(sorted((a, b)))]
                edge_id = side_edge.get("edge_id")
                cap_edge = (
                    cap_edges[n].get(edge_id)
                    if isinstance(edge_id, int) and not isinstance(edge_id, bool)
                    else None
                )
                if cap_edge is None:
                    raise incidence_error(
                        "unmatched_side_edge",
                        n,
                        side_edge=side_edge,
                        side_face_id=face_id,
                    )
                wanted = cap_edge["vertices_um"]
                actual = side_edge["vertices_um"]
                if not any(
                    all(
                        _native_coordinates_close(p, q)
                        for p, q in zip(actual, ordering)
                    )
                    for ordering in (wanted, wanted[::-1])
                ):
                    raise incidence_error(
                        "linked_edge_geometry_mismatch", n, cap_edge, side_edge, face_id
                    )
                if edge_id in side_owners[n]:
                    raise incidence_error(
                        "duplicate_side_incidence",
                        n,
                        cap_edge,
                        side_edge,
                        face_id,
                        side_owners[n][edge_id],
                    )
                side_owners[n][edge_id] = face_id
        boundary_edges.extend(zip(corners, (*corners[1:], corners[0])))
        observations.append(
            {
                "object_name": obj.name,
                "extrusion_side_vertices_um": [list(map(float, p)) for p in rings[0]],
                "edges": edge_observations,
            }
        )
    for vertex in obj.vertices:
        position = vertex.position
        if (
            position is None
            or len(position) != 3
            or not _native_z_may_contact(
                (position[2], position[2]), (min(cap_z[0]), max(cap_z[1]))
            )
            or not any(_native_on_segment(position, a, b) for a, b in boundary_edges)
        ):
            raise RuntimeError(
                "native solid vertex differs from actual extrusion boundary"
            )
    for n, edges in enumerate(cap_edges):
        for edge_id, edge in edges.items():
            if edge_id not in side_owners[n]:
                raise incidence_error("missing_side_incidence", n, cap_edge=edge)


def _native_junction_live_readback(
    app: Any, source: Mapping[str, Any], junction: PlanarJunction
) -> dict[str, Any]:
    """Verify live CAD/electrical state before authoring; line awaits native save.

    Face edges supply complete planar footprints, rather than bounding-box
    contact guesses. Actual returned geometry is preserved. Coordinate identity,
    straightness, Z and contact comparisons use absolute 1e-5 um, rel_tol=0;
    correspondence never substitutes source vertices into contact geometry.
    Below-resolution gaps/curvature remain numerically unresolved. Missing live
    RLC properties or ambiguous topology are explicit readback failures. The
    transaction-local result is not a completed partition_readback; the owned
    runtime must verify the saved assigned line before publishing that record.
    """
    from ansys.aedt.core.generic.constants import AEDT_UNITS
    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value
    from ansys.aedt.core.modules.boundary.common import BoundaryObject

    record = source["junction_partitions"][junction.junction_id]
    if app.modeler.model_units != "um":
        raise RuntimeError(
            "junction native readback requires source micrometre model units"
        )
    polygons = {p["polygon_id"]: p for p in source["polygons"]}
    z_um = float(record["z_um"])
    central_name = _native_name("junction", junction.junction_id)
    central = app.modeler.get_object_from_name(central_name)
    if central is None or len(central.faces) != 1 or len(central.faces[0].edges) != 4:
        raise RuntimeError("native central junction is not one four-edge face")
    observations, deviations = [], []
    central_loops = _native_planar_loops(central, z_um, observations)
    expected = polygons[record["central_polygon_id"]]
    central_matches = _native_polygon_correspondence(
        central_loops, expected, central_name, deviations, subdivisions=False
    )
    central_z = tuple(
        operation(p[2] for p in observations[0]["face_vertices_um"])
        for operation in (min, max)
    )

    def assignment(boundary_name, target, kind):
        if _native_boundary_type(app, boundary_name) != kind:
            raise RuntimeError(
                f"native junction boundary type differs: {boundary_name!r}"
            )
        raw = app.oboundary.GetBoundaryAssignment(boundary_name)
        if raw is None:
            raise RuntimeError(
                f"native junction boundary target unavailable: {boundary_name!r}"
            )
        ids = [int(v) for v in raw]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("native junction boundary assignment IDs are invalid")
        _, _, names = _resolve_native_assignment(app, ids)
        if names != {target}:
            raise RuntimeError(
                f"native junction boundary has wrong targets: {boundary_name!r}"
            )

    native_by_net: dict[str, list] = {}
    potential_by_net: dict[str, list] = {}
    native_by_polygon: dict[tuple[str, str], list] = {}
    native_z_by_polygon = {}
    entity_nets = {
        entity["semantic_id"]: entity["net_id"] for entity in source["conductors"]
    }
    authored_ids = {
        partition["authored_polygon_id"]
        for partition in source["junction_partitions"].values()
    }
    for entity in source["conductors"]:
        representation, effective_z_min_um, effective_z_max_um = (
            _native_conductor_geometry(source, entity)
        )
        plane = effective_z_min_um
        for pid in entity["polygon_ids"]:
            if pid in authored_ids:
                continue
            name = _native_entity_name(source, "conductor", entity["semantic_id"], pid)
            obj = app.modeler.get_object_from_name(name)
            if obj is None:
                raise RuntimeError(
                    f"native junction source conductor unavailable: {name!r}"
                )
            loops = _native_planar_loops(obj, plane, observations)
            wanted = polygons[pid]
            bottom_matches = _native_polygon_correspondence(
                loops, wanted, name, deviations
            )
            bottom_observation = observations[-1]
            bottom_xyz = {
                tuple(Fraction(str(v)) for v in p[:2]): p
                for p in bottom_observation["face_vertices_um"]
            }
            if representation == "sheet":
                assignment(
                    _native_name("pec_boundary", entity["semantic_id"], pid),
                    name,
                    "Perfect E",
                )
            else:
                if (
                    native_object_property(obj, "Material").strip('"').casefold()
                    != "pec"
                    or _native_object_boolean_property(obj, "Solve Inside") is not False
                ):
                    raise RuntimeError(
                        f"native junction conductor lacks finite PEC lowering: {name!r}"
                    )
                zs = [float(vertex.position[2]) for vertex in obj.vertices]
                if not zs or not _native_coordinates_close(
                    (min(zs), max(zs)),
                    (effective_z_min_um, effective_z_max_um),
                ):
                    raise RuntimeError(
                        f"native junction source material/Z lineage differs: {name!r}"
                    )
                top = _native_planar_loops(obj, max(zs), observations)
                top_matches = _native_polygon_correspondence(
                    top, wanted, name, deviations
                )
                try:
                    _native_polygon_correspondence(
                        top, {"exterior": loops[0][0], "holes": loops[0][1:]}, name, []
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "native junction actual top/bottom extrusion differs"
                    ) from exc
                top_xyz = {
                    tuple(Fraction(str(v)) for v in p[:2]): p
                    for p in observations[-1]["face_vertices_um"]
                }
                thickness = effective_z_max_um - effective_z_min_um
                if any(
                    not _native_coordinates_close(
                        bottom_matches[point], top_matches[point]
                    )
                    or not _native_coordinates_close(
                        (
                            top_xyz[top_matches[point]][2]
                            - bottom_xyz[bottom_matches[point]][2],
                        ),
                        (thickness,),
                    )
                    for point in bottom_matches
                ):
                    raise RuntimeError(
                        "native junction actual top/bottom extrusion differs"
                    )
                _native_extrusion_sides(
                    obj,
                    loops,
                    top,
                    (effective_z_min_um, effective_z_max_um),
                    thickness,
                    observations,
                )
            solid = representation != "sheet"
            native_z = (
                (min(zs), max(zs))
                if solid
                else tuple(
                    operation(p[2] for p in bottom_observation["face_vertices_um"])
                    for operation in (min, max)
                )
            )
            key = entity["semantic_id"], pid
            native_by_polygon[key] = loops
            native_z_by_polygon[key] = native_z, solid
            if _native_z_contact(central_z, False, native_z, solid):
                native_by_net.setdefault(entity["net_id"], []).extend(loops)
            if _native_z_may_contact(central_z, native_z):
                potential_by_net.setdefault(entity["net_id"], []).extend(loops)

    # Correspondence orders actual vertices in the authored A->B frame.
    points = [
        central_matches[tuple(Fraction(str(float(v))) for v in p)]
        for p in expected["exterior"]
    ]
    central_midpoint = tuple(sum(p[i] for p in points) / 4 for i in (0, 1))
    for bodies in potential_by_net.values():
        for rings in bodies:
            if any(_native_strict_inside(p, (points,)) for p in rings[0]) or any(
                _native_strict_inside(p, rings) for p in (*points, central_midpoint)
            ):
                raise RuntimeError("native PEC occupies central RLC interior")
            for ring in rings:
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    for c, d in zip(points, (*points[1:], points[0])):
                        if (
                            _native_cross_sign(a, b, c) * _native_cross_sign(a, b, d)
                            < 0
                            and _native_cross_sign(c, d, a)
                            * _native_cross_sign(c, d, b)
                            < 0
                        ):
                            raise RuntimeError(
                                "native PEC crosses central RLC interior"
                            )
    derived_ids = {
        end["polygon_id"]
        for partition in source["junction_partitions"].values()
        for end in partition["ends"]
        if end["polygon_id"] is not None
    }

    for end in record["ends"]:
        if end["polygon_id"] is None:
            continue
        end_loops = native_by_polygon[end["source_entity_id"], end["polygon_id"]]
        end_z, end_solid = native_z_by_polygon[
            end["source_entity_id"], end["polygon_id"]
        ]
        arms = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if eid == end["source_entity_id"]
            and pid not in derived_ids
            and _native_z_contact(end_z, end_solid, *native_z_by_polygon[eid, pid])
            for rings in loops
        ]
        if not _native_boundary_contact(end_loops, arms):
            raise RuntimeError(
                "native derived end lacks positive-length source-arm contact"
            )
        foreign = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if entity_nets[eid] != end["net_id"]
            and _native_z_may_contact(end_z, native_z_by_polygon[eid, pid][0])
            for rings in loops
        ]
        if _native_closed_contact(end_loops, foreign):
            raise RuntimeError("native derived end contacts another net")
    edge_specs = (
        (points[3], points[0], junction.terminal_a_net),
        (points[1], points[2], junction.terminal_b_net),
        (points[0], points[1], None),
        (points[2], points[3], None),
    )
    edge_evidence = []
    for start, end, correct_net in edge_specs:
        if correct_net is not None and not _native_edge_covered(
            native_by_net.get(correct_net, []), start, end
        ):
            raise RuntimeError(
                "native designated junction end edge is not wholly covered"
            )
        for net, bodies in potential_by_net.items():
            for rings in bodies:
                intervals, contacts = _native_edge_section([rings], start, end)
                if correct_net is not None:
                    forbidden = net != correct_net and bool(intervals or contacts)
                else:
                    forbidden = bool(intervals) or any(0 < v < 1 for v in contacts)
                    # Open side corners belong to their respective end edge.
                    for v in contacts:
                        point = start if v == 0 else end
                        own_corner = (
                            junction.terminal_a_net
                            if point in (points[0], points[3])
                            else junction.terminal_b_net
                        )
                        forbidden |= net != own_corner
                if forbidden:
                    raise RuntimeError(
                        "native junction has forbidden side/wrong-net corner contact"
                    )
        edge_evidence.append(
            {
                "start_um": list(map(float, start)),
                "end_um": list(map(float, end)),
                "terminal_net": correct_net,
            }
        )

    boundary_name = _native_name("junction_rlc", junction.junction_id)
    assignment(boundary_name, central_name, "Lumped RLC")
    live = BoundaryObject(app, boundary_name, auto_update=False)._child_object
    if live is None:
        raise RuntimeError("native live junction RLC property object unavailable")
    available = [str(v) for v in live.GetPropNames()]
    required = (
        "RLC Type",
        "Use Induct",
        "Inductance",
        "Use Cap",
        "Use Resist",
    )
    if any(name not in available for name in required):
        raise RuntimeError(
            f"native live RLC readback lacks required properties; available={available!r}"
        )
    values = {name: live.GetPropValue(name) for name in required}

    def enabled(value):
        if value in (True, "true", "True", 1):
            return True
        if value in (False, "false", "False", 0):
            return False
        raise RuntimeError("native RLC enabled property is invalid")

    def scalar(value, unit):
        # Exact decimal SI equality avoids binary unit-conversion roundoff;
        # the length comparison policy never applies to electrical parameters.
        number, observed_unit = decompose_variable_value(value)
        system = {"H": "Inductance", "F": "Capacitance"}[unit]
        if (
            observed_unit not in AEDT_UNITS[system]
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
        ):
            raise RuntimeError("native RLC scalar unit/value unavailable")
        return Decimal(str(number)) * Decimal(str(AEDT_UNITS[system][observed_unit]))

    if (
        values["RLC Type"] != "Parallel"
        or not enabled(values["Use Induct"])
        or enabled(values["Use Resist"])
        or scalar(values["Inductance"], "H") != Decimal(str(junction.inductance_h))
    ):
        raise RuntimeError("native junction parallel L/R parameters differ")
    if enabled(values["Use Cap"]) != bool(junction.capacitance_f):
        raise RuntimeError("native junction capacitance enable state differs")
    if junction.capacitance_f and (
        "Capacitance" not in available
        or scalar(live.GetPropValue("Capacitance"), "F")
        != Decimal(str(junction.capacitance_f))
    ):
        raise RuntimeError("native junction capacitance differs")
    deviations.extend(
        observation["maximum_plane_coordinate_deviation_um"]
        for observation in observations
        if "maximum_plane_coordinate_deviation_um" in observation
    )
    return {
        "coordinate_comparison_policy": {
            "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
            "scope": "SCGSim native junction readback; not an AEDT precision guarantee",
            "geometry_moved": False,
        },
        "maximum_source_coordinate_deviation_um": max(deviations, default=0.0),
        "native_geometry_observations": observations,
        "edge_contacts": edge_evidence,
        "boundary_name": boundary_name,
    }


def _read_saved_hfss_design(project_path, design_name, *, purpose):
    """Read one saved HFSS design without a cached parser or project mutation."""
    from ansys.aedt.core.internal.load_aedt_file import load_entire_aedt_file

    path = Path(project_path).resolve(strict=True)
    digest = file_sha256(path)
    native = load_entire_aedt_file(path)
    if file_sha256(path) != digest:
        raise RuntimeError(f"saved {purpose} native project changed during readback")
    project = native.get("AnsoftProject")
    designs = project.get("HFSSModel") if isinstance(project, Mapping) else None
    if isinstance(designs, Mapping):
        designs = [designs]
    if not isinstance(designs, list):
        raise TypeError(f"saved {purpose} design records are unavailable")
    matching = [
        d for d in designs if isinstance(d, Mapping) and d.get("Name") == design_name
    ]
    if len(matching) != 1:
        raise RuntimeError(f"saved {purpose} design is missing or ambiguous")
    return digest, matching[0]


def verify_saved_native_surface_references(
    project_path, design_name, references
) -> list[dict[str, Any]]:
    """Bind reference membership to the existing pre-Analyze native save.

    Created PyAEDT List properties hold the authored assignment. Only the saved
    GeometryEntityListOperation supplies native serialization membership here;
    this function neither saves nor changes the model or the reference support.
    """
    verified = detached(references)
    if not verified:
        return verified
    digest, design = _read_saved_hfss_design(
        project_path, design_name, purpose="native surface reference"
    )
    setup = design.get("ModelSetup")
    core = setup.get("GeometryCore") if isinstance(setup, Mapping) else None
    operations = core.get("GeometryOperations") if isinstance(core, Mapping) else None
    lists = operations.get("GeometryEntityLists") if isinstance(operations, Mapping) else None
    records = lists.get("GeometryEntityListOperation") if isinstance(lists, Mapping) else None
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, list):
        raise RuntimeError("saved native surface reference list records are unavailable")
    for reference in verified:
        requested = reference["face_list"]
        name = requested["name"]
        matching = [
            record for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("Attributes"), Mapping)
            and record["Attributes"].get("Name") == name
        ]
        expected = {
            "name": name, "native_id": requested["native_id"],
            "entity_type": "Face", "face_ids": requested["face_ids"],
        }
        if len(matching) != 1:
            raise RuntimeError(
                f"saved native surface reference list missing or ambiguous: "
                f"requested={expected!r}; matching_records={matching!r}"
            )
        record = matching[0]
        parameters = record.get("GeometryEntityListParameters")
        actual = {
            "name": record["Attributes"]["Name"], "native_id": record.get("ID"),
            "entity_type": parameters.get("EntityType") if isinstance(parameters, Mapping) else None,
            "face_ids": parameters.get("EntityList") if isinstance(parameters, Mapping) else None,
        }
        if (
            actual != expected
            or type(actual["native_id"]) is not int
            or not isinstance(actual["face_ids"], list)
            or any(type(value) is not int for value in actual["face_ids"])
        ):
            raise RuntimeError(
                f"saved native surface reference membership differs: "
                f"requested={expected!r}; actual={actual!r}"
            )
        requested["saved_membership"] = {
            "basis": "saved_native_project_geometry_entity_list",
            "project_sha256": digest, "design_name": design_name, **actual,
        }
    return verified


def _read_saved_junction_lines(app, project_path, design_name, prepared, geometry):
    """Complete junction evidence from one uncached, identity-bound native save.

    PyAEDT1.3 configurations._update_boundaries interprets GeometryPosition XYZ
    in global model units. Attached EdgeCenter instead references actual edges;
    XYZ can be zero placeholders. Only these two position kinds are supported.
    IDs are resolved against freshly read central edges in the current object,
    with no cross-session ID persistence assumption or geometry guess.
    """
    source = detached(prepared.source)
    partitions = source.get("junction_partitions", {})
    if not partitions:
        return  # Historical geometry retains its existing binding contract.
    path = Path(project_path).resolve(strict=True)
    if (
        Path(app.project_file).resolve(strict=True) != path
        or app.design_name != design_name
        or app.design_type != "HFSS"
    ):
        raise RuntimeError("saved junction project/design identity differs")
    digest, design = _read_saved_hfss_design(
        path, design_name, purpose="junction"
    )
    model_setup = design.get("ModelSetup")
    model = (
        model_setup.get("GeometryCore") if isinstance(model_setup, Mapping) else None
    )
    if (
        not isinstance(model, Mapping)
        or model.get("Units") != app.modeler.model_units
        or model.get("Units") != "um"
    ):
        raise RuntimeError(
            "saved junction model units differ from live micrometre geometry"
        )
    boundary_setup = design.get("BoundarySetup")
    boundaries = (
        boundary_setup.get("Boundaries")
        if isinstance(boundary_setup, Mapping)
        else None
    )
    if not isinstance(boundaries, Mapping):
        raise TypeError("saved junction boundary records are unavailable")
    polygons = {p["polygon_id"]: p for p in source["polygons"]}
    bindings = {j["junction_id"]: j for j in geometry["junctions"]}
    completed = []
    for junction in prepared.junctions:
        if junction.junction_id not in partitions:
            continue
        binding = bindings[junction.junction_id]
        evidence = binding["_partition_live_readback"]
        boundary = boundaries.get(binding["boundary_name"])
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("BoundType") != "Lumped RLC"
        ):
            raise RuntimeError("saved junction boundary identity/type differs")
        raw_line = boundary.get("CurrentLine")
        positions = (
            raw_line.get("GeometryPosition") if isinstance(raw_line, Mapping) else None
        )
        if not isinstance(positions, list) or len(positions) != 2:
            raise RuntimeError(
                "saved junction line requires two ordered GeometryPosition records"
            )
        record = partitions[junction.junction_id]
        z_um = float(record["z_um"])
        expected = polygons[record["central_polygon_id"]]
        central = app.modeler.get_object_from_name(binding["object_name"])
        if (
            central is None
            or len(central.faces) != 1
            or len(central.faces[0].edges) != 4
        ):
            raise RuntimeError("saved junction central face/edge topology differs")
        central_id = int(app.modeler.oeditor.GetObjectIDByName(central.name))
        if boundary.get("Objects") != [central_id]:
            raise RuntimeError(
                "saved junction boundary target differs from actual central object"
            )
        observations, deviations = [], []
        loops = _native_planar_loops(central, z_um, observations)
        matches = _native_polygon_correspondence(
            loops, expected, central.name, deviations, subdivisions=False
        )
        points = [
            matches[tuple(Fraction(str(float(v))) for v in p)]
            for p in expected["exterior"]
        ]
        edge_ids = [edge["edge_id"] for edge in observations[0]["edges"]]
        if len(edge_ids) != len(set(edge_ids)):
            raise RuntimeError("saved junction native central edge IDs are ambiguous")
        terminal_edges = []
        for a, b in ((points[3], points[0]), (points[1], points[2])):
            found = [
                edge
                for edge in observations[0]["edges"]
                if {tuple(Fraction(str(v)) for v in p[:2]) for p in edge["vertices_um"]}
                == {a, b}
            ]
            if len(found) != 1:
                raise RuntimeError(
                    "saved junction actual terminal edge identity is unavailable"
                )
            if isinstance(found[0]["edge_id"], bool) or not isinstance(
                found[0]["edge_id"], int
            ):
                raise TypeError("saved junction actual terminal edge ID is invalid")
            terminal_edges.append(found[0])
        observed_line = []
        for position, edge in zip(positions, terminal_edges):
            if not isinstance(position, Mapping):
                raise TypeError("saved junction line position is malformed")
            if (
                position.get("IsAttachedToEntity") is True
                and position.get("PositionType") == "EdgeCenter"
            ):
                entity_id = position.get("EntityID")
                if (
                    isinstance(entity_id, bool)
                    or not isinstance(entity_id, int)
                    or entity_id != edge["edge_id"]
                ):
                    raise RuntimeError(
                        "saved junction line attachment references wrong terminal edge"
                    )
                observed_line.append(list(edge["midpoint_um"]))
            elif (
                position.get("IsAttachedToEntity") is False
                and position.get("PositionType") == "AbsolutePosition"
            ):
                raw = [position.get(axis + "Position") for axis in "XYZ"]
                try:
                    coordinate = [float(v) for v in raw]
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "saved junction absolute line coordinates are invalid"
                    ) from exc
                if any(
                    isinstance(v, bool) or not isinstance(v, (str, int, float))
                    for v in raw
                ) or not all(math.isfinite(v) for v in coordinate):
                    raise RuntimeError(
                        "saved junction absolute line coordinates are invalid"
                    )
                observed_line.append(coordinate)
            else:
                raise RuntimeError("saved junction line attachment kind is unsupported")
        line, _ = _junction_terminal_line(expected, junction, z_um)
        actual_midpoints = [edge["midpoint_um"] for edge in terminal_edges]
        if any(
            not _native_coordinates_close(a, b) for a, b in zip(observed_line, line)
        ):
            raise RuntimeError("saved junction integration line differs")
        if any(
            not _native_coordinates_close(a, b)
            for a, b in zip(observed_line, actual_midpoints)
        ):
            raise RuntimeError(
                "saved junction integration line differs from actual terminal edges"
            )
        deviations.extend(
            abs(a - b)
            for actual, wanted in zip(observed_line, line)
            for a, b in zip(actual, wanted)
        )
        completed.append(
            (
                binding,
                {
                    **evidence,
                    "method": "native_face_edges_live_electrical_and_saved_line.v3",
                    "maximum_source_coordinate_deviation_um": max(
                        evidence["maximum_source_coordinate_deviation_um"], *deviations
                    ),
                    "integration_line_um": observed_line,
                    "actual_terminal_edge_midpoints_um": actual_midpoints,
                    "raw_integration_line": detached(raw_line),
                    "saved_line_verification_project_sha256": digest,
                    "saved_line_native_geometry_observations": observations,
                },
            )
        )
    for binding, readback in completed:
        binding.pop("_partition_live_readback")
        binding["partition_readback"] = readback


def _region_bounds(region: Any) -> tuple[float, ...]:
    bounds = tuple(float(value) for value in region.bounding_box)
    if (
        len(bounds) != 6
        or not all(math.isfinite(value) for value in bounds)
        or any(bounds[index] >= bounds[index + 3] for index in range(3))
    ):
        raise RuntimeError("native EPR Region bounds are invalid")
    return bounds


def _verified_region_bounds(region: Any, plan: Mapping[str, Any]) -> tuple[float, ...]:
    bounds = _region_bounds(region)
    loop = tuple(
        tuple(float(value) for value in point)
        for point in plan["envelope_outer_loop_um"]
    )
    z_min, z_max = plan["z_range_um"]
    expected = (
        min(point[0] for point in loop),
        min(point[1] for point in loop),
        float(z_min),
        max(point[0] for point in loop),
        max(point[1] for point in loop),
        float(z_max),
    )
    if any(
        not math.isclose(
            actual,
            wanted,
            rel_tol=0.0,
            abs_tol=max(1e-6, 1e-9 * max(abs(actual), abs(wanted))),
        )
        for actual, wanted in zip(bounds, expected)
    ):
        raise RuntimeError("native EPR Region bounds differ from source envelope")
    return bounds


def _create_epr_region(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    plan = source["native_region"]
    if app.modeler.get_object_from_name("Region") is not None:
        raise RuntimeError("new EPR model already has Region")
    region = app.modeler.create_region(
        pad_value=list(plan["padding_um"]),
        pad_type="Absolute Offset",
        name="Region",
    )
    if region is False or region is None or region.name != "Region":
        raise RuntimeError("native EPR Region creation failed")
    # HFSS partitions overlapping interior bodies from Region in solver and
    # Calculator volume support; its special Region object rejects CAD subtract.
    region.material_name = str(plan["material_id"])
    observed_material = native_object_property(region, "Material").strip('"')
    if observed_material.casefold() != str(plan["material_id"]).casefold():
        raise RuntimeError("native EPR Region vacuum material differs")
    if not _native_object_boolean_property(region, "Solve Inside"):
        raise RuntimeError("native EPR Region must solve inside vacuum")
    evidence = _native_object_evidence(app, "Region")
    if evidence["native_object_type"] != "Solid":
        raise RuntimeError("native EPR Region is not solid")
    return {
        "kind": "solution_domain",
        "semantic_id": "Region",
        "material_id": plan["material_id"],
        "logical_vacuum_ids": list(plan["logical_vacuum_ids"]),
        "object_name": "Region",
        "padding_um": list(plan["padding_um"]),
        "native_solve_inside": True,
        "native_bounding_box_um": list(_verified_region_bounds(region, plan)),
        **evidence,
    }


def _enclosure_faces(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the Region's six outer faces without assigning a boundary."""

    region = app.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("native EPR Region is unavailable")
    bounds = _verified_region_bounds(region, source["native_region"])
    z_min, z_max = source["native_region"]["z_range_um"]
    face_records: list[dict[str, Any]] = []
    plane_labels = ("x_minus", "y_minus", "bottom", "x_plus", "y_plus", "top")
    for face in region.faces:
        center = tuple(float(value) for value in face.center)
        matches = [
            plane_labels[index]
            for index in range(6)
            if math.isclose(center[index % 3], bounds[index], rel_tol=0.0, abs_tol=1e-6)
        ]
        if len(matches) == 1:
            face_records.append(
                {
                    "face_id": int(face.id),
                    "object_name": "Region",
                    "center_um": list(center),
                    "enclosure_plane": matches[0],
                }
            )
    planes = [item["enclosure_plane"] for item in face_records]
    if sorted(planes) != sorted(plane_labels):
        raise RuntimeError("native closed-enclosure face selection is incomplete")
    face_ids = [item["face_id"] for item in face_records]
    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError("native closed-enclosure face selection repeats a face")
    return {
        "face_records": face_records,
        "envelope_outer_loop_um": _plain(
            source["native_region"]["envelope_outer_loop_um"]
        ),
        "z_range_um": [z_min, z_max],
        "native_bounding_box_um": list(bounds),
    }


def _verify_no_explicit_region_boundary(app: Any) -> None:
    """Require native assignment evidence that leaves Region to HFSS defaults."""

    raw = [str(value) for value in app.oboundary.GetBoundaries()]
    if len(raw) % 2:
        raise RuntimeError("native HFSS boundary list is invalid")
    names = raw[::2]
    if len(names) != len(set(names)):
        raise RuntimeError("native HFSS boundary names are not unique")
    for name in names:
        assigned = app.oboundary.GetBoundaryAssignment(name)
        if assigned is None:
            raise RuntimeError(
                f"native boundary assignment is unavailable for {name!r}"
            )
        try:
            raw_ids = [int(value) for value in assigned]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"native boundary assignment IDs are invalid for {name!r}"
            ) from exc
        if not raw_ids or len(raw_ids) != len(set(raw_ids)):
            raise RuntimeError(
                f"native boundary assignment IDs are invalid for {name!r}"
            )
        _, _, objects = _resolve_native_assignment(app, raw_ids)
        if "Region" in objects:
            raise RuntimeError(f"Region has an explicit native boundary {name!r}")


def _implicit_closed_enclosure(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
    faces = _enclosure_faces(app, source)
    _verify_no_explicit_region_boundary(app)
    return {
        "boundary_policy": "hfss_default.v1",
        "explicit_assignment": False,
        **faces,
    }


def _native_lumped_supports(
    app: Any, prepared: PreparedPlanarGeometry, *, saved: bool = False
) -> list[dict[str, Any]]:
    """Bind neutral authored supports without inventing conductor electrodes."""
    records = []
    for support in prepared.source.get("lumped_supports", ()):
        value = detached(support)
        effective = value["effective"]
        name = _native_name("lumped_support", value["support_id"])
        if saved:
            obj = app.modeler[name]
        else:
            obj = _polygon_sheet(app, effective, name=name, z_um=effective["z_um"])
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet" or len(obj.faces) != 1:
            raise RuntimeError(
                f"neutral support is not one native Sheet face: {value['support_id']!r}"
            )
        records.append({**value, **evidence, "object_name": obj.name})
    return records


def native_conductor_objects(
    prepared: PreparedPlanarGeometry,
    native: Mapping[str, Any],
    entity_ids: Sequence[str],
) -> list[str]:
    """Resolve explicit source Entities to all their created conductor bodies."""
    source_entities = prepared.source["conductors"]
    names = []
    for entity_id in entity_ids:
        semantic_ids = {
            item["semantic_id"]
            for item in source_entities
            if entity_id in {item["semantic_id"], item["source_semantic_id"]}
        }
        matches = [
            item["object_name"]
            for item in native["objects"]
            if item["kind"] == "conductor" and item["semantic_id"] in semantic_ids
        ]
        if not matches:
            raise RuntimeError(f"source conductor has no native bodies: {entity_id!r}")
        for name in matches:
            if name not in names:
                names.append(name)
    return names


def prepare_native_planar_geometry(
    app: Any, prepared: PreparedPlanarGeometry
) -> dict[str, Any]:
    """Create bodies and live evidence; runtime completes lines after native save."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if prepared.modeling not in {"solid", "thin_film"}:
        raise ValueError("new EPR native construction requires explicit modeling")
    source = detached(prepared.source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("EPR geometry predates single Region; reprepare the handoff")
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
    _validate_native_modeling_source(prepared, source)
    polygons = {item["polygon_id"]: item for item in source["polygons"]}
    junction_regions = {
        item["source_polygon_id"]: item for item in source["junction_regions"]
    }
    polygons.update(
        {
            source_id: {
                "polygon_id": source_id,
                "layer": item["source_layer"],
                "exterior": item["exterior"],
                "holes": item["holes"],
                "object_name": None,
                "net_name": None,
                "port_name": item["port_sheet_id"],
            }
            for source_id, item in junction_regions.items()
        }
    )
    entities = source["conductors"]
    bindings: list[dict[str, Any]] = []
    phase_seconds: dict[str, float] = {}

    started = time.perf_counter()
    material_readback = _install_material_catalog(
        app, _native_solution_materials(source)
    )
    phase_seconds["materials_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    for entity in source["solution_regions"]:
        if entity["metadata"].get("is_auto_vacuum_region"):
            continue
        obj = _solution_body(app, entity, source)
        evidence = _native_object_evidence(app, obj.name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(
                f"solution domain {entity['semantic_id']!r} is not solid"
            )
        bindings.append(
            {
                "kind": "solution_domain",
                "semantic_id": entity["semantic_id"],
                "material_id": entity["material_id"],
                "object_name": obj.name,
                **evidence,
            }
        )
    phase_seconds["interior_solution_bodies_seconds"] = time.perf_counter() - started

    junction_polygons = {item.source_polygon_id for item in prepared.junctions}
    started = time.perf_counter()
    for entity in entities:
        representation, z_min_um, z_max_um = _native_conductor_geometry(
            source, entity
        )
        is_sheet = representation == "sheet"
        z_um = z_min_um
        thickness_um = z_max_um - z_min_um
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygons:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            boundary_name: str | None = None
            if is_sheet:
                obj = _polygon_sheet(app, polygons[polygon_id], name=name, z_um=z_um)
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                boundary = app.assign_perfect_e(obj.name, name=boundary_name)
                if boundary is False or boundary is None:
                    raise RuntimeError(f"failed to assign Perfect E to {obj.name!r}")
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"Perfect E readback failed for {obj.name!r}")
            else:
                if thickness_um <= 0.0:
                    raise ValueError(
                        f"finite conductor {entity['semantic_id']!r} requires positive thickness"
                    )
                obj = _swept_z_solid(
                    app,
                    polygons[polygon_id],
                    name=name,
                    z_min_um=z_min_um,
                    z_max_um=z_max_um,
                )
                obj.material_name = "pec"
                obj.solve_inside = False
            evidence = _native_object_evidence(app, obj.name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"native conductor type mismatch for {obj.name!r}")
            observed: dict[str, Any] = {}
            if not is_sheet:
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": _native_object_boolean_property(
                        obj, "Solve Inside"
                    ),
                }
                if (
                    observed["native_material_name"].casefold() != "pec"
                    or observed["native_solve_inside"] is not False
                ):
                    raise RuntimeError(
                        f"native finite PEC readback mismatch for {obj.name!r}"
                    )
            bindings.append(
                {
                    "kind": "conductor",
                    "semantic_id": entity["semantic_id"],
                    "source_polygon_id": polygon_id,
                    **_prepared_modeling_identity(prepared),
                    "physical_layer": detached(entity["physical_layer"]),
                    "object_name": obj.name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )
    phase_seconds["conductors_seconds"] = time.perf_counter() - started

    junction_bindings: list[dict[str, Any]] = []
    started = time.perf_counter()
    for junction in prepared.junctions:
        polygon, partition = _junction_polygon(source, junction, polygons)
        region = junction_regions.get(junction.source_polygon_id)
        if partition is not None:
            owner_ids = [end["source_entity_id"] for end in partition["ends"]]
        elif region is None:
            owner_ids = [
                item["semantic_id"]
                for item in entities
                if junction.source_polygon_id in item["polygon_ids"]
            ]
        else:
            owner_ids = list(region["host_semantic_ids"])
        owners = [item for item in entities if item["semantic_id"] in owner_ids]
        if not owners or {item["net_id"] for item in owners} != {
            junction.terminal_a_net,
            junction.terminal_b_net,
        }:
            raise RuntimeError(
                f"junction {junction.junction_id!r} source owners do not match its nets"
            )
        z_values = [
            _native_conductor_geometry(source, owner)[1]
            for owner in owners
        ]
        if max(z_values) - min(z_values) > 1e-9:
            raise RuntimeError(
                f"junction {junction.junction_id!r} owners are not coplanar"
            )
        z_um = partition["z_um"] if partition is not None else z_values[0]
        name = _native_name("junction", junction.junction_id)
        sheet = _polygon_sheet(app, polygon, name=name, z_um=z_um)
        line, span_um = _junction_terminal_line(polygon, junction, z_um)
        boundary_name = _native_name("junction_rlc", junction.junction_id)
        boundary = app.assign_lumped_rlc_to_sheet(
            sheet.name,
            line,
            name=boundary_name,
            rlc_type="Parallel",
            inductance=junction.inductance_h,
            capacitance=(junction.capacitance_f or None),
        )
        if boundary is False or boundary is None:
            raise RuntimeError(f"failed to assign junction {junction.junction_id!r}")
        evidence = _native_object_evidence(app, sheet.name)
        partition_readback = (
            _native_junction_live_readback(app, source, junction)
            if partition is not None
            else None
        )
        junction_bindings.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": sheet.name,
                "boundary_name": boundary_name,
                "terminal_a_net": junction.terminal_a_net,
                "terminal_b_net": junction.terminal_b_net,
                "integration_line_um": line,
                "terminal_span_um": span_um,
                "width_um": junction.width_um,
                "inductance_h": junction.inductance_h,
                "capacitance_f": junction.capacitance_f,
                **(
                    {"_partition_live_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **(
                    {
                        "central_polygon_id": partition["central_polygon_id"],
                        "partition_method": partition["method"],
                    }
                    if partition is not None
                    else {}
                ),
                **evidence,
            }
        )
    phase_seconds["junctions_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bindings.insert(0, _create_epr_region(app, source))
    enclosure = _implicit_closed_enclosure(app, source)
    phase_seconds["region_boundary_seconds"] = time.perf_counter() - started

    surface_selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    base_names = _base_sheet_plan(prepared.surface_bindings)
    base_points: dict[str, list[list[float]]] = {}
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        if name not in base_points:
            sheet, points = _analysis_surface_sheet(
                app, binding["geometry_ref"], name=name
            )
            base_points[name] = points
        else:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                raise RuntimeError(f"shared analysis surface {name!r} vanished")
            points = base_points[name]
        native = _native_object_evidence(app, sheet.name)
        if native["native_object_type"] != "Sheet" or len(sheet.faces) != 1:
            raise RuntimeError(f"analysis surface {name!r} is not one native sheet")
        normal = sheet.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"analysis surface {name!r} normal is unavailable")
        unit_normal = tuple(float(value) for value in normal)
        centroid = tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )
        field_side = str(binding["contribution"]["side"])
        desired_normal, adjacent_side = _native_sheet_side_binding(
            unit_normal,
            field_side,
            prepared.modeling,
            unknown_side_error=f"analysis surface {name!r} has unknown field side",
            ambiguous_side_error=(
                f"analysis surface {name!r} has ambiguous effective-domain side"
            ),
        )
        surface_selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": sheet.name,
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                "field_side": field_side,
                "native_normal": list(unit_normal),
                "selection_centroid_um": list(centroid),
                "desired_field_side_normal": list(desired_normal),
                **native,
            }
        )
    phase_seconds["base_analysis_sheets_seconds"] = time.perf_counter() - started

    object_names = [item["object_name"] for item in bindings] + [
        item["object_name"] for item in junction_bindings
    ]
    if len(object_names) != len(set(object_names)):
        raise RuntimeError("native planar object names are not unique")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        **_prepared_modeling_identity(prepared),
        "source_sha256": prepared.source_sha256,
        "objects": bindings,
        **({"lumped_supports": _native_lumped_supports(app, prepared)}
           if source.get("lumped_supports") else {}),
        "junctions": junction_bindings,
        "surface_selections": surface_selections,
        "material_readback": material_readback,
        "closed_enclosure": enclosure,
        "geometry_phases": {
            key: round(value, 6) for key, value in phase_seconds.items()
        },
    }


def bind_saved_planar_geometry(
    app: Any, prepared: PreparedPlanarGeometry
) -> dict[str, Any]:
    """Bind deterministic native objects in a saved project without creating CAD."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    _validate_native_modeling_source(prepared, source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("saved EPR geometry predates single Region; reprepare")
    objects: list[dict[str, Any]] = []
    for domain in source["solution_regions"]:
        if domain["metadata"].get("is_auto_vacuum_region"):
            continue
        name = _native_entity_name(source, "domain", domain["semantic_id"])
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(f"saved solution domain {name!r} is not solid")
        objects.append(
            {
                "kind": "solution_domain",
                "semantic_id": domain["semantic_id"],
                "material_id": domain["material_id"],
                "object_name": name,
                **evidence,
            }
        )
    region_plan = source["native_region"]
    saved_region = app.modeler.get_object_from_name("Region")
    if saved_region is None:
        raise RuntimeError("saved EPR Region is unavailable")
    region_evidence = _native_object_evidence(app, "Region")
    if region_evidence["native_object_type"] != "Solid":
        raise RuntimeError("saved EPR Region is not solid")
    saved_material = native_object_property(saved_region, "Material").strip('"')
    if saved_material.casefold() != str(region_plan["material_id"]).casefold():
        raise RuntimeError("saved EPR Region vacuum material differs")
    if not _native_object_boolean_property(saved_region, "Solve Inside"):
        raise RuntimeError("saved EPR Region does not solve inside vacuum")
    objects.insert(
        0,
        {
            "kind": "solution_domain",
            "semantic_id": "Region",
            "material_id": region_plan["material_id"],
            "logical_vacuum_ids": list(region_plan["logical_vacuum_ids"]),
            "object_name": "Region",
            "padding_um": list(region_plan["padding_um"]),
            "native_solve_inside": True,
            "native_bounding_box_um": list(
                _verified_region_bounds(saved_region, region_plan)
            ),
            **region_evidence,
        },
    )

    junction_polygon_ids = {item.source_polygon_id for item in prepared.junctions}
    for entity in source["conductors"]:
        representation, _, _ = _native_conductor_geometry(source, entity)
        is_sheet = representation == "sheet"
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygon_ids:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            evidence = _native_object_evidence(app, name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"saved conductor {name!r} has wrong object type")
            boundary_name: str | None = None
            observed: dict[str, Any] = {}
            if is_sheet:
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"saved conductor {name!r} lacks Perfect E")
                assigned = app.oboundary.GetBoundaryAssignment(boundary_name)
                if assigned is None:
                    raise RuntimeError(f"saved conductor {name!r} lacks PEC assignment")
                try:
                    raw_ids = [int(value) for value in assigned]
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"saved conductor {name!r} has invalid PEC IDs"
                    ) from exc
                if not raw_ids or len(raw_ids) != len(set(raw_ids)):
                    raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs")
                _, _, assigned_objects = _resolve_native_assignment(app, raw_ids)
                if assigned_objects != {name}:
                    raise RuntimeError(f"saved conductor {name!r} PEC target differs")
            else:
                obj = app.modeler.get_object_from_name(name)
                if obj is None:
                    raise RuntimeError(f"saved conductor {name!r} is unavailable")
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": _native_object_boolean_property(
                        obj, "Solve Inside"
                    ),
                }
                if (
                    observed["native_material_name"].casefold() != "pec"
                    or observed["native_solve_inside"] is not False
                ):
                    raise RuntimeError(f"saved conductor {name!r} PEC readback differs")
            objects.append(
                {
                    "kind": "conductor",
                    "semantic_id": entity["semantic_id"],
                    "source_polygon_id": polygon_id,
                    **_prepared_modeling_identity(prepared),
                    **(
                        {"physical_layer": detached(entity["physical_layer"])}
                        if isinstance(entity.get("physical_layer"), Mapping)
                        else {}
                    ),
                    "object_name": name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )

    junctions: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        name = _native_name("junction", junction.junction_id)
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved junction {name!r} is not a sheet")
        partition_readback = (
            _native_junction_live_readback(app, source, junction)
            if junction.junction_id in source.get("junction_partitions", {})
            else None
        )
        junctions.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": name,
                "boundary_name": _native_name("junction_rlc", junction.junction_id),
                **(
                    {"_partition_live_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **evidence,
            }
        )

    selections: list[dict[str, Any]] = []
    base_names = _base_sheet_plan(prepared.surface_bindings)
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        obj = app.modeler.get_object_from_name(name)
        if obj is None or len(obj.faces) != 1:
            raise RuntimeError(f"saved analysis surface {name!r} is unavailable")
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved analysis surface {name!r} is not a sheet")
        normal = obj.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"saved analysis surface {name!r} normal is unavailable")
        native_normal = tuple(float(value) for value in normal)
        side = str(binding["contribution"]["side"])
        desired, adjacent_side = _native_sheet_side_binding(
            native_normal,
            side,
            prepared.modeling,
            unknown_side_error=f"saved analysis surface {name!r} has unknown side",
            ambiguous_side_error=(
                f"saved analysis surface {name!r} side is ambiguous"
            ),
        )
        selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": name,
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": adjacent_side,
                "field_side": side,
                "native_normal": list(native_normal),
                "desired_field_side_normal": list(desired),
                **evidence,
            }
        )
    boundary_policy = region_plan.get("outer_boundary_policy")
    if boundary_policy == "hfss_default.v1":
        enclosure = {
            **_implicit_closed_enclosure(app, source),
            "binding_stage": "saved_project_readback_without_cad_creation",
        }
    elif "outer_boundary_policy" not in region_plan:
        # Historical prepared source records an explicit named enclosure.
        closed_type = _native_boundary_type(app, "SCGSimClosedEnclosure")
        if closed_type != "Perfect E":
            raise RuntimeError("saved EPR closed enclosure boundary differs")
        enclosure = {
            "boundary_name": "SCGSimClosedEnclosure",
            "native_boundary_type": closed_type,
            "binding_stage": "saved_project_readback_without_cad_creation",
        }
    else:
        raise ValueError("saved EPR Region boundary policy is unknown")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        **_prepared_modeling_identity(prepared),
        "source_sha256": prepared.source_sha256,
        "objects": objects,
        **({"lumped_supports": _native_lumped_supports(app, prepared, saved=True)}
           if source.get("lumped_supports") else {}),
        "junctions": junctions,
        "surface_selections": selections,
        "material_readback": _material_readback(
            app, _native_solution_materials(source)
        ),
        "closed_enclosure": enclosure,
        "binding_stage": "saved_project_readback_without_cad_creation",
    }
