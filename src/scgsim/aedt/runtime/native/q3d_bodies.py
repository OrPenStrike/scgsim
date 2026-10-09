"""Q3D v3 native body construction and recorded binding evidence."""

from __future__ import annotations


import json

from typing import Any

from scgsim.aedt.runtime.native.common import (
    _native_object_evidence,
    native_object_property,
)

from scgsim.aedt.specs.q3d import Q3dBodySpec, Q3dSpec


def q3d_terminal_segments(body: Q3dBodySpec, side: str) -> list[int]:
    """Select one contiguous source exterior chain, never a hole or nearest edge."""
    if side not in {"+X", "-X", "+Y", "-Y"}:
        raise ValueError("Q3D Sheet terminals require an in-plane +/-X or +/-Y side")
    ring = body.exterior_um
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    axis = 0 if side[-1] == "X" else 1
    extreme = (max if side[0] == "+" else min)(point[axis] for point in ring)
    selected = [
        index
        for index, point in enumerate(ring)
        if point[axis] == extreme and ring[(index + 1) % len(ring)][axis] == extreme
    ]
    starts = [index for index in selected if (index - 1) % len(ring) not in selected]
    if not selected or len(starts) != 1:
        raise ValueError(
            f"Q3D Sheet terminal {body.body_id!r}/{side} has no unambiguous exterior chain"
        )
    return selected


def _hole_name(body_index: int, hole_index: int) -> str:
    return f"SCGSimQ3DHoleCut_{body_index:06d}_{hole_index:06d}"


def _native_incidence(obj: Any, native_face_ids: list[int]) -> dict[str, Any]:
    """Return only face/edge/vertex IDs actually exposed by PyAEDT."""
    try:
        faces = {int(face.id): face for face in obj.faces}
        if set(faces) != set(native_face_ids):
            raise RuntimeError("PyAEDT face IDs differ from editor face IDs")
        face_edge_ids: dict[str, list[int]] = {}
        edge_vertex_ids: dict[str, list[int]] = {}
        for face_id in native_face_ids:
            edge_ids: list[int] = []
            for edge in faces[face_id].edges:
                edge_id = int(edge.id)
                vertices = sorted(int(vertex.id) for vertex in edge.vertices)
                if edge_id < 0 or any(vertex_id < 0 for vertex_id in vertices):
                    raise RuntimeError("PyAEDT returned a non-native incidence ID")
                previous = edge_vertex_ids.get(str(edge_id))
                if previous is not None and previous != vertices:
                    raise RuntimeError(
                        f"PyAEDT returned inconsistent endpoints for edge {edge_id}"
                    )
                edge_vertex_ids[str(edge_id)] = vertices
                edge_ids.append(edge_id)
            face_edge_ids[str(face_id)] = edge_ids
        return {
            "status": "observed",
            "face_ids": list(native_face_ids),
            "face_edge_ids": face_edge_ids,
            "edge_vertex_ids": edge_vertex_ids,
        }
    except Exception as exc:  # noqa: BLE001 -- optional topology evidence.
        return {
            "status": "unavailable",
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }


def _note_native_failure(
    failure: Exception,
    app: Any,
    spec: Q3dSpec,
    evidence: dict[str, Any],
) -> None:
    """Attach operation context without replacing the native primary failure."""
    try:
        messages = app.logger.get_messages(
            project_name=spec.project_name,
            design_name=spec.design_name,
            level=1,
            aedt_messages=True,
        )
        evidence["native_messages"] = {
            "scope": "owned project/design plus native Desktop context",
            "warning_level": list(messages.warning_level),
            "error_level": list(messages.error_level),
        }
    except Exception as diagnostic_error:  # noqa: BLE001 -- preserve primary failure.
        evidence["native_messages_unavailable"] = (
            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
        )
    try:
        failure.add_note(
            "Q3D body-stage evidence: " + json.dumps(evidence, sort_keys=True)
        )
    except Exception:  # noqa: BLE001 -- preserve primary failure.
        failure.add_note("Q3D body-stage evidence could not be serialized")


def _make_sheet(
    app: Any, ring: tuple[tuple[float, float], ...], *, name: str, z_um: float
) -> Any:
    points = [[float(x), float(y), float(z_um)] for x, y in ring]
    sheet = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=False,
    )
    if sheet is False or sheet is None:
        raise RuntimeError(f"failed to create Q3D covered sheet {name!r}")
    actual = app.modeler.get_object_from_name(name)
    if actual is None or actual.name != name:
        raise RuntimeError(f"Q3D covered sheet readback failed for {name!r}")
    return actual


def _construct_body(
    app: Any,
    body: Q3dBodySpec,
    *,
    effective: dict[str, Any],
    temporary_names: tuple[str, ...],
    operations: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    phase = "outer_sheet"
    operations.append({"phase": phase, "operation": "create_covered_exterior"})
    sheet = _make_sheet(
        app, body.exterior_um, name=body.body_id, z_um=effective["effective_z_min_um"]
    )
    operations[-1].update({"object_name": sheet.name, "returned": True})

    hole_records: list[dict[str, Any]] = []
    for hole_index, ring in enumerate(body.holes_um):
        cutter_name = temporary_names[hole_index]
        phase = "hole_sheet"
        operations.append({"phase": phase, "operation": "create_covered_hole_cutter"})
        _make_sheet(app, ring, name=cutter_name, z_um=effective["effective_z_min_um"])
        operations[-1].update({"object_name": cutter_name, "returned": True})

        phase = "hole_subtraction"
        operation = {
            "phase": phase,
            "operation": "subtract_hole_cutter",
            "blank_object_name": body.body_id,
            "tool_object_name": cutter_name,
            "keep_originals": False,
        }
        operations.append(operation)
        if not app.modeler.subtract(body.body_id, cutter_name, keep_originals=False):
            raise RuntimeError(f"Q3D hole subtraction failed for {body.body_id!r}")
        if app.modeler.get_object_from_name(body.body_id) is None:
            raise RuntimeError(
                f"Q3D body disappeared during hole subtraction: {body.body_id!r}"
            )
        if app.modeler.get_object_from_name(cutter_name) is not None:
            raise RuntimeError(
                f"Q3D hole cutter remained after subtraction: {cutter_name!r}"
            )
        operation["returned"] = True
        hole_records.append(
            {
                "cutter_object_name": cutter_name,
                "subtract_keep_originals": False,
                "subtract_returned": True,
            }
        )

    if effective["representation"] == "sheet":
        return app.modeler.get_object_from_name(body.body_id), {
            "outer_sheet": {"object_name": body.body_id, "created": True},
            "hole_subtractions": hole_records,
            "effective_geometry": effective,
        }
    phase = "positive_z_sweep"
    span_um = effective["effective_z_max_um"] - effective["effective_z_min_um"]
    operation = {
        "phase": phase,
        "operation": "sweep_body_along_positive_z",
        "object_name": body.body_id,
        "vector_um": [0.0, 0.0, span_um],
    }
    operations.append(operation)
    swept = app.modeler.sweep_along_vector(
        body.body_id, ["0um", "0um", f"{span_um:.17g}um"]
    )
    if swept is False or swept is None or isinstance(swept, list):
        raise RuntimeError(
            f"Q3D body sweep did not return one native body for {body.body_id!r}"
        )
    obj = app.modeler.get_object_from_name(body.body_id)
    if obj is None or obj.name != body.body_id or swept.name != body.body_id:
        raise RuntimeError(f"Q3D swept body name readback failed for {body.body_id!r}")
    operation["returned_object_name"] = swept.name
    operation["returned"] = True
    return obj, {
        "outer_sheet": {"object_name": body.body_id, "created": True},
        "hole_subtractions": hole_records,
        "sweep": {
            "object_name": body.body_id,
            "vector_um": [0.0, 0.0, span_um],
        },
    }


def _assign_material(app: Any, body: Q3dBodySpec, material: Any) -> str:
    obj = app.modeler.get_object_from_name(body.body_id)
    if obj is None:
        raise RuntimeError(
            f"Q3D body disappeared before material assignment: {body.body_id!r}"
        )
    if material.is_superconducting:
        obj.material_name = "pec"
        expected_name = "pec"
    else:
        if not app.materials.exists_material(material.library_name):
            raise RuntimeError(
                f"AEDT library material is unavailable: {material.library_name!r}"
            )
        obj.material_name = material.library_name
        expected_name = material.library_name
    observed = native_object_property(obj, "Material").strip('"')
    if observed.casefold() != expected_name.casefold():
        raise RuntimeError(f"Q3D material readback mismatch for {body.body_id!r}")
    return observed


def construct_q3d_bodies(app: Any, spec: Q3dSpec) -> list[dict[str, Any]]:
    """Build every declared body, inventory native identities, then assign materials."""
    effective_bodies = {record["body_id"]: record for record in spec.effective_bodies}
    for body in spec.bodies:
        if (
            effective_bodies[body.body_id]["representation"] == "sheet"
            and not spec.materials[body.material_id].is_superconducting
        ):
            raise ValueError("Q3D ThinConductor requires a declared PDK superconductor")
    body_ids = [body.body_id for body in spec.bodies]
    cutter_names = tuple(
        _hole_name(body_index, hole_index)
        for body_index, body in enumerate(spec.bodies, start=1)
        for hole_index, _ in enumerate(body.holes_um, start=1)
    )
    reserved_names = body_ids + list(cutter_names)
    if len(set(reserved_names)) != len(reserved_names):
        raise RuntimeError("Q3D body and temporary hole-cutter names collide")

    existing_names = {str(name) for name in app.modeler.object_names}
    conflicts = sorted(existing_names & set(reserved_names))
    if conflicts:
        raise RuntimeError(f"Q3D body names already exist in the design: {conflicts!r}")

    records: dict[str, dict[str, Any]] = {}
    current_body_id: str | None = None
    phase = "body_construction"
    operations: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"stage": "body_construction"}
    try:
        for body_index, body in enumerate(spec.bodies, start=1):
            current_body_id = body.body_id
            phase = "outer_sheet"
            operations = []
            body_cutter_names = tuple(
                _hole_name(body_index, hole_index)
                for hole_index, _ in enumerate(body.holes_um, start=1)
            )
            _, construction = _construct_body(
                app,
                body,
                effective=effective_bodies[body.body_id],
                temporary_names=body_cutter_names,
                operations=operations,
            )
            records[body.body_id] = {"construction": construction}

        phase = "complete_body_inventory"
        current_body_id = None
        operations = []
        app.modeler.refresh_all_ids()
        actual_names = {str(name) for name in app.modeler.object_names}
        if actual_names != set(body_ids):
            raise RuntimeError(
                "Q3D native body inventory differs from the declared bodies: "
                f"expected={sorted(body_ids)!r}, actual={sorted(actual_names)!r}"
            )

        native_ids: set[int] = set()
        for body in spec.bodies:
            current_body_id = body.body_id
            phase = "body_native_readback"
            obj = app.modeler.get_object_from_name(body.body_id)
            if obj is None:
                raise RuntimeError(f"Q3D native body is unavailable: {body.body_id!r}")
            native = _native_object_evidence(app, body.body_id)
            native_id = native["native_object_id"]
            if native_id <= 0 or native_id in native_ids:
                raise RuntimeError(
                    f"Q3D native body ID is invalid or duplicated: {native_id}"
                )
            native_ids.add(native_id)
            expected_type = (
                "Sheet"
                if effective_bodies[body.body_id]["representation"] == "sheet"
                else "Solid"
            )
            if native["native_object_type"] != expected_type:
                raise RuntimeError(
                    f"Q3D body is not a native {expected_type}: {body.body_id!r}"
                )
            native_face_ids = list(native["native_face_ids"])
            if any(face_id <= 0 for face_id in native_face_ids):
                raise RuntimeError(
                    f"Q3D native face IDs are invalid for {body.body_id!r}"
                )
            records[body.body_id]["body_binding"] = {
                "body": body.to_payload(),
                "effective_geometry": effective_bodies[body.body_id],
                "native_object_id": native_id,
                "object_name": body.body_id,
                "native_object_type": native["native_object_type"],
                "native_face_ids": native_face_ids,
                "incidence": _native_incidence(obj, native_face_ids),
                "construction": records[body.body_id]["construction"],
            }

        observed: list[dict[str, Any]] = []
        phase = "material_assignment"
        for body in spec.bodies:
            current_body_id = body.body_id
            material = spec.materials[body.material_id]
            effective = effective_bodies[body.body_id]
            evidence.pop("thin_conductor_readback", None)
            if effective["representation"] == "sheet":
                boundary_name = f"{body.body_id}ThinConductor"
                thickness = f"{effective['physical_thickness_um']:.17g}um"
                boundary = app.assign_thin_conductor(
                    assignment=body.body_id,
                    material="pec",
                    thickness=thickness,
                    name=boundary_name,
                )
                if not boundary or boundary.name != boundary_name:
                    raise RuntimeError(
                        f"Q3D thin conductor assignment failed: {body.body_id!r}"
                    )
                # Sheet PEC belongs to ThinConductor, not the Solid Material property.
                child = boundary.child_object
                property_names = list(child.GetPropNames())
                if (
                    "Material" not in property_names
                    or "Thickness" not in property_names
                ):
                    raise RuntimeError(
                        f"Q3D ThinConductor native properties unavailable: {property_names!r}"
                    )
                native_id = records[body.body_id]["body_binding"]["native_object_id"]
                readback = {
                    "boundary_name": boundary_name,
                    "native_object_id": native_id,
                    "expected": {
                        "material": "pec",
                        "thickness": thickness,
                        "thickness_um": effective["physical_thickness_um"],
                        "assignment_ids": [native_id],
                    },
                    "observed": {},
                }
                # Retain each native return before the next query or comparison can fail.
                evidence["thin_conductor_readback"] = readback
                actual = readback["observed"]
                actual["material"] = child.GetPropValue("Material")
                actual["thickness"] = child.GetPropValue("Thickness")
                actual["assignment_ids"] = list(
                    app.oboundary.GetExcitationAssignment(boundary_name)
                )
                assigned = [int(value) for value in actual["assignment_ids"]]
                raw_material = actual["material"]
                if (
                    isinstance(raw_material, (list, tuple))
                    and len(raw_material) == 2
                    and raw_material[0] == "Material:="
                ):
                    raw_material = raw_material[1]
                if not isinstance(raw_material, str):
                    raise RuntimeError(
                        "Q3D ThinConductor Material has unknown native shape"
                    )
                boundary_material = raw_material.strip('"')
                from ansys.aedt.core.generic.numbers_utils import Quantity

                if not isinstance(actual["thickness"], str):
                    raise RuntimeError(
                        "Q3D ThinConductor Thickness is not a native length"
                    )
                quantity = Quantity(actual["thickness"])
                if quantity.unit_system != "Length":
                    raise RuntimeError(
                        "Q3D ThinConductor Thickness is not a native length"
                    )
                measured_thickness_um = quantity.to("um").value
                actual["thickness_um"] = measured_thickness_um
                differs = {}
                if boundary_material.casefold() != "pec":
                    differs["material"] = {
                        "expected": "pec",
                        "observed": actual["material"],
                    }
                if measured_thickness_um != effective["physical_thickness_um"]:
                    differs["thickness"] = {
                        "expected_um": effective["physical_thickness_um"],
                        "observed": actual["thickness"],
                        "observed_um": measured_thickness_um,
                    }
                if assigned != [native_id]:
                    differs["assignment_ids"] = {
                        "expected": [native_id],
                        "observed": actual["assignment_ids"],
                    }
                if differs:
                    raise RuntimeError(
                        f"Q3D thin conductor readback differs: {body.body_id!r}; "
                        f"fields={differs!r}"
                    )
                records[body.body_id]["body_binding"]["thin_conductor"] = {
                    "name": boundary_name,
                    "bound_type": "ThinConductor",
                    "material": boundary_material,
                    "material_raw": actual["material"],
                    "thickness": actual["thickness"],
                    "thickness_um": measured_thickness_um,
                    "native_assignment_ids_raw": actual["assignment_ids"],
                    "native_object_ids": assigned,
                }
                observed_material = {
                    "native_material_property": "not_applicable_sheet",
                    "native_boundary_material_name": boundary_material,
                }
            else:
                observed_material = {
                    "native_material_name": _assign_material(app, body, material)
                }
            observed.append(
                {
                    "object_name": body.body_id,
                    "role": body.physical_role,
                    "material_id": material.material_id,
                    "kind": material.kind,
                    "is_superconducting": material.is_superconducting,
                    "requested_library_name": material.library_name,
                    "observed": observed_material,
                    "body_binding": records[body.body_id]["body_binding"],
                }
            )
        return observed
    except Exception as exc:  # noqa: BLE001 -- keep the native primary error.
        evidence.update(
            {
                "phase": (
                    phase
                    if phase == "material_assignment"
                    else operations[-1]["phase"]
                    if operations
                    else phase
                ),
                "body_id": current_body_id,
                "operations": operations,
            }
        )
        _note_native_failure(exc, app, spec, evidence)
        raise
