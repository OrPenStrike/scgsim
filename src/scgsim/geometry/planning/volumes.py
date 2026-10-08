"""Plan source-bound bodies, shells, volume memberships and cut hosts from shared surfaces; no native fallback volume discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import hypot

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.entities import _entity_by_id, _entity_z_range_um
from scgsim.geometry._primitives.entity_validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
)
from scgsim.geometry._primitives.geometry_refs import (
    _entity_geometry_ref,
    _route_entity_geometry_ref,
)
from scgsim.geometry._primitives.loops import _clean_loop
from scgsim.geometry._primitives.spatial import _vector_cross, _vector_subtract
from scgsim.geometry._primitives.surface_records import (
    _conductor_boundary_surface_id,
    _surface_boundary_volume_ids,
)
from scgsim.geometry.models.common import RouteLiteral, SurfaceOrientationLiteral
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    CutHostOperationRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    MMContactRecord,
    RouteABVolumePlanRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.domain import (
    _component_is_boundary_attached,
    _conductor_sidewall_adjacent_solution_id,
    _entity_occupied_region,
    _required_host_solution_id,
    _solution_bounds,
)
from scgsim.geometry.planning.interfaces import _is_fully_normalized_face_metal
from scgsim.geometry.planning.surfaces import (
    _conductor_sidewall_geometry_refs,
    _contact_patches_by_entity_face,
    _subtract_contact_patches_from_face,
)
from scgsim.geometry.planning.tags import _entity_physical_group_id
from scgsim.geometry.planning.topology import (
    _structured_surface_boundary_volume_ids,
    _surface_owner_ids,
    _surface_ring3d_specs,
)


def plan_route_construction_bodies(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
) -> tuple[ConstructionBodyPlanRecord, ...]:
    """Plan Route A/B cutter bodies without making them final geometry.

    Route A only gives construction bodies to conductors represented as
    `cutout_boundary_shell` such as bumps/posts. Route A `surface_sheet`
    conductors are not cutters. Route B gives every `cutout_boundary_shell`
    conductor a construction body. Route C has no construction bodies because
    material volumes survive.
    """
    if route == "C":
        return ()

    contact_faces = _contact_patches_by_entity_face(interfaces)
    records: list[ConstructionBodyPlanRecord] = []
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            continue
        representation = entity.route_representations.get(route)
        if representation != "cutout_boundary_shell":
            continue

        host_id = _required_host_solution_id(build_input, entity)
        records.append(
            ConstructionBodyPlanRecord(
                construction_body_id=f"CBODY__{route}__{entity.semantic_id}",
                owner_semantic_id=entity.semantic_id,
                host_semantic_id=host_id,
                representation=representation,
                geometry_ref=_route_entity_geometry_ref(
                    build_input,
                    route,
                    entity,
                    representation=representation,
                    interfaces=interfaces,
                ),
                expected_surface_ids=(
                    ()
                    if route == "B"
                    and _is_fully_normalized_face_metal(build_input, entity)
                    else _cutout_shell_surface_ids(
                        build_input,
                        route,
                        entity,
                        interfaces=interfaces,
                        contact_faces=contact_faces,
                    )
                ),
                valid_routes=(route,),
            )
        )
    return tuple(records)


def plan_route_volumes(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
) -> tuple[VolumePlanRecord, ...]:
    """Plan volumes only after all boundary surfaces have stable ids.

    The same rule applies to Route A/B/C: solution domains and retained Route C
    conductors are closed by planned surfaces, then lowered through
    `addSurfaceLoop()` and `addVolume()`. This function must not create a
    volume from `domain_bounds_um`, `outer_loop`, or `thickness_um`; those values
    are only audit/planning metadata once surface ids exist.
    """
    entity_ids = {entity.semantic_id for entity in build_input.entities}
    surfaces_by_owner: dict[str, list[SurfacePlanRecord]] = {}
    for surface in surfaces:
        owner_ids = (
            _structured_surface_boundary_volume_ids(
                surface,
                _surface_owner_ids(surface),
            )
            if surface.metadata.get("embedded_surface_sheet")
            else _surface_boundary_volume_ids(
                surface,
                known_entity_ids=entity_ids,
            )
        )
        for owner_id in owner_ids:
            surfaces_by_owner.setdefault(str(owner_id), []).append(surface)

    records: list[VolumePlanRecord] = []
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            physical_owner_semantic_id = (
                "VACUUM_REGION"
                if bool(entity.metadata.get("is_auto_vacuum_region"))
                and entity.material_kind == "vacuum"
                else None
            )
            all_refs = tuple(
                SurfaceRefRecord(
                    surface_id=surface.surface_id,
                    orientation=_surface_orientation_for_volume(surface, entity),
                    role="planned_boundary",
                )
                for surface in surfaces_by_owner.get(entity.semantic_id, ())
            )
            void_surfaces: dict[str, list[SurfaceRefRecord]] = {}
            component_members: dict[str, set[str]] = {}
            contact_ids_by_component: dict[str, list[str]] = {}
            for contact in mm_contacts:
                component_members.setdefault(
                    contact.conductor_component_id, set()
                ).update((contact.lower_entity_id, contact.upper_entity_id))
                contact_ids_by_component.setdefault(
                    contact.conductor_component_id, []
                ).append(contact.contact_id)
            boundary_components = {
                str(surface.metadata["conductor_component_id"])
                for surface in surfaces_by_owner.get(entity.semantic_id, ())
                if surface.metadata.get("conductor_component_id") is not None
                and _component_is_boundary_attached(
                    build_input, solution=entity, surface=surface
                )
            }
            exterior_refs: list[SurfaceRefRecord] = []
            for surface, ref in zip(
                surfaces_by_owner.get(entity.semantic_id, ()), all_refs, strict=True
            ):
                is_sheet_cap = (
                    route == "A"
                    and bool(surface.metadata.get("sheet_contact_cap"))
                    and _is_vacuum_solution_entity(entity)
                )
                component_id = str(
                    surface.metadata.get(
                        "conductor_component_id",
                        f"COMP__{surface.owner_semantic_id}",
                    )
                )
                is_component_boundary = component_id in boundary_components
                if (
                    surface.surface_role == "cutout_boundary_shell" or is_sheet_cap
                ) and not is_component_boundary:
                    component_members.setdefault(component_id, set()).add(
                        surface.owner_semantic_id
                    )
                    void_surfaces.setdefault(component_id, []).append(
                        replace(
                            ref,
                            orientation=_inner_void_orientation(
                                build_input,
                                surface=surface,
                                component_members=component_members[component_id],
                            ),
                        )
                    )
                else:
                    exterior_refs.append(ref)
            void_shells = tuple(
                InnerPecVoidShellRecord(
                    shell_id=f"VOID__{entity.semantic_id}__{component_id}",
                    surface_refs=tuple(refs),
                    conductor_component_id=component_id,
                    owner_semantic_ids=tuple(sorted(component_members[component_id])),
                )
                for component_id, refs in void_surfaces.items()
            )
            if route in {"A", "B"}:
                records.append(
                    RouteABVolumePlanRecord(
                        volume_id=f"VOL__{entity.semantic_id}",
                        owner_semantic_id=entity.semantic_id,
                        material_id=entity.material_id,
                        surface_refs=all_refs,
                        exterior_surface_refs=tuple(exterior_refs),
                        inner_pec_void_shells=void_shells,
                        valid_routes=(route,),
                        metadata={
                            "representation": "solution_volume",
                            "material_kind": entity.material_kind,
                            "geometry_ref": dict(entity.geometry),
                            "physical_owner_semantic_id": physical_owner_semantic_id,
                            "inner_pec_void_shell_contact_ids": {
                                component_id: tuple(
                                    contact_ids_by_component.get(component_id, ())
                                )
                                for component_id in void_surfaces
                            },
                        },
                    )
                )
            else:
                records.append(
                    VolumePlanRecord(
                        volume_id=f"VOL__{entity.semantic_id}",
                        owner_semantic_id=entity.semantic_id,
                        material_id=entity.material_id,
                        surface_refs=all_refs,
                        valid_routes=(route,),
                        metadata={
                            "representation": "solution_volume",
                            "material_kind": entity.material_kind,
                            "geometry_ref": dict(entity.geometry),
                            "physical_owner_semantic_id": physical_owner_semantic_id,
                        },
                    )
                )
            continue
        representation = entity.route_representations.get(route)
        if route == "C" and representation == "material_volume":
            physical_owner_semantic_id = _entity_physical_group_id(entity)
            records.append(
                VolumePlanRecord(
                    volume_id=f"VOL__{entity.semantic_id}",
                    owner_semantic_id=entity.semantic_id,
                    material_id=entity.material_id,
                    surface_refs=tuple(
                        SurfaceRefRecord(
                            surface_id=surface.surface_id,
                            orientation=_surface_orientation_for_volume(
                                surface,
                                entity,
                            ),
                            role="planned_boundary",
                        )
                        for surface in surfaces_by_owner.get(
                            entity.semantic_id,
                            (),
                        )
                    ),
                    valid_routes=(route,),
                    metadata={
                        "representation": representation,
                        "material_kind": entity.material_kind,
                        "geometry_ref": _entity_geometry_ref(
                            entity,
                            representation=representation,
                        ),
                        "physical_owner_semantic_id": physical_owner_semantic_id,
                    },
                )
            )
    return tuple(records)


def _inner_void_orientation(
    build_input: GeometryBuildInput,
    *,
    surface: SurfacePlanRecord,
    component_members: set[str],
) -> SurfaceOrientationLiteral:
    """Orient from conductor center, then reverse for a solution inner shell."""
    centers = tuple(
        _entity_volume_center_um(_entity_by_id(build_input, member))
        for member in component_members
    )
    component_center = tuple(
        sum(center[index] for center in centers) / len(centers) for index in range(3)
    )
    normal = _surface_normal_vector(surface)
    centroid = _surface_centroid(surface)
    dot = sum(
        normal[index] * (centroid[index] - component_center[index])
        for index in range(3)
    )
    if abs(dot) <= 1e-9:
        # A symmetric perforated conductor can put a sidewall centroid exactly
        # on the component center.  Classify the normal locally against the
        # component's occupied planar region instead of inventing a global
        # orientation convention for that degenerate center-vector case.
        normal_xy_length = hypot(normal[0], normal[1])
        if normal_xy_length > _TOPOLOGY_EPS_UM:
            import gdstk

            regions = tuple(
                region
                for member in component_members
                for region in _entity_occupied_region(
                    gdstk,
                    _entity_by_id(build_input, member),
                )
            )
            if regions:
                epsilon = max(1e-6, normal_xy_length * 1e-6)
                plus = (
                    centroid[0] + epsilon * normal[0] / normal_xy_length,
                    centroid[1] + epsilon * normal[1] / normal_xy_length,
                )
                minus = (
                    centroid[0] - epsilon * normal[0] / normal_xy_length,
                    centroid[1] - epsilon * normal[1] / normal_xy_length,
                )
                plus_inside, minus_inside = gdstk.inside((plus, minus), regions)
                if plus_inside != minus_inside:
                    # A forward normal points out of the conductor precisely
                    # when its positive local probe is outside the component.
                    return "reversed" if not plus_inside else "forward"
        raise ValueError(
            f"{surface.surface_id} has ambiguous component void orientation"
        )
    return "reversed" if dot > 0 else "forward"


def plan_cut_host_operations(
    *,
    route: RouteLiteral,
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
) -> tuple[CutHostOperationRecord, ...]:
    """Group Route A/B construction bodies into host-cut operation plans.

    This is semantic grouping and provenance in the current v1 backend. Exposed
    shell surfaces are already planned as `SurfacePlanRecord`s; these records
    explain which construction bodies belong to each host exclusion policy
    without asking the backend to discover new surfaces through boolean cuts.
    """
    if route == "C":
        return ()

    bodies_by_host: dict[str, list[ConstructionBodyPlanRecord]] = {}
    for body in construction_bodies:
        bodies_by_host.setdefault(body.host_semantic_id, []).append(body)

    return tuple(
        CutHostOperationRecord(
            operation_id=f"CUT__{route}__{host_id}",
            host_semantic_id=host_id,
            construction_body_ids=tuple(body.construction_body_id for body in bodies),
            exposed_surface_ids=tuple(
                surface_id
                for body in bodies
                for surface_id in body.expected_surface_ids
            ),
            valid_routes=(route,),
        )
        for host_id, bodies in bodies_by_host.items()
    )


def _reconcile_construction_body_surface_ids(
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    *,
    surfaces: Sequence[SurfacePlanRecord],
) -> tuple[ConstructionBodyPlanRecord, ...]:
    """Bind preplanned cutter bodies to their final local surface children."""
    surface_ids_by_body: dict[str, list[str]] = {}
    for surface in surfaces:
        body_id = surface.geometry_ref.get("construction_body_id")
        if isinstance(body_id, str) and body_id:
            surface_ids_by_body.setdefault(body_id, []).append(surface.surface_id)
    return tuple(
        replace(
            body,
            expected_surface_ids=tuple(
                sorted(surface_ids_by_body.get(body.construction_body_id, ()))
            ),
        )
        if body.construction_body_id in surface_ids_by_body
        else body
        for body in construction_bodies
    )


def _cutout_shell_surface_ids(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    *,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
    contact_faces: Mapping[
        tuple[str, str],
        tuple[tuple[tuple[float, float], ...], ...],
    ]
    | None = None,
) -> tuple[str, ...]:
    contact_faces = contact_faces or {}
    top_bottom_ids: list[str] = []
    for shell_part in ("top", "bottom"):
        base_surface_id = _conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            shell_part,
        )
        base_geometry_ref = {
            **_route_entity_geometry_ref(
                build_input,
                route,
                entity,
                representation="cutout_boundary_shell",
                interfaces=interfaces,
            ),
            "shell_part": shell_part,
        }
        face_geometry_refs = _subtract_contact_patches_from_face(
            base_geometry_ref,
            contact_faces.get((entity.semantic_id, shell_part), ()),
        )
        if len(face_geometry_refs) == 1:
            top_bottom_ids.append(base_surface_id)
        else:
            top_bottom_ids.extend(
                f"{base_surface_id}__P{index:04d}"
                for index, _ in enumerate(face_geometry_refs)
            )
    sidewall_adjacent_id = (
        _conductor_sidewall_adjacent_solution_id(build_input, entity)
        if route == "C"
        else None
    )
    sidewall_refs = (
        ()
        if route == "C" and sidewall_adjacent_id is None
        else _conductor_sidewall_geometry_refs(
            build_input,
            route=route,
            entity=entity,
            representation="cutout_boundary_shell",
            adjacent_solution_id=sidewall_adjacent_id,
            interfaces=interfaces,
        )
    )
    sidewall_ids = tuple(
        _conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            f"sidewall_{edge_index:04d}",
        )
        for edge_index, _ in enumerate(sidewall_refs)
    )
    return (*top_bottom_ids, *sidewall_ids)


def _surface_orientation_for_volume(
    surface: SurfacePlanRecord,
    entity: SemanticEntitySpec,
) -> SurfaceOrientationLiteral:
    """Orient a planned surface as an outward face of one owning volume."""
    normal = _surface_normal_vector(surface)
    surface_centroid = _surface_centroid(surface)
    volume_center = _entity_volume_center_um(entity)
    outward = tuple(
        surface_centroid[index] - volume_center[index] for index in range(3)
    )
    dot = sum(normal[index] * outward[index] for index in range(3))
    if abs(dot) <= 1e-9:
        raise ValueError(
            f"{surface.surface_id} has ambiguous orientation for {entity.semantic_id}"
        )
    return "forward" if dot > 0 else "reversed"


def _surface_normal_vector(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    if surface.normal_hint is not None:
        return tuple(float(value) for value in surface.normal_hint)
    ring = _surface_ring3d_specs(surface)[0][2]
    origin = ring[0]
    for first_index in range(1, len(ring) - 1):
        first = _vector_subtract(ring[first_index], origin)
        for second_index in range(first_index + 1, len(ring)):
            second = _vector_subtract(ring[second_index], origin)
            normal = _vector_cross(first, second)
            length_sq = sum(value * value for value in normal)
            if length_sq > 1e-18:
                return normal
    raise ValueError(f"{surface.surface_id} has no nondegenerate normal")


def _surface_centroid(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    points = tuple(
        coordinate
        for _, _, ring in _surface_ring3d_specs(surface)
        for coordinate in ring
    )
    if not points:
        raise ValueError(f"{surface.surface_id} has no coordinates")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _entity_volume_center_um(
    entity: SemanticEntitySpec,
) -> tuple[float, float, float]:
    if _is_solution_entity(entity):
        bounds = _solution_bounds(entity)
        return (
            (float(bounds["x_min_um"]) + float(bounds["x_max_um"])) / 2.0,
            (float(bounds["y_min_um"]) + float(bounds["y_max_um"])) / 2.0,
            (float(entity.geometry["z_min_um"]) + float(entity.geometry["z_max_um"]))
            / 2.0,
        )
    if "outer_loop" not in entity.geometry:
        raise ValueError(f"{entity.semantic_id} requires outer_loop for volume center")
    loop = _clean_loop(entity.geometry["outer_loop"])
    z_min_um, z_max_um = _entity_z_range_um(entity)
    return (
        (min(point[0] for point in loop) + max(point[0] for point in loop)) / 2.0,
        (min(point[1] for point in loop) + max(point[1] for point in loop)) / 2.0,
        (z_min_um + z_max_um) / 2.0,
    )
