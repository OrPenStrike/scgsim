"""Recognize interfaces and conductor contact/MM ownership from semantic source facts before backend construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from scgsim.geometry._primitives.constants import (
    _INTERFACE_KIND_ORDER,
    _TOPOLOGY_EPS_UM,
)
from scgsim.geometry._primitives.entities import (
    _entity_z_range_um,
    _intent_supports_route,
    _unique_ids,
)
from scgsim.geometry._primitives.entity_validation import (
    _resolve_contact_pad_attachment,
)
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _canonical_loop_sort_key,
    _clean_loop,
    _loop_signature,
    _loops_touch_without_area,
    _polygon_area,
    _ring_edges,
)
from scgsim.geometry._primitives.spatial import _bounds_overlap, _same_z, _z_key
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    _active_route_conductor_entities,
    _entity_loop_bounds,
    _entity_occupied_region,
    _required_host_solution_id,
    _route_a_sheet_boundary_volume_ids_from_solutions,
    _route_a_sheet_plane_z_um_from_solutions,
    _solution_entities,
)
from scgsim.geometry.planning.topology import (
    _surface_interface_record_owners,
    _surface_owner_ids,
)


def complete_interface_plan_from_surfaces(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Backfill InterfacePlan records for planned adjacency surfaces."""
    known = {interface.interface_id for interface in interfaces}
    completed = list(interfaces)
    for surface in surfaces:
        if surface.interface_id is None or surface.interface_id in known:
            continue
        kind = surface.interface_id.split("__", 1)[0]
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid surface interface id: {surface.interface_id}")
        owners = _surface_owner_ids(surface)
        record_owners = _surface_interface_record_owners(surface, owners)
        if len(record_owners) != 2:
            raise ValueError(
                f"{surface.surface_id} interface needs two owners, got {owners!r}"
            )
        completed.append(
            InterfacePlanRecord(
                interface_id=surface.interface_id,
                kind=kind,  # type: ignore[arg-type]
                owner_semantic_ids=(record_owners[0], record_owners[1]),
                recognition_rule=str(
                    surface.metadata.get(
                        "recognition_rule",
                        "planned_surface_adjacency",
                    )
                ),
                solver_use=surface.solver_use,
                metadata={
                    "generated_from_surface_id": surface.surface_id,
                    **dict(surface.metadata),
                },
            )
        )
        known.add(surface.interface_id)
    return tuple(completed)


def recognize_route_interfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize interfaces before creating OCC geometry.

    Route-specific recognizers should own each interface rule: draw/ground
    shared edges, XY footprint overlap contacts, stack domain boundaries, ports,
    and exposed EM boundaries. Current fixture metadata may seed those rules
    through `interface_intents_2d`, including generic `interfaces` entries with
    explicit `kind`.
    """
    intents = dict(build_input.metadata.get("interface_intents_2d", {}))
    records: list[InterfacePlanRecord] = []
    for index, intent in enumerate(intents.get("interfaces", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("interfaces entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        kind = str(intent.get("kind", ""))
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid interface kind: {kind!r}")
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid interface owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=str(
                    intent.get("interface_id") or f"{kind}__INTENT__{index:04d}"
                ),
                kind=kind,
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule=str(intent.get("recognition_rule", "explicit")),
                source_polygon_ids=tuple(
                    str(value) for value in intent.get("source_polygon_ids", ())
                ),
                metadata=dict(intent),
            )
        )
    for index, intent in enumerate(intents.get("metal_metal_contact_edges", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_metal_contact_edges entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid MM edge intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_EDGE__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="draw_edge_overlaps_ground_mask_cutout_edge",
                source_polygon_ids=tuple(
                    str(intent[key])
                    for key in ("source_polygon_id", "ground_polygon_id")
                    if key in intent
                ),
                metadata=dict(intent),
            )
        )
    for index, intent in enumerate(intents.get("metal_ground_contact_patches", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_ground_contact_patches entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid contact patch intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_PATCH__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="projected_xy_footprint_overlap",
                source_polygon_ids=tuple(
                    str(intent[key])
                    for key in ("source_polygon_id", "ground_polygon_id")
                    if key in intent
                ),
                metadata=dict(intent),
            )
        )
    return tuple(records)


def plan_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize coplanar conductor contact patches from planned conductors.

    This is the small v1 contact planner: it partitions each opposite-face
    overlap into connected conductor islands and emits one interface per
    validated contact component.
    """
    if route == "C":
        return _plan_route_c_conductor_contact_patches(
            build_input, route=route, interfaces=interfaces
        )
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    solution_entities = _solution_entities(build_input)
    top_faces: dict[
        float,
        list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]],
    ] = {}
    bottom_faces: dict[
        float,
        list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]],
    ] = {}

    for entity in _active_route_conductor_entities(build_input, route):
        region = _entity_occupied_region(gdstk, entity)
        if not region:
            continue
        bounds = _entity_loop_bounds(entity)
        z_min_um, z_max_um = _entity_z_range_um(entity)
        bottom_faces.setdefault(_z_key(z_min_um), []).append((entity, region, bounds))
        top_faces.setdefault(_z_key(z_max_um), []).append((entity, region, bounds))

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower, lower_region, lower_bounds in lower_faces:
            for upper, upper_region, upper_bounds in bottom_faces.get(z_key, ()):
                if lower.semantic_id == upper.semantic_id:
                    continue
                if not _bounds_overlap(lower_bounds, upper_bounds):
                    if _loops_touch_without_area(
                        lower.geometry["outer_loop"], upper.geometry["outer_loop"]
                    ):
                        raise ValueError(
                            f"{lower.semantic_id}/{upper.semantic_id} has "
                            "edge/point-only conductor contact"
                        )
                    continue
                overlap_region = _boolean_gdstk_region(
                    gdstk,
                    lower_region,
                    upper_region,
                    "and",
                )
                if not overlap_region:
                    if _loops_touch_without_area(
                        lower.geometry["outer_loop"], upper.geometry["outer_loop"]
                    ):
                        raise ValueError(
                            f"{lower.semantic_id}/{upper.semantic_id} has "
                            "edge/point-only conductor contact"
                        )
                    continue
                contact_loops = _contact_patch_loops(
                    overlap_region,
                    lower.semantic_id,
                    upper.semantic_id,
                )
                # When the upper footprint is wholly the contact patch, keep
                # its authored ring verbatim.  The lower top hole and upper
                # authored sidewall then share canonical curves without
                # replacing either body's sidewall geometry.
                upper_remainder = _boolean_gdstk_region(
                    gdstk, upper_region, overlap_region, "not"
                )
                if not upper_remainder and not upper.geometry.get("hole_loops"):
                    contact_loops = (_clean_loop(upper.geometry["outer_loop"]),)
                for contact_loop in contact_loops:
                    signature = (
                        lower.semantic_id,
                        upper.semantic_id,
                        _z_key(float(z_key)),
                        _loop_signature(contact_loop),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    metadata = _contact_patch_metadata(
                        build_input,
                        route=route,
                        solution_entities=solution_entities,
                        lower=lower,
                        upper=upper,
                        contact_z_um=float(z_key),
                        contact_loop=contact_loop,
                        contact_index=index,
                    )
                    generated.append(
                        InterfacePlanRecord(
                            interface_id=(
                                f"MM__CONTACT__{lower.semantic_id}__"
                                f"{upper.semantic_id}__{index:04d}"
                            ),
                            kind="MM",
                            owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                            recognition_rule="coplanar_conductor_contact_patch",
                            source_polygon_ids=(
                                *lower.polygon_ids,
                                *upper.polygon_ids,
                            ),
                            metadata=metadata,
                        )
                    )
                    index += 1
    return (*interfaces, *generated)


def _plan_route_c_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Preserve the pre-A/B Route-C rectangular-contact behavior exactly."""
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    solution_entities = _solution_entities(build_input)
    top_faces: dict[
        float, list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]]
    ] = {}
    bottom_faces: dict[
        float, list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]]
    ] = {}
    for entity in _active_route_conductor_entities(build_input, route):
        region = _entity_occupied_region(gdstk, entity)
        if not region:
            continue
        bounds = _entity_loop_bounds(entity)
        z_min_um, z_max_um = _entity_z_range_um(entity)
        bottom_faces.setdefault(_z_key(z_min_um), []).append((entity, region, bounds))
        top_faces.setdefault(_z_key(z_max_um), []).append((entity, region, bounds))

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower, lower_region, lower_bounds in lower_faces:
            for upper, upper_region, upper_bounds in bottom_faces.get(z_key, ()):
                if lower.semantic_id == upper.semantic_id or not _bounds_overlap(
                    lower_bounds, upper_bounds
                ):
                    continue
                overlap_region = _boolean_gdstk_region(
                    gdstk, lower_region, upper_region, "and"
                )
                if not overlap_region:
                    continue
                contact_loop = _single_rectangular_contact_loop(
                    overlap_region, lower.semantic_id, upper.semantic_id
                )
                signature = (
                    lower.semantic_id,
                    upper.semantic_id,
                    _z_key(float(z_key)),
                    _loop_signature(contact_loop),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                metadata = _route_c_contact_patch_metadata(
                    build_input,
                    solution_entities=solution_entities,
                    lower=lower,
                    upper=upper,
                    contact_z_um=float(z_key),
                    contact_loop=contact_loop,
                )
                generated.append(
                    InterfacePlanRecord(
                        interface_id=(
                            f"MM__CONTACT__{lower.semantic_id}__"
                            f"{upper.semantic_id}__{index:04d}"
                        ),
                        kind="MM",
                        owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                        recognition_rule="coplanar_conductor_contact_patch",
                        source_polygon_ids=(*lower.polygon_ids, *upper.polygon_ids),
                        metadata=metadata,
                    )
                )
                index += 1
    return (*interfaces, *generated)


def _route_c_contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    solution_entities: Sequence[SemanticEntitySpec],
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    contact_z_um: float,
    contact_loop: tuple[tuple[float, float], ...],
) -> Mapping[str, Any]:
    """Exact pre-A/B Route-C contact metadata shape."""
    del build_input, solution_entities
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": "retained_material_contact",
        "export_surface": True,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "contact_plane": {"axis": "z", "value_um": contact_z_um},
        "plane": {"axis": "z", "value_um": contact_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_kinds": ("MM",),
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": (lower.semantic_id, upper.semantic_id),
        "valid_routes": ("C",),
    }


def plan_mm_contact_records(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[tuple[InterfacePlanRecord, ...], tuple[MMContactRecord, ...]]:
    """Turn finite-area same-net contact into hidden component provenance.

    Gmsh/OCC must never infer this graph from coincident bodies.  The contact
    record remains present for both Route A and Route B while its internal face
    is deliberately absent from solver physical groups.
    """
    entities = {
        entity.semantic_id: entity
        for entity in _active_route_conductor_entities(build_input, route)
    }
    contacts: list[
        tuple[InterfacePlanRecord, SemanticEntitySpec, SemanticEntitySpec]
    ] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower = entities.get(lower_id)
        upper = entities.get(upper_id)
        if lower is None or upper is None:
            raise ValueError(f"{interface.interface_id} references non-live conductors")
        if not lower.net_id or not upper.net_id:
            raise ValueError(f"{interface.interface_id} contact requires resolved nets")
        if lower.net_id != upper.net_id:
            raise ValueError(
                f"{interface.interface_id} shorts different nets "
                f"{lower.net_id!r} and {upper.net_id!r}"
            )
        contacts.append((interface, lower, upper))

    normalizations = _validate_volumetric_conductor_contacts(
        build_input, route=route, entities=entities
    )
    parent = {entity_id: entity_id for entity_id in entities}

    def find(entity_id: str) -> str:
        while parent[entity_id] != entity_id:
            parent[entity_id] = parent[parent[entity_id]]
            entity_id = parent[entity_id]
        return entity_id

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for _, lower, upper in contacts:
        union(lower.semantic_id, upper.semantic_id)
    for lower, upper, _ in normalizations:
        union(lower.semantic_id, upper.semantic_id)

    component_ids = {entity_id: f"COMP__{find(entity_id)}" for entity_id in entities}
    component_values: dict[str, tuple[str | None, str | None]] = {}
    for component_id in set(component_ids.values()):
        members = [
            entity
            for entity_id, entity in entities.items()
            if component_ids[entity_id] == component_id
        ]
        nets = {entity.net_id for entity in members if entity.net_id}
        equipotentials = {
            str(entity.metadata["equipotential_id"])
            for entity in members
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(nets) > 1 or len(equipotentials) > 1:
            raise ValueError(f"{component_id} has conflicting net/equipotential ids")
        component_values[component_id] = (
            next(iter(nets), None),
            next(iter(equipotentials), None),
        )
    records: list[MMContactRecord] = []
    enriched: list[InterfacePlanRecord] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            enriched.append(interface)
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower, upper = entities[lower_id], entities[upper_id]
        loop = _clean_loop(interface.metadata["outer_loop"])
        component_id = component_ids[lower_id]
        component_net, component_equipotential = component_values[component_id]
        contact_id = str(interface.metadata.get("contact_id") or interface.interface_id)
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.append(
            MMContactRecord(
                contact_id=contact_id,
                lower_entity_id=lower_id,
                upper_entity_id=upper_id,
                lower_source_face_id=f"{lower_id}__{lower_face}",
                upper_source_face_id=f"{upper_id}__{upper_face}",
                lower_source_fragment_ids=lower.polygon_ids,
                upper_source_fragment_ids=upper.polygon_ids,
                outer_loop=loop,
                area_um2=abs(_polygon_area(loop)),
                normal=(0.0, 0.0, 1.0),
                conductor_component_id=component_id,
                net_id=component_net,
                equipotential_id=component_equipotential,
                layer_provenance={
                    "lower": _source_layer_provenance(lower),
                    "upper": _source_layer_provenance(upper),
                },
                material_provenance={
                    "lower": lower.material_id,
                    "upper": upper.material_id,
                },
                source_provenance={
                    "interface_id": interface.interface_id,
                    "route": route,
                    "recognition_rule": interface.recognition_rule,
                    "source_polygon_ids": interface.source_polygon_ids,
                },
            )
        )
        enriched.append(
            replace(
                interface,
                metadata={
                    **dict(interface.metadata),
                    "conductor_component_id": component_id,
                    "net_id": component_net,
                    "equipotential_id": component_equipotential,
                    "hidden_solver_contact": route in {"A", "B", "_effective"},
                },
            )
        )
    for lower, upper, loops in normalizations:
        for index, loop in enumerate(loops):
            records.append(
                MMContactRecord(
                    contact_id=(
                        f"NORMALIZED__{lower.semantic_id}__{upper.semantic_id}__"
                        f"{index:04d}"
                    ),
                    lower_entity_id=lower.semantic_id,
                    upper_entity_id=upper.semantic_id,
                    lower_source_face_id=f"{lower.semantic_id}__normalized_overlap",
                    upper_source_face_id=f"{upper.semantic_id}__normalized_overlap",
                    lower_source_fragment_ids=lower.polygon_ids,
                    upper_source_fragment_ids=upper.polygon_ids,
                    outer_loop=loop,
                    area_um2=abs(_polygon_area(loop)),
                    normal=(0.0, 0.0, 1.0),
                    conductor_component_id=component_ids[lower.semantic_id],
                    net_id=component_values[component_ids[lower.semantic_id]][0],
                    equipotential_id=component_values[component_ids[lower.semantic_id]][
                        1
                    ],
                    layer_provenance={
                        "lower": _source_layer_provenance(lower),
                        "upper": _source_layer_provenance(upper),
                    },
                    material_provenance={
                        "lower": lower.material_id,
                        "upper": upper.material_id,
                    },
                    source_provenance={
                        "route": route,
                        "source_polygon_ids": (*lower.polygon_ids, *upper.polygon_ids),
                    },
                )
            )
    return tuple(enriched), tuple(records)


def _source_layer_provenance(entity: SemanticEntitySpec) -> dict[str, Any]:
    """Exact authored source layer identity; no layer/datatype collapsing."""
    return {
        "source_layer_name": entity.metadata.get("source_layer_name"),
        "gds_layer": entity.geometry.get("gds_layer"),
        "gds_datatype": entity.geometry.get("gds_datatype"),
    }


def _validate_volumetric_conductor_contacts(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entities: Mapping[str, SemanticEntitySpec],
) -> tuple[
    tuple[
        SemanticEntitySpec,
        SemanticEntitySpec,
        tuple[tuple[tuple[float, float], ...], ...],
    ],
    ...,
]:
    """Reject ambiguous body overlap before a route can hide it as a PEC void."""
    import gdstk

    ordered = tuple(sorted(entities.values(), key=lambda entity: entity.semantic_id))
    normalized: list[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ]
    ] = []
    # An authored contact_pad is not merely a label: it must normalize a real,
    # finite same-z overlap with the explicitly attached face-metal entity (or
    # semantic group).  A disconnected pad would otherwise be silently kept
    # as an unrelated PEC body under an attachment provenance claim.
    for pad in ordered:
        if pad.part_role != "contact_pad":
            continue
        _resolve_contact_pad_attachment(pad, ordered)
    for index, lower in enumerate(ordered):
        lower_region = _entity_occupied_region(gdstk, lower)
        lower_min, lower_max = _entity_z_range_um(lower)
        for upper in ordered[index + 1 :]:
            upper_min, upper_max = _entity_z_range_um(upper)
            if (
                min(lower_max, upper_max) - max(lower_min, upper_min)
                <= _TOPOLOGY_EPS_UM
            ):
                continue
            overlap = _boolean_gdstk_region(
                gdstk, lower_region, _entity_occupied_region(gdstk, upper), "and"
            )
            if not overlap:
                continue
            _validate_volumetric_overlap_ownership(lower, upper, ordered)
            normalized.append(
                (
                    lower,
                    upper,
                    _contact_patch_loops(overlap, lower.semantic_id, upper.semantic_id),
                )
            )
    return tuple(normalized)


def _validate_volumetric_overlap_ownership(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> None:
    """Apply contact ownership after the caller establishes actual body overlap."""
    if not lower.net_id or not upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "requires resolved nets"
        )
    if lower.net_id != upper.net_id:
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "shorts different nets"
        )
    if not _has_explicit_volumetric_normalization(
        lower, upper, entities, finite_overlap=finite_overlap
    ):
        raise ValueError(
            f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
            "requires explicit same-net UBM/M1/In normalization provenance"
        )


def _has_explicit_volumetric_normalization(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
) -> bool:
    lower_z_min, lower_z_max = _entity_z_range_um(lower)
    upper_z_min, upper_z_max = _entity_z_range_um(upper)
    if not (_same_z(lower_z_min, upper_z_min) and _same_z(lower_z_max, upper_z_max)):
        return False
    return any(
        pad.part_role == "contact_pad"
        and face.part_role == "face_metal"
        and _resolve_contact_pad_attachment(
            pad, entities, finite_overlap=finite_overlap
        )
        is face
        for pad, face in ((lower, upper), (upper, lower))
    )


def _is_fully_normalized_face_metal(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> bool:
    if entity.part_role != "face_metal":
        return False
    import gdstk

    face = _entity_occupied_region(gdstk, entity)
    pads = tuple(
        _entity_occupied_region(gdstk, pad)
        for pad in build_input.entities
        if pad.part_role == "contact_pad"
        and _resolve_contact_pad_attachment(pad, build_input.entities) is entity
    )
    covered = _boolean_gdstk_region(
        gdstk, face, tuple(polygon for region in pads for polygon in region), "not"
    )
    return not covered


def _contact_signature(interface: InterfacePlanRecord) -> tuple[Any, ...]:
    if interface.recognition_rule != "coplanar_conductor_contact_patch":
        return ()
    return (
        str(interface.metadata.get("lower_entity_id", "")),
        str(interface.metadata.get("upper_entity_id", "")),
        _z_key(float(interface.metadata.get("contact_z_um", 0.0))),
        _loop_signature(interface.metadata.get("outer_loop", ())),
    )


def _contact_patch_loops(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if not polygons:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce at least one "
            f"polygon, got {len(polygons)}"
        )
    loops: list[tuple[tuple[float, float], ...]] = []
    for polygon in polygons:
        loop = _clean_loop(polygon.points)
        loops.append(loop)
    return _sort_contact_loops(tuple(loops))


def _single_rectangular_contact_loop(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[float, float], ...]:
    """Route-C legacy contact recognizer: exactly one rectangular patch."""
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if len(polygons) != 1:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce one polygon, got "
            f"{len(polygons)}"
        )
    loop = _clean_loop(polygons[0].points)
    if not _is_axis_aligned_rectangle(loop):
        raise ValueError(
            f"{lower_id} and {upper_id} contact is not a rectangular patch"
        )
    return loop


def _is_axis_aligned_rectangle(loop: tuple[tuple[float, float], ...]) -> bool:
    return len(loop) == 4 and all(
        _same_z(start[0], end[0]) or _same_z(start[1], end[1])
        for start, end in _ring_edges(loop)
    )


def _sort_contact_loops(
    loops: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    def _stable_loop_key(loop: tuple[tuple[float, float], ...]) -> tuple:
        clean = _clean_loop(loop)
        xs = tuple(point[0] for point in clean)
        ys = tuple(point[1] for point in clean)
        canonical = _canonical_loop_sort_key(clean)
        return (
            min(xs),
            max(xs),
            min(ys),
            max(ys),
            round(abs(_polygon_area(clean)), 12),
            canonical,
        )

    return tuple(
        sorted(
            {_loop_signature(loop): loop for loop in loops}.values(),
            key=_stable_loop_key,
        )
    )


def _contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution_entities: Sequence[SemanticEntitySpec],
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    contact_z_um: float,
    contact_loop: tuple[tuple[float, float], ...],
    contact_index: int,
) -> Mapping[str, Any]:
    plane_z_um = contact_z_um
    boundary_volume_ids: tuple[str, ...]
    interface_kinds: tuple[str, ...] = ("MM",)
    export_surface = route != "B"

    if route in {"A", "_effective"}:
        sheet = next(
            (
                entity
                for entity in (lower, upper)
                if entity.route_representations.get(route) == "surface_sheet"
            ),
            None,
        )
        if sheet is not None:
            plane_z_um = _route_a_sheet_plane_z_um_from_solutions(
                sheet,
                solution_entities,
            )
            boundary_volume_ids = _route_a_sheet_boundary_volume_ids_from_solutions(
                sheet,
                solution_entities,
            )
            interface_kinds = ("MM", "MS")
        else:
            boundary_volume_ids = ()
            export_surface = False
    elif route == "C":
        boundary_volume_ids = (lower.semantic_id, upper.semantic_id)
    else:
        boundary_volume_ids = _unique_ids(
            (
                _required_host_solution_id(build_input, lower),
                _required_host_solution_id(build_input, upper),
            )
        )

    contact_policy = {
        "A": "sheet_contact_patch",
        "_effective": "sheet_contact_patch",
        "B": "hidden_cutout_contact",
        "C": "retained_material_contact",
    }[route]
    interface_type = "_".join(interface_kinds)
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": contact_policy,
        "export_surface": export_surface,
        "contact_kind": "MM",
        "contact_index": contact_index,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "face_kinds": ("top", "bottom"),
        "route": route,
        "contact_representation": "circuit_contact_patch",
        "contact_id": (
            f"{lower.semantic_id}__{upper.semantic_id}__"
            f"{contact_policy}__{contact_index:04d}"
        ),
        "contact_plane": {"axis": "z", "value_um": plane_z_um},
        "plane": {"axis": "z", "value_um": plane_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_type": interface_type,
        "interface_kinds": interface_kinds,
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": boundary_volume_ids,
        "conductor_component_id": f"{lower.semantic_id}__{upper.semantic_id}",
        "valid_routes": (route,),
    }
