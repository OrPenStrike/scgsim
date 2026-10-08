"""Plan partitions, sheet regions and contribution surfaces using canonical source/interface ownership."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

from scgsim.geometry._primitives.constants import (
    _INTERFACE_KIND_ORDER,
    _TOPOLOGY_EPS_UM,
)
from scgsim.geometry._primitives.entities import (
    _component_metadata,
    _entity_by_id,
    _entity_z_range_um,
    _geometry_ref_from_metadata,
    _unique_ids,
)
from scgsim.geometry._primitives.entity_validation import _is_solution_entity
from scgsim.geometry._primitives.geometry_refs import (
    _route_entity_geometry_ref,
    _sidewall_geometry_refs,
)
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _canonical_loop_sort_key,
    _clean_loop,
    _contact_holes_are_simple,
    _gdstk_surface_region,
    _geometry_refs_from_gdstk_region,
    _loop_signature,
    _ring_edges,
    _same_loop_geometry,
)
from scgsim.geometry._primitives.spatial import (
    _interpolate_3d,
    _interval_complement,
    _line_key_2d,
    _same_z,
    _segment_overlap_interval,
)
from scgsim.geometry._primitives.surface_records import (
    _conductor_boundary_surface_id,
    _is_route_a_sheet_interface,
    _RouteASheetPatch,
    _surface_contribution_provenance,
)
from scgsim.geometry.compiler.validation import validate_selected_route
from scgsim.geometry.models.common import (
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    RouteLiteral,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import (
    InterfacePlanRecord,
    MMContactRecord,
    SurfacePlanRecord,
)
from scgsim.geometry.planning.domain import (
    _adjacent_auto_vacuum_sidewall_component_id,
    _auto_vacuum_sidewall_ref_solution_id,
    _conductor_face_solution_pieces,
    _conductor_sidewall_adjacent_solution_id,
    _interface_surface_kinds,
    _merge_solution_sidewall_interfaces,
    _plan_substrate_air_surfaces,
    _planar_side_solution_regions,
    _prepare_auto_vacuum_solution_regions,
    _reconcile_solution_domain_boundaries,
    _route_a_sheet_boundary_volume_ids,
    _route_a_sheet_plane_z_um,
    _sidewall_is_auto_vacuum_envelope_edge,
    _sidewall_on_solution_outer_boundary,
)
from scgsim.geometry.planning.evidence import (
    build_semantic_evidence_facade,
    conductor_solution_evidence,
    metal_metal_evidence,
    require_semantic_evidence_facade,
)
from scgsim.geometry.planning.interfaces import (
    plan_conductor_contact_patches,
    plan_mm_contact_records,
    recognize_route_interfaces,
)
from scgsim.geometry.planning.tags import (
    _entity_physical_group_id,
    _physical_group_owner_ids,
    _surface_physical_owner_ids,
)
from scgsim.geometry.planning.topology import _surface_owner_ids
from scgsim.geometry.source.intents import _route_a_sheet_interfaces
from scgsim.semantics import EvidenceResult, SemanticEvidenceFacade
from scgsim.semantics.ownership import interface_surface_owner_ids


def plan_surface_contribution_patches(
    build_input: GeometryBuildInput, *, route: RouteLiteral
) -> tuple[GeometryBuildInput, tuple[SurfacePlanRecord, ...]]:
    """Derive local physical-side evidence without a backend construction plan.

    This non-Gmsh seam reuses the polygon Boolean and Semantic Core
    classifiers, but does not plan cutter bodies, canonical mesh topology,
    port lowering, volumes, tags, or backend identifiers.
    """

    build_input = _prepare_auto_vacuum_solution_regions(build_input, route=route)
    if route == "A":
        build_input = _refresh_generated_route_a_sheet_interfaces(build_input)
    validate_selected_route(build_input, route)
    semantic_facts = build_semantic_evidence_facade(build_input, route=route)
    interfaces = recognize_route_interfaces(build_input, route=route)
    interfaces = plan_conductor_contact_patches(
        build_input, route=route, interfaces=interfaces
    )
    if route in {"A", "B"}:
        interfaces, mm_contacts = plan_mm_contact_records(
            build_input, route=route, interfaces=interfaces
        )
    else:
        mm_contacts = ()
    surfaces = plan_route_surfaces(
        build_input,
        route=route,
        interfaces=interfaces,
        surface_partitions=plan_surface_partitions(
            build_input, route=route, interfaces=interfaces
        ),
        construction_bodies=(),
        mm_contacts=mm_contacts,
        semantic_facts=semantic_facts,
    )
    surfaces = _reconcile_solution_domain_boundaries(build_input, surfaces=surfaces)
    return build_input, _merge_solution_sidewall_interfaces(
        build_input,
        surfaces=surfaces,
        semantic_facts=semantic_facts,
    )


def _refresh_generated_route_a_sheet_interfaces(
    build_input: GeometryBuildInput,
) -> GeometryBuildInput:
    """Rebuild adapter-generated Route A sheet footprints from final geometry.

    Junction partitioning can replace authored conductor polygons with PEC
    ends. Refresh only the generated sheet intents so their footprints follow
    that final geometry; caller-authored interface declarations stay intact.
    """
    intents_value = build_input.metadata.get("interface_intents_2d", {})
    if not isinstance(intents_value, Mapping):
        return build_input
    intents = dict(intents_value)
    existing = tuple(intents.get("interfaces", ()))
    generated_owner_ids = {
        str(owners[0])
        for intent in existing
        if isinstance(intent, Mapping)
        and intent.get("intent_origin") == "generated_route_a_surface_sheet"
        and isinstance((owners := intent.get("owner_semantic_ids")), Sequence)
        and not isinstance(owners, (str, bytes))
        and owners
    }
    explicit = tuple(
        intent
        for intent in existing
        if not (
            isinstance(intent, Mapping)
            and intent.get("intent_origin") == "generated_route_a_surface_sheet"
        )
    )
    generated = tuple(
        intent
        for intent in _route_a_sheet_interfaces(
            build_input.entities, build_input.polygons
        )["interfaces"]
        if intent["owner_semantic_ids"][0] in generated_owner_ids
    )
    intents["interfaces"] = (*explicit, *generated)
    return replace(
        build_input,
        metadata={**build_input.metadata, "interface_intents_2d": intents},
    )


def plan_surface_partitions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[SurfacePartitionRecord, ...]:
    """Plan parent-interface partitions before live surfaces are created.

    This function only consumes explicit `build_input.metadata["surface_partitions"]`
    records.

    Child regions are partition intent, not backend geometry. Each returned
    `SurfacePartitionRecord` must point to a child `SurfacePlanRecord` that
    will be created directly by the backend. The parent interface itself is a
    semantic aggregate and must not be cut by OCC after creation.
    """
    interface_ids = {interface.interface_id for interface in interfaces}
    raw_partitions = build_input.metadata.get("surface_partitions", ())
    if raw_partitions in (None, ()):
        return ()
    if not isinstance(raw_partitions, tuple | list):
        raise TypeError("surface_partitions metadata must be a sequence")

    records: list[SurfacePartitionRecord] = []
    for index, intent in enumerate(raw_partitions):
        if not isinstance(intent, Mapping):
            raise TypeError("surface_partitions entries must be mappings")
        valid_routes = tuple(
            str(value) for value in intent.get("valid_routes", (route,))
        )
        if route not in valid_routes:
            continue
        parent_interface_id = str(intent.get("parent_interface_id", ""))
        if parent_interface_id not in interface_ids:
            raise ValueError(
                f"surface partition references unknown interface "
                f"{parent_interface_id!r}"
            )
        label = str(intent.get("label", "")).strip()
        if not label:
            raise ValueError("surface partition label must be non-empty")
        child_surface_id = str(
            intent.get("child_surface_id") or f"SURF__{parent_interface_id}__{label}"
        )
        records.append(
            SurfacePartitionRecord(
                partition_id=str(
                    intent.get("partition_id")
                    or f"PART__{parent_interface_id}__{index:04d}"
                ),
                parent_interface_id=parent_interface_id,
                child_surface_id=child_surface_id,
                label=label,
                valid_routes=valid_routes,
                metadata=dict(intent),
            )
        )
    return tuple(records)


def _plan_route_a_sheet_patches(
    build_input: GeometryBuildInput,
    *,
    interfaces: Sequence[InterfacePlanRecord],
    semantic_facts: SemanticEvidenceFacade,
) -> tuple[_RouteASheetPatch, ...]:
    """Partition each Route-A sheet by its exact ordered local domains."""
    import gdstk

    raw_patches: list[
        tuple[
            InterfacePlanRecord,
            SemanticEntitySpec,
            str,
            str,
            dict[str, Any],
        ]
    ] = []
    for interface in interfaces:
        if not _is_route_a_sheet_interface("A", interface):
            continue
        sheet = _entity_by_id(build_input, interface.owner_semantic_ids[0])
        plane_z_um = _route_a_sheet_plane_z_um(build_input, sheet)
        parent_geometry_ref = {
            "from_interface_id": interface.interface_id,
            "source_polygon_ids": interface.source_polygon_ids,
            **_geometry_ref_from_metadata(interface.metadata),
            "plane": {"axis": "z", "value_um": plane_z_um},
            "representation": "surface_sheet",
        }
        sheet_region = _gdstk_surface_region(parent_geometry_ref)
        if not sheet_region:
            raise ValueError(
                f"{sheet.semantic_id} Route A sheet has no occupied region"
            )
        bottom_regions = _route_a_sheet_side_solution_regions(
            build_input,
            sheet=sheet,
            sheet_region=sheet_region,
            plane_z_um=plane_z_um,
            side="bottom",
        )
        top_regions = _route_a_sheet_side_solution_regions(
            build_input,
            sheet=sheet,
            sheet_region=sheet_region,
            plane_z_um=plane_z_um,
            side="top",
        )
        pair_region: tuple[Any, ...] = ()
        has_distinct_side_domains = False
        for bottom_id, bottom_region in bottom_regions:
            for top_id, top_region in top_regions:
                overlap = _boolean_gdstk_region(
                    gdstk,
                    bottom_region,
                    top_region,
                    "and",
                )
                if not overlap:
                    continue
                has_distinct_side_domains |= bottom_id != top_id
                refs = _geometry_refs_from_gdstk_region(parent_geometry_ref, overlap)
                for geometry_ref in refs:
                    raw_patches.append(
                        (interface, sheet, bottom_id, top_id, geometry_ref)
                    )
                pair_region = (
                    overlap
                    if not pair_region
                    else _boolean_gdstk_region(gdstk, pair_region, overlap, "or")
                )
        uncovered = _boolean_gdstk_region(
            gdstk,
            sheet_region,
            pair_region,
            "not",
        )
        if uncovered:
            raise ValueError(
                f"{sheet.semantic_id} local Route A patches do not cover the full sheet."
            )
        if not has_distinct_side_domains:
            raise ValueError(
                f"{sheet.semantic_id} has no positive-area physical support from "
                "distinct side domains."
            )

    ordered = sorted(
        raw_patches,
        key=lambda item: (
            item[0].interface_id,
            item[2],
            item[3],
            _loop_signature(item[4]["outer_loop"]),
        ),
    )
    counts: Counter[str] = Counter(interface.interface_id for interface, *_ in ordered)
    indexes: Counter[str] = Counter()
    records: list[_RouteASheetPatch] = []
    for interface, sheet, bottom_id, top_id, geometry_ref in ordered:
        component_index = indexes[interface.interface_id]
        indexes[interface.interface_id] += 1
        patch_suffix = (
            "" if counts[interface.interface_id] == 1 else f":{component_index:04d}"
        )
        patch_id = f"route-a-local:{interface.interface_id}{patch_suffix}"
        bottom = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:bottom:{bottom_id}",
            patch_id=f"planned:{patch_id}:bottom",
            conductor_id=sheet.semantic_id,
            solution_id=bottom_id,
            side="bottom",
        )
        top = conductor_solution_evidence(
            semantic_facts,
            contribution_id=f"{patch_id}:top:{top_id}",
            patch_id=f"planned:{patch_id}:top",
            conductor_id=sheet.semantic_id,
            solution_id=top_id,
            side="top",
        )
        records.append(
            _RouteASheetPatch(
                parent_interface_id=interface.interface_id,
                sheet_entity_id=sheet.semantic_id,
                patch_id=patch_id,
                geometry_ref=geometry_ref,
                bottom=bottom,
                top=top,
            )
        )
    return tuple(records)


def _route_a_sheet_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    sheet: SemanticEntitySpec,
    sheet_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Return nonoverlapping exact solution coverage for one sheet side."""
    return _planar_side_solution_regions(
        build_input,
        owner_id=sheet.semantic_id,
        occupied_region=sheet_region,
        plane_z_um=plane_z_um,
        side=side,
    )


def plan_route_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
    surface_partitions: tuple[SurfacePartitionRecord, ...],
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Plan route-specific surfaces without building geometry.

    Route A should plan thin sheet interfaces and PEC shell/contact surfaces. Route B
    should plan host-owned cutout shell surfaces. Route C should plan retained
    material top/bottom/sidewall/contact surfaces.

    Partitioned interfaces must already be represented as child live surfaces
    in this stage. The backend should only receive surfaces it can build
    directly from point/curve/loop metadata.

    Route A `surface_sheet` conductors are not standalone surfaces here. They
    must appear as interface-owned `MS`, `MA`, `MM`, or `SA` surfaces, so metal
    coverage replaces the bare substrate-air interface instead of overlapping it.
    """
    semantic_facts = require_semantic_evidence_facade(
        build_input,
        route=route,
        facade=semantic_facts,
    )
    partitions_by_interface: dict[str, list[SurfacePartitionRecord]] = {}
    for partition in surface_partitions:
        partitions_by_interface.setdefault(
            partition.parent_interface_id,
            [],
        ).append(partition)

    route_a_sheet_patches = (
        _plan_route_a_sheet_patches(
            build_input,
            interfaces=interfaces,
            semantic_facts=semantic_facts,
        )
        if route == "A"
        else ()
    )
    sheet_patches_by_interface: dict[str, list[_RouteASheetPatch]] = {}
    for patch in route_a_sheet_patches:
        sheet_patches_by_interface.setdefault(patch.parent_interface_id, []).append(
            patch
        )
    records: list[SurfacePlanRecord] = []
    records.extend(
        _plan_substrate_air_surfaces(
            build_input,
            route=route,
            semantic_facts=semantic_facts,
            route_a_sheet_patches=route_a_sheet_patches,
        )
    )
    contact_faces = _contact_patches_by_entity_face(interfaces)
    sheet_contacts_by_face = (
        _route_a_sheet_contacts_by_face_metal(build_input, mm_contacts)
        if route == "A"
        else {}
    )
    normalized_sheet_loops = _sheet_contact_loops_by_face_metal(
        build_input,
        mm_contacts,
        include_direct_bump_contacts=False,
    )
    for interface in interfaces:
        # Same-net conductor contacts are component provenance only.  Neither
        # Route A nor Route B may lower an internal MM face as solver geometry.
        if _is_hidden_contact_interface(route, interface):
            continue
        if _is_route_a_sheet_interface(route, interface):
            patches = tuple(sheet_patches_by_interface.get(interface.interface_id, ()))
            if not patches:
                raise ValueError(
                    f"{interface.interface_id} has no local Route A sheet patches"
                )
            records.extend(
                _route_a_sheet_patch_surfaces(
                    build_input,
                    interface=interface,
                    patches=patches,
                    sheet_contacts=sheet_contacts_by_face.get(
                        patches[0].sheet_entity_id, ()
                    ),
                    semantic_facts=semantic_facts,
                )
            )
            continue
        geometry_ref = {
            "from_interface_id": interface.interface_id,
            "source_polygon_ids": interface.source_polygon_ids,
            **_geometry_ref_from_metadata(interface.metadata),
        }
        route_a_sheet_evidence: tuple[EvidenceResult, ...] = ()
        primary_kind = next(
            (
                result.classification
                for result in route_a_sheet_evidence
                if interface.owner_semantic_ids[1]
                in (result.effective_domain_ids or ())
            ),
            None,
        )
        surface_interface_id = _surface_interface_id(
            build_input,
            route=route,
            interface=interface,
            primary_kind=primary_kind,
        )
        interface_kinds = _interface_surface_kinds(
            build_input,
            route=route,
            interface=interface,
            route_a_evidence=route_a_sheet_evidence,
        )
        owner_semantic_ids = _interface_surface_owner_ids(
            build_input,
            route=route,
            interface=interface,
            route_a_evidence=route_a_sheet_evidence,
        )
        boundary_volume_ids = _interface_boundary_volume_ids(
            build_input,
            route=route,
            interface=interface,
            route_a_evidence=route_a_sheet_evidence,
        )
        physical_owner_semantic_ids = _physical_group_owner_ids(
            build_input,
            owner_semantic_ids,
        )
        embedded_sheet = _is_route_a_sheet_interface(route, interface)
        partitions = partitions_by_interface.get(interface.interface_id, ())
        if partitions:
            parent_surface_id = f"SURF__{surface_interface_id}"
            for partition in partitions:
                child_geometry_ref = {
                    **geometry_ref,
                    "partition_id": partition.partition_id,
                    "parent_interface_id": partition.parent_interface_id,
                    **_geometry_ref_from_metadata(partition.metadata),
                }
                records.append(
                    SurfacePlanRecord(
                        surface_id=partition.child_surface_id,
                        owner_semantic_id=interface.owner_semantic_ids[0],
                        surface_role=f"{route}_planned_interface_partition",
                        geometry_ref=child_geometry_ref,
                        interface_id=surface_interface_id,
                        parent_surface_id=parent_surface_id,
                        partition_label=partition.label,
                        solver_use=interface.solver_use or "solver_active",
                        valid_routes=(route,),
                        metadata={
                            "interface_kinds": interface_kinds,
                            "owner_semantic_ids": owner_semantic_ids,
                            "physical_owner_semantic_ids": physical_owner_semantic_ids,
                            "boundary_volume_ids": boundary_volume_ids,
                            "embedded_surface_sheet": embedded_sheet,
                        },
                    )
                )
            continue
        records.append(
            SurfacePlanRecord(
                surface_id=f"SURF__{surface_interface_id}",
                owner_semantic_id=interface.owner_semantic_ids[0],
                surface_role=f"{route}_planned_interface",
                geometry_ref=geometry_ref,
                interface_id=surface_interface_id,
                solver_use=interface.solver_use or "solver_active",
                valid_routes=(route,),
                metadata={
                    "interface_kinds": interface_kinds,
                    "owner_semantic_ids": owner_semantic_ids,
                    "physical_owner_semantic_ids": physical_owner_semantic_ids,
                    "boundary_volume_ids": boundary_volume_ids,
                    "embedded_surface_sheet": embedded_sheet,
                },
            )
        )

    construction_body_by_surface_id = {
        surface_id: body
        for body in construction_bodies
        for surface_id in body.expected_surface_ids
    }
    normalized_pad_loops = _sheet_contact_loops_by_pad(build_input, mm_contacts)
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            continue
        representation = entity.route_representations.get(route)
        if representation in {"cutout_boundary_shell", "material_volume"}:
            for shell_part in ("top", "bottom"):
                base_surface_id = _conductor_boundary_surface_id(
                    route,
                    entity,
                    representation,
                    shell_part,
                )
                base_geometry_ref = {
                    **_route_entity_geometry_ref(
                        build_input,
                        route,
                        entity,
                        representation=representation,
                        interfaces=interfaces,
                    ),
                    "shell_part": shell_part,
                }
                face_geometry_refs = _subtract_contact_patches_from_face(
                    base_geometry_ref,
                    (
                        *contact_faces.get((entity.semantic_id, shell_part), ()),
                        *(
                            normalized_sheet_loops.get(entity.semantic_id, ())
                            if route == "B" and entity.part_role == "face_metal"
                            else normalized_pad_loops.get(entity.semantic_id, ())
                            if route == "A" and shell_part == "bottom"
                            else ()
                        ),
                    ),
                )
                if not face_geometry_refs:
                    continue
                face_pieces = _conductor_face_solution_pieces(
                    build_input,
                    route=route,
                    entity=entity,
                    shell_part=shell_part,
                    geometry_refs=face_geometry_refs,
                )
                for face_index, (adjacent_id, face_geometry_ref) in enumerate(
                    face_pieces
                ):
                    contribution_id = (
                        f"conductor-solution:{entity.semantic_id}:"
                        f"{adjacent_id}:{shell_part}:{face_index:04d}"
                    )
                    interface_evidence = conductor_solution_evidence(
                        semantic_facts,
                        contribution_id=contribution_id,
                        patch_id=f"planned:{contribution_id}",
                        conductor_id=entity.semantic_id,
                        solution_id=adjacent_id,
                        side=shell_part,
                    )
                    interface_kind = interface_evidence.classification
                    surface_id = (
                        base_surface_id
                        if len(face_pieces) == 1
                        else f"{base_surface_id}__P{face_index:04d}"
                    )
                    body = construction_body_by_surface_id.get(surface_id)
                    if body is None:
                        body = construction_body_by_surface_id.get(base_surface_id)
                    face_owner_ids = interface_evidence.source_owner_ids
                    records.append(
                        SurfacePlanRecord(
                            surface_id=surface_id,
                            owner_semantic_id=entity.semantic_id,
                            surface_role=(
                                "cutout_boundary_shell"
                                if representation == "cutout_boundary_shell"
                                else "material_interface"
                            ),
                            geometry_ref={
                                **face_geometry_ref,
                                "construction_body_id": (
                                    body.construction_body_id
                                    if body is not None
                                    else None
                                ),
                            },
                            interface_id=_conductor_face_interface_id(
                                interface_kind,
                                entity.semantic_id,
                                adjacent_id,
                                shell_part,
                                None if len(face_pieces) == 1 else face_index,
                            ),
                            valid_routes=(route,),
                            solver_use="solver_active",
                            metadata={
                                "interface_kinds": (interface_kind,),
                                "owner_semantic_ids": face_owner_ids,
                                "physical_owner_semantic_ids": (
                                    _physical_group_owner_ids(
                                        build_input,
                                        face_owner_ids,
                                    )
                                ),
                                "boundary_volume_ids": (
                                    _route_conductor_boundary_volume_ids(
                                        build_input,
                                        route=route,
                                        entity=entity,
                                        face=shell_part,
                                        adjacent_solution_id=adjacent_id,
                                    )
                                ),
                                "exposed_surface_role": shell_part,
                                "source_provenance": (
                                    _surface_contribution_provenance(
                                        parent_interface_id=(
                                            f"conductor-face:{entity.semantic_id}:"
                                            f"{shell_part}"
                                        ),
                                        patch_id=contribution_id,
                                        contributions=(interface_evidence,),
                                    )
                                ),
                            },
                        )
                    )
            sidewall_adjacent_id = (
                _conductor_sidewall_adjacent_solution_id(build_input, entity)
                if route == "C"
                else None
            )
            sidewall_geometry_refs = (
                ()
                if route == "C" and sidewall_adjacent_id is None
                else _conductor_sidewall_geometry_refs(
                    build_input,
                    route=route,
                    entity=entity,
                    representation=representation,
                    adjacent_solution_id=sidewall_adjacent_id,
                    interfaces=interfaces,
                    excluded_footprints=(
                        normalized_sheet_loops.get(entity.semantic_id, ())
                        if route == "B" and entity.part_role == "face_metal"
                        else ()
                    ),
                )
            )
            for edge_index, geometry_ref in enumerate(sidewall_geometry_refs):
                shell_part = f"sidewall_{edge_index:04d}"
                surface_id = _conductor_boundary_surface_id(
                    route,
                    entity,
                    representation,
                    shell_part,
                )
                body = construction_body_by_surface_id.get(surface_id)
                sidewall_adjacent_owner_id = str(
                    geometry_ref.get(
                        "adjacent_conductor_semantic_id",
                        geometry_ref.get("adjacent_solution_id", sidewall_adjacent_id),
                    )
                )
                if (
                    not sidewall_adjacent_owner_id
                    or sidewall_adjacent_owner_id == "None"
                ):
                    raise ValueError(
                        f"{entity.semantic_id} sidewall lacks exact adjacent solution provenance."
                    )
                contribution_id = (
                    f"sidewall:{entity.semantic_id}:"
                    f"{sidewall_adjacent_owner_id}:{shell_part}"
                )
                if "adjacent_conductor_semantic_id" in geometry_ref:
                    sidewall_evidence = metal_metal_evidence(
                        semantic_facts,
                        contribution_id=contribution_id,
                        patch_id=f"planned:{contribution_id}",
                        lower_id=entity.semantic_id,
                        upper_id=sidewall_adjacent_owner_id,
                        side="sidewall",
                    )
                else:
                    sidewall_evidence = conductor_solution_evidence(
                        semantic_facts,
                        contribution_id=contribution_id,
                        patch_id=f"planned:{contribution_id}",
                        conductor_id=entity.semantic_id,
                        solution_id=sidewall_adjacent_owner_id,
                        side="sidewall",
                    )
                sidewall_interface_kind = sidewall_evidence.classification
                sidewall_interface_id = (
                    None
                    if geometry_ref.get("solution_exterior_boundary")
                    else (
                        f"{sidewall_interface_kind}__{entity.semantic_id}__"
                        f"{sidewall_adjacent_owner_id}__"
                        f"{shell_part.upper()}"
                    )
                )
                sidewall_boundary_volume_ids = (
                    (entity.semantic_id,)
                    if geometry_ref.get("solution_exterior_boundary")
                    else _conductor_boundary_volume_ids(
                        route,
                        entity,
                        sidewall_adjacent_owner_id,
                    )
                )
                sidewall_owner_ids = (
                    (entity.semantic_id,)
                    if geometry_ref.get("solution_exterior_boundary")
                    else sidewall_evidence.source_owner_ids
                )
                sidewall_physical_owner_ids = _physical_group_owner_ids(
                    build_input,
                    sidewall_owner_ids,
                )
                sidewall_physical_name = (
                    f"MA__{'__'.join(sidewall_physical_owner_ids)}__SIDEWALL"
                    if entity.part_role == "bump_body"
                    and sidewall_interface_kind == "MA"
                    else None
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=surface_id,
                        owner_semantic_id=entity.semantic_id,
                        surface_role=(
                            "cutout_boundary_shell"
                            if representation == "cutout_boundary_shell"
                            else "material_interface"
                        ),
                        geometry_ref={
                            **geometry_ref,
                            "from_semantic_id": entity.semantic_id,
                            "geometry_kind": entity.geometry_kind,
                            "part_role": entity.part_role,
                            "representation": representation,
                            "source_polygon_ids": entity.polygon_ids,
                            "shell_part": shell_part,
                            "construction_body_id": (
                                body.construction_body_id if body is not None else None
                            ),
                        },
                        interface_id=sidewall_interface_id,
                        valid_routes=(route,),
                        solver_use="solver_active",
                        metadata={
                            "physical_name": sidewall_physical_name,
                            "interface_kinds": (
                                ()
                                if sidewall_interface_id is None
                                else (sidewall_interface_kind,)
                            ),
                            "owner_semantic_ids": sidewall_owner_ids,
                            "physical_owner_semantic_ids": (
                                sidewall_physical_owner_ids
                            ),
                            "boundary_volume_ids": sidewall_boundary_volume_ids,
                            "exposed_surface_role": shell_part,
                            **(
                                {
                                    "source_provenance": (
                                        _surface_contribution_provenance(
                                            parent_interface_id=(
                                                f"conductor-sidewall:"
                                                f"{entity.semantic_id}"
                                            ),
                                            patch_id=contribution_id,
                                            contributions=(sidewall_evidence,),
                                        )
                                    )
                                }
                                if sidewall_interface_id is not None
                                else {}
                            ),
                        },
                    )
                )
    if route == "C":
        return tuple(records)
    return _with_surface_contract_metadata(
        build_input,
        route=route,
        interfaces=interfaces,
        mm_contacts=mm_contacts,
        surfaces=tuple(records),
    )


def _route_a_sheet_patch_surfaces(
    build_input: GeometryBuildInput,
    *,
    interface: InterfacePlanRecord,
    patches: Sequence[_RouteASheetPatch],
    sheet_contacts: Sequence[MMContactRecord],
    semantic_facts: SemanticEvidenceFacade,
) -> tuple[SurfacePlanRecord, ...]:
    """Lower local Route-A adjacency without assigning one pair to a whole sheet."""
    import gdstk

    sheet = _entity_by_id(build_input, interface.owner_semantic_ids[0])
    contact_loops_by_patch: dict[str, list[tuple[tuple[float, float], ...]]] = {
        patch.patch_id: [] for patch in patches
    }
    records: list[SurfacePlanRecord] = []
    for contact_index, contact in enumerate(sheet_contacts):
        contact_region = (gdstk.Polygon(_clean_loop(contact.outer_loop)),)
        matches: list[_RouteASheetPatch] = []
        for patch in patches:
            overlap = _boolean_gdstk_region(
                gdstk,
                contact_region,
                _gdstk_surface_region(patch.geometry_ref),
                "and",
            )
            if overlap:
                matches.append(patch)
        if len(matches) != 1:
            raise ValueError(
                f"{contact.contact_id} crosses {len(matches)} local Route A "
                "sheet patches; one contact requires one ordered domain pair."
            )
        patch = matches[0]
        uncovered = _boolean_gdstk_region(
            gdstk,
            contact_region,
            _gdstk_surface_region(patch.geometry_ref),
            "not",
        )
        if uncovered:
            raise ValueError(
                f"{contact.contact_id} is not contained by its local Route A patch."
            )
        contact_loops_by_patch[patch.patch_id].append(_clean_loop(contact.outer_loop))
        cap_face = _route_a_sheet_contact_cap_face(
            build_input,
            sheet_entity=sheet,
            contact=contact,
        )
        parent_evidence = patch.bottom if cap_face == "bottom" else patch.top
        cap_evidence = _route_a_sheet_child_evidence(
            semantic_facts,
            patch=patch,
            parent=parent_evidence,
            child_label=f"contact-cap:{contact.contact_id}",
        )
        cap_solution_id = _one_effective_domain(cap_evidence)
        cap_kind = cap_evidence.classification
        cap_owner_ids = cap_evidence.source_owner_ids
        cap_interface_id = (
            f"{cap_kind}__{sheet.semantic_id}__SHEET_CONTACT_CAP__{contact_index:04d}"
        )
        cap_geometry_ref = {
            **dict(patch.geometry_ref),
            "outer_loop": _clean_loop(contact.outer_loop),
            "hole_loops": (),
            "source_polygon_ids": _unique_ids(
                (
                    *contact.lower_source_fragment_ids,
                    *contact.upper_source_fragment_ids,
                )
            ),
        }
        records.append(
            SurfacePlanRecord(
                surface_id=f"SURF__{cap_interface_id}",
                owner_semantic_id=sheet.semantic_id,
                surface_role="route_a_sheet_contact_cap",
                geometry_ref=cap_geometry_ref,
                interface_id=cap_interface_id,
                valid_routes=("A",),
                metadata={
                    "physical_name": (
                        f"{cap_kind}__{_entity_physical_group_id(sheet)}"
                        "__SHEET_CONTACT_CAP"
                    ),
                    "interface_kinds": (cap_kind,),
                    "owner_semantic_ids": cap_owner_ids,
                    "physical_owner_semantic_ids": _physical_group_owner_ids(
                        build_input, cap_owner_ids
                    ),
                    "boundary_volume_ids": (cap_solution_id,),
                    "embedded_surface_sheet": True,
                    "exposed_surface_role": "sheet_contact_cap",
                    "sheet_contact_cap": True,
                    "source_contact_id": contact.contact_id,
                    "source_contact_owner_semantic_ids": (
                        contact.lower_entity_id,
                        contact.upper_entity_id,
                    ),
                    "source_provenance": _surface_contribution_provenance(
                        parent_interface_id=interface.interface_id,
                        patch_id=patch.patch_id,
                        contributions=(cap_evidence,),
                    ),
                },
            )
        )

    candidates: list[
        tuple[_RouteASheetPatch, dict[str, Any], tuple[EvidenceResult, EvidenceResult]]
    ] = []
    for patch in patches:
        geometry_refs = _subtract_contact_patches_from_face(
            patch.geometry_ref,
            contact_loops_by_patch[patch.patch_id],
        )
        for child_index, geometry_ref in enumerate(geometry_refs):
            child_label = f"live:{child_index:04d}"
            bottom = _route_a_sheet_child_evidence(
                semantic_facts,
                patch=patch,
                parent=patch.bottom,
                child_label=child_label,
            )
            top = _route_a_sheet_child_evidence(
                semantic_facts,
                patch=patch,
                parent=patch.top,
                child_label=child_label,
            )
            candidates.append((patch, geometry_ref, (bottom, top)))

    interface_ids = [
        _route_a_sheet_patch_interface_id(interface, patch, contributions)
        for patch, _, contributions in candidates
    ]
    interface_counts = Counter(interface_ids)
    interface_indexes: Counter[str] = Counter()
    for (patch, geometry_ref, contributions), surface_interface_id in zip(
        candidates, interface_ids, strict=True
    ):
        child_index = interface_indexes[surface_interface_id]
        interface_indexes[surface_interface_id] += 1
        surface_id = f"SURF__{surface_interface_id}"
        parent_surface_id = None
        partition_label = None
        if interface_counts[surface_interface_id] > 1:
            parent_surface_id = surface_id
            partition_label = f"LOCAL_{child_index:04d}"
            surface_id = f"{surface_id}__{partition_label}"
        interface_kinds = _route_a_sheet_patch_interface_kinds(
            interface,
            contributions,
        )
        owner_ids = _unique_ids(
            owner_id for result in contributions for owner_id in result.source_owner_ids
        )
        records.append(
            SurfacePlanRecord(
                surface_id=surface_id,
                owner_semantic_id=sheet.semantic_id,
                surface_role="A_planned_interface",
                geometry_ref=dict(geometry_ref),
                interface_id=surface_interface_id,
                parent_surface_id=parent_surface_id,
                partition_label=partition_label,
                solver_use=interface.solver_use or "solver_active",
                valid_routes=("A",),
                metadata={
                    "interface_kinds": interface_kinds,
                    "owner_semantic_ids": owner_ids,
                    "physical_owner_semantic_ids": _physical_group_owner_ids(
                        build_input, owner_ids
                    ),
                    "boundary_volume_ids": patch.boundary_volume_ids,
                    "embedded_surface_sheet": True,
                    "source_provenance": _surface_contribution_provenance(
                        parent_interface_id=interface.interface_id,
                        patch_id=patch.patch_id,
                        contributions=contributions,
                    ),
                },
            )
        )
    return tuple(records)


def _route_a_sheet_child_evidence(
    semantic_facts: SemanticEvidenceFacade,
    *,
    patch: _RouteASheetPatch,
    parent: EvidenceResult,
    child_label: str,
) -> EvidenceResult:
    solution_id = _one_effective_domain(parent)
    return conductor_solution_evidence(
        semantic_facts,
        contribution_id=(f"{patch.patch_id}:{child_label}:{parent.side}:{solution_id}"),
        patch_id=f"planned:{patch.patch_id}:{child_label}:{parent.side}",
        conductor_id=patch.sheet_entity_id,
        solution_id=solution_id,
        side=parent.side or "unknown",
    )


def _one_effective_domain(result: EvidenceResult) -> str:
    if result.effective_domain_ids is None or len(result.effective_domain_ids) != 1:
        raise ValueError(
            f"{result.contribution_id} requires one effective solution domain."
        )
    return result.effective_domain_ids[0]


def _route_a_sheet_patch_interface_id(
    interface: InterfacePlanRecord,
    patch: _RouteASheetPatch,
    contributions: tuple[EvidenceResult, EvidenceResult],
) -> str:
    preferred_domain = interface.owner_semantic_ids[1]
    primary_kind = next(
        (
            result.classification
            for result in contributions
            if preferred_domain in (result.effective_domain_ids or ())
        ),
        contributions[0].classification,
    )
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return (
        f"{primary_kind}__{patch.sheet_entity_id}__"
        f"{'__'.join(patch.boundary_volume_ids)}__{suffix}"
    )


def _route_a_sheet_patch_interface_kinds(
    interface: InterfacePlanRecord,
    contributions: Sequence[EvidenceResult],
) -> tuple[str, ...]:
    derived = {result.classification for result in contributions}
    raw = interface.metadata.get("interface_kinds")
    if (
        interface.metadata.get("intent_origin") != "generated_route_a_surface_sheet"
        and raw is not None
    ):
        declared = {str(raw)} if isinstance(raw, str) else {str(value) for value in raw}
        if declared != derived:
            raise ValueError(
                f"{interface.interface_id} explicit interface kinds "
                f"{sorted(declared)!r} contradict local evidence {sorted(derived)!r}."
            )
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in derived)


def _with_surface_contract_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Attach the explicit fields required by final surface export."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    interface_by_id = {interface.interface_id: interface for interface in interfaces}
    # This is a projection of the completed MMContactRecord ledger, including
    # same-z contact-pad normalizations.  Do not infer a second union graph
    # from exposed surface adjacency.
    component_ledger = _component_metadata(mm_contacts)
    component_by_entity = {
        entity_id: str(component["conductor_component_id"])
        for component in component_ledger
        for entity_id in component["members"]
    }
    component_net = {
        str(component["conductor_component_id"]): component["net_id"]
        for component in component_ledger
    }
    component_equipotential = {
        str(component["conductor_component_id"]): component["equipotential_id"]
        for component in component_ledger
    }

    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        owner_ids = _surface_owner_ids(surface)
        structured_owner_ids = _surface_physical_owner_ids(surface)
        conductors = tuple(
            entities[owner_id]
            for owner_id in owner_ids
            if owner_id in entities and not _is_solution_entity(entities[owner_id])
        )
        net_ids = {entity.net_id for entity in conductors if entity.net_id}
        if len(net_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous conductor net")
        component_ids = {
            component_by_entity[entity.semantic_id]
            for entity in conductors
            if entity.semantic_id in component_by_entity
        }
        if conductors and not component_ids:
            component_ids = {f"COMP__{conductors[0].semantic_id}"}
        if len(component_ids) > 1:
            raise ValueError(
                f"{surface.surface_id} spans independent conductor components"
            )
        equipotential_ids = {
            str(entity.metadata["equipotential_id"])
            for entity in conductors
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(equipotential_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous equipotential id")
        interface = interface_by_id.get(surface.interface_id or "")
        raw_interface_kinds = surface.metadata.get("interface_kinds", ())
        interface_kinds = (
            (str(raw_interface_kinds),)
            if isinstance(raw_interface_kinds, str)
            else tuple(str(kind) for kind in raw_interface_kinds)
        )
        face_kind = str(
            surface.metadata.get(
                "exposed_surface_role",
                surface.metadata.get("boundary_role", "interface"),
            )
        )
        if face_kind.startswith("sidewall_"):
            face_kind = "sidewall"
        source_layer_names = {
            str(entity.metadata["source_layer_name"])
            for entity in conductors
            if entity.metadata.get("source_layer_name") is not None
        }
        if len(source_layer_names) > 1:
            raise ValueError(
                f"{surface.surface_id} has ambiguous conductor source_layer_name"
            )
        existing_source_provenance = surface.metadata.get("source_provenance", {})
        if not isinstance(existing_source_provenance, Mapping):
            raise TypeError(f"{surface.surface_id} source_provenance must be a mapping")
        result.append(
            replace(
                surface,
                metadata={
                    **dict(surface.metadata),
                    "route": route,
                    "representation": str(
                        surface.geometry_ref.get(
                            "representation",
                            "surface_sheet" if conductors else "solution_surface",
                        )
                    ),
                    "interface_type": "_".join(interface_kinds)
                    if interface_kinds
                    else "boundary",
                    "contact_kind": (
                        str(interface.metadata.get("contact_kind", "MM"))
                        if interface is not None
                        and interface.recognition_rule
                        == "coplanar_conductor_contact_patch"
                        else None
                    ),
                    "face_kind": face_kind,
                    # Physical groups can intentionally aggregate split
                    # entities under a stable semantic group id.  Keep the
                    # exact split owners in source provenance, while the
                    # solver-live owner field names the actual group owner.
                    "owner_semantic_ids": (
                        owner_ids
                        if any(
                            boundary_id not in structured_owner_ids
                            for boundary_id in surface.metadata.get(
                                "boundary_volume_ids", ()
                            )
                        )
                        else structured_owner_ids
                    ),
                    "net_id": (
                        component_net[next(iter(component_ids))]
                        if component_ids and next(iter(component_ids)) in component_net
                        else next(iter(net_ids), None)
                    ),
                    "conductor_component_id": next(iter(component_ids), None),
                    "equipotential_id": (
                        component_equipotential[next(iter(component_ids))]
                        if component_ids
                        and next(iter(component_ids)) in component_equipotential
                        else next(iter(equipotential_ids), None)
                    ),
                    "source_provenance": {
                        **dict(existing_source_provenance),
                        "source_polygon_ids": _normalized_source_polygon_ids(
                            build_input,
                            owner_ids,
                            surface.geometry_ref,
                            mm_contacts=mm_contacts,
                        ),
                        "interface_id": surface.interface_id,
                        "route": route,
                        "conductor_source_layer_name": next(
                            iter(source_layer_names), None
                        ),
                    },
                },
            )
        )
    return tuple(result)


def _normalized_source_polygon_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
    geometry_ref: Mapping[str, Any],
    *,
    mm_contacts: Sequence[MMContactRecord],
) -> tuple[str, ...]:
    """Keep only spatially participating M1 contact provenance."""
    ids = list(geometry_ref.get("source_polygon_ids", ()))
    owner_set = set(owner_ids)
    component_id = geometry_ref.get("conductor_component_id")
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    for record in mm_contacts:
        if not _is_face_pad_mm_contact(record, entities):
            continue
        if component_id is not None and record.conductor_component_id != component_id:
            continue
        if not owner_set.intersection((record.lower_entity_id, record.upper_entity_id)):
            continue
        ids.extend(record.lower_source_fragment_ids)
        ids.extend(record.upper_source_fragment_ids)
    return _unique_ids(ids)


def _contact_patches_by_entity_face(
    interfaces: tuple[InterfacePlanRecord, ...],
) -> dict[tuple[str, str], tuple[tuple[tuple[float, float], ...], ...]]:
    records: dict[tuple[str, str], list[tuple[tuple[float, float], ...]]] = {}
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        loop = _clean_loop(interface.metadata["outer_loop"])
        lower_id = str(interface.metadata["lower_entity_id"])
        upper_id = str(interface.metadata["upper_entity_id"])
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.setdefault((lower_id, lower_face), []).append(loop)
        records.setdefault((upper_id, upper_face), []).append(loop)
    return {key: tuple(value) for key, value in records.items()}


def _sheet_contact_loops_by_pad(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite MM ledger loops for the contact-pad side of an M1 join."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        if not _is_face_pad_mm_contact(record, entities):
            continue
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        pad = lower if lower.part_role == "contact_pad" else upper
        result.setdefault(pad.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def _is_face_pad_mm_contact(
    record: MMContactRecord,
    entities: Mapping[str, SemanticEntitySpec],
) -> bool:
    return {
        entities[record.lower_entity_id].part_role,
        entities[record.upper_entity_id].part_role,
    } == {"contact_pad", "face_metal"}


def _sheet_contact_loops_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
    *,
    include_direct_bump_contacts: bool,
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite M1 contact loops that require Route-A sheet cap topology."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if "face_metal" not in roles or (
            roles != {"face_metal", "contact_pad"}
            and not (
                include_direct_bump_contacts and roles == {"face_metal", "bump_body"}
            )
        ):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def _route_a_sheet_contacts_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[MMContactRecord, ...]]:
    """Project typed finite MM records into Route-A sheet cap authority."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[MMContactRecord]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if roles not in ({"face_metal", "contact_pad"}, {"face_metal", "bump_body"}):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record)
    return {
        entity_id: tuple(
            sorted(
                contacts,
                key=lambda contact: (
                    _canonical_loop_sort_key(_clean_loop(contact.outer_loop)),
                    contact.contact_id,
                ),
            )
        )
        for entity_id, contacts in result.items()
    }


def _route_a_sheet_contact_cap_face(
    build_input: GeometryBuildInput,
    *,
    sheet_entity: SemanticEntitySpec,
    contact: MMContactRecord,
) -> str:
    """Select the sheet side opposite its typed finite MM contact direction."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    lower = entities.get(contact.lower_entity_id)
    upper = entities.get(contact.upper_entity_id)
    if (
        lower is None
        or upper is None
        or sheet_entity.semantic_id
        not in {
            contact.lower_entity_id,
            contact.upper_entity_id,
        }
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact members")
    normal = tuple(float(value) for value in contact.normal)
    if len(normal) != 3 or not all(isfinite(value) for value in normal):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    roles = {lower.part_role, upper.part_role}
    if roles == {"face_metal", "contact_pad"}:
        # A typed contact_pad is the same-z under-sheet normalization authority.
        return "bottom"
    if roles != {"face_metal", "bump_body"}:
        raise ValueError(f"{contact.contact_id} has incompatible sheet contact roles")
    if (
        abs(normal[0]) > _TOPOLOGY_EPS_UM
        or abs(normal[1]) > _TOPOLOGY_EPS_UM
        or abs(normal[2]) <= _TOPOLOGY_EPS_UM
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    if sheet_entity.semantic_id == lower.semantic_id:
        contact_face = "top" if normal[2] > 0 else "bottom"
        source_face_id = contact.lower_source_face_id
        expected_source_face_id = f"{lower.semantic_id}__{contact_face}"
    else:
        contact_face = "bottom" if normal[2] > 0 else "top"
        source_face_id = contact.upper_source_face_id
        expected_source_face_id = f"{upper.semantic_id}__{contact_face}"
    expected_lower_face_id = (
        f"{lower.semantic_id}__{'top' if normal[2] > 0 else 'bottom'}"
    )
    expected_upper_face_id = (
        f"{upper.semantic_id}__{'bottom' if normal[2] > 0 else 'top'}"
    )
    if (
        source_face_id != expected_source_face_id
        or contact.lower_source_face_id != expected_lower_face_id
        or contact.upper_source_face_id != expected_upper_face_id
    ):
        raise ValueError(f"{contact.contact_id} has inconsistent sheet contact face")
    return "bottom" if contact_face == "top" else "top"


def _subtract_contact_patches_from_face(
    geometry_ref: Mapping[str, Any],
    contact_loops: Sequence[tuple[tuple[float, float], ...]],
) -> tuple[dict[str, Any], ...]:
    if not contact_loops:
        return (dict(geometry_ref),)
    outer_loop = _clean_loop(geometry_ref["outer_loop"])
    existing_holes = tuple(
        _clean_loop(hole_loop) for hole_loop in geometry_ref.get("hole_loops", ())
    )
    remaining_holes: list[tuple[tuple[float, float], ...]] = list(existing_holes)
    for contact_loop in contact_loops:
        if _same_loop_geometry(contact_loop, outer_loop):
            return ()
        remaining_holes.append(_clean_loop(contact_loop))
    simple_ref = {
        **dict(geometry_ref),
        "hole_loops": tuple(remaining_holes),
        "contact_hole_loops": tuple(_clean_loop(loop) for loop in contact_loops),
    }
    if _contact_holes_are_simple(outer_loop, remaining_holes):
        return (simple_ref,)

    import gdstk

    live_region = _boolean_gdstk_region(
        gdstk,
        _gdstk_surface_region(geometry_ref),
        tuple(gdstk.Polygon(loop) for loop in contact_loops),
        "not",
    )
    return _geometry_refs_from_gdstk_region(geometry_ref, live_region)


def _interface_surface_owner_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
) -> tuple[str, ...]:
    raw_owner_ids = interface.metadata.get("surface_owner_semantic_ids")
    route_a_owner_ids = None
    if _is_route_a_sheet_interface(route, interface):
        route_a_owner_ids = _unique_ids(
            owner_id
            for result in route_a_evidence
            for owner_id in (
                *result.source_owner_ids,
                *(result.effective_domain_ids or ()),
            )
        )
        if not route_a_owner_ids:
            entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
            route_a_owner_ids = (
                entity.semantic_id,
                *_route_a_sheet_boundary_volume_ids(build_input, entity),
            )
    return interface_surface_owner_ids(
        raw_owner_ids,
        route_a_sheet_owner_ids=route_a_owner_ids,
        fallback_owner_ids=interface.owner_semantic_ids,
    )


def _interface_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
) -> tuple[str, ...]:
    raw_boundary_ids = interface.metadata.get("boundary_volume_ids")
    if raw_boundary_ids is not None:
        if isinstance(raw_boundary_ids, str):
            return (raw_boundary_ids,)
        return _unique_ids(raw_boundary_ids)
    if _is_route_a_sheet_interface(route, interface):
        if route_a_evidence:
            return _unique_ids(
                domain_id
                for result in route_a_evidence
                for domain_id in (result.effective_domain_ids or ())
            )
        return _route_a_sheet_boundary_volume_ids(
            build_input,
            _entity_by_id(build_input, interface.owner_semantic_ids[0]),
        )
    return _unique_ids(interface.owner_semantic_ids)


def _surface_interface_id(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
    primary_kind: str | None = None,
) -> str:
    if not _is_route_a_sheet_interface(route, interface):
        return interface.interface_id
    entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
    boundary_ids = _route_a_sheet_boundary_volume_ids(build_input, entity)
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return (
        f"{primary_kind or interface.kind}__{entity.semantic_id}__"
        f"{'__'.join(boundary_ids)}__{suffix}"
    )


def _is_hidden_contact_interface(
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> bool:
    return (
        interface.recognition_rule == "coplanar_conductor_contact_patch"
        and route in {"A", "B"}
        and bool(interface.metadata.get("hidden_solver_contact"))
    )


def _conductor_face_interface_id(
    interface_kind: str,
    semantic_id: str,
    adjacent_id: str,
    shell_part: str,
    face_index: int | None,
) -> str:
    suffix = "" if face_index is None else f"__P{face_index:04d}"
    return (
        f"{interface_kind}__{semantic_id}__{adjacent_id}__{shell_part.upper()}{suffix}"
    )


def _conductor_boundary_volume_ids(
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    if route == "C":
        return (entity.semantic_id, adjacent_solution_id)
    return (adjacent_solution_id,)


def _route_conductor_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    face: str,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    """Keep each finite conductor face with its directly adjacent solution.

    Route-A boundary-attached components join an exterior shell through their
    exposed side/top faces; their retained bottom MS face belongs only to the
    substrate-side solution and is never duplicated into air.
    """
    del build_input, face
    return _conductor_boundary_volume_ids(route, entity, adjacent_solution_id)


def _conductor_sidewall_geometry_refs(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    representation: str,
    adjacent_solution_id: str | None,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
    excluded_footprints: Sequence[tuple[tuple[float, float], ...]] = (),
) -> tuple[dict[str, Any], ...]:
    base_ref = _route_entity_geometry_ref(
        build_input,
        route,
        entity,
        representation=representation,
        interfaces=interfaces,
    )
    face_refs = _subtract_contact_patches_from_face(base_ref, excluded_footprints)
    refs = tuple(
        side_ref
        for face_ref in face_refs
        for side_ref in _sidewall_geometry_refs(face_ref)
    )
    edge_index = _conductor_boundary_edge_index(
        build_input,
        route=route,
        entity=entity,
    )
    if route == "C":
        if adjacent_solution_id is None:
            raise ValueError(
                f"{entity.semantic_id} Route C sidewall requires a solution."
            )
        return _route_c_conductor_sidewall_geometry_refs(
            entity=entity,
            adjacent_solution=_entity_by_id(build_input, adjacent_solution_id),
            refs=refs,
            edge_index=edge_index,
        )
    exposed_refs: list[dict[str, Any]] = []
    for raw_geometry_ref in refs:
        if _sidewall_is_auto_vacuum_envelope_edge(
            build_input,
            entity=entity,
            geometry_ref=raw_geometry_ref,
        ):
            continue
        # Exact same-z conductor contacts remove their shared MM segment
        # before solution adjacency is selected.  One raw sidewall can span
        # contact and vacuum portions; only each retained topology segment is
        # eligible for a unique auto-vacuum child.
        for geometry_ref in _trim_sidewall_ref_against_route_conductors(
            geometry_ref=raw_geometry_ref,
            edge_index=edge_index,
        ):
            auto_parent, ref_adjacent_solution_id = (
                _auto_vacuum_sidewall_ref_solution_id(
                    build_input,
                    entity=entity,
                    geometry_ref=geometry_ref,
                )
            )
            if auto_parent:
                if ref_adjacent_solution_id is None:
                    raise ValueError(
                        f"{entity.semantic_id} sidewall has no exact auto-vacuum child."
                    )
            else:
                if adjacent_solution_id is None:
                    adjacent_solution_id = _conductor_sidewall_adjacent_solution_id(
                        build_input,
                        entity,
                    )
                if adjacent_solution_id is None:
                    continue
                ref_adjacent_solution_id = adjacent_solution_id
            adjacent_solution = _entity_by_id(build_input, ref_adjacent_solution_id)
            # The explicit parent-envelope check above is the only exterior rule
            # for auto-vacuum children. A disconnected child's rectangular bounds
            # can coincide with a conductor hole edge, which is not an exterior
            # boundary and must retain its exact child adjacency.
            if not auto_parent and _sidewall_on_solution_outer_boundary(
                geometry_ref, adjacent_solution
            ):
                ref_adjacent_solution_id = _adjacent_auto_vacuum_sidewall_component_id(
                    build_input,
                    entity=entity,
                    geometry_ref=geometry_ref,
                )
                if ref_adjacent_solution_id is None:
                    continue
            exposed_refs.append(
                {
                    **geometry_ref,
                    "adjacent_solution_id": ref_adjacent_solution_id,
                }
            )
    return tuple(exposed_refs)


def _route_c_conductor_sidewall_geometry_refs(
    *,
    entity: SemanticEntitySpec,
    adjacent_solution: SemanticEntitySpec,
    refs: tuple[dict[str, Any], ...],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for geometry_ref in refs:
        points = geometry_ref.get("quad_points", ())
        if len(points) != 4:
            result.append(dict(geometry_ref))
            continue
        start = (float(points[0][0]), float(points[0][1]))
        end = (float(points[1][0]), float(points[1][1]))
        exterior_intervals = (
            ((0.0, 1.0),)
            if _sidewall_on_solution_outer_boundary(geometry_ref, adjacent_solution)
            else ()
        )
        contact_intervals = _route_c_contact_intervals(
            entity=entity,
            start=start,
            end=end,
            edge_index=edge_index,
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"solution_exterior_boundary": True},
            )
            for interval_start, interval_end in exterior_intervals
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"adjacent_conductor_semantic_id": adjacent_id},
            )
            for (
                interval_start,
                interval_end,
                adjacent_id,
                create_surface,
            ) in contact_intervals
            if create_surface
        )
        blocked = (
            *exterior_intervals,
            *((start, end) for start, end, _, _ in contact_intervals),
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
            )
            for interval_start, interval_end in _interval_complement(blocked)
        )
    return tuple(result)


def _conductor_boundary_edge_index(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
) -> dict[
    tuple[tuple[float, float], float],
    tuple[tuple[tuple[float, float], tuple[float, float], str], ...],
]:
    if entity.part_role in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return {}
    index: dict[
        tuple[tuple[float, float], float],
        list[tuple[tuple[float, float], tuple[float, float], str]],
    ] = {}
    for other in build_input.entities:
        if not _can_trim_against_route_conductor(
            route,
            entity=entity,
            other=other,
        ):
            continue
        for other_start, other_end in _entity_boundary_edges(other):
            line_key = _line_key_2d(other_start, other_end)
            if line_key is None:
                continue
            index.setdefault(line_key, []).append(
                (other_start, other_end, other.semantic_id)
            )
    return {line_key: tuple(records) for line_key, records in index.items()}


def _candidate_boundary_edges(
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
    start: tuple[float, float],
    end: tuple[float, float],
) -> Sequence[tuple[tuple[float, float], tuple[float, float], str]]:
    line_key = _line_key_2d(start, end)
    if line_key is None:
        return ()
    return edge_index.get(line_key, ())


def _route_c_contact_intervals(
    entity: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[tuple[float, float, str, bool], ...]:
    records: list[tuple[float, float, str, bool]] = []
    for other_start, other_end, other_semantic_id in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = _segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        records.append(
            (
                interval[0],
                interval[1],
                other_semantic_id,
                entity.semantic_id < other_semantic_id,
            )
        )
    return tuple(records)


def _trim_sidewall_ref_against_route_conductors(
    *,
    geometry_ref: Mapping[str, Any],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return (dict(geometry_ref),)
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    covered_intervals: list[tuple[float, float]] = []
    for other_start, other_end, _ in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = _segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        covered_intervals.append(interval)
    if not covered_intervals:
        return (dict(geometry_ref),)
    return tuple(
        _sidewall_subsegment_geometry_ref(
            geometry_ref,
            start_parameter,
            end_parameter,
        )
        for start_parameter, end_parameter in _interval_complement(
            covered_intervals,
        )
        if end_parameter - start_parameter > _TOPOLOGY_EPS_UM
    )


def _can_trim_against_route_conductor(
    route: RouteLiteral,
    *,
    entity: SemanticEntitySpec,
    other: SemanticEntitySpec,
) -> bool:
    if (
        other.semantic_id == entity.semantic_id
        or _is_solution_entity(other)
        or other.route_representations.get(route) is None
        or "outer_loop" not in other.geometry
    ):
        return False
    z_min_um, z_max_um = _entity_z_range_um(entity)
    other_z_min_um, other_z_max_um = _entity_z_range_um(other)
    return _same_z(z_min_um, other_z_min_um) and _same_z(z_max_um, other_z_max_um)


def _entity_boundary_edges(
    entity: SemanticEntitySpec,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        edge
        for loop in (
            entity.geometry["outer_loop"],
            *entity.geometry.get("hole_loops", ()),
        )
        for edge in _ring_edges(_clean_loop(loop))
    )


def _sidewall_subsegment_geometry_ref(
    geometry_ref: Mapping[str, Any],
    start_parameter: float,
    end_parameter: float,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in geometry_ref["quad_points"]
    )
    return {
        **dict(geometry_ref),
        "quad_points": (
            _interpolate_3d(points[0], points[1], start_parameter),
            _interpolate_3d(points[0], points[1], end_parameter),
            _interpolate_3d(points[3], points[2], end_parameter),
            _interpolate_3d(points[3], points[2], start_parameter),
        ),
        "trimmed_by_conductor_contact": True,
        **dict(extra or {}),
    }
