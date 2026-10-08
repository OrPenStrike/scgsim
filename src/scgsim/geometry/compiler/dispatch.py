"""Own the single polygon/curved route dispatcher and existing planning stage order. Native construction remains downstream."""

from __future__ import annotations

import time
from typing import Any

from scgsim.geometry._primitives.entities import _component_metadata
from scgsim.geometry.compiler.validation import (
    validate_curve_plan_coverage,
    validate_interface_surface_source_of_truth,
    validate_no_surface_overlap,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_deduplication,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_surface_use_counts,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.construction import (
    ConstructionPlanRecord,
    RouteABConstructionPlanRecord,
)
from scgsim.geometry.models.input import GeometryBuildInput
from scgsim.geometry.planning.domain import (
    _merge_solution_sidewall_interfaces,
    _prepare_auto_vacuum_solution_regions,
    _reconcile_solution_domain_boundaries,
)
from scgsim.geometry.planning.evidence import build_semantic_evidence_facade
from scgsim.geometry.planning.interfaces import (
    complete_interface_plan_from_surfaces,
    plan_conductor_contact_patches,
    plan_mm_contact_records,
    recognize_route_interfaces,
)
from scgsim.geometry.planning.ports import (
    _bind_route_b_port_volume_plans,
    _lower_port_sheet_regions,
    _validate_route_b_port_sheet_sidewall_topology,
)
from scgsim.geometry.planning.surfaces import (
    plan_route_surfaces,
    plan_surface_partitions,
)
from scgsim.geometry.planning.tags import plan_route_tags
from scgsim.geometry.planning.topology import plan_canonical_topology
from scgsim.geometry.planning.volumes import (
    _reconcile_construction_body_surface_ids,
    plan_cut_host_operations,
    plan_route_construction_bodies,
    plan_route_volumes,
)


def build_route_construction_plan(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> ConstructionPlanRecord:
    """Build the route-aware plan consumed by bottom-up OCC construction.

    The plan is the semantic center of v1. Interface recognition, surface
    ownership, partition ownership, and tag ownership are decided before OCC
    geometry exists.

    The backend must be able to build from this record without global
    `occ.fragment()`: surfaces carry loop geometry, volumes reference those
    surfaces, Route A/B cuts are explicit `CutHostOperationRecord`s, and
    `TagPlanRecord`s define the physical names before dim-tags exist.

    Tags are planned before backend construction. Every backend-live
    `SurfacePlanRecord` or `VolumePlanRecord` must either get a `TagPlanRecord`
    or be explicitly marked `construction_only`.
    """
    if build_input.boundary_curves or build_input.boundary_reconstruction:
        if route == "C":
            raise NotImplementedError(
                "active source curves support Palace Route A/B, not Route C"
            )
        from scgsim.geometry.construction.curved import build_curved_construction_plan

        return build_curved_construction_plan(build_input, route=route)

    timings: list[dict[str, Any]] = []
    build_input = _prepare_auto_vacuum_solution_regions(build_input, route=route)
    _timed(
        timings,
        "validate_selected_route",
        lambda: validate_selected_route(build_input, route),
    )
    semantic_facts = build_semantic_evidence_facade(build_input, route=route)
    interfaces = _timed(
        timings,
        "recognize_route_interfaces",
        lambda: recognize_route_interfaces(build_input, route=route),
    )
    interfaces = _timed(
        timings,
        "plan_conductor_contact_patches",
        lambda: plan_conductor_contact_patches(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    if route in {"A", "B"}:
        interfaces, mm_contacts = _timed(
            timings,
            "plan_mm_contact_records",
            lambda: plan_mm_contact_records(
                build_input,
                route=route,
                interfaces=interfaces,
            ),
        )
    else:
        mm_contacts = ()
    surface_partitions = _timed(
        timings,
        "plan_surface_partitions",
        lambda: plan_surface_partitions(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    construction_bodies = _timed(
        timings,
        "plan_route_construction_bodies",
        lambda: plan_route_construction_bodies(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    surfaces = _timed(
        timings,
        "plan_route_surfaces",
        lambda: plan_route_surfaces(
            build_input,
            route=route,
            interfaces=interfaces,
            surface_partitions=surface_partitions,
            construction_bodies=construction_bodies,
            mm_contacts=mm_contacts,
            semantic_facts=semantic_facts,
        ),
    )
    surfaces = _timed(
        timings,
        "reconcile_solution_domain_boundaries",
        lambda: _reconcile_solution_domain_boundaries(
            build_input,
            surfaces=surfaces,
        ),
    )
    if route in {"A", "B"} and build_input.port_sheet_regions:
        surfaces = _timed(
            timings,
            "lower_port_sheet_regions",
            lambda: _lower_port_sheet_regions(
                build_input,
                route=route,
                mm_contacts=mm_contacts,
                planned_surfaces=surfaces,
            ),
        )
    _timed(
        timings,
        "validate_surface_sheet_interface_coverage",
        lambda: validate_surface_sheet_interface_coverage(
            build_input,
            route=route,
            surfaces=surfaces,
        ),
    )
    surfaces = _timed(
        timings,
        "merge_solution_sidewall_interfaces",
        lambda: _merge_solution_sidewall_interfaces(
            build_input,
            surfaces=surfaces,
            semantic_facts=semantic_facts,
        ),
    )
    interfaces = _timed(
        timings,
        "complete_interface_plan_from_surfaces",
        lambda: complete_interface_plan_from_surfaces(
            interfaces=interfaces,
            surfaces=surfaces,
        ),
    )
    points, curves, surface_loops, surfaces = _timed(
        timings,
        "plan_canonical_topology",
        lambda: plan_canonical_topology(surfaces=surfaces),
    )
    _timed(
        timings,
        "validate_route_b_port_sheet_sidewall_topology",
        lambda: _validate_route_b_port_sheet_sidewall_topology(
            route=route,
            points=points,
            curves=curves,
            surface_loops=surface_loops,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_curve_plan_coverage",
        lambda: validate_curve_plan_coverage(
            points=points,
            curves=curves,
            surface_loops=surface_loops,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_deduplication",
        lambda: validate_surface_deduplication(surfaces=surfaces),
    )
    _timed(
        timings,
        "validate_no_surface_overlap",
        lambda: validate_no_surface_overlap(surfaces=surfaces),
    )
    volumes = _timed(
        timings,
        "plan_route_volumes",
        lambda: plan_route_volumes(
            build_input,
            route=route,
            surfaces=surfaces,
            mm_contacts=mm_contacts,
        ),
    )
    if route == "B":
        surfaces = _bind_route_b_port_volume_plans(build_input, surfaces, volumes)
    _timed(
        timings,
        "validate_route_volume_surface_refs",
        lambda: validate_route_volume_surface_refs(route=route, volumes=volumes),
    )
    _timed(
        timings,
        "validate_volume_surface_closure",
        lambda: validate_volume_surface_closure(
            volumes=volumes,
            surfaces=surfaces,
            surface_loops=surface_loops,
        ),
    )
    _timed(
        timings,
        "validate_surface_use_counts",
        lambda: validate_surface_use_counts(volumes=volumes, surfaces=surfaces),
    )
    construction_bodies = _reconcile_construction_body_surface_ids(
        construction_bodies,
        surfaces=surfaces,
    )
    cut_operations = _timed(
        timings,
        "plan_cut_host_operations",
        lambda: plan_cut_host_operations(
            route=route,
            construction_bodies=construction_bodies,
        ),
    )
    tags = _timed(
        timings,
        "plan_route_tags",
        lambda: plan_route_tags(route=route, surfaces=surfaces, volumes=volumes),
    )
    _timed(
        timings,
        "validate_route_operation_coverage",
        lambda: validate_route_operation_coverage(
            construction_bodies=construction_bodies,
            cut_operations=cut_operations,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_partition_coverage",
        lambda: validate_surface_partition_coverage(
            interfaces=interfaces,
            surface_partitions=surface_partitions,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_interface_surface_source_of_truth",
        lambda: validate_interface_surface_source_of_truth(
            interfaces=interfaces,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_tag_plan_coverage",
        lambda: validate_tag_plan_coverage(
            surfaces=surfaces,
            volumes=volumes,
            tags=tags,
        ),
    )
    plan_type = (
        RouteABConstructionPlanRecord if route in {"A", "B"} else ConstructionPlanRecord
    )
    plan_kwargs = {
        "route": route,
        "interfaces": interfaces,
        "surface_partitions": surface_partitions,
        "points": points,
        "curves": curves,
        "surface_loops": surface_loops,
        "surfaces": surfaces,
        "volumes": volumes,
        "construction_bodies": construction_bodies,
        "cut_operations": cut_operations,
        "port_sheet_regions": build_input.port_sheet_regions,
        "tags": tags,
        "metadata": {
            "backend_strategy": "surface_plan_first_bottom_up_occ",
            "port_sheet_region_layer": {
                "source": "GeometryBuildInput.port_sheet_regions",
                "backend_live_surface_records": (
                    bool(build_input.port_sheet_regions) and route in {"A", "B"}
                ),
                "allowed_overlap": "palace_lumped_port_sheet_only",
                "lowering_status": (
                    "lowered" if build_input.port_sheet_regions else "none"
                ),
            },
            "timings": timings,
            **(
                {"conductor_components": _component_metadata(mm_contacts)}
                if route in {"A", "B"}
                else {}
            ),
        },
    }
    if route in {"A", "B"}:
        plan_kwargs["mm_contacts"] = mm_contacts
    return plan_type(**plan_kwargs)


def _timed(
    timings: list[dict[str, Any]],
    stage: str,
    fn: Any,
) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception:
        timings.append(
            {
                "stage": stage,
                "seconds": round(time.perf_counter() - started, 6),
                "status": "failed",
            }
        )
        raise
    timings.append(
        {
            "stage": stage,
            "seconds": round(time.perf_counter() - started, 6),
            "status": "done",
        }
    )
    return result
