"""Gmsh/OpenCASCADE bottom-up construction backend.

This backend consumes `ConstructionPlanRecord`. It must not discover interfaces
by running global `occ.fragment()` over arbitrary volumes.

Implemented lowering shape:

1. consume planned canonical curves/surface loops and create live surfaces;
2. reuse the same surface tag for planned conformal interfaces;
3. assemble every backend-live volume with `occ.addSurfaceLoop()` and
   `occ.addVolume()`;
4. recover backend dim-tags by `SurfacePlanRecord.surface_id` and
   `VolumePlanRecord.volume_id`;
5. write those tags to `BackendEntityTagRecord`;
6. attach physical groups only from `TagPlanRecord`;
7. write XAO/metadata with the same stable source ids;
8. fail when the plan cannot be built conformally.

`ConstructionBodyPlanRecord` and `CutHostOperationRecord` are route-policy
provenance in the current v1 path. They explain why cutout-shell surfaces exist
and how host interiors should be excluded, but this backend does not run
boolean cutter batches or use them to discover new surfaces.

Concrete Gmsh/OCC lowering target:

- For each unique point referenced by `CurvePlanRecord`, call
  `gmsh.model.occ.addPoint()`.
- For each `CurvePlanRecord`, call
  `gmsh.model.occ.addLine()`.
- For each `SurfaceLoopRecord`, call `gmsh.model.occ.addCurveLoop()` using
  planned `CurveRefRecord` order. Do not rely on negative OCC curve tags as the
  only correctness mechanism; loop closure and normals must already be
  validated by the compiler plan and audited after lowering.
- For each planned surface, call `gmsh.model.occ.addPlaneSurface()` with the
  planned outer loop and any hole loops in one call.
- For each backend-live volume, call `gmsh.model.occ.addSurfaceLoop()` with the
  planned `SurfaceRefRecord`s, then `gmsh.model.occ.addVolume()`.
- After `gmsh.model.occ.synchronize()`, recover dim-tags by source record id,
  call `gmsh.model.addPhysicalGroup()` for each `TagPlanRecord`, call
  `gmsh.model.setPhysicalName()` with the planned physical name, and write one
  route XAO file.
- After lowering, audit `getBoundary()` results for surfaces and volumes
  against planned curves/surfaces before treating the XAO as solver-ready.

Do not create volumes directly with boxes, extrusions, or `geometry_ref`
fallbacks. `domain_bounds_um` may help the planner create outer boundary
surfaces, but the backend only accepts volumes whose boundary surfaces already
exist in the plan.

Ground-plane subtraction should be handled as planned surface geometry in the
OCC kernel, not by preprocessing GDS into fragmented positive polygons.
Surface partitions are not backend cuts: the backend should receive child
`SurfacePlanRecord`s already expanded and build those child surfaces directly.

OCC backend comment block:

- `SurfacePlanRecord.outer_loop_ref` and `hole_loop_refs` are the v1 backend
  contract. `geometry_ref` is retained as audit/source metadata only. Raw-loop
  lowering remains disabled because canonical point/curve/surface-loop records
  are the conformal-topology contract.
- `TagPlanRecord` plus `BackendEntityTagRecord` is the tag ledger. Plan ids are
  stable; OCC tags and physical group ids are lowering results. If two live
  source ids map to one OCC dim-tag, canonicalization failed and the backend
  must report it instead of merging identities.
- Indium-bump or airbridge contact with a ground plane must become one planned
  contact patch surface plus one or more remainder surfaces carrying that patch
  as a hole loop. The contact patch is the shared conformal surface; the hole is
  only how the remainder face avoids overlap.
- Partitioned interfaces must arrive as child surface plans with their own
  loops. Build those child surfaces directly and do not create overlay masks or
  cut a live parent surface after the fact.
- Route A surface sheets are interface surfaces, not extra standalone faces.
  If the plan has an `MS`, `MA`, `MM`, or `SA` surface owned by a sheet
  conductor, build that one live surface and tag it; do not add another sheet on
  top of it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from scgsim.sgb.engine_gates import (
    engine_gate_gmsh_brep_conformality,
)
from scgsim.sgb.models import (
    BackendEntityTagRecord,
    ConstructionPlanRecord,
    GmshDimTag,
    SurfaceLoopRecord,
    TagPlanRecord,
)


def write_occ_geometry_from_plan(
    plan: ConstructionPlanRecord,
    *,
    xao_path: Path,
) -> ConstructionPlanRecord:
    """Write one XAO and return the plan with backend tags attached.

    The implementation attaches physical groups before writing `xao_path` and
    returns a copy of `plan` with `backend_entity_tags` populated. It refuses
    any volume that lacks planned `surface_refs`, because constructing that
    volume directly would bypass semantic surface ownership.
    """
    _validate_conformal_plan(plan)
    xao_path.parent.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []
    debug_logging = False
    gmsh = None
    was_initialized = True
    try:
        with _debug_stage(
            debug_logging,
            "gmsh import/initialize/clear/model setup",
            timings,
        ):
            import gmsh as gmsh_module

            gmsh = gmsh_module
            was_initialized = bool(gmsh.isInitialized())
            if not was_initialized:
                gmsh.initialize()
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 1 if debug_logging else 0)
            _apply_gmsh_number_option(
                gmsh,
                "Geometry.OCCAutoFix",
                "SGB_GMSH_OCC_AUTO_FIX",
                default=0.0,
            )
            gmsh.model.add(f"semantic_geometry_route_{plan.route.lower()}")

        point_tags: dict[str, int] = {}
        with _debug_stage(
            debug_logging,
            f"add {len(plan.points)} OCC points",
            timings,
        ):
            for point in plan.points:
                point_tags[point.point_id] = gmsh.model.occ.addPoint(*point.coordinate)
        curve_tags: dict[str, int] = {}
        with _debug_stage(
            debug_logging,
            f"add {len(plan.curves)} OCC curves",
            timings,
        ):
            for curve in plan.curves:
                curve_tags[curve.curve_id] = gmsh.model.occ.addLine(
                    point_tags[curve.start_point_id],
                    point_tags[curve.end_point_id],
                )
        loop_tags: dict[str, int] = {}
        surface_orientations = (
            _chosen_surface_orientations(plan) if plan.route in {"A", "B"} else {}
        )
        with _debug_stage(
            debug_logging,
            f"add {len(plan.surface_loops)} OCC curve loops",
            timings,
        ):
            for loop in plan.surface_loops:
                if debug_logging and len(loop.curve_refs) > 512:
                    print(
                        "[sgb:gmsh] "
                        f"large loop {loop.loop_id}: {len(loop.curve_refs)} curves",
                        flush=True,
                    )
                if plan.route in {"A", "B"}:
                    reverse = (
                        surface_orientations.get(loop.surface_id, "forward")
                        == "reversed"
                    )
                    loop_tags[loop.loop_id] = _add_curve_loop_from_plan(
                        gmsh,
                        loop,
                        curve_tags,
                        reverse=reverse,
                    )
                else:
                    loop_tags[loop.loop_id] = _add_curve_loop_from_plan(
                        gmsh,
                        loop,
                        curve_tags,
                    )
        source_tags: dict[tuple[str, str], list[GmshDimTag]] = {}
        live_surfaces = tuple(
            surface for surface in plan.surfaces if not surface.construction_only
        )
        with _debug_stage(
            debug_logging,
            f"add {len(live_surfaces)} OCC plane surfaces",
            timings,
        ):
            for surface in live_surfaces:
                if debug_logging:
                    edge_count = _surface_edge_count(surface, plan.surface_loops)
                    if edge_count > 512:
                        print(
                            "[sgb:gmsh] "
                            f"large surface {surface.surface_id}: {edge_count} edges",
                            flush=True,
                        )
                surface_tag = gmsh.model.occ.addPlaneSurface(
                    [
                        loop_tags[surface.outer_loop_ref],
                        *(loop_tags[loop_id] for loop_id in surface.hole_loop_refs),
                    ]
                )
                source_tags.setdefault(("surface", surface.surface_id), []).append(
                    (2, surface_tag)
                )

        live_volumes = tuple(
            volume for volume in plan.volumes if not volume.construction_only
        )
        with _debug_stage(
            debug_logging,
            f"add {len(live_volumes)} OCC volumes",
            timings,
        ):
            largest_volume_boundary: tuple[int, str] = (0, "")
            for volume in live_volumes:
                if plan.route in {"A", "B"}:
                    exterior_refs = (
                        getattr(volume, "exterior_surface_refs", ())
                        or volume.surface_refs
                    )
                    exterior_tags = _surface_ref_tags(
                        volume.volume_id, exterior_refs, source_tags
                    )
                    shell_tags = [
                        gmsh.model.occ.addSurfaceLoop(exterior_tags, sewing=True)
                    ]
                    shell_tags.extend(
                        gmsh.model.occ.addSurfaceLoop(
                            _surface_ref_tags(
                                void.shell_id, void.surface_refs, source_tags
                            ),
                            sewing=True,
                        )
                        for void in getattr(volume, "inner_pec_void_shells", ())
                    )
                else:
                    exterior_tags = _surface_ref_tags(
                        volume.volume_id, volume.surface_refs, source_tags
                    )
                    shell_tags = [gmsh.model.occ.addSurfaceLoop(exterior_tags)]
                if len(exterior_tags) > largest_volume_boundary[0]:
                    largest_volume_boundary = (len(exterior_tags), volume.volume_id)
                volume_tag = gmsh.model.occ.addVolume(shell_tags)
                source_tags.setdefault(("volume", volume.volume_id), []).append(
                    (3, volume_tag)
                )
            if debug_logging and largest_volume_boundary[1]:
                print(
                    "[sgb:gmsh] "
                    f"largest volume boundary {largest_volume_boundary[1]}: "
                    f"{largest_volume_boundary[0]} surfaces",
                    flush=True,
                )

        with _debug_stage(debug_logging, "synchronize OCC model", timings):
            gmsh.model.occ.synchronize()
        with _debug_stage(debug_logging, "engine gate gmsh_brep_conformality", timings):
            gmsh_brep_gate = engine_gate_gmsh_brep_conformality(
                plan,
                gmsh=gmsh,
                source_tags=source_tags,
                curve_tags=curve_tags,
            )
        backend_tags = _backend_entity_tags(source_tags)
        grouped_tag_plans = _group_tag_plans(plan.tags)
        with _debug_stage(
            debug_logging,
            f"add {len(grouped_tag_plans)} physical groups",
            timings,
        ):
            for tag_plans in grouped_tag_plans:
                first_tag = tag_plans[0]
                entity_tags = _physical_entity_tags(tag_plans, source_tags)
                if not entity_tags:
                    raise ValueError(
                        f"{first_tag.physical_name} has no backend entity tags"
                    )
                group_tag = gmsh.model.addPhysicalGroup(
                    first_tag.dimension,
                    entity_tags,
                )
                gmsh.model.setPhysicalName(
                    first_tag.dimension,
                    group_tag,
                    first_tag.physical_name,
                )
        with _debug_stage(debug_logging, f"write XAO {xao_path}", timings):
            gmsh.write(str(xao_path))
        return replace(
            plan,
            backend_entity_tags=backend_tags,
            metadata={
                **dict(plan.metadata),
                "xao_path": str(xao_path),
                "backend_timings": timings,
                "engine_gate_gmsh_brep_conformality": gmsh_brep_gate,
            },
        )
    finally:
        if gmsh is not None and not was_initialized:
            with _debug_stage(debug_logging, "finalize gmsh", timings):
                gmsh.finalize()


def _apply_gmsh_number_option(
    gmsh: Any,
    option_name: str,
    environment_name: str,
    *,
    default: float,
) -> None:
    value = os.environ.get(environment_name)
    gmsh.option.setNumber(option_name, default if value is None else float(value))


@contextmanager
def _debug_stage(
    enabled: bool,
    label: str,
    timings: list[dict[str, Any]],
) -> Iterator[None]:
    started = time.perf_counter()
    status = "done"
    if enabled:
        print(f"[sgb:gmsh] start {label}", flush=True)
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - started
        timings.append(
            {
                "stage": label,
                "seconds": round(elapsed, 6),
                "status": status,
            }
        )
        if enabled:
            print(f"[sgb:gmsh] {status} {label} in {elapsed:.3f}s", flush=True)


def _validate_conformal_plan(plan: ConstructionPlanRecord) -> None:
    """Reject any backend-live geometry that lacks canonical topology refs."""
    if any(not surface.construction_only for surface in plan.surfaces) and (
        not plan.points or not plan.curves or not plan.surface_loops
    ):
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live surfaces require "
            "planned PointPlan/CurvePlan/SurfaceLoop records before lowering"
        )
    missing_loop_refs = [
        surface.surface_id
        for surface in plan.surfaces
        if not surface.construction_only and surface.outer_loop_ref is None
    ]
    if missing_loop_refs:
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live surfaces require "
            f"outer_loop_ref before lowering {missing_loop_refs!r}"
        )
    unpartitioned_volumes = [
        volume.volume_id
        for volume in plan.volumes
        if not volume.construction_only and not volume.surface_refs
    ]
    if unpartitioned_volumes:
        raise NotImplementedError(
            "nonconformal OCC build refused: backend-live volumes must be "
            "assembled from planned surface_refs with addSurfaceLoop/addVolume "
            f"{unpartitioned_volumes!r}. Building these directly would create "
            "standalone surfaces instead of shared conformal topology."
        )


def _add_curve_loop_from_plan(
    gmsh: Any,
    loop: SurfaceLoopRecord,
    curve_tags: Mapping[str, int],
    *,
    reverse: bool = False,
) -> int:
    signed_curve_tags = [
        curve_tags[curve_ref.curve_id] * curve_ref.orientation * (-1 if reverse else 1)
        for curve_ref in loop.curve_refs
    ]
    try:
        return gmsh.model.occ.addCurveLoop(signed_curve_tags)
    except Exception as exc:
        raise ValueError(
            f"{loop.loop_id} could not be lowered to an OCC curve loop"
        ) from exc


def _chosen_surface_orientations(plan: ConstructionPlanRecord) -> dict[str, str]:
    """Choose one planned direction per OCC surface before it is created."""
    uses: dict[str, set[str]] = {}
    for volume in plan.volumes:
        exterior_refs = (
            getattr(volume, "exterior_surface_refs", ()) or volume.surface_refs
        )
        for ref in exterior_refs:
            uses.setdefault(ref.surface_id, set()).add(ref.orientation)
        for void in getattr(volume, "inner_pec_void_shells", ()):
            for ref in void.surface_refs:
                uses.setdefault(ref.surface_id, set()).add(ref.orientation)
    for orientations in uses.values():
        if len(orientations) > 2:
            raise ValueError("surface has ambiguous planned volume orientations")
    # A shared interface may be opposite in two volumes; its plane is created
    # once and OCC sewing supplies the shell-local sense. Prefer forward
    # deterministically, otherwise lower its single reversed use.
    return {
        surface_id: "forward" if "forward" in orientations else "reversed"
        for surface_id, orientations in uses.items()
    }


def _surface_edge_count(
    surface: Any,
    surface_loops: Sequence[SurfaceLoopRecord],
) -> int:
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    loop_ids = (
        *((surface.outer_loop_ref,) if surface.outer_loop_ref is not None else ()),
        *surface.hole_loop_refs,
    )
    return sum(len(loops_by_id[loop_id].curve_refs) for loop_id in loop_ids)


def _surface_ref_tags(
    owner_id: str,
    surface_refs: Sequence[Any],
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> list[int]:
    """Return OCC surface tags for `addSurfaceLoop()`.

    OCC's Python `addSurfaceLoop()` rejects negative tags.  Orientation is
    retained in the plan/audit ledger; lowering relies on each planned surface
    loop's canonical ring direction.
    """
    tags = [
        tag
        for surface_ref in surface_refs
        for dim, tag in source_tags.get(("surface", surface_ref.surface_id), ())
        if dim == 2
    ]
    if len(tags) != len(surface_refs):
        missing = [
            surface_ref.surface_id
            for surface_ref in surface_refs
            if not source_tags.get(("surface", surface_ref.surface_id), ())
        ]
        raise ValueError(f"{owner_id} missing surface tags: {missing!r}")
    return tags


def _backend_entity_tags(
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> tuple[BackendEntityTagRecord, ...]:
    return tuple(
        BackendEntityTagRecord(
            source_record_kind=source_kind,
            source_record_id=source_id,
            dim_tag=dim_tag,
        )
        for (source_kind, source_id), dim_tags in source_tags.items()
        for dim_tag in dim_tags
    )


def _group_tag_plans(
    tags: tuple[TagPlanRecord, ...],
) -> tuple[tuple[TagPlanRecord, ...], ...]:
    grouped: dict[tuple[str, int, str, str], list[TagPlanRecord]] = {}
    for tag in tags:
        grouped.setdefault(
            (tag.physical_name, tag.dimension, tag.role, tag.solver_use),
            [],
        ).append(tag)
    return tuple(tuple(items) for items in grouped.values())


def _physical_entity_tags(
    tag_plans: tuple[TagPlanRecord, ...],
    source_tags: Mapping[tuple[str, str], Sequence[GmshDimTag]],
) -> list[int]:
    tags: list[int] = []
    seen: set[int] = set()
    for tag_plan in tag_plans:
        for dimension, tag in source_tags.get(
            (tag_plan.source_record_kind, tag_plan.source_record_id),
            (),
        ):
            if dimension != tag_plan.dimension or tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
    return tags
