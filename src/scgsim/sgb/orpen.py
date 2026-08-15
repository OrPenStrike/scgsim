"""Public OrPen SC PDK semantic-stack adapter for the Xmon example."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil, cos, isfinite, radians, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

_LEVELS = (
    "D0_SUBSTRATE",
    "D0_TO_D1_GAP",
    "D1_SUBSTRATE",
    "OUTER_VACUUM",
    "D0_TOP_M1",
    "D1_BOTTOM_M1",
    "D0_D1_INDIUM_BUMP",
)

# This adapter is intentionally tied to the public examples extra pin.  The
# receipt records the source identity alongside the PDK-owned primitive/spec.
_ORPEN_SC_PDK_REVISION = "8be8fc48c26f6bed06298cab673d49b9a6afbe7f"
_FILL_OWNER_SEMANTIC_IDS = ("D0_TOP_GROUND_PLANE", "D1_BOTTOM_GROUND_PLANE")


def build_kosen2024_flip_chip_xmon_stack(
    *,
    component: Any,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    d0_top_ground_mask_layer: tuple[int, int],
    indium_bump_layer: tuple[int, int],
    coupon_padding_um: float,
    include_airbox: bool = True,
    air_below_thickness_um: float | None = None,
    air_above_thickness_um: float | None = None,
) -> dict[str, Any]:
    """Project public PDK facts into the one SCGSim Route-A/B stack contract.

    PDK records remain the authority for material, z, thickness and layers.
    Signal identities come only from the documented topology anchor and authored
    component ports; no layout-name, bbox, or residual-policy inference occurs.
    """
    if not isinstance(include_airbox, bool):
        raise TypeError("include_airbox must be a bool.")
    levels = getattr(layer_stack, "layers", None)
    if not isinstance(levels, Mapping):
        raise TypeError("layer_stack must expose public LayerStack.layers.")
    required_levels = _LEVELS if include_airbox else _LEVELS[:3] + _LEVELS[4:]
    missing = [name for name in required_levels if name not in levels]
    if missing:
        raise ValueError(f"public OrPen LayerStack lacks {missing!r}.")
    d0_substrate, gap, d1_substrate = (levels[name] for name in _LEVELS[:3])
    outer = levels["OUTER_VACUUM"] if include_airbox else None
    d0_top, d1_bottom, bump = (levels[name] for name in _LEVELS[4:])
    coupon_bounds_um = _coupon_domain_bounds(component, coupon_padding_um)
    d0_z_min = _zmin(d0_substrate)
    d0_z_max = d0_z_min + _thickness(d0_substrate)
    gap_z_min = _zmin(gap)
    gap_z_max = gap_z_min + _thickness(gap)
    d1_z_min = _zmin(d1_substrate)
    d1_z_max = d1_z_min + _thickness(d1_substrate)
    if (abs(d0_z_max - gap_z_min) > 1e-9) or (abs(gap_z_max - d1_z_min) > 1e-9):
        raise ValueError("public OrPen D0/gap/D1 solution regions must be contiguous.")
    air_below_thickness = None
    air_above_thickness = None
    if include_airbox:
        air_below_thickness = _positive_number(
            air_below_thickness_um, "air_below_thickness_um"
        )
        air_above_thickness = _positive_number(
            air_above_thickness_um, "air_above_thickness_um"
        )
    info = getattr(component, "info", {})
    try:
        layers = info["layers"]
    except (KeyError, TypeError):
        layers = {}
    d1_draw = _layer_tuple(layers.get("q_chip_draw"))
    d1_mask = _layer_tuple(layers.get("q_chip_ground_mask"))
    d0_mask = _layer_tuple(d0_top_ground_mask_layer)
    bump_layer = _layer_tuple(indium_bump_layer)
    if not all((d1_draw, d1_mask, d0_mask, bump_layer)):
        raise ValueError("public PDK lacks required D0/D1/bump GDS-layer facts.")

    materials = _materials(material_records)
    signal_group = "D1_BOTTOM_SIGNAL_GROUP"
    common = _metal_record(
        layer=d1_draw,
        level=d1_bottom,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D1_BOTTOM_M1_DRAW",
    )
    selectors = _signal_selectors(component)
    signal_layers = [
        {
            **common,
            "semantic_id": semantic_id,
            "net_id": net_id,
            "geometry": {**common["geometry"], "selector_point_um": point},
            "metadata": {
                **common["metadata"],
                "semantic_group_id": signal_group,
                "logical_layer_id": "D1_BOTTOM_M1",
                "source_selector": source,
            },
        }
        for semantic_id, net_id, point, source in selectors
    ]
    d0_ground = _metal_record(
        layer=d0_mask,
        level=d0_top,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D0_TOP_GROUND_MASK",
    )
    d1_ground = _metal_record(
        layer=d1_mask,
        level=d1_bottom,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D1_BOTTOM_GROUND_MASK",
    )
    d0_ground.update(
        {
            "semantic_id": "D0_TOP_GROUND_PLANE",
            "net_id": "Ground",
            "geometry": {
                **d0_ground["geometry"],
                "geometry_source": "die_face_minus_ground_mask",
                "plane_bounds_ref": "D0_SUBSTRATE",
            },
            "metadata": {
                **d0_ground["metadata"],
                "equipotential_id": "Ground",
            },
        }
    )
    d1_ground.update(
        {
            "semantic_id": "D1_BOTTOM_GROUND_PLANE",
            "net_id": "Ground",
            "geometry": {
                **d1_ground["geometry"],
                "geometry_source": "die_face_minus_ground_mask",
                "mask_layer": d1_mask,
                "include_layer": d1_draw,
                "include_selector_points_um": [(-62.0, -37.0)],
                "plane_bounds_ref": "D1_SUBSTRATE",
            },
            "metadata": {
                **d1_ground["metadata"],
                "logical_layer_id": "D1_BOTTOM_M1",
                "source_selector": "public lower-left junction ground-arm anchor",
                "equipotential_id": "Ground",
            },
        }
    )
    bumps = _metal_record(
        layer=bump_layer,
        level=bump,
        host="D0_TO_D1_GAP",
        part_role="bump_body",
        representations={"A": "cutout_boundary_shell", "B": "cutout_boundary_shell"},
        source_layer_name="D0_D1_INDIUM_BUMP",
    )
    bumps.update(
        {
            "semantic_id": "D0_D1_INDIUM_BUMP",
            "net_id": "Ground",
            "geometry": {**bumps["geometry"], "split_polygons_as_entities": True},
            "metadata": {
                **bumps["metadata"],
                # Coalescing is family-scoped; the adapter still retains each
                # source polygon as a distinct structured entity.
                "semantic_group_id": "D0_D1_INDIUM_BUMP",
                "source_kind": "authored",
                "source_semantic_id": "D0_D1_INDIUM_BUMP",
            },
        }
    )
    solution_regions = {
        "D0_SUBSTRATE": _solution(
            material_id=str(d0_substrate.material),
            z_min_um=d0_z_min,
            z_max_um=d0_z_max,
            domain_bounds_um=coupon_bounds_um,
        ),
        "D0_TO_D1_GAP": _solution(
            material_id=str(gap.material),
            z_min_um=gap_z_min,
            z_max_um=gap_z_max,
            domain_bounds_um=coupon_bounds_um,
        ),
        "D1_SUBSTRATE": _solution(
            material_id=str(d1_substrate.material),
            z_min_um=d1_z_min,
            z_max_um=d1_z_max,
            domain_bounds_um=coupon_bounds_um,
        ),
    }
    if include_airbox:
        assert outer is not None
        assert air_below_thickness is not None
        assert air_above_thickness is not None
        solution_regions = {
            "AIR_BELOW": _solution(
                material_id=str(outer.material),
                z_min_um=d0_z_min - air_below_thickness,
                z_max_um=d0_z_min,
                domain_bounds_um=coupon_bounds_um,
                is_airbox=True,
                airbox_side="below",
            ),
            **solution_regions,
            "AIR_ABOVE": _solution(
                material_id=str(outer.material),
                z_min_um=d1_z_max,
                z_max_um=d1_z_max + air_above_thickness,
                domain_bounds_um=coupon_bounds_um,
                is_airbox=True,
                airbox_side="above",
            ),
        }
    metadata = {
        "adapter": "scgsim.sgb.orpen.build_kosen2024_flip_chip_xmon_stack",
        "component_contract": "kosen2024_flip_chip_xmon_qubit public zero-argument cell",
        "excluded_layers": ("D1_BOTTOM_JJ_DRAW", "D0_D1_UNDER_BUMP"),
        "signal_group": signal_group,
        "coupon_domain_bounds_um": coupon_bounds_um,
        "coupon_padding_um": float(coupon_padding_um),
    }
    if include_airbox:
        metadata.update(
            {
                "air_below_thickness_um": air_below_thickness,
                "air_above_thickness_um": air_above_thickness,
            }
        )
    return {
        "solution_regions": solution_regions,
        "materials": materials,
        "layers": [d0_ground, d1_ground, *signal_layers, bumps],
        "metadata": metadata,
    }


def _prepare_indium_ground_bump_fill(
    *,
    component: Any,
    stack: Mapping[str, Any],
    fill: bool,
    fill_pitch_um: float,
    fill_clearance_um: float,
) -> dict[str, Any]:
    """Materialize PDK-defined ground-bump fill without mutating its input cell.

    The public PDK owns the bump primitive, physical origin, and keepout-layer
    meaning.  SCGSim only enumerates the explicit lattice and carries its
    resulting per-site semantic records through the existing Route-A/B planner.
    """
    if not isinstance(fill, bool):
        raise TypeError("fill must be a bool.")
    pitch = _positive_number(fill_pitch_um, "fill_pitch_um")
    clearance = _non_negative_number(fill_clearance_um, "fill_clearance_um")
    try:
        from orpen_sc_pdk import get_indium_ground_bump_spec
        from orpen_sc_pdk.cells.indium import indium_bump
    except ImportError as exc:
        raise RuntimeError(
            "set_indium_ground_bumps() requires the public orpen-sc-pdk examples extra."
        ) from exc

    spec = dict(get_indium_ground_bump_spec())
    if spec.get("schema_identity") != "orpen.indium_ground_bump_spec.v1":
        raise ValueError("unsupported public indium ground bump spec identity.")
    origin = _point2(spec.get("lattice_origin_um"), "lattice_origin_um")
    bump_layer, under_bump_layer = _spec_layers(spec)
    settings = spec.get("canonical_component_settings")
    if not isinstance(settings, Mapping):
        raise TypeError("public indium bump spec settings must be a mapping.")

    import gdstk

    from scgsim.sgb.adapter import build_gds_stack_geometry_input
    from scgsim.sgb.models import LayoutPolygonSpec, SemanticEntitySpec

    result_stack = deepcopy(dict(stack))
    with TemporaryDirectory(prefix="scgsim-indium-fill-") as temporary:
        temporary_path = Path(temporary)
        source_gds = temporary_path / "source.gds"
        source_stack = temporary_path / "stack.json"
        component.write_gds(source_gds)
        source_stack.write_text(_json_text(result_stack), encoding="utf-8")
        build_input = build_gds_stack_geometry_input(
            gds_file=source_gds,
            stack_file=source_stack,
            top_cell_name=str(getattr(component, "name", "") or "") or None,
        )
        library = gdstk.read_gds(str(source_gds))
        cell = _gdstk_cell(library, str(getattr(component, "name", "") or ""))
        audit = _audit_authored_indium_sites(
            build_input=build_input,
            cell=cell,
            bump_layer=bump_layer,
            under_bump_layer=under_bump_layer,
            spec=spec,
        )
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "rule": {
                "identity": "scgsim.indium_ground_bump_fill.v1",
                "version": 1,
            },
            "source": "scgsim.sgb.orpen._prepare_indium_ground_bump_fill",
            "pdk_revision": _ORPEN_SC_PDK_REVISION,
            "pdk_spec": spec,
            "input": {
                "component_gds_sha256": _sha256_file(source_gds),
                "canonical_stack_sha256": _sha256_file(source_stack),
            },
            "controls": {
                "fill": fill,
                "fill_pitch_um": pitch,
                "fill_clearance_um": clearance,
            },
            "authored_sites": audit["authored_sites"],
            "accepted": [],
            "rejected": [],
            "semantic_contract": audit["semantic_contract"],
        }
        if not fill:
            receipt["counts"] = {
                "authored": len(audit["authored_sites"]),
                "generated": 0,
                "rejected": 0,
            }
            _set_indium_fill_metadata(result_stack, receipt)
            return {
                "component": component,
                "stack": result_stack,
                "injected_entities": (),
                "injected_polygons": (),
                "receipt": receipt,
            }

        ground = audit["ground_intersection"]
        blockers = list(audit["blockers"])
        bounds = _coupon_bounds(result_stack)
        footprint_size = max(
            float(settings["indium_bump_size"]),
            float(settings["under_bump_size"]),
        )
        accepted: list[tuple[int, int, float, float]] = []
        rejected: list[dict[str, Any]] = []
        for i in range(
            ceil((bounds["x_min_um"] - origin[0]) / pitch),
            ceil((bounds["x_max_um"] - origin[0]) / pitch) + 1,
        ):
            for j in range(
                ceil((bounds["y_min_um"] - origin[1]) / pitch),
                ceil((bounds["y_max_um"] - origin[1]) / pitch) + 1,
            ):
                x, y = origin[0] + i * pitch, origin[1] + j * pitch
                candidate = gdstk.rectangle(
                    (x - footprint_size / 2, y - footprint_size / 2),
                    (x + footprint_size / 2, y + footprint_size / 2),
                )
                expanded = (
                    gdstk.offset((candidate,), clearance, join="round", tolerance=1e-6)
                    if clearance
                    else [candidate]
                )
                candidate_id = _fill_id(i, j)
                blocker_ids: tuple[str, ...] = ()
                if gdstk.boolean(expanded, ground, "not", precision=1e-9):
                    reason = "outside_d0_d1_ground_intersection"
                    blocker_ids = ("D0_D1_GROUND_INTERSECTION",)
                else:
                    blocker_ids = _intersecting_blocker_ids(expanded, blockers)
                    reason = "intersects_blocking_footprint" if blocker_ids else None
                if reason:
                    rejected.append(
                        {
                            "semantic_id": candidate_id,
                            "lattice_index": [i, j],
                            "center_um": [x, y],
                            "geometry_placement_sha256": _placement_digest(
                                semantic_id=candidate_id,
                                lattice_index=(i, j),
                                center_um=(x, y),
                                footprint_size_um=footprint_size,
                            ),
                            "owner_semantic_ids": audit["semantic_contract"][
                                "owner_semantic_ids"
                            ],
                            "net_id": audit["semantic_contract"]["net_id"],
                            "equipotential_id": audit["semantic_contract"][
                                "equipotential_id"
                            ],
                            "host_solution_volume_id": audit["semantic_contract"][
                                "host_solution_volume_id"
                            ],
                            "source_kind": "fill",
                            "source_semantic_id": candidate_id,
                            "disposition": "rejected",
                            "reason": reason,
                            "blocker_source_ids": list(blocker_ids),
                        }
                    )
                else:
                    accepted.append((i, j, x, y))
                    # This exact canonical footprint becomes a blocker for all
                    # later lattice sites; it is not a separate meshing path.
                    blockers.append((f"generated_site:{candidate_id}", (candidate,)))

    derived = _component_with_canonical_bumps(
        component, indium_bump, settings, accepted, pitch
    )
    _assert_generated_center_selectors(derived, bump_layer, accepted)
    _exclude_generated_bumps_from_authored_layer(result_stack, accepted)
    fill_entities, fill_polygons = _fill_semantic_records(
        accepted=accepted,
        bump_layer=bump_layer,
        bump_size_um=float(settings["indium_bump_size"]),
        result_stack=result_stack,
        semantic_contract=receipt["semantic_contract"],
        LayoutPolygonSpec=LayoutPolygonSpec,
        SemanticEntitySpec=SemanticEntitySpec,
    )
    receipt["accepted"] = [
        {
            "semantic_id": _fill_id(i, j),
            "lattice_index": [i, j],
            "center_um": [x, y],
            "geometry_placement_sha256": _placement_digest(
                semantic_id=_fill_id(i, j),
                lattice_index=(i, j),
                center_um=(x, y),
                footprint_size_um=footprint_size,
            ),
            "owner_semantic_ids": receipt["semantic_contract"]["owner_semantic_ids"],
            "net_id": receipt["semantic_contract"]["net_id"],
            "equipotential_id": receipt["semantic_contract"]["equipotential_id"],
            "host_solution_volume_id": receipt["semantic_contract"][
                "host_solution_volume_id"
            ],
            "source_kind": "fill",
            "source_semantic_id": _fill_id(i, j),
            "disposition": "accepted",
            "source_provenance": "public_pdk_indium_ground_fill",
        }
        for i, j, x, y in accepted
    ]
    receipt["rejected"] = rejected
    receipt["counts"] = {
        "authored": len(receipt["authored_sites"]),
        "generated": len(accepted),
        "rejected": len(rejected),
    }
    _set_indium_fill_metadata(result_stack, receipt)
    return {
        "component": derived,
        "stack": result_stack,
        "injected_entities": tuple(fill_entities),
        "injected_polygons": tuple(fill_polygons),
        "receipt": receipt,
    }


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _placement_digest(
    *,
    semantic_id: str,
    lattice_index: tuple[int, int],
    center_um: tuple[float, float],
    footprint_size_um: float,
) -> str:
    payload = {
        "semantic_id": semantic_id,
        "lattice_index": list(lattice_index),
        "center_um": list(center_um),
        "collision_footprint_size_um": footprint_size_um,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audit_authored_indium_sites(
    *,
    build_input: Any,
    cell: Any,
    bump_layer: tuple[int, int],
    under_bump_layer: tuple[int, int],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit authored sites and semantic ownership before either fill mode.

    Layout geometry only supplies footprint mechanics.  Ground ownership, net,
    role and host remain structured adapter records and ambiguous facts fail
    before a candidate can be emitted.
    """
    import gdstk

    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    owners = []
    for semantic_id in _FILL_OWNER_SEMANTIC_IDS:
        entity = entities.get(semantic_id)
        if entity is None:
            raise ValueError(f"indium fill requires structured owner {semantic_id!r}.")
        if (
            entity.role != "metal"
            or entity.part_role != "face_metal"
            or entity.net_id != "Ground"
            or not isinstance(entity.host_void_semantic_id, str)
            or not entity.host_void_semantic_id
            or entity.metadata.get("equipotential_id") != "Ground"
        ):
            raise ValueError(
                f"structured owner {semantic_id!r} lacks exact Ground role/net/equipotential/host facts."
            )
        selected = [
            polygons_by_id[key] for key in entity.polygon_ids if key in polygons_by_id
        ]
        if len(selected) != len(entity.polygon_ids) or not selected:
            raise ValueError(f"structured owner {semantic_id!r} lacks exact polygons.")
        owners.append((entity, _gdstk_polygons(selected)))
    owner_nets = {entity.net_id for entity, _ in owners}
    owner_hosts = {entity.host_void_semantic_id for entity, _ in owners}
    owner_equipotentials = {
        entity.metadata.get("equipotential_id") for entity, _ in owners
    }
    if (
        owner_nets != {"Ground"}
        or owner_equipotentials != {"Ground"}
        or len(owner_hosts) != 1
    ):
        raise ValueError(
            "D0/D1 structured ground owners disagree on exact Ground net/equipotential or host."
        )
    ground = gdstk.boolean(owners[0][1], owners[1][1], "and", precision=1e-9)
    if not ground:
        raise ValueError("D0/D1 structured Ground intersection is empty.")

    owner_ids = tuple(entity.semantic_id for entity, _ in owners)
    owner_net = next(iter(owner_nets))
    owner_host = next(iter(owner_hosts))
    semantic_contract = {
        "owner_semantic_ids": list(owner_ids),
        "net_id": owner_net,
        "equipotential_id": "Ground",
        "host_solution_volume_id": owner_host,
        "owners": [
            {
                "semantic_id": entity.semantic_id,
                "role": entity.role,
                "part_role": entity.part_role,
                "net_id": entity.net_id,
                "equipotential_id": entity.metadata.get("equipotential_id"),
                "host_solution_volume_id": entity.host_void_semantic_id,
            }
            for entity, _ in owners
        ],
        "source_provenance": "public_pdk_indium_ground_fill",
    }

    blockers: list[tuple[str, tuple[Any, ...]]] = []
    for record in spec.get("keepout_records", ()):
        if not isinstance(record, Mapping):
            raise TypeError("public indium keepout records must be mappings.")
        consumers = record.get("consumers", ())
        if isinstance(consumers, str | bytes) or not isinstance(consumers, Sequence):
            raise TypeError("public indium keepout consumers must be a sequence.")
        if "indium_ground_bump" not in consumers:
            continue
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("public indium keepout requires an exact source name.")
        layer = _point2(record.get("layer"), "keepout layer")
        polygons = tuple(_gds_layer_polygons(cell, (int(layer[0]), int(layer[1]))))
        if polygons:
            blockers.append((f"pdk_keepout:{name}", polygons))

    for entity in build_input.entities:
        if entity.material_kind != "conductor":
            continue
        if not isinstance(entity.net_id, str) or not entity.net_id:
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown net semantics for indium audit."
            )
        if not isinstance(entity.part_role, str) or not entity.part_role:
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown part-role semantics for indium audit."
            )
        if (
            not isinstance(entity.host_void_semantic_id, str)
            or not entity.host_void_semantic_id
        ):
            raise ValueError(
                f"conductor {entity.semantic_id!r} has unknown host semantics for indium audit."
            )
        if entity.semantic_id in owner_ids or entity.net_id == owner_net:
            continue
        selected = [
            polygons_by_id[key] for key in entity.polygon_ids if key in polygons_by_id
        ]
        if len(selected) != len(entity.polygon_ids):
            raise ValueError(
                f"conductor {entity.semantic_id!r} polygon records are ambiguous for indium audit."
            )
        polygons = tuple(_gdstk_polygons(selected))
        if polygons:
            blockers.append((f"structured_conductor:{entity.semantic_id}", polygons))

    bump_polygons = _gds_layer_polygons(cell, bump_layer)
    under_polygons = _gds_layer_polygons(cell, under_bump_layer)
    if not bump_polygons and not under_polygons:
        pairs: list[tuple[Any, Any, str, str]] = []
    elif len(bump_polygons) != len(under_polygons):
        raise ValueError("authored indium audit requires equal bump and UBM counts.")
    else:
        pairs = []
        used_under: set[int] = set()
        for bump in bump_polygons:
            matching = [
                index
                for index, under in enumerate(under_polygons)
                if not gdstk.boolean((bump,), (under,), "not", precision=1e-9)
            ]
            if len(matching) != 1 or matching[0] in used_under:
                raise ValueError(
                    "each authored bump requires one exact unique UBM footprint."
                )
            used_under.add(matching[0])
            source_entity_id, source_polygon_id = _authored_bump_source_identity(
                build_input=build_input,
                bump_layer=bump_layer,
                bump=bump,
            )
            pairs.append(
                (bump, under_polygons[matching[0]], source_entity_id, source_polygon_id)
            )
        if len(used_under) != len(under_polygons):
            raise ValueError("each authored UBM must belong to one exact bump site.")

    ordered_pairs = sorted(pairs, key=lambda item: _polygon_center(item[0]))
    authored_sites: list[dict[str, Any]] = []
    for index, (bump, under, source_entity_id, source_polygon_id) in enumerate(
        ordered_pairs
    ):
        site_id = f"AUTHORED_INDIUM_SITE__{index:04d}"
        center = _polygon_center(bump)
        footprint = (under,)
        if gdstk.boolean(footprint, ground, "not", precision=1e-9):
            raise ValueError(f"authored site {site_id!r} is outside D0/D1 Ground.")
        sources = _intersecting_blocker_ids(footprint, blockers)
        if sources:
            raise ValueError(
                f"authored site {site_id!r} intersects blocking sources {list(sources)!r}."
            )
        for other_index, (_, other_under, _, _) in enumerate(ordered_pairs):
            if index == other_index:
                continue
            if gdstk.boolean(footprint, (other_under,), "and", precision=1e-9):
                raise ValueError(
                    f"authored site {site_id!r} overlaps another authored UBM."
                )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "semantic_id": site_id,
                    "center_um": list(center),
                    "bump": _polygon_points(bump),
                    "ubm": _polygon_points(under),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        authored_sites.append(
            {
                "semantic_id": site_id,
                "center_um": list(center),
                "owner_semantic_ids": list(owner_ids),
                "net_id": owner_net,
                "host_solution_volume_id": owner_host,
                "bump_layer": list(bump_layer),
                "under_bump_layer": list(under_bump_layer),
                "geometry_placement_sha256": digest,
                "source_kind": "authored",
                "source_semantic_id": source_entity_id,
                "source_polygon_id": source_polygon_id,
            }
        )
        blockers.append((f"authored_site:{site_id}", footprint))
    return {
        "ground_intersection": tuple(ground),
        "blockers": blockers,
        "authored_sites": authored_sites,
        "semantic_contract": semantic_contract,
    }


def _polygon_center(polygon: Any) -> tuple[float, float]:
    points = _polygon_points(polygon)
    return (
        (min(point[0] for point in points) + max(point[0] for point in points)) / 2,
        (min(point[1] for point in points) + max(point[1] for point in points)) / 2,
    )


def _polygon_points(polygon: Any) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in polygon.points]


def _intersecting_blocker_ids(
    footprint: Sequence[Any], blockers: Sequence[tuple[str, Sequence[Any]]]
) -> tuple[str, ...]:
    import gdstk

    return tuple(
        source_id
        for source_id, polygons in blockers
        if polygons and gdstk.boolean(footprint, polygons, "and", precision=1e-9)
    )


def _point2(value: Any, name: str) -> tuple[float, float]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"{name} must be a two-item numeric sequence.")
    return float(value[0]), float(value[1])


def _spec_layers(spec: Mapping[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    settings = spec.get("canonical_component_settings")
    if not isinstance(settings, Mapping):
        raise TypeError("public indium bump spec settings must be a mapping.")
    bump = _point2(settings.get("indium_bump_layer"), "indium_bump_layer")
    under = _point2(settings.get("under_bump_layer"), "under_bump_layer")
    return (int(bump[0]), int(bump[1])), (int(under[0]), int(under[1]))


def _gdstk_cell(library: Any, name: str) -> Any:
    matches = [cell for cell in library.cells if cell.name == name] if name else []
    if len(matches) == 1:
        return matches[0]
    tops = library.top_level()
    if len(tops) != 1:
        raise ValueError("generated component GDS needs one exact top cell.")
    return tops[0]


def _gds_layer_polygons(cell: Any, layer: tuple[int, int]) -> list[Any]:
    return list(
        cell.get_polygons(layer=layer[0], datatype=layer[1], apply_repetitions=True)
    )


def _gdstk_polygons(polygons: Sequence[Any]) -> list[Any]:
    import gdstk

    result = []
    for polygon in polygons:
        outer = [gdstk.Polygon(polygon.exterior)]
        holes = [gdstk.Polygon(hole) for hole in polygon.holes]
        result.extend(
            outer if not holes else gdstk.boolean(outer, holes, "not", precision=1e-9)
        )
    return result


def _authored_bump_source_identity(
    *,
    build_input: Any,
    bump_layer: tuple[int, int],
    bump: Any,
) -> tuple[str, str]:
    """Return the one structured source record that exactly owns a bump.

    This is a lossless source-record lookup: GDS geometry identifies only the
    already-authored structured polygon, never a net, owner, or host.
    """
    import gdstk

    layer_name = f"{bump_layer[0]}/{bump_layer[1]}"
    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    matches: list[tuple[str, str]] = []
    for entity in build_input.entities:
        if entity.metadata.get("source_kind") != "authored":
            continue
        for polygon_id in entity.polygon_ids:
            polygon = polygons_by_id.get(polygon_id)
            if polygon is None or polygon.layer != layer_name:
                continue
            candidate = tuple(_gdstk_polygons((polygon,)))
            if not gdstk.boolean(
                (bump,), candidate, "not", precision=1e-9
            ) and not gdstk.boolean(candidate, (bump,), "not", precision=1e-9):
                matches.append((entity.semantic_id, polygon_id))
    if len(matches) != 1:
        raise ValueError(
            "authored bump geometry must resolve to exactly one structured source "
            f"record; got {matches!r}."
        )
    return matches[0]


def _coupon_bounds(stack: Mapping[str, Any]) -> Mapping[str, float]:
    regions = stack.get("solution_regions")
    if not isinstance(regions, Mapping) or "D0_SUBSTRATE" not in regions:
        raise ValueError("fill requires D0_SUBSTRATE solution-region bounds.")
    raw = regions["D0_SUBSTRATE"].get("geometry", {}).get("domain_bounds_um")
    if not isinstance(raw, Mapping):
        raise TypeError("D0_SUBSTRATE requires explicit domain_bounds_um.")
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in raw for key in required):
        raise ValueError("D0_SUBSTRATE bounds must define x/y min/max.")
    return {key: float(raw[key]) for key in required}


def _component_with_canonical_bumps(
    component: Any,
    factory: Any,
    settings: Mapping[str, Any],
    accepted: Sequence[tuple[int, int, float, float]],
    pitch: float,
) -> Any:
    import gdsfactory as gf

    derived = gf.Component()
    derived << component
    bump = factory(**dict(settings))
    rows: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for i, j, x, y in accepted:
        rows.setdefault((j, 0), []).append((i, x, y))
    for (j, _), sites in sorted(rows.items()):
        sites.sort()
        start = 0
        while start < len(sites):
            end = start + 1
            while end < len(sites) and sites[end][0] == sites[end - 1][0] + 1:
                end += 1
            run = sites[start:end]
            if len(run) == 1:
                reference = derived << bump
            else:
                reference = derived << gf.components.array(
                    component=bump,
                    columns=len(run),
                    rows=1,
                    column_pitch=pitch,
                    row_pitch=pitch,
                )
            reference.move((run[0][1], run[0][2]))
            start = end
    derived.ports.clear()
    return derived


def _assert_generated_center_selectors(
    component: Any,
    bump_layer: tuple[int, int],
    accepted: Sequence[tuple[int, int, float, float]],
) -> None:
    import gdstk

    with TemporaryDirectory(prefix="scgsim-indium-fill-verify-") as temporary:
        path = Path(temporary) / "derived.gds"
        component.write_gds(path)
        cell = _gdstk_cell(
            gdstk.read_gds(str(path)), str(getattr(component, "name", "") or "")
        )
        polygons = _gds_layer_polygons(cell, bump_layer)
        for i, j, x, y in accepted:
            matches = [
                polygon
                for polygon in polygons
                if _point_in_polygon((x, y), polygon.points)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"generated bump selector {_fill_id(i, j)!r} matched {len(matches)} canonical bump polygons."
                )


def _point_in_polygon(point: tuple[float, float], vertices: Any) -> bool:
    x, y = point
    inside = False
    points = list(vertices)
    for first, second in zip(points, [*points[1:], points[0]], strict=True):
        x1, y1 = first
        x2, y2 = second
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _exclude_generated_bumps_from_authored_layer(
    stack: dict[str, Any], accepted: Sequence[tuple[int, int, float, float]]
) -> None:
    layers = stack.get("layers")
    if not isinstance(layers, list):
        raise TypeError("fill stack needs a mutable layer list.")
    for record in layers:
        if (
            isinstance(record, dict)
            and record.get("semantic_id") == "D0_D1_INDIUM_BUMP"
        ):
            geometry = dict(record.get("geometry", {}))
            geometry.update(
                {
                    "route_ab_fused_selector_mode": True,
                    "exclude_selector_points_um": [(x, y) for _, _, x, y in accepted],
                }
            )
            record["geometry"] = geometry
            record["metadata"] = {
                **dict(record.get("metadata", {})),
                "source_provenance": "authored_indium_bump",
            }
            return
    raise ValueError("fill stack lacks the authored D0_D1_INDIUM_BUMP layer.")


def _fill_semantic_records(
    *,
    accepted: Sequence[tuple[int, int, float, float]],
    bump_layer: tuple[int, int],
    bump_size_um: float,
    result_stack: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
    LayoutPolygonSpec: Any,
    SemanticEntitySpec: Any,
) -> tuple[list[Any], list[Any]]:
    import gdstk

    template = next(
        record
        for record in result_stack["layers"]
        if record.get("semantic_id") == "D0_D1_INDIUM_BUMP"
    )
    net_id = semantic_contract.get("net_id")
    equipotential_id = semantic_contract.get("equipotential_id")
    host_solution_volume_id = semantic_contract.get("host_solution_volume_id")
    owner_semantic_ids = semantic_contract.get("owner_semantic_ids")
    if (
        not isinstance(net_id, str)
        or not net_id
        or equipotential_id != "Ground"
        or not isinstance(host_solution_volume_id, str)
        or not host_solution_volume_id
        or isinstance(owner_semantic_ids, str | bytes)
        or not isinstance(owner_semantic_ids, Sequence)
        or len(owner_semantic_ids) != 2
        or any(
            not isinstance(owner_id, str) or not owner_id
            for owner_id in owner_semantic_ids
        )
    ):
        raise ValueError("indium fill requires an exact audited semantic contract.")
    semantic_group_id = str(
        template.get("metadata", {}).get("semantic_group_id")
        or template.get("semantic_id")
        or ""
    )
    if not semantic_group_id:
        raise ValueError("indium fill requires an exact authored bump family id.")
    entities, polygons = [], []
    for i, j, x, y in accepted:
        semantic_id = _fill_id(i, j)
        polygon_id = f"{semantic_id}__P0000"
        square = gdstk.rectangle(
            (x - bump_size_um / 2, y - bump_size_um / 2),
            (x + bump_size_um / 2, y + bump_size_um / 2),
        )
        polygon = LayoutPolygonSpec(
            polygon_id=polygon_id,
            layer=f"{bump_layer[0]}/{bump_layer[1]}",
            exterior=tuple((float(px), float(py)) for px, py in square.points),
            object_name=semantic_id,
            net_name=net_id,
            metadata={
                "source": "public_pdk_indium_ground_fill",
                "lattice_index": (i, j),
                "center_um": (x, y),
            },
        )
        entities.append(
            SemanticEntitySpec(
                semantic_id=semantic_id,
                role="metal",
                material_id=str(template["material_id"]),
                material_kind="conductor",
                priority=int(template.get("priority", 0)),
                geometry_kind="layout_extrusion",
                part_role="bump_body",
                net_id=net_id,
                polygon_ids=(polygon_id,),
                host_void_semantic_id=host_solution_volume_id,
                route_representations={
                    "A": "cutout_boundary_shell",
                    "B": "cutout_boundary_shell",
                },
                geometry={
                    **{
                        key: value
                        for key, value in dict(template["geometry"]).items()
                        if key
                        not in {
                            "include_selector_points_um",
                            "exclude_selector_points_um",
                        }
                    },
                    "outer_loop": polygon.exterior,
                    "source_provenance": "public_pdk_indium_ground_fill",
                },
                metadata={
                    "semantic_group_id": semantic_group_id,
                    "source_layer_name": str(template["metadata"]["source_layer_name"]),
                    "source_provenance": "public_pdk_indium_ground_fill",
                    "source_kind": "fill",
                    "source_semantic_id": semantic_id,
                    "owner_semantic_ids": tuple(owner_semantic_ids),
                    "equipotential_id": equipotential_id,
                    "lattice_index": (i, j),
                },
            )
        )
        polygons.append(polygon)
    return entities, polygons


def _fill_id(i: int, j: int) -> str:
    return f"D0_D1_INDIUM_GROUND_FILL__I{i:+05d}__J{j:+05d}"


def _set_indium_fill_metadata(
    stack: dict[str, Any], receipt: Mapping[str, Any]
) -> None:
    metadata = dict(stack.get("metadata", {}))
    metadata["indium_ground_bumps"] = {
        "controls": dict(receipt["controls"]),
        "counts": dict(receipt["counts"]),
        "source": receipt["source"],
    }
    stack["metadata"] = metadata


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _materials(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    kinds = {
        "superconductor": "conductor",
        "conductive": "conductor",
        "conductor": "conductor",
        "vacuum": "vacuum",
        "dielectric": "dielectric",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        raw_kind = record.get("material_kind")
        if raw_kind not in kinds:
            continue
        normalized = kinds[raw_kind]
        item: dict[str, Any] = {"kind": normalized}
        if normalized != "conductor":
            permittivity = record.get("relative_permittivity")
            if isinstance(permittivity, (int, float)):
                item["permittivity"] = float(permittivity)
        loss_tangent = record.get("loss_tangent")
        if isinstance(loss_tangent, (int, float)):
            item["loss_tangent"] = float(loss_tangent)
        result[str(name)] = item
    if not result:
        raise ValueError(
            "public PDK material records contain no supported material kinds."
        )
    return result


def _solution(
    *,
    material_id: str,
    z_min_um: float,
    z_max_um: float,
    domain_bounds_um: Mapping[str, float],
    is_airbox: bool = False,
    airbox_side: str | None = None,
) -> dict[str, Any]:
    geometry: dict[str, Any] = {
        "z_min_um": z_min_um,
        "z_max_um": z_max_um,
        "domain_bounds_um": dict(domain_bounds_um),
    }
    record: dict[str, Any] = {
        "role": "solution_region",
        "is_airbox": is_airbox,
        "material_id": material_id,
        "geometry_kind": "domain",
        "geometry": geometry,
    }
    if airbox_side is not None:
        record["airbox_side"] = airbox_side
    return record


def _coupon_domain_bounds(component: Any, padding_um: float) -> dict[str, float]:
    padding = _positive_number(padding_um, "coupon_padding_um")
    bbox = component.dbbox()
    if bbox is None:
        raise ValueError("public Xmon component must have a physical-unit dbbox.")
    return {
        "x_min_um": float(bbox.left) - padding,
        "y_min_um": float(bbox.bottom) - padding,
        "x_max_um": float(bbox.right) + padding,
        "y_max_um": float(bbox.top) + padding,
    }


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _metal_record(
    *,
    layer: tuple[int, int],
    level: Any,
    host: str,
    part_role: str,
    representations: Mapping[str, str],
    source_layer_name: str,
) -> dict[str, Any]:
    return {
        "layer": layer[0],
        "datatype": layer[1],
        "role": "metal",
        "material_id": str(level.material),
        "priority": int(getattr(level, "mesh_order", 0) or 0),
        "part_role": part_role,
        "host_void_semantic_id": host,
        "geometry_kind": "layout_extrusion",
        "geometry": {
            "z_um": _zmin(level),
            "thickness_um": _thickness(level),
            "geometry_source": "gds_polygon",
            "route_ab_fused_selector_mode": True,
        },
        "route_representations": dict(representations),
        "metadata": {
            "source_layer_name": source_layer_name,
            "source_authority": "orpen-sc-pdk public LayerStack",
        },
    }


def _signal_selectors(
    component: Any,
) -> tuple[tuple[str, str, tuple[float, float], str], ...]:
    ports = getattr(component, "ports", None)
    if ports is None:
        raise TypeError("public Xmon component must expose authored ports.")
    result = [("D1_XMON_PAD", "xmon_pad", (0.0, 0.0), "public topology anchor (0, 0)")]
    for index, name in enumerate(("o1", "o2", "o3", "o4"), 1):
        port = ports[name]
        center = tuple(float(value) for value in port.center)
        distance = float(port.width) / 2
        angle = radians(float(port.orientation))
        result.append(
            (
                f"D1_COUPLER_{index}",
                f"coupler_{index}",
                (center[0] - distance * cos(angle), center[1] - distance * sin(angle)),
                f"authored component port {name}",
            )
        )
    return tuple(result)


def _layer_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if hasattr(value, "layer") and hasattr(value, "datatype"):
        return int(value.layer), int(value.datatype)
    candidates = (value, getattr(value, "layer", None))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            layout = candidate.layout
            raw = int(getattr(candidate, "layer", candidate))
            info = layout.get_info(raw)
            return int(info.layer), int(info.datatype)
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def _zmin(level: Any) -> float:
    return float(level.zmin)


def _thickness(level: Any) -> float:
    value = float(level.thickness)
    if value <= 0:
        raise ValueError("public PDK LayerLevel thickness must be positive.")
    return value


__all__ = ["build_kosen2024_flip_chip_xmon_stack"]
