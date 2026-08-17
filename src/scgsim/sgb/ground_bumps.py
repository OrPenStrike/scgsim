"""PDK-driven deterministic ground-bump fill."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _prepare_indium_ground_bump_fill(
    *,
    component: Any,
    stack: Mapping[str, Any],
    fill: bool,
    fill_pitch_um: float,
    fill_clearance_um: float,
) -> dict[str, Any]:
    """Materialize PDK-defined ground-bump fill without mutating its input cell.

    The PDK owns bump polygons, layers, physical origin, and keepout meaning.
    SCGSim only enumerates the explicit lattice and carries the resulting
    per-site records through the existing Route-A/B planner.
    """
    if not isinstance(fill, bool):
        raise TypeError("fill must be a bool.")
    pitch = _positive_number(fill_pitch_um, "fill_pitch_um")
    clearance = _non_negative_number(fill_clearance_um, "fill_clearance_um")
    result_stack = deepcopy(dict(stack))
    layers = result_stack.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, str | bytes):
        raise TypeError("indium ground bumps require a semantic layer sequence.")
    has_authored_bumps = any(
        isinstance(record, Mapping) and record.get("part_role") == "bump_body"
        for record in layers
    )
    if not fill and not has_authored_bumps:
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "rule": {
                "identity": "scgsim.indium_ground_bump_fill.v1",
                "version": 1,
            },
            "source": "scgsim.sgb.ground_bumps._prepare_indium_ground_bump_fill",
            "pdk_spec": None,
            "pdk_spec_sha256": None,
            "controls": {
                "fill": False,
                "fill_pitch_um": pitch,
                "fill_clearance_um": clearance,
            },
            "authored_sites": [],
            "accepted": [],
            "rejected": [],
            "semantic_contract": None,
            "counts": {"authored": 0, "generated": 0, "rejected": 0},
        }
        _set_indium_fill_metadata(result_stack, receipt)
        return {
            "component": component,
            "stack": result_stack,
            "injected_entities": (),
            "injected_polygons": (),
            "receipt": receipt,
        }

    import gdstk

    from scgsim.sgb.adapter import build_gds_stack_geometry_input
    from scgsim.sgb.models import LayoutPolygonSpec, SemanticEntitySpec

    template, spec = _ground_bump_fill_spec(result_stack)
    origin = _point2(spec["lattice_origin_um"], "lattice_origin_um")
    bump_layer = _layer_pair(spec["body_layer"], "body_layer")
    under_bump_layer = _layer_pair(spec["contact_layer"], "contact_layer")
    body_polygon = _relative_polygon(spec["body_polygon_um"], "body_polygon_um")
    contact_polygon = _relative_polygon(
        spec["contact_polygon_um"], "contact_polygon_um"
    )
    template_semantic_id = str(template["semantic_id"])
    fill_prefix = f"{template_semantic_id}_GROUND_FILL"
    geometry_digest = hashlib.sha256(
        json.dumps(
            {
                "body_layer": bump_layer,
                "body_polygon_um": body_polygon,
                "contact_layer": under_bump_layer,
                "contact_polygon_um": contact_polygon,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
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
            semantic_contract=_bump_semantic_contract(template),
        )
        receipt: dict[str, Any] = {
            "schema_version": 2,
            "rule": {
                "identity": "scgsim.indium_ground_bump_fill.v1",
                "version": 1,
            },
            "source": "scgsim.sgb.ground_bumps._prepare_indium_ground_bump_fill",
            "pdk_spec": spec,
            "pdk_spec_sha256": geometry_digest,
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
                candidate = (
                    gdstk.Polygon(_translate_polygon(body_polygon, x, y)),
                    gdstk.Polygon(_translate_polygon(contact_polygon, x, y)),
                )
                expanded = (
                    gdstk.offset(candidate, clearance, join="round", tolerance=1e-6)
                    if clearance
                    else list(candidate)
                )
                candidate_id = _fill_id(fill_prefix, i, j)
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
                                geometry_digest=geometry_digest,
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
                    blockers.append((f"generated_site:{candidate_id}", candidate))

    derived = _component_with_canonical_bumps(component, spec, accepted, pitch)
    _assert_generated_center_selectors(derived, bump_layer, accepted)
    _exclude_generated_bumps_from_authored_layer(
        result_stack, template_semantic_id, accepted
    )
    fill_entities, fill_polygons = _fill_semantic_records(
        accepted=accepted,
        fill_prefix=fill_prefix,
        template_semantic_id=template_semantic_id,
        bump_layer=bump_layer,
        body_polygon_um=body_polygon,
        result_stack=result_stack,
        semantic_contract=receipt["semantic_contract"],
        LayoutPolygonSpec=LayoutPolygonSpec,
        SemanticEntitySpec=SemanticEntitySpec,
    )
    receipt["accepted"] = [
        {
            "semantic_id": _fill_id(fill_prefix, i, j),
            "lattice_index": [i, j],
            "center_um": [x, y],
            "geometry_placement_sha256": _placement_digest(
                semantic_id=_fill_id(fill_prefix, i, j),
                lattice_index=(i, j),
                center_um=(x, y),
                geometry_digest=geometry_digest,
            ),
            "owner_semantic_ids": receipt["semantic_contract"]["owner_semantic_ids"],
            "net_id": receipt["semantic_contract"]["net_id"],
            "equipotential_id": receipt["semantic_contract"]["equipotential_id"],
            "host_solution_volume_id": receipt["semantic_contract"][
                "host_solution_volume_id"
            ],
            "source_kind": "fill",
            "source_semantic_id": _fill_id(fill_prefix, i, j),
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
    geometry_digest: str,
) -> str:
    payload = {
        "semantic_id": semantic_id,
        "lattice_index": list(lattice_index),
        "center_um": list(center_um),
        "pdk_geometry_sha256": geometry_digest,
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
    semantic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit authored sites and semantic ownership before either fill mode.

    Layout geometry only supplies footprint mechanics.  Ground ownership, net,
    role and host remain structured adapter records and ambiguous facts fail
    before a candidate can be emitted.
    """
    import gdstk

    polygons_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    owner_ids = tuple(semantic_contract["owner_semantic_ids"])
    expected_net = semantic_contract["net_id"]
    expected_equipotential = semantic_contract["equipotential_id"]
    expected_host = semantic_contract["host_solution_volume_id"]
    owners = []
    for semantic_id in owner_ids:
        entity = entities.get(semantic_id)
        if entity is None:
            raise ValueError(f"indium fill requires structured owner {semantic_id!r}.")
        if (
            entity.role != "metal"
            or entity.part_role != "face_metal"
            or entity.net_id != expected_net
            or entity.host_void_semantic_id != expected_host
            or entity.metadata.get("equipotential_id") != expected_equipotential
        ):
            raise ValueError(
                f"structured owner {semantic_id!r} disagrees with the authored bump contract."
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
        owner_nets != {expected_net}
        or owner_equipotentials != {expected_equipotential}
        or owner_hosts != {expected_host}
    ):
        raise ValueError(
            "structured ground owners disagree on exact net/equipotential or host."
        )
    ground = gdstk.boolean(owners[0][1], owners[1][1], "and", precision=1e-9)
    if not ground:
        raise ValueError("D0/D1 structured Ground intersection is empty.")

    owner_net = next(iter(owner_nets))
    owner_host = next(iter(owner_hosts))
    resolved_contract = {
        "owner_semantic_ids": list(owner_ids),
        "net_id": owner_net,
        "equipotential_id": expected_equipotential,
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
        "source_provenance": "pdk_ground_bump_fill",
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
        "semantic_contract": resolved_contract,
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


def _ground_bump_fill_spec(
    stack: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    layers = stack.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, str | bytes):
        raise TypeError("ground-bump fill requires a semantic layer sequence.")
    matches = [
        record
        for record in layers
        if isinstance(record, Mapping)
        and isinstance(record.get("metadata"), Mapping)
        and "ground_bump_fill_spec" in record["metadata"]
    ]
    if len(matches) != 1:
        raise ValueError(
            "ground-bump fill requires exactly one PDK ground_bump_fill_spec."
        )
    template = matches[0]
    raw = template["metadata"]["ground_bump_fill_spec"]
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("unsupported PDK ground_bump_fill_spec schema.")
    required = {
        "schema_version",
        "body_layer",
        "body_polygon_um",
        "contact_layer",
        "contact_polygon_um",
        "lattice_origin_um",
        "keepout_records",
    }
    if set(raw) != required:
        raise ValueError("PDK ground_bump_fill_spec fields are not canonical.")
    return template, deepcopy(dict(raw))


def _bump_semantic_contract(template: Mapping[str, Any]) -> dict[str, Any]:
    metadata = template.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("ground-bump template metadata must be a mapping.")
    owners = metadata.get("owner_semantic_ids")
    if (
        isinstance(owners, str | bytes)
        or not isinstance(owners, Sequence)
        or len(owners) != 2
        or any(not isinstance(owner, str) or not owner for owner in owners)
    ):
        raise ValueError("ground-bump template requires exactly two structured owners.")
    net_id = template.get("net_id")
    equipotential_id = metadata.get("equipotential_id")
    host = template.get("host_void_semantic_id")
    if any(
        not isinstance(value, str) or not value
        for value in (net_id, equipotential_id, host)
    ):
        raise ValueError(
            "ground-bump template lacks exact net/equipotential/host facts."
        )
    return {
        "owner_semantic_ids": tuple(owners),
        "net_id": net_id,
        "equipotential_id": equipotential_id,
        "host_solution_volume_id": host,
    }


def _layer_pair(value: Any, name: str) -> tuple[int, int]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise TypeError(f"{name} must be a two-item integer layer pair.")
    return int(value[0]), int(value[1])


def _relative_polygon(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) < 3
    ):
        raise TypeError(f"{name} must contain at least three 2D points.")
    points = tuple(_point2(point, name) for point in value)
    area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, (*points[1:], points[0]), strict=True)
    )
    if not isfinite(area) or abs(area) <= 1e-12:
        raise ValueError(f"{name} must define a finite nondegenerate polygon.")
    return points


def _translate_polygon(
    points: Sequence[tuple[float, float]], x: float, y: float
) -> tuple[tuple[float, float], ...]:
    return tuple((px + x, py + y) for px, py in points)


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
    metadata = stack.get("metadata")
    raw = (
        metadata.get("coupon_domain_bounds_um")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(raw, Mapping):
        raise TypeError("ground-bump fill requires explicit coupon_domain_bounds_um.")
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in raw for key in required):
        raise ValueError("coupon bounds must define x/y min/max.")
    return {key: float(raw[key]) for key in required}


def _component_with_canonical_bumps(
    component: Any,
    spec: Mapping[str, Any],
    accepted: Sequence[tuple[int, int, float, float]],
    pitch: float,
) -> Any:
    import gdsfactory as gf

    derived = gf.Component()
    derived << component
    bump = gf.Component()
    bump.add_polygon(
        _relative_polygon(spec["body_polygon_um"], "body_polygon_um"),
        layer=_layer_pair(spec["body_layer"], "body_layer"),
    )
    bump.add_polygon(
        _relative_polygon(spec["contact_polygon_um"], "contact_polygon_um"),
        layer=_layer_pair(spec["contact_layer"], "contact_layer"),
    )
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
                    f"generated bump at lattice index {(i, j)!r} matched "
                    f"{len(matches)} canonical body polygons."
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
    stack: dict[str, Any],
    template_semantic_id: str,
    accepted: Sequence[tuple[int, int, float, float]],
) -> None:
    layers = stack.get("layers")
    if not isinstance(layers, list):
        raise TypeError("fill stack needs a mutable layer list.")
    for record in layers:
        if (
            isinstance(record, dict)
            and record.get("semantic_id") == template_semantic_id
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
    raise ValueError("fill stack lacks its authored ground-bump layer.")


def _fill_semantic_records(
    *,
    accepted: Sequence[tuple[int, int, float, float]],
    fill_prefix: str,
    template_semantic_id: str,
    bump_layer: tuple[int, int],
    body_polygon_um: Sequence[tuple[float, float]],
    result_stack: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
    LayoutPolygonSpec: Any,
    SemanticEntitySpec: Any,
) -> tuple[list[Any], list[Any]]:
    import gdstk

    template = next(
        record
        for record in result_stack["layers"]
        if record.get("semantic_id") == template_semantic_id
    )
    net_id = semantic_contract.get("net_id")
    equipotential_id = semantic_contract.get("equipotential_id")
    host_solution_volume_id = semantic_contract.get("host_solution_volume_id")
    owner_semantic_ids = semantic_contract.get("owner_semantic_ids")
    if (
        not isinstance(net_id, str)
        or not net_id
        or not isinstance(equipotential_id, str)
        or not equipotential_id
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
        semantic_id = _fill_id(fill_prefix, i, j)
        polygon_id = f"{semantic_id}__P0000"
        body = gdstk.Polygon(_translate_polygon(body_polygon_um, x, y))
        polygon = LayoutPolygonSpec(
            polygon_id=polygon_id,
            layer=f"{bump_layer[0]}/{bump_layer[1]}",
            exterior=tuple((float(px), float(py)) for px, py in body.points),
            object_name=semantic_id,
            net_name=net_id,
            metadata={
                "source": "pdk_ground_bump_fill",
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
                    "source_provenance": "pdk_ground_bump_fill",
                },
                metadata={
                    "semantic_group_id": semantic_group_id,
                    "source_layer_name": str(template["metadata"]["source_layer_name"]),
                    "source_provenance": "pdk_ground_bump_fill",
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


def _fill_id(prefix: str, i: int, j: int) -> str:
    return f"{prefix}__I{i:+05d}__J{j:+05d}"


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


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result
