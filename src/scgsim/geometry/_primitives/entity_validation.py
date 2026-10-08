"""Existing input Entity predicates and contact attachment operations shared below source/compiler validation. Preserve their exact errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from scgsim.geometry.models.input import SemanticEntitySpec

_BOOLEAN_AREA_EPS_UM2 = 1e-8


_BOOLEAN_RELATIVE_AREA_EPS = 1e-12


def _resolve_contact_pad_attachment(
    pad: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
    *,
    finite_overlap: Callable[[SemanticEntitySpec, SemanticEntitySpec], bool]
    | None = None,
    defer_overlap: bool = False,
) -> SemanticEntitySpec | None:
    """Resolve attachment, or check declarations before native curved overlap exists."""
    attached_id = pad.attached_face_metal_semantic_id
    candidates = tuple(
        candidate
        for candidate in entities
        if attached_id
        in {
            candidate.semantic_id,
            str(candidate.metadata.get("semantic_group_id", "")),
        }
    )
    if not candidates:
        raise ValueError(
            f"{pad.semantic_id} contact_pad requires attached face metal id"
        )
    if any(candidate.part_role != "face_metal" for candidate in candidates):
        raise ValueError(
            f"{pad.semantic_id} contact_pad attachment must resolve face_metal"
        )
    matches = tuple(
        candidate
        for candidate in candidates
        if _same_entity_z_range(pad, candidate)
        and (
            defer_overlap
            or (finite_overlap or _entities_have_finite_overlap)(pad, candidate)
        )
    )
    if not matches:
        raise ValueError(
            f"{pad.semantic_id} contact_pad requires finite same-z overlap "
            "with attached face_metal"
        )
    if defer_overlap:
        # Actual curved overlap selects the attachment and proves uniqueness
        # later. Here only authored identity, role, Z and Net can be checked.
        if not pad.net_id or not any(
            candidate.net_id == pad.net_id for candidate in matches
        ):
            raise ValueError(
                f"{pad.semantic_id} contact_pad and attached M1 require one equal net"
            )
        return None
    if len(matches) != 1:
        raise ValueError(f"{pad.semantic_id} contact_pad attachment is ambiguous")
    attached = matches[0]
    if not pad.net_id or not attached.net_id or attached.net_id != pad.net_id:
        raise ValueError(
            f"{pad.semantic_id} contact_pad and attached M1 require one equal net"
        )
    return attached


def _same_entity_z_range(first: SemanticEntitySpec, second: SemanticEntitySpec) -> bool:
    first_min, first_max = _entity_z_range(first)
    second_min, second_max = _entity_z_range(second)
    return abs(first_min - second_min) <= 1e-9 and abs(first_max - second_max) <= 1e-9


def _entity_z_range(entity: SemanticEntitySpec) -> tuple[float, float]:
    z_min = float(entity.geometry.get("z_min_um", entity.geometry.get("z_um", 0.0)))
    return z_min, z_min + float(entity.geometry.get("thickness_um", 0.0))


def _entities_have_finite_overlap(
    first: SemanticEntitySpec, second: SemanticEntitySpec
) -> bool:
    import gdstk

    overlap = gdstk.boolean(
        _entity_occupied_region(first),
        _entity_occupied_region(second),
        "and",
        precision=1e-9,
    )
    return any(
        abs(_polygon_area(polygon.points))
        > max(_BOOLEAN_AREA_EPS_UM2, abs(polygon.area()) * _BOOLEAN_RELATIVE_AREA_EPS)
        for polygon in overlap or ()
    )


def _entity_occupied_region(entity: SemanticEntitySpec) -> tuple[Any, ...]:
    import gdstk

    outer_loop = entity.geometry.get("outer_loop")
    if outer_loop is None:
        return ()
    outer = gdstk.Polygon(outer_loop)
    holes = tuple(gdstk.Polygon(loop) for loop in entity.geometry.get("hole_loops", ()))
    if not holes:
        return (outer,)
    return tuple(gdstk.boolean((outer,), holes, "not", precision=1e-9) or ())


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    ring = tuple((float(point[0]), float(point[1])) for point in points)
    return 0.5 * sum(
        start[0] * end[1] - start[1] * end[0]
        for start, end in zip(ring, (*ring[1:], ring[0]), strict=True)
    )


def _is_solution_entity(entity: SemanticEntitySpec) -> bool:
    return entity.role == "solution_region" and entity.material_kind in {
        "vacuum",
        "dielectric",
    }


def _is_vacuum_solution_entity(entity: SemanticEntitySpec) -> bool:
    return _is_solution_entity(entity) and entity.material_kind == "vacuum"


def _requires_route_representation(entity: SemanticEntitySpec) -> bool:
    if _is_solution_entity(entity):
        return False
    return entity.material_kind == "conductor"
