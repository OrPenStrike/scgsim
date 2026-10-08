"""Pure EPR source-selection rules shared by request and analysis contracts."""

from __future__ import annotations


import math

import hashlib

import json

from collections.abc import Mapping

from typing import Any

from scgsim.aedt.epr.models import (
    EprAnalysisRequest,
    PreparedPlanarGeometry,
    detached,
    surface_evaluations,
)


def _finite(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _exact_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def surface_integral_groups(
    prepared: PreparedPlanarGeometry,
    request: EprAnalysisRequest | None,
) -> tuple[dict[str, Any], ...]:
    """Partition physical bindings only where one exact film coefficient applies."""

    selected = (
        {item.contribution_id for item in prepared.contributions}
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    specs = {item.contribution_id: item for item in prepared.contributions}
    domains = {
        item["semantic_id"]: item for item in prepared.source["solution_regions"]
    }
    conductor_ids = {item["semantic_id"] for item in prepared.source["conductors"]}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    evaluation_policy = prepared.source.get("surface_evaluation_policy")
    for binding in prepared.surface_bindings:
        record = _exact_mapping(binding["contribution"], "surface provenance")
        contribution_id = record["contribution_id"]
        if contribution_id not in selected:
            continue
        spec = specs[contribution_id]
        if spec.interface_kind == "MM":
            continue
        source_owners = tuple(record["source_owner_ids"])
        if spec.interface_kind == "SA":
            owner_id = binding["substrate_domain_id"]
            if (
                owner_id not in source_owners
                or owner_id not in domains
                or prepared.source["materials"][domains[owner_id]["material_id"]][
                    "kind"
                ]
                != "dielectric"
            ):
                raise ValueError("SA group lacks its physical dielectric owner")
        else:
            owner_id = binding["owner_semantic_id"]
            if owner_id not in source_owners or owner_id not in conductor_ids:
                raise ValueError("metal group lacks its physical conductor owner")
        material = (
            binding["substrate_material"]
            if spec.interface_kind in {"MS", "SA"}
            else binding["effective_material"]
        )
        epsilon_s = _finite(material["permittivity"], "group substrate permittivity")
        if epsilon_s <= 0.0:
            raise ValueError("group substrate permittivity must be positive")
        signature = (
            owner_id,
            spec.interface_kind,
            spec.film_thickness_m,
            spec.film_relative_permittivity,
            epsilon_s,
        )
        member = {
            "binding_id": binding["binding_id"],
            "contribution_id": contribution_id,
            "source_polygon_id": spec.source_polygon_id,
            "source_owner_ids": list(source_owners),
            "field_side": spec.field_side,
            "effective_domain_id": binding["effective_domain_id"],
            "effective_material_id": binding["effective_material_id"],
            "substrate_domain_id": binding["substrate_domain_id"],
            "geometry_ref": detached(binding["geometry_ref"]),
        }
        for evaluation_kind, margin in surface_evaluations(
            spec.margins_um, policy=evaluation_policy
        ):
            evaluation_key = (
                (evaluation_kind, float(margin))
                if evaluation_policy is not None
                else (float(margin),)
            )
            groups.setdefault((*signature, *evaluation_key), []).append(member)
    result: list[dict[str, Any]] = []
    for signature, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        member_assumptions = {
            (
                specs[member["contribution_id"]].loss_tangent,
                specs[member["contribution_id"]].source,
                specs[member["contribution_id"]].preset,
            )
            for member in members
        }
        if len(member_assumptions) != 1:
            raise ValueError(
                "one physical surface group has conflicting film provenance"
            )
        owner_id, interface_kind, thickness, film_epsilon, substrate_epsilon = (
            signature[:5]
        )
        if evaluation_policy is None:
            evaluation_kind = None
            margin = signature[5]
        else:
            evaluation_kind, margin = signature[5:]
        members.sort(key=lambda item: (item["binding_id"], item["contribution_id"]))
        if len({item["binding_id"] for item in members}) != len(members):
            raise ValueError("surface group repeats a physical binding")
        group_identity = {
            "owner_id": owner_id,
            "interface_kind": interface_kind,
            "film_thickness_m": thickness,
            "film_relative_permittivity": film_epsilon,
            "substrate_relative_permittivity": substrate_epsilon,
            "margin_um": margin,
        }
        if evaluation_kind is not None:
            group_identity["evaluation_kind"] = evaluation_kind
        group_id = (
            "surface_group_"
            + hashlib.sha256(
                json.dumps(
                    group_identity, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()[:24]
        )
        result.append({"group_id": group_id, **group_identity, "members": members})
    if len({item["group_id"] for item in result}) != len(result):
        raise ValueError("surface group identities are not unique")
    return tuple(result)
