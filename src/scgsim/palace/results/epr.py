"""Detached Palace EPR values, serialization, and offline reanalysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from scgsim.semantics.epr import film_assumptions, inverse_quality_factor

from .resolve import ResolvedPalaceResult


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return float(value)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


def _freeze(value: Any, *, verified: Any = None) -> Any:
    if verified is not None and value is verified:
        return verified
    if isinstance(value, Mapping):
        prior = verified if isinstance(verified, Mapping) else {}
        return MappingProxyType(
            {
                str(key): _freeze(item, verified=prior.get(str(key)))
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        prior = verified if isinstance(verified, (tuple, list)) else ()
        return tuple(
            _freeze(item, verified=prior[index] if index < len(prior) else None)
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported EPR value {type(value).__name__}")


@dataclass(frozen=True)
class PalaceEprResult:
    """Portable exact-mode snapshot without a native run or mutable Sim handle."""

    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    _verified_source: PalaceEprResult | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("Palace EPR requires non-empty row mappings")
        identities: set[int] = set()
        group_sources: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            mode = row.get("mode")
            if type(mode) is not int or mode <= 0 or mode in identities:
                raise ValueError("Palace EPR modes must be unique positive integers")
            identities.add(mode)
            _number(row.get("frequency_hz"), "frequency_hz", positive=True)
            _number(
                row.get("normalization_energy_j"),
                "normalization_energy_j",
                positive=True,
            )
            for name in ("surface", "bulk", "junction"):
                if not isinstance(row.get(name), (list, tuple)) or any(
                    not isinstance(item, Mapping) for item in row[name]
                ):
                    raise TypeError(f"Palace EPR {name} must contain mappings")
            for surface in row["surface"]:
                for name in (
                    "group_id",
                    "surface_id",
                    "interface_kind",
                    "field_side",
                    "evaluation_kind",
                ):
                    if not isinstance(surface.get(name), str) or not surface[name]:
                        raise ValueError(f"Palace EPR surface {name} is missing")
                if surface["interface_kind"] not in {"MA", "MS", "SA"} or surface[
                    "evaluation_kind"
                ] not in {"unmasked_baseline", "requested_margin"}:
                    raise ValueError("Palace EPR surface identity is invalid")
                if not isinstance(surface.get("members"), Mapping):
                    raise TypeError("Palace EPR source provenance must be a mapping")
                owners = surface.get("owner_semantic_ids")
                if (
                    not isinstance(owners, (list, tuple))
                    or not owners
                    or any(not isinstance(owner, str) or not owner for owner in owners)
                ):
                    raise ValueError("Palace EPR surface owners are invalid")
                _number(surface.get("margin_um"), "surface margin")
                _number(surface.get("original_participation"), "original participation")
                _number(surface.get("participation"), "participation")
                assumption = film_assumptions(
                    surface.get("assumptions"), allow_unknown=True
                )
                original = film_assumptions(
                    surface.get("original_assumptions"), allow_unknown=True
                )
                expected_q = inverse_quality_factor(
                    surface["participation"], assumption["loss_tangent"]
                )
                actual_q = surface.get("inverse_q")
                if (expected_q is None) != (actual_q is None) or (
                    expected_q is not None
                    and not math.isclose(
                        _number(actual_q, "inverse_q"),
                        expected_q,
                        rel_tol=1e-12,
                        abs_tol=0.0,
                    )
                ):
                    raise ValueError(
                        "Palace EPR inverse_q disagrees with participation and loss"
                    )
                source = (
                    surface["surface_id"],
                    surface["interface_kind"],
                    tuple(owners),
                    tuple(sorted(original.items())),
                )
                prior = group_sources.setdefault(surface["group_id"], source)
                if source != prior:
                    raise ValueError(
                        "Palace EPR original group identity has conflicting provenance"
                    )
            for bulk in row["bulk"]:
                if not isinstance(bulk.get("domain_id"), str) or not bulk["domain_id"]:
                    raise ValueError("Palace EPR bulk domain_id is invalid")
                _number(bulk.get("participation"), "bulk participation")
            for port in row["junction"]:
                if not isinstance(port.get("port_name"), str) or not port["port_name"]:
                    raise ValueError("Palace EPR port_name is invalid")
                value = port.get("participation")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Palace EPR signed port participation is invalid")
        if not isinstance(self.provenance, Mapping) or not isinstance(
            self.provenance.get("handoff_id"), str
        ):
            raise ValueError("Palace EPR provenance lacks handoff identity")
        if self._verified_source is not None and not isinstance(
            self._verified_source, PalaceEprResult
        ):
            raise TypeError("verified source must be a PalaceEprResult")
        prior_rows = (
            self._verified_source.rows if self._verified_source is not None else ()
        )
        object.__setattr__(
            self,
            "rows",
            tuple(
                _freeze(
                    row, verified=prior_rows[index] if index < len(prior_rows) else None
                )
                for index, row in enumerate(rows)
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze(
                self.provenance,
                verified=self._verified_source.provenance
                if self._verified_source is not None
                else None,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.palace.epr-result.v1",
            "rows": _plain(self.rows),
            "provenance": _plain(self.provenance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PalaceEprResult:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"schema_version", "rows", "provenance"}
            or payload.get("schema_version") != "scgsim.palace.epr-result.v1"
        ):
            raise ValueError("Palace EPR payload is not canonical")
        return cls(tuple(payload["rows"]), payload["provenance"])


def resolve_epr_result(path: str | Path) -> PalaceEprResult:
    """Load a detached Palace EPR record without reading a solver directory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(source)
    return PalaceEprResult.from_payload(json.loads(source.read_text(encoding="utf-8")))


def epr_result(result: ResolvedPalaceResult) -> PalaceEprResult:
    """Extract compact verified final Eigenmode quantities and source bindings."""

    if not isinstance(result, ResolvedPalaceResult) or result.problem != "Eigenmode":
        raise TypeError("a resolved Palace Eigenmode run is required")
    index = result.provenance.index_map
    config = result.provenance.config
    l0 = _number(config["Model"]["L0"], "Model.L0", positive=True)
    entries = index["entries"]
    surfaces = {
        item["index"]: item
        for item in entries
        if item["section"] == "Boundaries.Postprocessing.Dielectric"
    }
    domains = {
        item["index"]: item
        for item in entries
        if item["section"] == "Domains.Postprocessing.Energy"
    }
    junctions = {
        item["index"]: item
        for item in entries
        if item["section"] == "Boundaries.LumpedPort"
    }
    request = index.get(
        "epr_request",
        {
            "schema_version": "scgsim.palace.epr-selection.v1",
            "surface_interfaces": None,
            "bulk_domain_ids": None,
            "port_names": None,
        },
    )
    if (
        not isinstance(request, Mapping)
        or set(request)
        != {"schema_version", "surface_interfaces", "bulk_domain_ids", "port_names"}
        or request["schema_version"] != "scgsim.palace.epr-selection.v1"
    ):
        raise ValueError("Palace EPR selection payload is invalid")
    available = {
        "surface_interfaces": {
            item["metadata"]["interface_type"] for item in surfaces.values()
        },
        "bulk_domain_ids": {item["entry_name"] for item in domains.values()},
        "port_names": {item["port_name"] for item in junctions.values()},
    }
    selected: dict[str, set[str]] = {}
    for name, configured in available.items():
        values = request[name]
        if values is None:
            selected[name] = configured
        elif (
            not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
            or not set(values) <= configured
        ):
            raise ValueError(f"Palace EPR selection has unknown or invalid {name}")
        else:
            selected[name] = set(values)
    eig = result.tables["eig"].rows
    surface_table = (
        result.tables["surface-Q"].rows
        if surfaces
        else tuple({"m": item["m"]} for item in eig)
    )
    domain_table = result.tables["domain-E"].rows
    port_table = (
        result.tables["port-EPR"].rows
        if junctions
        else tuple({"m": item["m"]} for item in eig)
    )
    masks_requested = any("mask" in entry for entry in surfaces.values())
    mask_q_table = (
        result.tables["surface-mask-Q"].rows if masks_requested else (None,) * len(eig)
    )
    mask_energy_table = (
        result.tables["surface-mask-energy"].rows
        if masks_requested
        else (None,) * len(eig)
    )
    rows: list[dict[str, Any]] = []
    for mode_row, surface_row, domain_row, port_row, mask_q_row, mask_energy_row in zip(
        eig,
        surface_table,
        domain_table,
        port_table,
        mask_q_table,
        mask_energy_table,
        strict=True,
    ):
        mode = int(mode_row["m"])
        if any(
            int(item.get("m", item.get("i"))) != mode
            for item in (surface_row, domain_row, port_row)
        ):
            raise ValueError("Palace EPR mode tables do not align")
        if masks_requested and (
            mask_q_row["m"] != mode or mask_energy_row["m"] != mode
        ):
            raise ValueError("Palace EPR mask tables do not align with the mode")
        normalization = _number(domain_row["E_elec (J)"], "electric energy") + _number(
            domain_row["E_cap (J)"], "capacitive energy"
        )
        normalization = _number(normalization, "normalization energy", positive=True)
        surface_items: list[dict[str, Any]] = []
        for number, entry in sorted(surfaces.items()):
            metadata = entry["metadata"]
            if metadata["interface_type"] not in selected["surface_interfaces"]:
                continue
            spec = entry["epr_spec"]
            native_thickness = _number(
                spec["thickness"], "native film thickness", positive=True
            )
            recorded_l0 = _number(
                spec.get("model_l0_m", l0), "film Model.L0", positive=True
            )
            if recorded_l0 != l0:
                raise ValueError("film Model.L0 disagrees with resolved config")
            physical = _number(
                spec.get("film_thickness_m", native_thickness * l0),
                "SI film thickness",
                positive=True,
            )
            if not math.isclose(physical, native_thickness * l0, rel_tol=1e-12):
                raise ValueError("film SI/native thickness mismatch")
            recorded_loss = spec.get("loss_tangent")
            historical_ambiguous_zero = (
                spec.get("schema_version") != "scgsim.palace.surface-film.v2"
                and recorded_loss == 0
            )
            loss = None if historical_ambiguous_zero else recorded_loss
            assumption = film_assumptions(
                {
                    "film_thickness_m": physical,
                    "film_relative_permittivity": spec["permittivity"],
                    "loss_tangent": loss,
                    "source": spec.get(
                        "source",
                        "historical_ambiguous_zero"
                        if historical_ambiguous_zero
                        else "historical",
                    ),
                    "preset": spec.get("preset"),
                }
            )
            mask = entry.get("mask")
            participation = _number(
                (
                    mask_q_row[f"p_surf_mask[{number}]"]
                    if mask is not None
                    else surface_row[f"p_surf[{number}]"]
                ),
                "surface participation",
            )
            surface_items.append(
                {
                    "group_id": f"palace_surface_{entry.get('baseline_index', number)}",
                    "index": number,
                    "baseline_index": entry.get("baseline_index", number),
                    "owner_semantic_ids": metadata["owner_semantic_ids"],
                    "interface_kind": metadata["interface_type"],
                    "field_side": metadata["face_kind"],
                    "surface_id": metadata["surface_id"],
                    "members": metadata["source_provenance"],
                    "evaluation_kind": "requested_margin"
                    if mask is not None
                    else "unmasked_baseline",
                    "margin_um": float(mask["margin_um"]) if mask is not None else 0.0,
                    "margin_index": mask["margin_index"] if mask is not None else None,
                    "original_participation": participation,
                    "participation": participation,
                    "original_assumptions": assumption,
                    "assumptions": assumption,
                    "recorded_native_assumptions": {
                        "schema_version": spec.get("schema_version"),
                        "thickness_model_l0_units": native_thickness,
                        "model_l0_m": l0,
                        "relative_permittivity": spec["permittivity"],
                        "loss_tangent": recorded_loss,
                        "source": spec.get("source"),
                        "preset": spec.get("preset"),
                    },
                    "loss_provenance": (
                        "historical_ambiguous_zero"
                        if historical_ambiguous_zero
                        else "recorded"
                    ),
                    "inverse_q": inverse_quality_factor(participation, loss),
                }
            )
        bulk = [
            {
                "domain_id": item["entry_name"],
                "index": number,
                "material": item["metadata"]["material"],
                "participation": _number(
                    domain_row[f"p_elec[{number}]"], "bulk participation"
                ),
            }
            for number, item in sorted(domains.items())
            if item["entry_name"] in selected["bulk_domain_ids"]
        ]
        port = [
            {
                "port_name": item["port_name"],
                "owner_semantic_ids": item["metadata"]["owner_semantic_ids"],
                "index": number,
                "participation": float(port_row[f"p[{number}]"]),
                "signed_palace": True,
            }
            for number, item in sorted(junctions.items())
            if item["port_name"] in selected["port_names"]
        ]
        rows.append(
            {
                "mode": mode,
                "frequency_hz": _number(
                    mode_row["Re{f} (GHz)"], "frequency", positive=True
                )
                * 1e9,
                "normalization_energy_j": normalization,
                "surface": surface_items,
                "bulk": bulk,
                "junction": port,
            }
        )
    source_inputs = {"config.json", "metadata/palace_index_map.json"}
    source_outputs = {
        f"results/palace/{name}.csv"
        for name in (
            "eig",
            "domain-E",
            "surface-Q",
            "port-EPR",
            "surface-mask-Q",
            "surface-mask-energy",
        )
    }
    return PalaceEprResult(
        tuple(rows),
        {
            "handoff_id": result.returned_receipt.handoff_id,
            "route": result.route,
            "source": "verified_final_palace_tables",
            "analysis_request": request,
            "surface_scope": "recorded_native_MA_MS_SA; sidewalls_uncomputed_if_unselected",
            "source_reference": {
                "receipt_schema": result.returned_receipt.schema,
                "receipt_timestamp_utc": result.returned_receipt.timestamp_utc,
                "solver_identity": result.returned_receipt.solver_identity,
                "input_files": [
                    item
                    for item in result.returned_receipt.input_hashes
                    if item.get("path") in source_inputs
                ],
                "output_files": [
                    item
                    for item in result.returned_receipt.output_files
                    if item.get("path") in source_outputs
                ],
            },
        },
    )


def reanalyze_epr(
    result: PalaceEprResult,
    *,
    surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> PalaceEprResult:
    """Apply a new whole-group film snapshot to stored native participation."""

    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")
    defaults = (
        {}
        if surface_defaults is None
        else {
            key: film_assumptions(value, partial=True)
            for key, value in surface_defaults.items()
        }
    )
    overrides = (
        {}
        if group_overrides is None
        else {
            key: film_assumptions(value, partial=True)
            for key, value in group_overrides.items()
        }
    )
    if set(defaults) - {"MA", "MS", "SA"}:
        raise ValueError("surface_defaults supports only MA, MS, and SA")
    known = {item["group_id"] for row in result.rows for item in row["surface"]}
    if set(overrides) - known:
        raise ValueError("group_overrides contains an unknown group")
    rows: list[dict[str, Any]] = []
    for row in result.rows:
        surfaces: list[dict[str, Any]] = []
        for item in row["surface"]:
            original = film_assumptions(item["original_assumptions"])
            current = film_assumptions(item["assumptions"])
            interface_update = defaults.get(item["interface_kind"], {})
            group_update = overrides.get(item["group_id"], {})
            applied = {**interface_update, **group_update}
            candidate = {**current, **applied}
            if any(
                name in applied
                for name in (
                    "film_thickness_m",
                    "film_relative_permittivity",
                    "loss_tangent",
                )
            ):
                if "source" not in applied:
                    candidate["source"] = (
                        "offline_group_override"
                        if group_update
                        else "offline_interface_default"
                    )
                if "preset" not in applied:
                    candidate["preset"] = None
            updated = film_assumptions(candidate)
            ratio = updated["film_thickness_m"] / original["film_thickness_m"]
            if item["interface_kind"] in {"MA", "MS"}:
                ratio *= (
                    original["film_relative_permittivity"]
                    / updated["film_relative_permittivity"]
                )
            elif (
                updated["film_relative_permittivity"]
                != original["film_relative_permittivity"]
            ):
                raise ValueError(
                    "SA epsilon change requires stored normal/tangential integrals"
                )
            participation = item["original_participation"] * ratio
            surfaces.append(
                {
                    **item,
                    "assumptions": updated,
                    **(
                        {
                            "assumption_update": {
                                "scope": "group_override"
                                if group_update
                                else "interface_default",
                                "fields": sorted(applied),
                                "original_source": original["source"],
                                "original_preset": original["preset"],
                            }
                        }
                        if applied
                        else {}
                    ),
                    "participation": participation,
                    "inverse_q": inverse_quality_factor(
                        participation, updated["loss_tangent"]
                    ),
                }
            )
        rows.append({**row, "surface": surfaces})
    original_hash = result.provenance.get("original_result_sha256")
    if original_hash is None:
        original_hash = hashlib.sha256(
            json.dumps(
                result.to_payload(), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    return PalaceEprResult(
        tuple(rows),
        {
            **result.provenance,
            "original_result_sha256": original_hash,
            "film_analysis": {
                "schema_version": "scgsim.epr.film-analysis.v1",
                "surface_defaults": defaults,
                "group_overrides": overrides,
                "status": "derived_offline",
            },
        },
        _verified_source=result,
    )


def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
    """Plot one mode with separate surface, bulk, and signed Palace port axes."""

    if not isinstance(result, PalaceEprResult):
        raise TypeError("result must be PalaceEprResult")
    from ..presentation.epr import show_epr as render

    return render(result, mode=mode)
