"""Immutable records for the converging HFSS Eigenmode EPR lifecycle.

These records contain source intent, bound native evidence, and detached result
data.  Native application handles never enter this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

Route = Literal["A", "B"]
InterfaceKind = Literal["MA", "MS", "SA", "MM"]
FieldSide = Literal["top", "bottom", "sidewall"]


def surface_evaluations(
    margins_um: Sequence[float], *, policy: str | None
) -> tuple[tuple[str, float], ...]:
    """Keep the automatic baseline distinct from each requested mask."""

    requested = tuple(("requested_margin", float(value)) for value in margins_um)
    if policy is None:
        return requested  # Historical records had no automatic baseline.
    if policy != "unmasked_plus_requested.v1":
        raise ValueError("unsupported surface evaluation policy")
    return (("unmasked_baseline", 0.0), *requested)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum:g}")
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("EPR mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"EPR data contains unsupported {type(value).__name__}")


def detached(value: Any) -> Any:
    """Return recursively detached plain containers."""
    if isinstance(value, Mapping):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [detached(item) for item in value]
    if isinstance(value, list):
        return [detached(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        detached(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PlanarJunction:
    """One source-linked linearized Lumped RLC junction sheet."""

    junction_id: str
    source_polygon_id: str
    terminal_a_net: str
    terminal_b_net: str
    direction_xy: tuple[float, float]
    width_um: float
    inductance_h: float
    capacitance_f: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "junction_id",
            "source_polygon_id",
            "terminal_a_net",
            "terminal_b_net",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.terminal_a_net == self.terminal_b_net:
            raise ValueError("junction terminal nets must be distinct")
        if len(self.direction_xy) != 2:
            raise ValueError("junction direction_xy must contain two values")
        dx = _number(self.direction_xy[0], "direction_xy[0]", minimum=-math.inf)
        dy = _number(self.direction_xy[1], "direction_xy[1]", minimum=-math.inf)
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            raise ValueError("junction direction_xy must be nonzero")
        object.__setattr__(self, "direction_xy", (dx / norm, dy / norm))
        object.__setattr__(self, "width_um", _number(self.width_um, "width_um"))
        if self.width_um <= 0.0:
            raise ValueError("junction width_um must be > 0")
        object.__setattr__(
            self, "inductance_h", _number(self.inductance_h, "inductance_h")
        )
        object.__setattr__(
            self, "capacitance_f", _number(self.capacitance_f, "capacitance_f")
        )
        if self.inductance_h <= 0.0 or self.capacitance_f < 0.0:
            raise ValueError("junction L must be > 0 and C must be >= 0")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "source_polygon_id": self.source_polygon_id,
            "terminal_a_net": self.terminal_a_net,
            "terminal_b_net": self.terminal_b_net,
            "direction_xy": list(self.direction_xy),
            "width_um": self.width_um,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "metadata": detached(self.metadata),
        }

    @classmethod
    def from_payload(cls, value: Any) -> PlanarJunction:
        if not isinstance(value, Mapping):
            raise TypeError("junction payload must be a mapping")
        expected = {
            "junction_id",
            "source_polygon_id",
            "terminal_a_net",
            "terminal_b_net",
            "direction_xy",
            "width_um",
            "inductance_h",
            "capacitance_f",
            "metadata",
        }
        if set(value) != expected:
            raise ValueError("junction payload members are not canonical")
        return cls(**dict(value))


@dataclass(frozen=True)
class SurfaceEprSpec:
    """One requested physical-side contribution and its alternative margins."""

    contribution_id: str
    source_polygon_id: str
    interface_kind: InterfaceKind
    field_side: FieldSide
    margins_um: tuple[float, ...] = (0.0,)
    film_thickness_m: float | None = None
    film_relative_permittivity: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("contribution_id", "source_polygon_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.interface_kind not in {"MA", "MS", "SA", "MM"}:
            raise ValueError(f"unsupported EPR interface kind {self.interface_kind!r}")
        if self.field_side not in {"top", "bottom", "sidewall"}:
            raise ValueError(f"unsupported EPR field side {self.field_side!r}")
        margins = tuple(_number(value, "margin_um") for value in self.margins_um)
        if not margins or len(margins) != len(set(margins)):
            raise ValueError("margins_um must be non-empty and unique")
        object.__setattr__(self, "margins_um", margins)
        optional = (
            "film_thickness_m",
            "film_relative_permittivity",
        )
        for name in optional:
            value = getattr(self, name)
            if value is not None:
                number = _number(value, name)
                if number <= 0.0:
                    raise ValueError(f"{name} must be > 0 when provided")
                object.__setattr__(self, name, number)
        if self.interface_kind in {"MA", "MS", "SA"} and (
            self.film_thickness_m is None
            or self.film_relative_permittivity is None
        ):
            raise ValueError(f"{self.interface_kind} requires film thickness and permittivity")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "source_polygon_id": self.source_polygon_id,
            "interface_kind": self.interface_kind,
            "field_side": self.field_side,
            "margins_um": list(self.margins_um),
            "film_thickness_m": self.film_thickness_m,
            "film_relative_permittivity": self.film_relative_permittivity,
            "metadata": detached(self.metadata),
        }

    @classmethod
    def from_payload(cls, value: Any) -> SurfaceEprSpec:
        if not isinstance(value, Mapping):
            raise TypeError("surface EPR payload must be a mapping")
        expected = {
            "contribution_id",
            "source_polygon_id",
            "interface_kind",
            "field_side",
            "margins_um",
            "film_thickness_m",
            "film_relative_permittivity",
            "metadata",
        }
        if set(value) != expected:
            raise ValueError("surface EPR payload members are not canonical")
        return cls(**dict(value))


@dataclass(frozen=True)
class EprAnalysisRequest:
    """Optional analysis/cache selection independent of model construction."""

    mode_indices: tuple[int, ...] | None = None
    surface_contribution_ids: tuple[str, ...] | None = None
    junction_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("mode_indices", "surface_contribution_ids", "junction_ids"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{name} must be a sequence")
            values = tuple(value)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
            if name == "mode_indices":
                if any(type(item) is not int or item <= 0 for item in values):
                    raise ValueError("mode_indices must contain positive integers")
            else:
                values = tuple(_text(item, name) for item in values)
            object.__setattr__(self, name, values)
    def to_payload(self) -> dict[str, Any]:
        return {
            "mode_indices": (
                None if self.mode_indices is None else list(self.mode_indices)
            ),
            "surface_contribution_ids": (
                None
                if self.surface_contribution_ids is None
                else list(self.surface_contribution_ids)
            ),
            "junction_ids": (
                None if self.junction_ids is None else list(self.junction_ids)
            ),
        }

    @classmethod
    def from_payload(cls, value: Any) -> EprAnalysisRequest:
        if not isinstance(value, Mapping):
            raise TypeError("EPR analysis request must be a mapping")
        expected = {
            "mode_indices",
            "surface_contribution_ids",
            "junction_ids",
        }
        if set(value) != expected:
            raise ValueError("EPR analysis request members are not canonical")
        return cls(**dict(value))


@dataclass(frozen=True)
class PreparedPlanarGeometry:
    """Detached source request before any native AEDT object exists."""

    route: Route
    source: Mapping[str, Any]
    junctions: tuple[PlanarJunction, ...]
    contribution_catalog: tuple[Mapping[str, Any], ...]
    contributions: tuple[SurfaceEprSpec, ...]
    surface_bindings: tuple[Mapping[str, Any], ...]
    model_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.route not in {"A", "B"}:
            raise ValueError("EPR supports only planar Route A or Route B")
        junctions = tuple(self.junctions)
        catalog = tuple(self.contribution_catalog)
        contributions = tuple(self.contributions)
        bindings = tuple(self.surface_bindings)
        if any(not isinstance(item, PlanarJunction) for item in junctions):
            raise TypeError("junctions must contain PlanarJunction records")
        if any(not isinstance(item, SurfaceEprSpec) for item in contributions):
            raise TypeError("contributions must contain SurfaceEprSpec records")
        if any(not isinstance(item, Mapping) for item in catalog):
            raise TypeError("contribution_catalog must contain mappings")
        if any(not isinstance(item, Mapping) for item in bindings):
            raise TypeError("surface_bindings must contain mappings")
        object.__setattr__(self, "junctions", junctions)
        object.__setattr__(
            self, "contribution_catalog", tuple(_freeze(item) for item in catalog)
        )
        object.__setattr__(self, "contributions", contributions)
        source = _freeze(self.source)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "surface_bindings",
            tuple(_freeze(item) for item in bindings),
        )
        expected_model = canonical_sha256(
            {
                "route": self.route,
                "source": source,
                "junctions": [item.to_payload() for item in self.junctions],
            }
        )
        if self.model_sha256 != expected_model:
            raise ValueError("prepared planar model digest is inconsistent")
        expected = canonical_sha256(
            {
                "route": self.route,
                "source": source,
                "junctions": [item.to_payload() for item in self.junctions],
                "contribution_catalog": self.contribution_catalog,
                "contributions": [item.to_payload() for item in self.contributions],
                "surface_bindings": self.surface_bindings,
            }
        )
        if self.source_sha256 != expected:
            raise ValueError("prepared planar source digest is inconsistent")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.aedt.epr-planar.v1",
            "route": self.route,
            "source": detached(self.source),
            "junctions": [item.to_payload() for item in self.junctions],
            "contribution_catalog": [
                detached(item) for item in self.contribution_catalog
            ],
            "contributions": [item.to_payload() for item in self.contributions],
            "surface_bindings": [detached(item) for item in self.surface_bindings],
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_payload(cls, value: Any) -> PreparedPlanarGeometry:
        if not isinstance(value, Mapping):
            raise TypeError("prepared planar payload must be a mapping")
        expected = {
            "schema_version",
            "route",
            "source",
            "junctions",
            "contribution_catalog",
            "contributions",
            "surface_bindings",
            "model_sha256",
            "source_sha256",
        }
        if set(value) != expected or value.get("schema_version") != "scgsim.aedt.epr-planar.v1":
            raise ValueError("prepared planar payload is not canonical")
        return cls(
            route=value["route"],
            source=value["source"],
            junctions=tuple(PlanarJunction.from_payload(item) for item in value["junctions"]),
            contribution_catalog=tuple(value["contribution_catalog"]),
            contributions=tuple(
                SurfaceEprSpec.from_payload(item) for item in value["contributions"]
            ),
            surface_bindings=tuple(value["surface_bindings"]),
            model_sha256=value["model_sha256"],
            source_sha256=value["source_sha256"],
        )


@dataclass(frozen=True)
class SavedSolution:
    """Verified immutable saved project and complete result-directory inventory."""

    root: Path
    project_path: Path
    result_path: Path
    members: tuple[Mapping[str, Any], ...]
    content_sha256: str
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        project = Path(self.project_path).resolve()
        result = Path(self.result_path).resolve()
        if not project.is_relative_to(root) or not result.is_relative_to(root):
            raise ValueError("saved solution members must be contained by root")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "project_path", project)
        object.__setattr__(self, "result_path", result)
        if isinstance(self.members, (str, bytes)):
            raise TypeError("saved solution members must be a sequence of mappings")
        members = tuple(self.members)
        if any(not isinstance(item, Mapping) for item in members):
            raise TypeError("saved solution members must contain mappings")
        object.__setattr__(self, "members", tuple(_freeze(item) for item in members))
        if not isinstance(self.content_sha256, str) or len(self.content_sha256) != 64:
            raise ValueError("saved solution content_sha256 is invalid")
        identity = _freeze(self.identity)
        expected_identity = {
            "model_source_sha256",
            "project_name",
            "design_name",
            "setup_name",
            "physical_variation",
            "solver_last_completed_pass",
            "saved_fields_pass",
            "saved_fields",
        }
        if not isinstance(identity, Mapping) or set(identity) != expected_identity:
            raise ValueError("saved solution identity members are not canonical")
        for name in expected_identity - {
            "solver_last_completed_pass",
            "saved_fields_pass",
            "saved_fields",
        }:
            _text(identity[name], f"saved solution identity.{name}")
        for name in ("solver_last_completed_pass", "saved_fields_pass"):
            if type(identity[name]) is not int or identity[name] <= 0:
                raise ValueError(f"saved solution {name} must be positive")
        if identity["saved_fields"] is not True:
            raise ValueError("saved solution must explicitly attest saved fields")
        object.__setattr__(self, "identity", identity)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.aedt.saved-solution.v1",
            "project": self.project_path.relative_to(self.root).as_posix(),
            "results": self.result_path.relative_to(self.root).as_posix(),
            "members": [detached(item) for item in self.members],
            "content_sha256": self.content_sha256,
            "identity": detached(self.identity),
        }

    @classmethod
    def from_payload(cls, root: str | Path, value: Any) -> SavedSolution:
        if not isinstance(value, Mapping):
            raise TypeError("saved solution payload must be a mapping")
        expected = {
            "schema_version",
            "project",
            "results",
            "members",
            "content_sha256",
            "identity",
        }
        if set(value) != expected or value.get("schema_version") != "scgsim.aedt.saved-solution.v1":
            raise ValueError("saved solution payload is not canonical")
        base = Path(root)
        project = _contained_relative_path(value["project"], "saved solution project")
        results = _contained_relative_path(value["results"], "saved solution results")
        return cls(
            root=base,
            project_path=base / project,
            result_path=base / results,
            members=tuple(value["members"]),
            content_sha256=value["content_sha256"],
            identity=value["identity"],
        )


@dataclass(frozen=True)
class EprResult:
    """Detached partial-history or final saved-field EPR result."""

    result_kind: Literal["adaptive_history", "saved_field"]
    setup_name: str
    rows: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.result_kind not in {"adaptive_history", "saved_field"}:
            raise ValueError("unsupported EPR result kind")
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        if isinstance(self.rows, (str, bytes)):
            raise TypeError("EPR result rows must be a sequence of mappings")
        rows = tuple(self.rows)
        if any(not isinstance(row, Mapping) for row in rows):
            raise TypeError("EPR result rows must contain mappings")
        frozen_rows = tuple(_freeze(row) for row in rows)
        identities: set[tuple[int, int | None]] = set()
        complete_members = {
            "schema_version",
            "frequency_hz",
            "electric_energy_j",
            "magnetic_energy_j",
            "junction_inductive_energy_j",
            "magnetic_energy_balance_j",
            "junction_capacitive_energy_j",
            "normalization_energy_j",
            "electric_domains",
            "surface_contributions",
            "junctions",
        }
        for row in frozen_rows:
            mode = row.get("mode")
            if type(mode) is not int or mode <= 0:
                raise ValueError("EPR result row mode must be a positive integer")
            status = row.get("status")
            if status not in {"complete", "partial"}:
                raise ValueError("EPR result row status must be complete or partial")
            if "raw_integrals" in row and not isinstance(
                row["raw_integrals"], Mapping
            ):
                raise TypeError("EPR result row raw_integrals must be a mapping")
            if "raw_integral_evidence" in row:
                raw_evidence = row["raw_integral_evidence"]
                if isinstance(raw_evidence, (str, bytes)) or not isinstance(
                    raw_evidence, Sequence
                ) or any(not isinstance(item, Mapping) for item in raw_evidence):
                    raise TypeError(
                        "EPR result row raw_integral_evidence must contain mappings"
                    )
            native_pass: int | None = None
            if self.result_kind == "adaptive_history":
                native_pass = row.get("native_pass")
                if type(native_pass) is not int or native_pass <= 0:
                    raise ValueError(
                        "adaptive EPR result row native_pass must be a positive integer"
                    )
            elif "native_pass" in row:
                raise ValueError("saved-field EPR rows cannot claim an adaptive pass")
            identity = (mode, native_pass)
            if identity in identities:
                raise ValueError("EPR result rows repeat a mode/pass identity")
            identities.add(identity)
            if status == "complete":
                if not complete_members.issubset(row) or row.get(
                    "schema_version"
                ) != "scgsim.aedt.epr-mode-energy.v1":
                    raise ValueError("complete EPR result row lacks canonical energy members")
                for name in (
                    "frequency_hz",
                    "electric_energy_j",
                    "magnetic_energy_j",
                    "junction_inductive_energy_j",
                    "magnetic_energy_balance_j",
                    "junction_capacitive_energy_j",
                    "normalization_energy_j",
                ):
                    number = _number(row[name], f"EPR result row {name}")
                    if name in {"frequency_hz", "normalization_energy_j"} and number <= 0:
                        raise ValueError(f"complete EPR result row {name} must be > 0")
                for name in (
                    "electric_domains",
                    "surface_contributions",
                    "junctions",
                ):
                    value = row[name]
                    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                        raise TypeError(f"complete EPR result row {name} must be a sequence")
                    if any(not isinstance(item, Mapping) for item in value):
                        raise TypeError(
                            f"complete EPR result row {name} must contain mappings"
                        )
                for surface in row["surface_contributions"]:
                    _number(
                        surface.get("participation"),
                        "surface contribution participation",
                    )
                for junction in row["junctions"]:
                    _number(
                        junction.get("inductive_participation"),
                        "junction inductive participation",
                    )
                    _number(
                        junction.get("capacitive_participation"),
                        "junction capacitive participation",
                    )
        provenance = _freeze(self.provenance)
        if not isinstance(provenance, Mapping):
            raise TypeError("EPR result provenance must be a mapping")
        for name in ("model_source_sha256", "analysis_source_sha256"):
            value = provenance.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"EPR result provenance {name} is invalid")
        requested_modes = provenance.get("requested_modes")
        if (
            not isinstance(requested_modes, tuple)
            or not requested_modes
            or any(type(item) is not int or item <= 0 for item in requested_modes)
            or len(set(requested_modes)) != len(requested_modes)
        ):
            raise ValueError("EPR result provenance requested_modes is invalid")
        object.__setattr__(self, "rows", frozen_rows)
        object.__setattr__(self, "provenance", provenance)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "scgsim.aedt.epr-result.v1",
            "result_kind": self.result_kind,
            "setup_name": self.setup_name,
            "rows": [detached(row) for row in self.rows],
            "provenance": detached(self.provenance),
        }

    @classmethod
    def from_payload(cls, value: Any) -> EprResult:
        if not isinstance(value, Mapping):
            raise TypeError("EPR result payload must be a mapping")
        expected = {
            "schema_version",
            "result_kind",
            "setup_name",
            "rows",
            "provenance",
        }
        if set(value) != expected or value.get("schema_version") != "scgsim.aedt.epr-result.v1":
            raise ValueError("EPR result payload is not canonical")
        return cls(
            result_kind=value["result_kind"],
            setup_name=value["setup_name"],
            rows=tuple(value["rows"]),
            provenance=value["provenance"],
        )


def _contained_relative_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path


__all__ = [
    "EprAnalysisRequest",
    "EprResult",
    "PlanarJunction",
    "PreparedPlanarGeometry",
    "SavedSolution",
    "SurfaceEprSpec",
]
