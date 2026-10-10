"""Immutable shared AEDT request records and schema constants."""

from __future__ import annotations


import math

from collections.abc import Mapping

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Literal

from scgsim.aedt.epr.models import LumpedRlc as LumpedRlc


HfssDrivenMode = Literal["terminal", "modal"]

LayerRole = Literal["ground", "signal", "substrate"]

PdkMaterialKind = Literal["vacuum", "dielectric", "superconductor"]

Side = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

SCHEMA_VERSION = "scgsim.aedt.hfss-driven.v1"

EIGENMODE_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode.v1"

EPR_EIGENMODE_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode-epr.v1"

EPR_ANALYSIS_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode-epr-analysis.v1"

EPR_EIGENMODE_SCHEMA_VERSION_V2 = "scgsim.aedt.hfss-eigenmode-epr.v2"

EPR_EIGENMODE_SCHEMA_VERSION_V3 = "scgsim.aedt.hfss-eigenmode-epr.v3"

DRIVEN_GEOMETRY_SCHEMA_VERSION = "scgsim.aedt.hfss-driven-geometry.v1"

EPR_ANALYSIS_SCHEMA_VERSION_V2 = "scgsim.aedt.hfss-eigenmode-epr-analysis.v2"

Q3D_SCHEMA_VERSION = "scgsim.aedt.q3d.v3"

Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION = "scgsim.aedt.q3d-geometry-source.v3"

Q2D_SCHEMA_VERSION = "scgsim.aedt.q2d.v1"

OFFICIAL_PYAEDT_SOURCE_URL = "https://github.com/ansys/pyaedt/tree/v1.3.0"

LOCKED_PYAEDT = "1.3.0"

REQUIRED_AEDT_VERSION = "2024.2"

POINT_COUNT = 20_000

SURFACE_APPROXIMATION_LEVEL = 9


@dataclass(frozen=True)
class AedtResources:
    """Explicit local AEDT solve resources, independent of model geometry."""

    cores: int
    ram_limit_percent: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.cores, bool)
            or not isinstance(self.cores, int)
            or self.cores <= 0
        ):
            raise ValueError("cores must be a positive integer")
        if (
            isinstance(self.ram_limit_percent, bool)
            or not isinstance(self.ram_limit_percent, int)
            or not 1 <= self.ram_limit_percent <= 100
        ):
            raise ValueError("ram_limit_percent must be an integer from 1 to 100")

    def to_payload(self) -> dict[str, int]:
        return {"cores": self.cores, "ram_limit_percent": self.ram_limit_percent}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _project_filename(value: Any, field: str) -> Path:
    project = Path(_text(value, field))
    if project.parent != Path(".") or project.name in {".", ".."}:
        raise ValueError("project_name must be a single project filename")
    if not project.stem:
        raise ValueError("project_name must normalize to a non-empty stem")
    return project


def _project_name_from_payload(value: Any) -> str:
    """Adapt one canonical payload name through constructor normalization."""
    return f"{_project_filename(value, 'project.name').name}.aedt"


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _adaptive_controls(
    maximum_passes: Any,
    minimum_passes: Any,
    minimum_converged_passes: Any,
    percent_refinement: Any,
) -> float:
    for field, value in (
        ("maximum_passes", maximum_passes),
        ("minimum_passes", minimum_passes),
        ("minimum_converged_passes", minimum_converged_passes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        if field != "maximum_passes" and value > maximum_passes:
            raise ValueError(f"{field} must not exceed maximum_passes")
    refinement = _number(percent_refinement, "percent_refinement")
    if not 0 < refinement <= 100:
        raise ValueError("percent_refinement must be in (0, 100]")
    return refinement


@dataclass(frozen=True)
class LayerImport:
    """One numeric GDS layer and its explicit PyAEDT destination layer group."""

    layer: int
    datatype: int
    layer_name: str
    z_min_um: float
    z_max_um: float
    physical_layer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _nonnegative_int(self.layer, "layer"))
        object.__setattr__(
            self, "datatype", _nonnegative_int(self.datatype, "datatype")
        )
        object.__setattr__(self, "layer_name", _text(self.layer_name, "layer_name"))
        low = _number(self.z_min_um, "z_min_um")
        high = _number(self.z_max_um, "z_max_um")
        if high < low:
            raise ValueError(
                "z_max_um must be >= z_min_um; equal values are conductor sheets"
            )
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)

    def to_payload(self) -> dict[str, Any]:
        return {
            **({"physical_layer_id": self.physical_layer_id}
               if self.physical_layer_id is not None else {}),
            "layer": self.layer,
            "datatype": self.datatype,
            "layer_name": self.layer_name,
            "z_min_um": self.z_min_um,
            "z_max_um": self.z_max_um,
        }


@dataclass(frozen=True)
class PdkMaterial:
    """PDK-owned material identity consumed by AEDT without numeric properties."""

    material_id: str
    kind: PdkMaterialKind
    is_superconducting: bool
    library_name: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "material_id", _text(self.material_id, "material_id"))
        if self.kind not in {"vacuum", "dielectric", "superconductor"}:
            raise ValueError(
                "material kind must be vacuum, dielectric, or superconductor"
            )
        if not isinstance(self.is_superconducting, bool):
            raise TypeError("is_superconducting must be boolean")
        if self.is_superconducting != (self.kind == "superconductor"):
            raise ValueError(
                "PDK material kind and is_superconducting must agree exactly"
            )
        library_name = (
            None
            if self.library_name is None
            else _text(self.library_name, "library_name")
        )
        if self.is_superconducting and library_name is not None:
            raise ValueError("superconducting PDK material must not name AEDT material")
        if not self.is_superconducting and library_name is None:
            raise ValueError(
                "non-superconducting PDK material requires AEDT library name"
            )
        object.__setattr__(self, "library_name", library_name)

    def to_payload(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "is_superconducting": self.is_superconducting,
            "library_name": self.library_name,
        }


@dataclass(frozen=True)
class ObjectBinding:
    """Exact object readback and PDK material reference after numeric-layer import."""

    object_name: str
    layer: int
    role: LayerRole
    material_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_name", _text(self.object_name, "object_name"))
        object.__setattr__(self, "layer", _nonnegative_int(self.layer, "binding.layer"))
        if self.role not in {"ground", "signal", "substrate"}:
            raise ValueError("role must be ground, signal, or substrate")
        object.__setattr__(self, "material_id", _text(self.material_id, "material_id"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "object_name": self.object_name,
            "layer": self.layer,
            "role": self.role,
            "material_id": self.material_id,
        }


@dataclass(frozen=True)
class LumpedTerminalPort:
    """Driven Terminal excitation on an explicit source support, not Region."""

    index: int
    name: str
    support_id: str
    signal_entity_ids: tuple[str, ...]
    reference_entity_ids: tuple[str, ...]
    impedance_ohm: float
    renormalize: bool = False
    deembed_um: float = 0.0

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        for key in ("name", "support_id"):
            object.__setattr__(self, key, _text(getattr(self, key), key))
        for key in ("signal_entity_ids", "reference_entity_ids"):
            values = tuple(_text(value, key) for value in getattr(self, key))
            if not values:
                raise ValueError(f"{key} requires explicit conductor Entities")
            object.__setattr__(self, key, values)
        if isinstance(self.impedance_ohm, bool) or not isinstance(self.impedance_ohm, (int, float)):
            raise TypeError("impedance_ohm must be explicitly numeric")
        if self.deembed_um != 0:
            raise NotImplementedError("Lumped Terminal supports deembed_um=0 only")
        if not isinstance(self.renormalize, bool):
            raise TypeError("renormalize must be bool")

    def to_payload(self) -> dict[str, Any]:
        return {"kind": "lumped", "index": self.index, "name": self.name,
                "support_id": self.support_id, "signal_entity_ids": list(self.signal_entity_ids),
                "reference_entity_ids": list(self.reference_entity_ids),
                "impedance_ohm": self.impedance_ohm, "renormalize": self.renormalize,
                "deembed_um": self.deembed_um}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> LumpedTerminalPort:
        record = dict(value)
        if record.pop("kind") != "lumped":
            raise ValueError("lumped terminal kind differs")
        return cls(**record)


@dataclass(frozen=True)
class TerminalPort:
    """Driven Terminal port facts; references are meaningful only in this mode."""

    index: int
    name: str
    side: Side
    reference_objects: tuple[str, ...]
    deembed_um: float = 0.0
    signal_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        object.__setattr__(self, "name", _text(self.name, "port.name"))
        if self.side not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError("port.side is invalid")
        references = tuple(
            _text(value, "reference_object") for value in self.reference_objects
        )
        if not references:
            raise ValueError("terminal port requires explicit reference_objects")
        object.__setattr__(self, "reference_objects", references)
        object.__setattr__(self, "signal_entity_ids", tuple(
            _text(value, "signal_entity_id") for value in self.signal_entity_ids))
        deembed = _number(self.deembed_um, "deembed_um")
        if deembed < 0:
            raise ValueError("deembed_um must be >= 0")
        object.__setattr__(self, "deembed_um", deembed)

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "side": self.side,
            "reference_objects": list(self.reference_objects),
            "deembed_um": self.deembed_um,
            **({"signal_entity_ids": list(self.signal_entity_ids)} if self.signal_entity_ids else {}),
        }


@dataclass(frozen=True)
class ModalPort:
    """Driven Modal port facts; an explicit integration line is required."""

    index: int
    name: str
    side: Side
    integration_line_um: tuple[tuple[float, float, float], tuple[float, float, float]]

    def __post_init__(self) -> None:
        if self.index not in {1, 2}:
            raise ValueError("two-port V1 uses port indices 1 and 2")
        object.__setattr__(self, "name", _text(self.name, "port.name"))
        if self.side not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
            raise ValueError("port.side is invalid")
        if len(self.integration_line_um) != 2 or any(
            len(point) != 3 for point in self.integration_line_um
        ):
            raise ValueError("integration_line_um must be two xyz points")
        object.__setattr__(
            self,
            "integration_line_um",
            tuple(
                tuple(_number(value, "integration_line_um") for value in point)
                for point in self.integration_line_um
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "side": self.side,
            "integration_line_um": [list(point) for point in self.integration_line_um],
        }


def _padding(
    values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError(
            "region_padding_um requires six PyAEDT face values: +X,-X,+Y,-Y,+Z,-Z"
        )
    result = tuple(_number(value, "region_padding_um") for value in values)
    if any(value < 0 for value in result):
        raise ValueError("region_padding_um values must be >= 0")
    return result


def _padding_2d(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("Q2D region_padding_um requires +X,-X,+Y,-Y values")
    result = tuple(_number(value, "region_padding_um") for value in values)
    if any(value < 0 for value in result):
        raise ValueError("region_padding_um values must be >= 0")
    return result


def _normalize_common_spec(spec: Any) -> dict[str, PdkMaterial]:
    if (
        str(spec.aedt_version) != REQUIRED_AEDT_VERSION
        or str(spec.pyaedt_version) != LOCKED_PYAEDT
    ):
        raise ValueError("V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
    project = _project_filename(spec.project_name, "project_name")
    object.__setattr__(spec, "project_name", project.stem)
    object.__setattr__(spec, "design_name", _text(spec.design_name, "design_name"))
    if not isinstance(spec.materials, Mapping):
        raise TypeError("materials must be a PDK material_id mapping")
    materials = dict(spec.materials)
    if (
        not materials
        or any(not isinstance(item, PdkMaterial) for item in materials.values())
        or set(materials) != {item.material_id for item in materials.values()}
    ):
        raise ValueError("materials must map each PDK material_id to its record")
    vacuum_id = _text(spec.vacuum_material_id, "vacuum_material_id")
    vacuum = materials.get(vacuum_id)
    if (
        vacuum is None
        or vacuum.kind != "vacuum"
        or vacuum.is_superconducting
        or vacuum.library_name is None
        or vacuum.library_name.lower() != "vacuum"
    ):
        raise ValueError(
            "vacuum_material_id must reference non-superconducting PDK vacuum library_name 'vacuum'"
        )
    object.__setattr__(spec, "materials", materials)
    object.__setattr__(spec, "vacuum_material_id", vacuum_id)
    return materials


def _normalize_gds_spec(spec: Any) -> tuple[set[str], set[str]]:
    """Normalize the shared PDK/GDS model contract for 3D AEDT families."""
    materials = _normalize_common_spec(spec)
    gds = Path(spec.gds_path)
    if not str(gds) or gds.name in {"", "."}:
        raise ValueError("gds_path must name a GDS file")
    object.__setattr__(spec, "gds_path", gds)
    imports = tuple(spec.layer_imports)
    if (
        not imports
        or len({(item.layer, item.datatype) for item in imports}) != len(imports)
        or len({item.layer_name for item in imports}) != len(imports)
        or any(
            other.layer_name.startswith(item.layer_name)
            for item in imports
            for other in imports
            if item is not other
        )
    ):
        raise ValueError(
            "layer_imports must contain unique non-prefixing numeric and destination layer mappings"
        )
    object.__setattr__(spec, "layer_imports", imports)
    bindings = tuple(spec.object_bindings)
    if (
        not bindings
        or len({item.object_name for item in bindings}) != len(bindings)
        or {item.layer for item in bindings} != {item.layer for item in imports}
        or any(item.material_id not in materials for item in bindings)
    ):
        raise ValueError(
            "object_bindings must be unique and refer to declared import layers"
        )
    object.__setattr__(spec, "object_bindings", bindings)
    grounds = {item.object_name for item in bindings if item.role == "ground"}
    signals = {item.object_name for item in bindings if item.role == "signal"}
    substrates = [item for item in bindings if item.role == "substrate"]
    if not grounds or not signals or not substrates:
        raise ValueError(
            "HFSS model requires explicit signal, ground, and substrate bindings"
        )
    if any(
        not materials[item.material_id].is_superconducting
        for item in bindings
        if item.role in {"signal", "ground"}
    ):
        raise ValueError("signal and ground bindings require PDK superconductors")
    if any(
        materials[item.material_id].is_superconducting
        or materials[item.material_id].kind != "dielectric"
        for item in substrates
    ):
        raise ValueError(
            "substrate bindings require non-superconducting PDK dielectrics"
        )
    return grounds, signals


@dataclass(frozen=True)
class MatrixRunControl:
    """One Q3D/Q2D matrix setup and shared adaptive convergence policy."""

    setup_name: str
    frequency_ghz: float
    maximum_passes: int
    convergence_percent: float = 1.0
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        frequency = _number(self.frequency_ghz, "frequency_ghz")
        if frequency <= 0:
            raise ValueError("matrix frequency_ghz must be > 0")
        refinement = _adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        convergence = _number(self.convergence_percent, "convergence_percent")
        if convergence <= 0:
            raise ValueError("matrix convergence_percent must be > 0")
        object.__setattr__(self, "frequency_ghz", frequency)
        object.__setattr__(self, "convergence_percent", convergence)
        object.__setattr__(self, "percent_refinement", refinement)

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "frequency_ghz": self.frequency_ghz,
            "maximum_passes": self.maximum_passes,
            "convergence_percent": self.convergence_percent,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
        }
