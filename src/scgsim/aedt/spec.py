"""Explicit, fail-closed input contracts for the AEDT runtime families."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Literal

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
        if isinstance(self.cores, bool) or not isinstance(self.cores, int) or self.cores <= 0:
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
class FrequencySweepSpec:
    """One HFSS Fast sweep with a caller-authored output count."""

    start_ghz: float
    stop_ghz: float
    points: int = POINT_COUNT

    def __post_init__(self) -> None:
        start = _number(self.start_ghz, "start_ghz")
        stop = _number(self.stop_ghz, "stop_ghz")
        if start <= 0 or stop <= start:
            raise ValueError("frequency sweep requires 0 < start_ghz < stop_ghz")
        if isinstance(self.points, bool) or not isinstance(self.points, Number):
            raise ValueError("points must be an exact whole-number numeric value")
        try:
            native_points = int(self.points)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("points must be an exact whole-number numeric value") from exc
        if self.points != native_points:
            raise ValueError("points must be an exact whole-number numeric value")
        object.__setattr__(self, "start_ghz", start)
        object.__setattr__(self, "stop_ghz", stop)

    def to_payload(self) -> dict[str, Any]:
        return {
            "start_ghz": self.start_ghz,
            "stop_ghz": self.stop_ghz,
            # Preserve authored count representation in historical payload identity.
            "points": self.points,
        }


@dataclass(frozen=True)
class HfssRunControl:
    """One Driven setup, adaptive convergence policy, and Fast sweep."""

    setup_name: str
    sweep_name: str
    sweep: FrequencySweepSpec
    maximum_passes: int = 99
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0
    maximum_delta_s: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        object.__setattr__(self, "sweep_name", _text(self.sweep_name, "sweep_name"))
        refinement = _adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        delta = _number(self.maximum_delta_s, "maximum_delta_s")
        if delta <= 0:
            raise ValueError("maximum_delta_s must be > 0")
        object.__setattr__(self, "percent_refinement", refinement)
        object.__setattr__(self, "maximum_delta_s", delta)

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "sweep_name": self.sweep_name,
            "sweep": self.sweep.to_payload(),
            "maximum_passes": self.maximum_passes,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
            "maximum_delta_s": self.maximum_delta_s,
        }


@dataclass(frozen=True)
class LayerImport:
    """One numeric GDS layer and its explicit PyAEDT destination layer group."""

    layer: int
    datatype: int
    layer_name: str
    z_min_um: float
    z_max_um: float

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
class TerminalPort:
    """Driven Terminal port facts; references are meaningful only in this mode."""

    index: int
    name: str
    side: Side
    reference_objects: tuple[str, ...]
    deembed_um: float = 0.0

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


@dataclass(frozen=True)
class LengthMeshSpec:
    """The explicit uniform-CPW exception to the ordinary ground mesh rule."""

    signal_objects: tuple[str, ...]
    ground_objects: tuple[str, ...]
    maximum_length_um: float
    uniform_cpw_mtl: Literal[True] = True

    def __post_init__(self) -> None:
        signals = tuple(_text(value, "signal_object") for value in self.signal_objects)
        grounds = tuple(_text(value, "ground_object") for value in self.ground_objects)
        if not signals or not grounds:
            raise ValueError(
                "length mesh requires explicit signal_objects and ground_objects"
            )
        maximum = _number(self.maximum_length_um, "maximum_length_um")
        if maximum <= 0:
            raise ValueError("maximum_length_um must be > 0")
        if self.uniform_cpw_mtl is not True:
            raise ValueError("length mesh requires uniform_cpw_mtl=True")
        object.__setattr__(self, "signal_objects", signals)
        object.__setattr__(self, "ground_objects", grounds)
        object.__setattr__(self, "maximum_length_um", maximum)

    def to_payload(self) -> dict[str, Any]:
        return {
            "uniform_cpw_mtl": True,
            "signal_objects": list(self.signal_objects),
            "ground_objects": list(self.ground_objects),
            "maximum_length_um": self.maximum_length_um,
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
    return result  # type: ignore[return-value]


def _padding_2d(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("Q2D region_padding_um requires +X,-X,+Y,-Y values")
    result = tuple(_number(value, "region_padding_um") for value in values)
    if any(value < 0 for value in result):
        raise ValueError("region_padding_um values must be >= 0")
    return result  # type: ignore[return-value]


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
        or len({item.layer for item in imports}) != len(imports)
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
class HfssDrivenSpec:
    """The complete two-port, one-mode, one-setup public CPW handoff spec."""

    mode: HfssDrivenMode
    gds_path: Path | str
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    ports: tuple[TerminalPort, TerminalPort] | tuple[ModalPort, ModalPort]
    run_control: HfssRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    def __post_init__(self) -> None:
        if self.mode not in {"terminal", "modal"}:
            raise ValueError("mode must be terminal or modal")
        grounds, signals = _normalize_gds_spec(self)
        if len(self.ports) != 2 or tuple(port.index for port in self.ports) != (1, 2):
            raise ValueError("V1 requires exactly ordered ports 1 and 2")
        if len({port.name for port in self.ports}) != 2 or {
            port.side for port in self.ports
        } != {"-X", "+X"}:
            raise ValueError("V1 requires unique ports on distinct -X and +X faces")
        if self.mode == "terminal" and not all(
            isinstance(port, TerminalPort) for port in self.ports
        ):
            raise ValueError("terminal mode requires TerminalPort entries")
        if self.mode == "modal" and not all(
            isinstance(port, ModalPort) for port in self.ports
        ):
            raise ValueError("modal mode requires ModalPort entries")
        if (
            self.mode == "terminal"
            and all(isinstance(port, TerminalPort) for port in self.ports)
            and any(
                not set(port.reference_objects).issubset(grounds) for port in self.ports
            )
        ):
            raise ValueError("terminal references must name declared ground objects")
        if (
            self.mode == "terminal"
            and all(isinstance(port, TerminalPort) for port in self.ports)
            and self.ports[0].reference_objects != self.ports[1].reference_objects
        ):
            raise ValueError(
                "terminal ports must share one ordered global reference conductor tuple"
            )
        padding = _padding(self.region_padding_um)
        # PyAEDT create_region consumes this native order unchanged.
        side_index = {"+X": 0, "-X": 1, "+Y": 2, "-Y": 3, "+Z": 4, "-Z": 5}
        if any(padding[side_index[port.side]] != 0 for port in self.ports):
            raise ValueError(
                "region padding must be exactly zero on each declared port side"
            )
        object.__setattr__(self, "region_padding_um", padding)
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "gds": {"path": self.gds_path.as_posix()},
            "layer_imports": [item.to_payload() for item in self.layer_imports],
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "ports": [item.to_payload() for item in self.ports],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None
    ) -> HfssDrivenSpec:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported HFSS driven schema")
        mode = _text(payload.get("mode"), "mode")
        gds = Path(_text(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        imports = tuple(
            LayerImport(**item) for item in payload.get("layer_imports", ())
        )
        bindings = tuple(
            ObjectBinding(**item) for item in payload.get("object_bindings", ())
        )
        raw_materials = payload.get("materials")
        if not isinstance(raw_materials, dict):
            raise TypeError("materials must be a JSON object")
        materials: dict[str, PdkMaterial] = {}
        for material_id, item in raw_materials.items():
            if not isinstance(material_id, str) or not isinstance(item, dict):
                raise TypeError(
                    "materials must map string material_id values to objects"
                )
            materials[material_id] = PdkMaterial(**item)
        port_type = TerminalPort if mode == "terminal" else ModalPort
        ports = tuple(port_type(**item) for item in payload.get("ports", ()))
        run = payload.get("run_control", {})
        sweep = FrequencySweepSpec(**run.get("sweep", {}))
        length = payload.get("length_mesh")
        return cls(
            mode=mode,  # type: ignore[arg-type]
            gds_path=gds,
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            layer_imports=imports,
            object_bindings=bindings,
            ports=ports,  # type: ignore[arg-type]
            run_control=HfssRunControl(
                _text(run.get("setup_name"), "setup_name"),
                _text(run.get("sweep_name"), "sweep_name"),
                sweep,
                maximum_passes=run.get("maximum_passes", 99),
                minimum_passes=run.get("minimum_passes", 1),
                minimum_converged_passes=run.get("minimum_converged_passes", 1),
                percent_refinement=run.get("percent_refinement", 30.0),
                maximum_delta_s=run.get("maximum_delta_s", 0.02),
            ),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            length_mesh=LengthMeshSpec(**length) if length is not None else None,
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )


@dataclass(frozen=True)
class EigenmodeRunControl:
    """One explicit HFSS Eigenmode adaptive setup."""

    setup_name: str
    minimum_frequency_ghz: float
    num_modes: int
    maximum_passes: int
    maximum_delta_frequency_percent: float
    minimum_passes: int = 1
    minimum_converged_passes: int = 1
    percent_refinement: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        minimum = _number(self.minimum_frequency_ghz, "minimum_frequency_ghz")
        delta = _number(
            self.maximum_delta_frequency_percent,
            "maximum_delta_frequency_percent",
        )
        if minimum <= 0 or delta <= 0:
            raise ValueError("Eigenmode frequency and convergence percent must be > 0")
        if not isinstance(self.num_modes, int) or self.num_modes <= 0:
            raise ValueError("num_modes must be a positive integer")
        refinement = _adaptive_controls(
            self.maximum_passes,
            self.minimum_passes,
            self.minimum_converged_passes,
            self.percent_refinement,
        )
        object.__setattr__(self, "minimum_frequency_ghz", minimum)
        object.__setattr__(self, "maximum_delta_frequency_percent", delta)
        object.__setattr__(self, "percent_refinement", refinement)

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "minimum_frequency_ghz": self.minimum_frequency_ghz,
            "num_modes": self.num_modes,
            "maximum_passes": self.maximum_passes,
            "maximum_delta_frequency_percent": self.maximum_delta_frequency_percent,
            "minimum_passes": self.minimum_passes,
            "minimum_converged_passes": self.minimum_converged_passes,
            "percent_refinement": self.percent_refinement,
        }


@dataclass(frozen=True)
class HfssEigenmodeSpec:
    """One port-free HFSS Eigenmode model with explicit native setup controls."""

    gds_path: Path | str
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    run_control: EigenmodeRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["eigenmode"]:
        return "eigenmode"

    def __post_init__(self) -> None:
        grounds, signals = _normalize_gds_spec(self)
        object.__setattr__(self, "region_padding_um", _padding(self.region_padding_um))
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EIGENMODE_SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "gds": {"path": self.gds_path.as_posix()},
            "layer_imports": [item.to_payload() for item in self.layer_imports],
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None
    ) -> HfssEigenmodeSpec:
        if payload.get("schema_version") != EIGENMODE_SCHEMA_VERSION:
            raise ValueError("unsupported HFSS Eigenmode schema")
        gds = Path(_text(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        raw_materials = payload.get("materials")
        if not isinstance(raw_materials, dict):
            raise TypeError("materials must be a JSON object")
        materials = {
            material_id: PdkMaterial(**item)
            for material_id, item in raw_materials.items()
        }
        run = payload.get("run_control")
        if not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object")
        length = payload.get("length_mesh")
        return cls(
            gds_path=gds,
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            layer_imports=tuple(
                LayerImport(**item) for item in payload.get("layer_imports", ())
            ),
            object_bindings=tuple(
                ObjectBinding(**item) for item in payload.get("object_bindings", ())
            ),
            run_control=EigenmodeRunControl(**run),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            length_mesh=LengthMeshSpec(**length) if length is not None else None,
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )


@dataclass(frozen=True)
class HfssEprSpec:
    """One body-first HFSS Eigenmode request with embedded planar authority."""

    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl
    epr_request: Any = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    _legacy_payload: bool = False
    expression_convergence: Any = None

    @property
    def mode(self) -> Literal["eigenmode"]:
        return "eigenmode"

    def __post_init__(self) -> None:
        from ._epr_models import (
            EprAnalysisRequest,
            ExpressionCacheConvergence,
            PreparedPlanarGeometry,
        )

        if (
            str(self.aedt_version) != REQUIRED_AEDT_VERSION
            or str(self.pyaedt_version) != LOCKED_PYAEDT
        ):
            raise ValueError("EPR V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
        project = _project_filename(self.project_name, "project_name")
        object.__setattr__(self, "project_name", project.stem)
        object.__setattr__(self, "design_name", _text(self.design_name, "design_name"))
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if not isinstance(self.run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if self.epr_request is not None and not isinstance(
            self.epr_request, EprAnalysisRequest
        ):
            raise TypeError("epr_request must be EprAnalysisRequest or None")
        if self.expression_convergence is not None and not isinstance(
            self.expression_convergence, ExpressionCacheConvergence
        ):
            raise TypeError(
                "expression_convergence must be ExpressionCacheConvergence or None"
            )
        if self._legacy_payload and self.expression_convergence is not None:
            raise ValueError("legacy EPR payload cannot add expression convergence")
        _validate_epr_selection(self)

    def to_payload(self) -> dict[str, Any]:
        schema_version = (
            EPR_EIGENMODE_SCHEMA_VERSION
            if self._legacy_payload
            else (
                EPR_EIGENMODE_SCHEMA_VERSION_V3
                if self.expression_convergence is not None
                else EPR_EIGENMODE_SCHEMA_VERSION_V2
            )
        )
        return {
            "schema_version": schema_version,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": self.run_control.to_payload(),
            "geometry": self.geometry.to_payload(),
            "epr": (
                None if self.epr_request is None else self.epr_request.to_payload()
            ),
            **(
                {"expression_convergence": self.expression_convergence.to_payload()}
                if self.expression_convergence is not None
                else {}
            ),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HfssEprSpec:
        from ._epr_models import (
            EprAnalysisRequest,
            ExpressionCacheConvergence,
            PreparedPlanarGeometry,
        )

        schema = payload.get("schema_version")
        if schema not in {
            EPR_EIGENMODE_SCHEMA_VERSION,
            EPR_EIGENMODE_SCHEMA_VERSION_V2,
            EPR_EIGENMODE_SCHEMA_VERSION_V3,
        }:
            raise ValueError("unsupported HFSS EPR schema")
        expected = {
            "schema_version",
            "mode",
            "aedt",
            "pyaedt",
            "project",
            "run_control",
            "geometry",
            "epr",
        }
        if schema == EPR_EIGENMODE_SCHEMA_VERSION_V3:
            expected.add("expression_convergence")
            if payload.get("expression_convergence") is None:
                raise ValueError(
                    "HFSS EPR V3 requires an expression convergence configuration"
                )
        if set(payload) != expected or payload.get("mode") != "eigenmode":
            raise ValueError("HFSS EPR payload members are not canonical")
        run = payload.get("run_control")
        if not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise TypeError("project must be a JSON object")
        return cls(
            project_name=_project_name_from_payload(project.get("name")),
            design_name=_text(project.get("design"), "project.design"),
            geometry=PreparedPlanarGeometry.from_payload(payload.get("geometry")),
            run_control=EigenmodeRunControl(**run),
            epr_request=(
                None
                if payload.get("epr") is None
                else EprAnalysisRequest.from_payload(payload.get("epr"))
            ),
            expression_convergence=(
                None
                if schema != EPR_EIGENMODE_SCHEMA_VERSION_V3
                else ExpressionCacheConvergence.from_payload(
                    payload.get("expression_convergence")
                )
            ),
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
            _legacy_payload=schema == EPR_EIGENMODE_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class HfssEprAnalysisSpec:
    """One saved-copy-only Eigenmode EPR analysis request."""

    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl
    epr_request: Any
    saved_solution: Any
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    _legacy_payload: bool = False

    @property
    def mode(self) -> Literal["eigenmode"]:
        return "eigenmode"

    def __post_init__(self) -> None:
        from ._epr_models import EprAnalysisRequest, PreparedPlanarGeometry, SavedSolution

        if (
            str(self.aedt_version) != REQUIRED_AEDT_VERSION
            or str(self.pyaedt_version) != LOCKED_PYAEDT
        ):
            raise ValueError("EPR V1 requires AEDT 2024.2 and PyAEDT 1.3.0")
        project = _project_filename(self.project_name, "project_name")
        object.__setattr__(self, "project_name", project.stem)
        object.__setattr__(self, "design_name", _text(self.design_name, "design_name"))
        if not isinstance(self.geometry, PreparedPlanarGeometry):
            raise TypeError("geometry must be PreparedPlanarGeometry")
        if not isinstance(self.run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if not isinstance(self.saved_solution, SavedSolution):
            raise TypeError("saved_solution must be SavedSolution")
        if not isinstance(self.epr_request, EprAnalysisRequest):
            raise TypeError("epr_request must be EprAnalysisRequest")
        _validate_epr_selection(self)
        if self.saved_solution.project_path.stem != self.project_name:
            raise ValueError("saved solution project does not match project_name")
        identity = self.saved_solution.identity
        expected = {
            "model_source_sha256": self.geometry.model_sha256,
            "project_name": self.project_name,
            "design_name": self.design_name,
            "setup_name": self.run_control.setup_name,
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError("saved solution identity does not match the analysis model")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                EPR_ANALYSIS_SCHEMA_VERSION
                if self._legacy_payload else EPR_ANALYSIS_SCHEMA_VERSION_V2
            ),
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "run_control": self.run_control.to_payload(),
            "geometry": self.geometry.to_payload(),
            "epr": self.epr_request.to_payload(),
            "saved_solution": self.saved_solution.to_payload(),
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None
    ) -> HfssEprAnalysisSpec:
        from ._epr_models import EprAnalysisRequest, PreparedPlanarGeometry, SavedSolution

        schema = payload.get("schema_version")
        if schema not in {EPR_ANALYSIS_SCHEMA_VERSION, EPR_ANALYSIS_SCHEMA_VERSION_V2}:
            raise ValueError("unsupported HFSS EPR analysis schema")
        if base_dir is None:
            raise ValueError("HFSS EPR analysis requires a bound base directory")
        expected = {
            "schema_version",
            "mode",
            "aedt",
            "pyaedt",
            "project",
            "run_control",
            "geometry",
            "epr",
            "saved_solution",
        }
        if set(payload) != expected or payload.get("mode") != "eigenmode":
            raise ValueError("HFSS EPR analysis payload members are not canonical")
        project = payload.get("project")
        run = payload.get("run_control")
        if not isinstance(project, dict) or not isinstance(run, dict):
            raise TypeError("project and run_control must be JSON objects")
        return cls(
            project_name=_project_name_from_payload(project.get("name")),
            design_name=_text(project.get("design"), "project.design"),
            geometry=PreparedPlanarGeometry.from_payload(payload.get("geometry")),
            run_control=EigenmodeRunControl(**run),
            epr_request=EprAnalysisRequest.from_payload(payload.get("epr")),
            saved_solution=SavedSolution.from_payload(
                base_dir / "saved", payload.get("saved_solution")
            ),
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"),
                "pyaedt.locked_version",
            ),
            _legacy_payload=schema == EPR_ANALYSIS_SCHEMA_VERSION,
        )


def _validate_epr_selection(spec: HfssEprSpec | HfssEprAnalysisSpec) -> None:
    from ._epr_models import NormalizedSurfaceEprTotal
    from ._epr_results import surface_integral_groups

    request = spec.epr_request
    if request is None:
        if getattr(spec, "expression_convergence", None) is not None:
            raise ValueError("expression convergence requires an EPR analysis request")
        return
    if any(item.field_side == "sidewall" for item in spec.geometry.contributions) or any(
        item["contribution"]["side"] == "sidewall"
        for item in spec.geometry.surface_bindings
    ):
        raise ValueError(
            "sidewall Surface-EPR is excluded; reprepare a horizontal-only "
            "handoff before EPR execution"
        )
    modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    if not modes or any(mode > spec.run_control.num_modes for mode in modes):
        raise ValueError("EPR mode_indices must select configured Eigenmodes")
    surface_ids = {item.contribution_id for item in spec.geometry.contributions}
    selected_surfaces = (
        surface_ids
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    if not selected_surfaces <= surface_ids:
        raise ValueError("EPR request selects an unknown surface contribution")
    junction_ids = {item.junction_id for item in spec.geometry.junctions}
    selected_junctions = (
        junction_ids if request.junction_ids is None else set(request.junction_ids)
    )
    if not selected_junctions <= junction_ids:
        raise ValueError("EPR request selects an unknown junction")
    source_domains = {
        item["semantic_id"] for item in spec.geometry.source["solution_regions"]
    }
    native_region = spec.geometry.source.get("native_region")
    if native_region is not None:
        source_domains.difference_update(native_region["logical_vacuum_ids"])
        source_domains.add("Region")
    if request.bulk_domain_ids is not None and not set(request.bulk_domain_ids) <= source_domains:
        raise ValueError("EPR request selects an unknown bulk domain")
    groups = surface_integral_groups(spec.geometry, request)
    convergence = getattr(spec, "expression_convergence", None)
    if convergence is None:
        return
    if convergence.mode not in modes:
        raise ValueError("expression convergence mode must be selected for EPR")
    target = convergence.target
    if isinstance(target, NormalizedSurfaceEprTotal):
        selected_groups = [
            group
            for group in groups
            if group["interface_kind"] == target.interface_kind
            and group.get("evaluation_kind", "requested_margin")
            == target.evaluation_kind
            and float(group["margin_um"]) == target.margin_um
        ]
        if not selected_groups:
            raise ValueError(
                "normalized surface convergence target selects no prepared owner groups"
            )


HfssSpec = HfssDrivenSpec | HfssEigenmodeSpec | HfssEprSpec | HfssEprAnalysisSpec


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


@dataclass(frozen=True)
class Q3dNetSpec:
    """One exact connected Q3D net and optional signal source/sink."""

    name: str
    net_type: Literal["Signal", "Ground"]
    object_names: tuple[str, ...]
    source_object: str | None = None
    source_side: Side | None = None
    sink_object: str | None = None
    sink_side: Side | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "net.name"))
        if self.net_type not in {"Signal", "Ground"}:
            raise ValueError("Q3D net_type must be Signal or Ground")
        objects = tuple(_text(value, "net.object_name") for value in self.object_names)
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Q3D net object_names must be nonempty and unique")
        object.__setattr__(self, "object_names", objects)
        terminal_values = (
            self.source_object,
            self.source_side,
            self.sink_object,
            self.sink_side,
        )
        if self.net_type == "Ground":
            if any(value is not None for value in terminal_values):
                raise ValueError("Q3D Ground nets must not define source or sink")
            return
        if all(value is None for value in terminal_values):
            return
        if any(value is None for value in terminal_values):
            raise ValueError(
                "Q3D Signal terminals must be all absent or a complete source/sink pair"
            )
        source = _text(self.source_object, "net.source_object")
        sink = _text(self.sink_object, "net.sink_object")
        if source not in objects or sink not in objects:
            raise ValueError("Q3D source and sink objects must belong to their net")
        if self.source_side not in {
            "+X",
            "-X",
            "+Y",
            "-Y",
            "+Z",
            "-Z",
        } or self.sink_side not in {
            "+X",
            "-X",
            "+Y",
            "-Y",
            "+Z",
            "-Z",
        }:
            raise ValueError("Q3D source and sink sides are invalid")
        object.__setattr__(self, "source_object", source)
        object.__setattr__(self, "sink_object", sink)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "net_type": self.net_type,
            "object_names": list(self.object_names),
            "source_object": self.source_object,
            "source_side": self.source_side,
            "sink_object": self.sink_object,
            "sink_side": self.sink_side,
        }


@dataclass(frozen=True)
class Q3dBodySpec:
    """One normalized connected planar body, including holes, swept along +Z.

    Body identity is independent of final Net ownership. Optional source fields
    are absent for explicitly authored source-free bodies, never inferred.
    """

    body_id: str
    exterior_um: tuple[tuple[float, float], ...]
    holes_um: tuple[tuple[tuple[float, float], ...], ...]
    z_min_um: float
    z_max_um: float
    material_id: str
    physical_role: LayerRole
    net_id: str | None
    source_entity_id: str | None = None
    source_polygon_id: str | None = None
    source_occurrence_path: str | None = None
    source_local_entity_id: str | None = None
    source_level: str | None = None
    source_layer_datatype: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for field in ("body_id", "material_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        if self.physical_role not in {"signal", "ground", "substrate"}:
            raise ValueError("Q3D body physical_role is invalid")
        for field in (
            "net_id",
            "source_entity_id",
            "source_polygon_id",
            "source_occurrence_path",
            "source_local_entity_id",
            "source_level",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _text(value, field))

        def ring(value: Any) -> tuple[tuple[float, float], ...]:
            points = tuple(tuple(point) for point in value)
            if len(points) < 3 or any(len(point) != 2 for point in points):
                raise ValueError("Q3D body rings require at least three XY points")
            return tuple((float(x), float(y)) for x, y in points)

        object.__setattr__(self, "exterior_um", ring(self.exterior_um))
        object.__setattr__(
            self, "holes_um", tuple(ring(hole) for hole in self.holes_um)
        )
        low, high = (
            _number(self.z_min_um, "z_min_um"),
            _number(self.z_max_um, "z_max_um"),
        )
        if high <= low:
            raise ValueError("Q3D bodies require positive thickness")
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)
        pair = self.source_layer_datatype
        if pair is not None:
            if len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in pair
            ):
                raise ValueError(
                    "source_layer_datatype must be a nonnegative integer pair"
                )
            object.__setattr__(self, "source_layer_datatype", tuple(pair))

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exterior_um"] = [list(point) for point in self.exterior_um]
        payload["holes_um"] = [
            [list(point) for point in hole] for hole in self.holes_um
        ]
        if self.source_layer_datatype is not None:
            payload["source_layer_datatype"] = list(self.source_layer_datatype)
        return payload


def _q3d_geometry_source(
    value: Mapping[str, Any],
    *,
    bodies: tuple[Q3dBodySpec, ...],
    nets: tuple[Q3dNetSpec, ...],
) -> dict[str, Any]:
    """Bind canonical source attachments and normalized body declarations."""
    source = dict(value)
    expected_keys = {
        "schema_version",
        "source_gds_sha256",
        "source_geometry_sha256",
        "source_stack_sha256",
        "source_dbu_um",
        "files",
        "physical_ground_nets",
        "bodies",
    }
    if set(source) != expected_keys:
        raise ValueError("Q3D geometry_source fields are invalid")
    if source["schema_version"] != Q3D_GEOMETRY_SOURCE_SCHEMA_VERSION:
        raise ValueError("Q3D geometry_source schema is unsupported")
    for field in (
        "source_gds_sha256",
        "source_geometry_sha256",
        "source_stack_sha256",
    ):
        digest = source[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Q3D geometry_source {field} is invalid")
    files = source["files"]
    if not isinstance(files, Mapping) or set(files) != {
        "canonical_gds",
        "stack",
        "trace",
    }:
        raise ValueError("Q3D geometry_source files are invalid")
    copied_files: dict[str, dict[str, str]] = {}
    for key, reference in files.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError(f"Q3D geometry_source file reference is invalid: {key}")
        path = reference["path"]
        digest = reference["sha256"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"Q3D geometry_source file path is invalid: {key}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Q3D geometry_source file hash is invalid: {key}")
        copied_files[key] = {"path": path, "sha256": digest}
    if copied_files["canonical_gds"]["sha256"] != source["source_gds_sha256"]:
        raise ValueError("Q3D source GDS digest differs from its file reference")
    if copied_files["stack"]["sha256"] != source["source_stack_sha256"]:
        raise ValueError("Q3D source stack digest differs from its file reference")

    physical_ground_nets = source["physical_ground_nets"]
    net_by_name = {net.name: net for net in nets}
    if (
        not isinstance(physical_ground_nets, list)
        or any(not isinstance(name, str) or not name for name in physical_ground_nets)
        or len(set(physical_ground_nets)) != len(physical_ground_nets)
        or not set(physical_ground_nets) <= set(net_by_name)
    ):
        raise ValueError("Q3D geometry_source physical_ground_nets are invalid")

    dbu = _number(source["source_dbu_um"], "geometry_source.source_dbu_um")
    if dbu <= 0:
        raise ValueError("source_dbu_um must be positive")
    if source["bodies"] != [body.to_payload() for body in bodies]:
        raise ValueError("Q3D geometry_source bodies differ from the declared bodies")
    if any(
        body.physical_role != "ground"
        for body in bodies
        if body.net_id in physical_ground_nets
    ):
        raise ValueError("Q3D physical ground bodies must retain the ground role")
    source["files"] = copied_files
    source["bodies"] = [body.to_payload() for body in bodies]
    source["physical_ground_nets"] = list(physical_ground_nets)
    return source


@dataclass(frozen=True)
class Q3dSpec:
    """One body-backed Q3D capacitance and optional AC R/L request (v3 only)."""

    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    bodies: tuple[Q3dBodySpec, ...]
    nets: tuple[Q3dNetSpec, ...]
    run_control: MatrixRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    solve_ac_rl: bool = True
    grounded_region_net: str | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    schema_version: Literal["scgsim.aedt.q3d.v3"] = Q3D_SCHEMA_VERSION
    geometry_source: Mapping[str, Any] | None = None

    @property
    def mode(self) -> Literal["q3d"]:
        return "q3d"

    def __post_init__(self) -> None:
        if self.schema_version != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        materials = _normalize_common_spec(self)
        bodies = tuple(self.bodies)
        if not bodies or any(not isinstance(body, Q3dBodySpec) for body in bodies):
            raise TypeError("Q3D bodies must contain Q3dBodySpec records")
        if len({body.body_id for body in bodies}) != len(bodies):
            raise ValueError("Q3D body identities must be unique")
        for body in bodies:
            material = materials[body.material_id]
            if body.physical_role == "substrate":
                if material.kind != "dielectric" or body.net_id is not None:
                    raise ValueError(
                        "Q3D substrate bodies require dielectric material and no Net"
                    )
            elif not material.is_superconducting or body.net_id is None:
                raise ValueError(
                    "Q3D conductor bodies require a PDK superconductor and final Net"
                )
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "region_padding_um", _padding(self.region_padding_um))
        nets = tuple(self.nets)
        if (
            not nets
            or len({net.name for net in nets}) != len(nets)
            or not any(net.net_type == "Signal" for net in nets)
        ):
            raise ValueError("Q3D requires at least one unique Signal net")
        owners = {name: net for net in nets for name in net.object_names}
        conductors = {
            body.body_id: body for body in bodies if body.physical_role != "substrate"
        }
        if len(owners) != sum(len(net.object_names) for net in nets) or set(
            owners
        ) != set(conductors):
            raise ValueError("Q3D nets must cover every conductor body exactly once")
        if any(owners[name].name != body.net_id for name, body in conductors.items()):
            raise ValueError("Q3D body final Net differs from Net membership")
        if not isinstance(self.solve_ac_rl, bool):
            raise TypeError("solve_ac_rl must be boolean")
        if self.solve_ac_rl and any(
            net.net_type == "Signal" and net.source_object is None for net in nets
        ):
            raise ValueError(
                "Q3D AC/RL requires complete terminals for every Signal net"
            )
        if self.geometry_source is not None:
            if not isinstance(self.geometry_source, Mapping):
                raise TypeError("geometry_source must be a JSON object")
            object.__setattr__(
                self,
                "geometry_source",
                _q3d_geometry_source(self.geometry_source, bodies=bodies, nets=nets),
            )
        grounded_region_net = self.grounded_region_net
        if grounded_region_net is not None:
            _text(grounded_region_net, "grounded_region_net")
            if self.solve_ac_rl:
                raise ValueError("grounded_region_net requires solve_ac_rl=False")
            if grounded_region_net in {net.name for net in nets}:
                raise ValueError(
                    "grounded_region_net must name a new dedicated enclosure net"
                )
        object.__setattr__(self, "nets", nets)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                key: item.to_payload() for key, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "bodies": [body.to_payload() for body in self.bodies],
            "nets": [net.to_payload() for net in self.nets],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "solve_ac_rl": self.solve_ac_rl,
            "grounded_region_net": self.grounded_region_net,
        }
        if self.geometry_source is not None:
            payload["geometry_source"] = dict(self.geometry_source)
        return payload

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None
    ) -> Q3dSpec:
        if payload.get("schema_version") != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
        if {"gds", "layer_imports", "object_bindings"} & set(payload):
            raise ValueError("Q3D v3 does not accept GDS-piece inputs")
        return cls(
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials={
                key: PdkMaterial(**item) for key, item in payload["materials"].items()
            },
            vacuum_material_id=payload["vacuum_material_id"],
            bodies=tuple(Q3dBodySpec(**body) for body in payload["bodies"]),
            nets=tuple(Q3dNetSpec(**net) for net in payload["nets"]),
            run_control=MatrixRunControl(**payload["run_control"]),
            region_padding_um=tuple(payload["region_padding_um"]),
            solve_ac_rl=payload.get("solve_ac_rl", True),
            grounded_region_net=payload.get("grounded_region_net"),
            aedt_version=payload["aedt"]["requested_version"],
            pyaedt_version=payload["pyaedt"]["locked_version"],
            geometry_source=payload.get("geometry_source"),
        )


@dataclass(frozen=True)
class Q2dRectangleSpec:
    """One explicit native Q2D cross-section rectangle."""

    name: str
    origin_um: tuple[float, float]
    size_um: tuple[float, float]
    material_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "rectangle.name"))
        if len(self.origin_um) != 2 or len(self.size_um) != 2:
            raise ValueError("Q2D rectangle origin_um and size_um require x,y pairs")
        origin = tuple(
            _number(value, "rectangle.origin_um") for value in self.origin_um
        )
        size = tuple(_number(value, "rectangle.size_um") for value in self.size_um)
        if any(value <= 0 for value in size):
            raise ValueError("Q2D rectangle size_um values must be > 0")
        object.__setattr__(self, "origin_um", origin)
        object.__setattr__(self, "size_um", size)
        object.__setattr__(self, "material_id", _text(self.material_id, "material_id"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin_um": list(self.origin_um),
            "size_um": list(self.size_um),
            "material_id": self.material_id,
        }


@dataclass(frozen=True)
class Q2dConductorSpec:
    """One exact Q2D signal or the single reference-ground group."""

    name: str
    conductor_type: Literal["SignalLine", "ReferenceGround"]
    object_names: tuple[str, ...]
    thickness_um: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "conductor.name"))
        if self.conductor_type not in {"SignalLine", "ReferenceGround"}:
            raise ValueError("Q2D conductor_type must be SignalLine or ReferenceGround")
        objects = tuple(
            _text(value, "conductor.object_name") for value in self.object_names
        )
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Q2D conductor object_names must be nonempty and unique")
        thickness = _number(self.thickness_um, "conductor.thickness_um")
        if thickness <= 0:
            raise ValueError("Q2D conductor thickness_um must be > 0")
        object.__setattr__(self, "object_names", objects)
        object.__setattr__(self, "thickness_um", thickness)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "conductor_type": self.conductor_type,
            "object_names": list(self.object_names),
            "thickness_um": self.thickness_um,
        }


@dataclass(frozen=True)
class Q2dSpec:
    """One explicit native Q2D cross-section matrix extraction."""

    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    rectangles: tuple[Q2dRectangleSpec, ...]
    conductors: tuple[Q2dConductorSpec, ...]
    run_control: MatrixRunControl
    region_padding_um: tuple[float, float, float, float]
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["q2d"]:
        return "q2d"

    def __post_init__(self) -> None:
        materials = _normalize_common_spec(self)
        rectangles = tuple(self.rectangles)
        if (
            not rectangles
            or len({item.name for item in rectangles}) != len(rectangles)
            or any(item.material_id not in materials for item in rectangles)
        ):
            raise ValueError(
                "Q2D rectangles must be unique and use declared PDK materials"
            )
        if any(materials[item.material_id].kind == "vacuum" for item in rectangles):
            raise ValueError("Q2D vacuum is owned by the Region, not a rectangle")
        conductors = tuple(self.conductors)
        if (
            not conductors
            or len({item.name for item in conductors}) != len(conductors)
            or sum(item.conductor_type == "ReferenceGround" for item in conductors) != 1
            or not any(item.conductor_type == "SignalLine" for item in conductors)
        ):
            raise ValueError(
                "Q2D requires SignalLine conductors and one ReferenceGround"
            )
        owners = {
            object_name: conductor
            for conductor in conductors
            for object_name in conductor.object_names
        }
        if len(owners) != sum(len(item.object_names) for item in conductors):
            raise ValueError(
                "Q2D conductor rectangles must belong to exactly one group"
            )
        superconductors = {
            item.name
            for item in rectangles
            if materials[item.material_id].is_superconducting
        }
        if set(owners) != superconductors:
            raise ValueError(
                "Q2D conductor groups must cover every superconducting rectangle exactly"
            )
        object.__setattr__(self, "rectangles", rectangles)
        object.__setattr__(self, "conductors", conductors)
        object.__setattr__(
            self, "region_padding_um", _padding_2d(self.region_padding_um)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": Q2D_SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "rectangles": [item.to_payload() for item in self.rectangles],
            "conductors": [item.to_payload() for item in self.conductors],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Q2dSpec:
        if payload.get("schema_version") != Q2D_SCHEMA_VERSION:
            raise ValueError("unsupported Q2D schema")
        raw_materials = payload.get("materials")
        if not isinstance(raw_materials, dict):
            raise TypeError("materials must be a JSON object")
        materials = {
            material_id: PdkMaterial(**item)
            for material_id, item in raw_materials.items()
        }
        run = payload.get("run_control")
        if not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object")
        return cls(
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            rectangles=tuple(
                Q2dRectangleSpec(**item) for item in payload.get("rectangles", ())
            ),
            conductors=tuple(
                Q2dConductorSpec(**item) for item in payload.get("conductors", ())
            ),
            run_control=MatrixRunControl(**run),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )


AedtSpec = HfssSpec | Q3dSpec | Q2dSpec


def parse_aedt_spec(
    payload: dict[str, Any], *, base_dir: Path | None = None
) -> AedtSpec:
    """Dispatch one explicit AEDT schema without inferring solver family."""
    if payload.get("schema_version") == SCHEMA_VERSION:
        return HfssDrivenSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == EIGENMODE_SCHEMA_VERSION:
        return HfssEigenmodeSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") in {
        EPR_EIGENMODE_SCHEMA_VERSION,
        EPR_EIGENMODE_SCHEMA_VERSION_V2,
        EPR_EIGENMODE_SCHEMA_VERSION_V3,
    }:
        return HfssEprSpec.from_payload(payload)
    if payload.get("schema_version") in {
        EPR_ANALYSIS_SCHEMA_VERSION,
        EPR_ANALYSIS_SCHEMA_VERSION_V2,
    }:
        return HfssEprAnalysisSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == Q3D_SCHEMA_VERSION:
        return Q3dSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == Q2D_SCHEMA_VERSION:
        return Q2dSpec.from_payload(payload)
    raise ValueError("unsupported AEDT schema")
