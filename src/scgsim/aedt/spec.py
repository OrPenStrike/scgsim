"""Explicit, fail-closed input contracts for the AEDT runtime families."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

HfssDrivenMode = Literal["terminal", "modal"]
LayerRole = Literal["ground", "signal", "substrate"]
PdkMaterialKind = Literal["vacuum", "dielectric", "superconductor"]
Side = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

SCHEMA_VERSION = "scgsim.aedt.hfss-driven.v1"
EIGENMODE_SCHEMA_VERSION = "scgsim.aedt.hfss-eigenmode.v1"
Q3D_SCHEMA_VERSION = "scgsim.aedt.q3d.v1"
Q2D_SCHEMA_VERSION = "scgsim.aedt.q2d.v1"
OFFICIAL_PYAEDT_SOURCE_URL = "https://github.com/ansys/pyaedt/tree/v1.3.0"
LOCKED_PYAEDT = "1.3.0"
REQUIRED_AEDT_VERSION = "2024.2"
POINT_COUNT = 20_000
SURFACE_APPROXIMATION_LEVEL = 9


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


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


@dataclass(frozen=True)
class FrequencySweepSpec:
    """One HFSS Fast sweep with the Human-fixed output count."""

    start_ghz: float
    stop_ghz: float
    points: int = POINT_COUNT

    def __post_init__(self) -> None:
        start = _number(self.start_ghz, "start_ghz")
        stop = _number(self.stop_ghz, "stop_ghz")
        if start <= 0 or stop <= start:
            raise ValueError("frequency sweep requires 0 < start_ghz < stop_ghz")
        if self.points != POINT_COUNT:
            raise ValueError(f"points must be exactly {POINT_COUNT}")
        object.__setattr__(self, "start_ghz", start)
        object.__setattr__(self, "stop_ghz", stop)

    def to_payload(self) -> dict[str, Any]:
        return {
            "start_ghz": self.start_ghz,
            "stop_ghz": self.stop_ghz,
            "points": self.points,
        }


@dataclass(frozen=True)
class HfssRunControl:
    """The only setup and sweep identity created by one V1 invocation."""

    setup_name: str
    sweep_name: str
    sweep: FrequencySweepSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        object.__setattr__(self, "sweep_name", _text(self.sweep_name, "sweep_name"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "sweep_name": self.sweep_name,
            "sweep": self.sweep.to_payload(),
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
    project = Path(_text(spec.project_name, "project_name"))
    if project.parent != Path(".") or project.name in {".", ".."}:
        raise ValueError("project_name must be a single project filename")
    if not project.stem:
        raise ValueError("project_name must normalize to a non-empty stem")
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
            project_name=_text(payload.get("project", {}).get("name"), "project.name"),
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
        if not isinstance(self.maximum_passes, int) or self.maximum_passes <= 0:
            raise ValueError("maximum_passes must be a positive integer")
        object.__setattr__(self, "minimum_frequency_ghz", minimum)
        object.__setattr__(self, "maximum_delta_frequency_percent", delta)

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "minimum_frequency_ghz": self.minimum_frequency_ghz,
            "num_modes": self.num_modes,
            "maximum_passes": self.maximum_passes,
            "maximum_delta_frequency_percent": self.maximum_delta_frequency_percent,
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
            project_name=_text(payload.get("project", {}).get("name"), "project.name"),
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


HfssSpec = HfssDrivenSpec | HfssEigenmodeSpec


@dataclass(frozen=True)
class MatrixRunControl:
    """One Q3D/Q2D matrix setup at one frequency."""

    setup_name: str
    frequency_ghz: float
    maximum_passes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup_name", _text(self.setup_name, "setup_name"))
        frequency = _number(self.frequency_ghz, "frequency_ghz")
        if frequency <= 0:
            raise ValueError("matrix frequency_ghz must be > 0")
        if not isinstance(self.maximum_passes, int) or self.maximum_passes <= 0:
            raise ValueError("matrix maximum_passes must be a positive integer")
        object.__setattr__(self, "frequency_ghz", frequency)

    def to_payload(self) -> dict[str, Any]:
        return {
            "setup_name": self.setup_name,
            "frequency_ghz": self.frequency_ghz,
            "maximum_passes": self.maximum_passes,
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
        if any(value is None for value in terminal_values):
            raise ValueError(
                "Q3D Signal nets require exact source and sink objects/sides"
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
class Q3dSpec:
    """One direct-GDS Q3D capacitance and AC R/L extraction."""

    gds_path: Path | str
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    nets: tuple[Q3dNetSpec, ...]
    run_control: MatrixRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["q3d"]:
        return "q3d"

    def __post_init__(self) -> None:
        grounds, signals = _normalize_gds_spec(self)
        object.__setattr__(self, "region_padding_um", _padding(self.region_padding_um))
        nets = tuple(self.nets)
        if (
            not nets
            or len({net.name for net in nets}) != len(nets)
            or not any(net.net_type == "Signal" for net in nets)
            or not any(net.net_type == "Ground" for net in nets)
        ):
            raise ValueError("Q3D requires unique Signal and Ground nets")
        owners = {object_name: net for net in nets for object_name in net.object_names}
        if len(owners) != sum(len(net.object_names) for net in nets):
            raise ValueError("Q3D conductor objects must belong to exactly one net")
        if set(owners) != grounds | signals:
            raise ValueError("Q3D nets must cover every declared conductor exactly")
        if any(
            (name in signals) != (net.net_type == "Signal")
            for name, net in owners.items()
        ):
            raise ValueError("Q3D net types must match structured conductor roles")
        layers = {item.layer: item for item in self.layer_imports}
        if any(
            layers[binding.layer].z_max_um <= layers[binding.layer].z_min_um
            for binding in self.object_bindings
            if binding.role in {"signal", "ground"}
        ):
            raise ValueError("Q3D conductor imports require positive finite thickness")
        object.__setattr__(self, "nets", nets)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": Q3D_SCHEMA_VERSION,
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
            "nets": [item.to_payload() for item in self.nets],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None
    ) -> Q3dSpec:
        if payload.get("schema_version") != Q3D_SCHEMA_VERSION:
            raise ValueError("unsupported Q3D schema")
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
        return cls(
            gds_path=gds,
            project_name=_text(payload.get("project", {}).get("name"), "project.name"),
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
            nets=tuple(Q3dNetSpec(**item) for item in payload.get("nets", ())),
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
            project_name=_text(payload.get("project", {}).get("name"), "project.name"),
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
    if payload.get("schema_version") == Q3D_SCHEMA_VERSION:
        return Q3dSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == Q2D_SCHEMA_VERSION:
        return Q2dSpec.from_payload(payload)
    raise ValueError("unsupported AEDT schema")
