"""HFSS request contracts and their family-specific decoding rules."""

from __future__ import annotations


from collections.abc import Mapping

from dataclasses import dataclass, replace

from numbers import Number

from pathlib import Path

from typing import Any, Literal

from scgsim.aedt.specs.modeling import (
    Modeling, PhysicalLayerSpec, _request_modeling, _modeling_payload, _effective_record, _source_record_payload, _verify_effective_records,
)

from scgsim.aedt.specs.common import (
    EIGENMODE_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    HfssDrivenMode,
    LOCKED_PYAEDT,
    LayerImport,
    ModalPort,
    OFFICIAL_PYAEDT_SOURCE_URL,
    ObjectBinding,
    POINT_COUNT,
    PdkMaterial,
    REQUIRED_AEDT_VERSION,
    SCHEMA_VERSION,
    TerminalPort,
    _adaptive_controls,
    _normalize_gds_spec,
    _number,
    _padding,
    _project_filename,
    _project_name_from_payload,
    _text,
)


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
            raise ValueError(
                "points must be an exact whole-number numeric value"
            ) from exc
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
class HfssDrivenSpec:
    """The complete two-port, one-mode, one-setup public CPW handoff spec."""

    mode: HfssDrivenMode
    gds_path: Path | str
    modeling: Modeling
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    ports: tuple[TerminalPort, TerminalPort] | tuple[ModalPort, ModalPort]
    run_control: HfssRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    physical_layers: tuple[PhysicalLayerSpec, ...] = ()
    _historical_modeling: bool = False
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    def __post_init__(self) -> None:
        _request_modeling(self)
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

    @property
    def effective_ports(self) -> tuple[TerminalPort, ...] | tuple[ModalPort, ...]:
        """Native port coordinates derived from immutable authored endpoints."""
        mapping = _request_modeling(self)
        if mapping is None:
            raise ValueError("historical requests have no new execution modeling map")
        return tuple(
            replace(port, integration_line_um=tuple(
                (x, y, mapping.map_z(z)) for x, y, z in port.integration_line_um
            )) if isinstance(port, ModalPort) else port
            for port in self.ports
        )

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        mapping = _request_modeling(self)
        if mapping is None:
            raise ValueError("historical requests have no new execution modeling map")
        pairs = {pair: index + 1 for index, pair in enumerate(sorted(
            (item.layer, item.datatype) for item in self.layer_imports))}
        domain_pairs = {(item.layer, item.datatype) for item in self.layer_imports
                        if any(binding.role == "substrate"
                               and binding.layer == item.layer
                               and binding.object_name.startswith(f"{item.layer_name}_")
                               for binding in self.object_bindings)}
        return tuple({**_effective_record(item, mapping, retained_domain=(item.layer, item.datatype) in domain_pairs),
                      "native_import_layer": pairs[(item.layer, item.datatype)]}
                     for item in self.layer_imports)

    def to_payload(self) -> dict[str, Any]:
        return {
            **_modeling_payload(self),
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
            "layer_imports": (list(self.effective_layer_imports) if self.modeling is not None
                              else [item.to_payload() for item in self.layer_imports]),
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "ports": [item.to_payload() for item in self.ports],
            **({"effective_ports": [item.to_payload() for item in self.effective_ports]}
               if self.modeling is not None else {}),
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None, allow_historical_modeling: bool = False
    ) -> HfssDrivenSpec:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported HFSS driven schema")
        mode = _text(payload.get("mode"), "mode")
        gds = Path(_text(payload.get("gds", {}).get("path"), "gds.path"))
        if base_dir is not None and not gds.is_absolute():
            gds = base_dir / gds
        imports = tuple(
            LayerImport(**_source_record_payload(item)) for item in payload.get("layer_imports", ())
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
        result = cls(
            modeling=payload.get("modeling"),
            physical_layers=tuple(PhysicalLayerSpec(**item) for item in payload.get("physical_layers", ())),
            _historical_modeling=allow_historical_modeling and "modeling" not in payload,
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
        if result.modeling is not None:
            _verify_effective_records(payload["layer_imports"], result.effective_layer_imports)
            _verify_effective_records(payload["effective_ports"],
                                      [item.to_payload() for item in result.effective_ports])
        return result


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
    modeling: Modeling
    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    layer_imports: tuple[LayerImport, ...]
    object_bindings: tuple[ObjectBinding, ...]
    run_control: EigenmodeRunControl
    region_padding_um: tuple[float, float, float, float, float, float]
    physical_layers: tuple[PhysicalLayerSpec, ...] = ()
    _historical_modeling: bool = False
    length_mesh: LengthMeshSpec | None = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["eigenmode"]:
        return "eigenmode"

    def __post_init__(self) -> None:
        _request_modeling(self)
        grounds, signals = _normalize_gds_spec(self)
        object.__setattr__(self, "region_padding_um", _padding(self.region_padding_um))
        if self.length_mesh is not None and (
            not set(self.length_mesh.ground_objects).issubset(grounds)
            or not set(self.length_mesh.signal_objects).issubset(signals)
        ):
            raise ValueError(
                "length mesh targets must be declared ground/signal objects"
            )

    @property
    def effective_layer_imports(self) -> tuple[dict[str, Any], ...]:
        mapping = _request_modeling(self)
        if mapping is None:
            raise ValueError("historical requests have no new execution modeling map")
        pairs = {pair: index + 1 for index, pair in enumerate(sorted(
            (item.layer, item.datatype) for item in self.layer_imports))}
        domain_pairs = {(item.layer, item.datatype) for item in self.layer_imports
                        if any(binding.role == "substrate"
                               and binding.layer == item.layer
                               and binding.object_name.startswith(f"{item.layer_name}_")
                               for binding in self.object_bindings)}
        return tuple({**_effective_record(item, mapping, retained_domain=(item.layer, item.datatype) in domain_pairs),
                      "native_import_layer": pairs[(item.layer, item.datatype)]}
                     for item in self.layer_imports)

    def to_payload(self) -> dict[str, Any]:
        return {
            **_modeling_payload(self),
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
            "layer_imports": (list(self.effective_layer_imports) if self.modeling is not None
                              else [item.to_payload() for item in self.layer_imports]),
            "object_bindings": [item.to_payload() for item in self.object_bindings],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
            "length_mesh": self.length_mesh.to_payload() if self.length_mesh else None,
        }

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], *, base_dir: Path | None = None, allow_historical_modeling: bool = False
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
        result = cls(
            modeling=payload.get("modeling"),
            physical_layers=tuple(PhysicalLayerSpec(**item) for item in payload.get("physical_layers", ())),
            _historical_modeling=allow_historical_modeling and "modeling" not in payload,
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
                LayerImport(**_source_record_payload(item)) for item in payload.get("layer_imports", ())
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
        if result.modeling is not None:
            _verify_effective_records(payload["layer_imports"], result.effective_layer_imports)
        return result


@dataclass(frozen=True)
class HfssEprSpec:
    """One body-first HFSS Eigenmode request with embedded planar authority."""

    modeling: Modeling
    project_name: str
    design_name: str
    geometry: Any
    run_control: EigenmodeRunControl
    physical_layers: tuple[PhysicalLayerSpec, ...] = ()
    _historical_modeling: bool = False
    epr_request: Any = None
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT
    _legacy_payload: bool = False
    expression_convergence: Any = None

    @property
    def mode(self) -> Literal["eigenmode"]:
        return "eigenmode"

    def __post_init__(self) -> None:
        _request_modeling(self)
        from scgsim.aedt.epr.models import (
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
        if not self._historical_modeling and self.geometry.modeling != self.modeling:
            raise ValueError("request and prepared geometry modeling differ")
        if not self._historical_modeling:
            declared = tuple(PhysicalLayerSpec(**item) for item in self.geometry.source["modeling"]["source_layers"])
            if self.physical_layers and {item.physical_layer_id: item for item in self.physical_layers} != {item.physical_layer_id: item for item in declared}:
                raise ValueError("request physical layers differ from prepared source map")
            object.__setattr__(self, "physical_layers", declared)

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
            **_modeling_payload(self),
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
    def from_payload(cls, payload: dict[str, Any], *, allow_historical_modeling: bool = False) -> HfssEprSpec:
        from scgsim.aedt.epr.models import (
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
        if "modeling" in payload:
            expected.update({"modeling", "physical_layers"})
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
            modeling=payload.get("modeling"),
            physical_layers=tuple(PhysicalLayerSpec(**item) for item in payload.get("physical_layers", ())),
            _historical_modeling=allow_historical_modeling and "modeling" not in payload,
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
        from scgsim.aedt.epr.models import (
            EprAnalysisRequest,
            PreparedPlanarGeometry,
            SavedSolution,
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
            raise ValueError(
                "saved solution identity does not match the analysis model"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                EPR_ANALYSIS_SCHEMA_VERSION
                if self._legacy_payload
                else EPR_ANALYSIS_SCHEMA_VERSION_V2
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
        from scgsim.aedt.epr.models import (
            EprAnalysisRequest,
            PreparedPlanarGeometry,
            SavedSolution,
        )

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
    from scgsim.aedt.epr.models import NormalizedSurfaceEprTotal
    from scgsim.aedt.epr.selection import surface_integral_groups

    request = spec.epr_request
    if request is None:
        if getattr(spec, "expression_convergence", None) is not None:
            raise ValueError("expression convergence requires an EPR analysis request")
        return
    if any(
        item.field_side == "sidewall" for item in spec.geometry.contributions
    ) or any(
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
    if (
        request.bulk_domain_ids is not None
        and not set(request.bulk_domain_ids) <= source_domains
    ):
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


HfssSpec = HfssDrivenSpec | HfssEigenmodeSpec | HfssEprSpec | HfssEprAnalysisSpec
