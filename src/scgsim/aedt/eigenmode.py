"""One mutable Eigenmode preparation facade over the existing AEDT handoff."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from scgsim.semantics.epr import film_assumptions
from scgsim.sgb import (
    GeometryBuildInput,
    GeometryPlanSnapshot,
    InputSummary,
    build_gds_stack_geometry_input,
    summarize_geometry_input,
)

from ._epr_geometry import prepare_planar_geometry_input, validate_geometry_workers
from ._epr_models import (
    EprAnalysisRequest,
    EprResult,
    ExpressionCacheConvergence,
    PlanarJunction,
    PreparedPlanarGeometry,
    SurfaceEprSpec,
)
from ._epr_results import reanalyze_epr, show_epr
from .handoff import HandoffPlan, prepare_hfss_eigenmode_from_geometry
from .resolve import ResolvedRun, resolve_results
from .spec import AedtResources, EigenmodeRunControl


@dataclass
class EigenmodeSim:
    """Configure ordinary and EPR Eigenmode on one body-first HFSS source."""

    component: Any | None = None
    build_input: GeometryBuildInput | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    project_name: str | None = None
    design_name: str | None = None
    run_control: EigenmodeRunControl | None = None
    route: Literal["A", "B"] = "B"
    route_a_profile: str | None = None
    resources: AedtResources | None = None
    geometry_workers: int | None = None
    surface_contributions: tuple[SurfaceEprSpec, ...] = ()
    surface_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    junctions: tuple[PlanarJunction, ...] = ()
    epr_request: EprAnalysisRequest | None = None
    expression_convergence: ExpressionCacheConvergence | None = None
    prepared_geometry: PreparedPlanarGeometry | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _plan_snapshot: GeometryPlanSnapshot | None = field(default=None, init=False, repr=False)

    def _invalidate_model(self) -> None:
        self.prepared_geometry = None
        self.handoff_plan = None

    def set_geometry(self, component: Any | GeometryBuildInput) -> None:
        if self._plan_snapshot is not None:
            raise ValueError("GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim")
        if isinstance(component, GeometryBuildInput):
            self.build_input, self.component = component, None
        elif callable(getattr(component, "write_gds", None)):
            self.component, self.build_input = component, None
        else:
            raise TypeError(
                "geometry must be a Component with write_gds or GeometryBuildInput"
            )
        self._plan_snapshot = None
        self._invalidate_model()

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Consume one immutable normalized source and its paired stack."""
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError("set_plan requires a GeometryPlanSnapshot")
        trial = EigenmodeSim()
        trial.set_stack(snapshot.stack)
        build_input = snapshot.geometry_input
        self.component = None
        self.build_input = build_input
        self.stack = trial.stack
        self._plan_snapshot = snapshot
        self._invalidate_model()

    def set_stack(self, stack: Mapping[str, Any] | str | Path) -> None:
        if self._plan_snapshot is not None:
            raise ValueError("GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim")
        if isinstance(stack, Mapping):
            payload = dict(stack)
        elif isinstance(stack, (str, Path)):
            source = Path(stack)
            if not source.is_file():
                raise FileNotFoundError(source)
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            raise TypeError("stack must be a mapping or JSON path")
        if not isinstance(payload, Mapping):
            raise TypeError("stack JSON root must be a mapping")
        for key in ("materials", "solution_regions", "layers", "metadata"):
            if key not in payload:
                raise ValueError(f"stack lacks {key!r}")
        materials = payload["materials"]
        regions = payload["solution_regions"]
        layers = payload["layers"]
        metadata = payload["metadata"]
        if (
            not isinstance(materials, Mapping)
            or not isinstance(regions, Mapping)
            or not isinstance(layers, Sequence)
            or isinstance(layers, (str, bytes))
            or not isinstance(metadata, Mapping)
        ):
            raise TypeError("stack materials, solution_regions, and metadata must be mappings; layers must be a sequence")
        if any(not isinstance(item, Mapping) for item in materials.values()):
            raise TypeError("stack materials must contain mappings")
        if any(not isinstance(item, Mapping) for item in regions.values()):
            raise TypeError("stack solution_regions must contain mappings")
        if any(not isinstance(item, Mapping) for item in layers):
            raise TypeError("stack layers must contain mappings")
        if any(record.get("material_id", semantic_id) not in materials for semantic_id, record in regions.items()):
            raise ValueError("stack solution region references an unknown material")
        if any(record.get("material_id") not in materials for record in layers if "material_id" in record):
            raise ValueError("stack layer references an unknown material")
        detached = json.loads(json.dumps(payload, allow_nan=False))
        self.stack = detached
        self._plan_snapshot = None
        self._invalidate_model()

    def set_output_dir(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("output_dir must be a path string or Path")
        resolved = Path(path).expanduser().resolve()
        self.output_dir = resolved
        self.handoff_plan = None

    def set_eigenmode(
        self,
        *,
        project_name: str,
        design_name: str,
        run_control: EigenmodeRunControl,
        route: Literal["A", "B"] = "B",
        route_a_profile: str | None = None,
        resources: AedtResources | None = None,
        geometry_workers: int | None = None,
    ) -> None:
        if (
            not isinstance(project_name, str)
            or not project_name.strip()
            or Path(project_name).name != project_name
            or Path(project_name).suffix
        ):
            raise ValueError("project_name must be a plain project stem")
        if not isinstance(design_name, str) or not design_name.strip():
            raise ValueError("design_name must be non-empty text")
        if not isinstance(run_control, EigenmodeRunControl):
            raise TypeError("run_control must be EigenmodeRunControl")
        if route not in {"A", "B"}:
            raise ValueError("route must be A or B")
        if route == "A" and route_a_profile not in {
            "substrate_face",
            "metal_gap_equivalent",
        }:
            raise ValueError("Route A requires an explicit supported thin-film profile")
        if route == "B" and route_a_profile is not None:
            raise ValueError("Route B cannot use route_a_profile")
        if resources is not None and not isinstance(resources, AedtResources):
            raise TypeError("resources must be AedtResources or None")
        validate_geometry_workers(geometry_workers)
        model_changed = (route, route_a_profile) != (self.route, self.route_a_profile)
        self.project_name, self.design_name, self.run_control = (
            project_name,
            design_name,
            run_control,
        )
        self.route, self.route_a_profile = route, route_a_profile
        self.resources, self.geometry_workers = resources, geometry_workers
        if model_changed:
            self._invalidate_model()
        else:
            self.handoff_plan = None

    def set_surface_epr(
        self,
        *,
        contributions: Sequence[SurfaceEprSpec] = (),
        defaults: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if isinstance(contributions, (str, bytes)):
            raise TypeError("contributions must be SurfaceEprSpec records")
        selected = tuple(contributions)
        if any(not isinstance(item, SurfaceEprSpec) for item in selected):
            raise TypeError("contributions must be SurfaceEprSpec records")
        if len({item.contribution_id for item in selected}) != len(selected):
            raise ValueError("surface contribution IDs must be unique")
        if defaults is None:
            normalized: dict[str, dict[str, Any]] = {}
        elif not isinstance(defaults, Mapping) or set(defaults) - {"MA", "MS", "SA"}:
            raise ValueError("surface defaults must be keyed by MA, MS, or SA")
        else:
            normalized = {
                kind: film_assumptions(update, partial=True)
                for kind, update in defaults.items()
            }
        resolved = tuple(
            replace(
                item,
                **{
                    name: value
                    for name, value in normalized.get(item.interface_kind, {}).items()
                    if name in {"loss_tangent", "source", "preset"}
                    and getattr(item, name) is None
                },
                _legacy_payload=False,
            )
            for item in selected
        )
        self.surface_contributions, self.surface_defaults = resolved, normalized
        self._invalidate_model()

    def add_junction(self, junction: PlanarJunction) -> None:
        if not isinstance(junction, PlanarJunction):
            raise TypeError("junction must be PlanarJunction")
        if junction.junction_id in {item.junction_id for item in self.junctions}:
            raise ValueError("junction ID is already configured")
        self.junctions = (*self.junctions, junction)
        self._invalidate_model()

    def set_epr_request(self, request: EprAnalysisRequest | None) -> None:
        if request is not None and not isinstance(request, EprAnalysisRequest):
            raise TypeError("request must be EprAnalysisRequest or None")
        self.epr_request = request
        self.handoff_plan = None

    def set_expression_cache_convergence(
        self, control: ExpressionCacheConvergence | None
    ) -> None:
        if control is not None and not isinstance(control, ExpressionCacheConvergence):
            raise TypeError("control must be ExpressionCacheConvergence or None")
        self.expression_convergence = control
        self.handoff_plan = None

    def _source_input(self) -> GeometryBuildInput:
        if self._plan_snapshot is not None:
            return self._plan_snapshot.geometry_input
        if self.build_input is not None:
            return self.build_input
        if self.component is None or self.stack is None:
            raise ValueError("set_geometry() and set_stack() are required")
        with TemporaryDirectory() as source_dir:
            directory = Path(source_dir)
            gds_path = directory / "design.gds"
            stack_path = directory / "design.stack.json"
            try:
                self.component.write_gds(gds_path)
            except TypeError:
                self.component.write_gds(str(gds_path))
            stack_path.write_text(
                json.dumps(dict(self.stack), sort_keys=True), encoding="utf-8"
            )
            prepared = build_gds_stack_geometry_input(
                gds_file=gds_path, stack_file=stack_path
            )
        metadata = dict(prepared.metadata)
        metadata.pop("gds_file", None)
        metadata.pop("stack_file", None)
        return replace(prepared, metadata=metadata)

    def input_summary(self) -> InputSummary:
        """Show configured normalized inputs without writing GDS or preparing HFSS."""

        if self._plan_snapshot is not None:
            source, stack = self._plan_snapshot.geometry_input, self._plan_snapshot.stack
        elif self.build_input is not None and self.stack is not None:
            source, stack = self.build_input, self.stack
        else:
            raise ValueError(
                "input_summary requires a GeometryPlan snapshot or an already-"
                "normalized GeometryBuildInput with stack; normalize a raw "
                "Component explicitly first"
            )
        data = summarize_geometry_input(source, prepared_stack=stack).data
        data["backend"] = {
            "name": "aedt",
            "route": self.route,
            "route_a_profile": self.route_a_profile,
            "project_name": self.project_name,
            "design_name": self.design_name,
            "run_control": (
                None if self.run_control is None else self.run_control.to_payload()
            ),
            "resources": None if self.resources is None else self.resources.to_payload(),
            "geometry_workers": self.geometry_workers,
            "surface_contributions": [
                item.to_payload() for item in self.surface_contributions
            ],
            "surface_defaults": self.surface_defaults,
            "junctions": [item.to_payload() for item in self.junctions],
            "epr_request": (
                None if self.epr_request is None else self.epr_request.to_payload()
            ),
            "expression_convergence": (
                None
                if self.expression_convergence is None
                else self.expression_convergence.to_payload()
            ),
        }
        return InputSummary(data)

    def prepare_handoff(self) -> HandoffPlan:
        source_stack = self._plan_snapshot.stack if self._plan_snapshot is not None else self.stack
        if (
            source_stack is None
            or self.output_dir is None
            or self.project_name is None
            or self.design_name is None
            or self.run_control is None
        ):
            raise ValueError(
                "geometry, stack, output_dir, and eigenmode controls are required"
            )
        geometry = self.prepared_geometry
        if geometry is None:
            source = self._source_input()
            geometry = prepare_planar_geometry_input(
                source,
                prepared_stack=source_stack,
                route=self.route,
                route_a_profile=self.route_a_profile,
                junctions=self.junctions,
                contributions=self.surface_contributions,
            )
        plan = prepare_hfss_eigenmode_from_geometry(
            geometry=geometry,
            project_name=self.project_name,
            design_name=self.design_name,
            run_control=self.run_control,
            output_dir=self.output_dir,
            epr_request=self.epr_request,
            expression_convergence=self.expression_convergence,
            geometry_workers=self.geometry_workers,
            resources=self.resources,
        )
        self.prepared_geometry, self.handoff_plan = geometry, plan
        return plan

    def resolve_results(self, run_dir: str | Path | None = None) -> ResolvedRun:
        if run_dir is None:
            if self.handoff_plan is None:
                raise ValueError("prepare_handoff() or an explicit run_dir is required")
            run_dir = self.handoff_plan.run_dir
        return resolve_results(run_dir)

    @staticmethod
    def reanalyze_epr(
        result: EprResult,
        *,
        surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> EprResult:
        return reanalyze_epr(
            result, surface_defaults=surface_defaults, group_overrides=group_overrides
        )

    @staticmethod
    def show_epr(
        result: EprResult, *, mode: int | None = None, native_pass: int | None = None,
        theme: str = "light",
    ) -> Any:
        return show_epr(result, mode=mode, native_pass=native_pass, theme=theme)
