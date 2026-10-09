"""EPR handoff and native saved-field workflow orchestration."""

from __future__ import annotations


import json

import shutil

import time

from dataclasses import dataclass

from pathlib import Path

from typing import Any

from scgsim.aedt._io import file_sha256, write_json

from scgsim.aedt.epr.analysis import combine_epr_mode

from scgsim.aedt.epr.cache import (
    _SURFACE_ANALYSIS_SCOPE,
    _adaptive_epr_result,
    _adaptive_mode_history,
    _attach_saved_native_surface_membership,
    _author_epr_expressions,
    _canonical_integral,
    _create_setup,
    _empty_raw_integrals,
    _integral_target,
    _prepare_expressions_and_cache,
    _read_cache,
    _read_setup,
    _store_integral,
)

from scgsim.aedt.epr.fields import install_variables, parse_native_scalar

from scgsim.aedt.epr.models import EprResult, detached

from scgsim.aedt.epr.native import (
    _read_saved_junction_lines,
    bind_saved_planar_geometry,
    prepare_native_planar_geometry,
    verify_saved_native_surface_references,
)

from scgsim.aedt.results.convergence.hfss import read_hfss_convergence

from scgsim.aedt.runtime.benchmark import export_simulation_benchmark

from scgsim.aedt.runtime.families.hfss import _export_eigenmode

from scgsim.aedt.preparation.handoff import _prepare_epr_analysis_handoff

from scgsim.aedt.epr.models import (
    EprAnalysisRequest,
    PreparedPlanarGeometry,
    SavedSolution,
)

from scgsim.aedt.runtime.native.common import (
    BoundAedtRequest,
    analyze_with_resources,
    detached_data,
    pyaedt_version,
)

from scgsim.aedt.specs.common import AedtResources, REQUIRED_AEDT_VERSION

from scgsim.aedt.specs.hfss import (
    EigenmodeRunControl,
    HfssEprAnalysisSpec,
    HfssEprSpec,
)


@dataclass(frozen=True)
class PreparedEprHfss:
    """Carrier for one mutable native application and detached preparation evidence."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    geometry: dict[str, Any]
    setup: dict[str, Any]
    expressions: tuple[dict[str, Any], ...]
    cache: dict[str, Any]
    timings: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedEprHfss requires a bound AEDT request")
        object.__setattr__(self, "geometry", detached_data(self.geometry))
        object.__setattr__(self, "setup", detached_data(self.setup))
        object.__setattr__(
            self, "expressions", tuple(detached_data(item) for item in self.expressions)
        )
        object.__setattr__(self, "cache", detached_data(self.cache))
        object.__setattr__(self, "timings", detached_data(self.timings))


def analyze_epr(
    *,
    saved_solution: SavedSolution,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None = None,
) -> Any:
    """Analyze one sealed saved-field cohort in an owned disposable workcopy."""

    plan = _prepare_epr_analysis_handoff(
        saved_solution=saved_solution,
        geometry=geometry,
        project_name=project_name,
        design_name=design_name,
        run_control=run_control,
        epr_request=epr_request,
        output_dir=output_dir,
        geometry_workers=geometry_workers,
    )
    from scgsim.aedt.results.epr import resolve_epr_result
    from scgsim.aedt.runtime.transaction import _execute

    previous = Path.cwd()
    try:
        exit_code = _execute(plan.metadata_path, analyze_epr=True)
    finally:
        import os

        os.chdir(previous)
    if exit_code != 0:
        raise RuntimeError(
            "saved EPR analysis failed; inspect the prepared run receipt for the exact native error"
        )
    return resolve_epr_result(plan.run_dir / "results/epr/epr-result.json")


def prepare_epr_hfss(
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    inset_plan: dict[tuple[str, float], dict[str, Any]] | None = None,
) -> PreparedEprHfss:
    """Create, save, and read back one body-first no-solve Eigenmode model."""

    request = BoundAedtRequest.bind(run_dir, spec)
    bound = request.parse()
    if not isinstance(bound, HfssEprSpec):
        raise TypeError("bound EPR request did not retain its schema")
    if bound.modeling not in {"solid", "thin_film"} or (
        bound.geometry.modeling != bound.modeling
    ):
        raise ValueError("new EPR HFSS preparation requires explicit modeling")
    project_path = request.workspace / f"{bound.project_name}.aedt"
    timings: dict[str, Any] = {}
    preparation_started = time.perf_counter()
    started = time.perf_counter()
    app = Hfss(
        project=str(project_path),
        design=bound.design_name,
        solution_type="Eigenmode",
        new_desktop=False,
        close_on_exit=False,
    )
    timings["native_open_seconds"] = round(time.perf_counter() - started, 6)
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("HFSS EPR did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    started = time.perf_counter()
    geometry = prepare_native_planar_geometry(app, bound.geometry)
    timings["geometry_seconds"] = round(time.perf_counter() - started, 6)
    timings["geometry_phases"] = geometry.pop("geometry_phases")
    started = time.perf_counter()
    _create_setup(app, bound)
    timings["setup_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if bound.epr_request is None:
        expressions: list[dict[str, Any]] = []
        cache = {
            "schema_version": "scgsim.aedt.epr-cache-request.v1",
            "status": "disabled",
            "reason": "no EPR analysis request was supplied",
        }
    else:
        expressions, cache, phase_timings = _prepare_expressions_and_cache(
            app, request.workspace, bound, geometry, inset_plan
        )
        timings["epr_authoring_phases"] = phase_timings
    timings["epr_authoring_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS EPR project was not saved after preparation")
    timings["save_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    _read_saved_junction_lines(
        app, project_path, bound.design_name, bound.geometry, geometry
    )
    setup = _read_setup(app, bound)
    if bound.epr_request is not None:
        cache["serialized_readback"] = _read_cache(app, bound, cache["items"])
        native_references = cache["native_surface_references"]
        if native_references:
            cache["native_surface_references"] = verify_saved_native_surface_references(
                project_path, bound.design_name, native_references
            )
    timings["readback_seconds"] = round(time.perf_counter() - started, 6)
    preparation_wall = time.perf_counter() - preparation_started
    timings["preparation_wall_seconds"] = round(preparation_wall, 6)
    timings["preparation_residual_seconds"] = round(
        max(
            0.0,
            preparation_wall
            - sum(
                timings[key]
                for key in (
                    "native_open_seconds",
                    "geometry_seconds",
                    "setup_seconds",
                    "epr_authoring_seconds",
                    "save_seconds",
                    "readback_seconds",
                )
            ),
        ),
        6,
    )
    return PreparedEprHfss(
        app,
        request,
        project_path,
        geometry,
        setup,
        tuple(expressions),
        cache,
        timings,
    )


def prepared_epr_result(prepared: PreparedEprHfss) -> dict[str, Any]:
    """Return the canonical no-solve result without impersonating completion."""

    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec):
        raise TypeError("bound EPR request did not retain its schema")
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR project final preparation save failed")
    native_references = prepared.cache.get("native_surface_references", [])
    if native_references:
        prepared.cache["native_surface_references"] = (
            verify_saved_native_surface_references(
                prepared.project_path,
                spec.design_name,
                native_references,
            )
        )
    relative = prepared.project_path.relative_to(prepared.request.workspace).as_posix()
    digest = file_sha256(prepared.project_path)
    result = {
        "workflow_status": "native_preparation_only",
        "solver_invoked": False,
        "outputs": {relative: digest},
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "expressions": detached_data(prepared.expressions),
        "cache": detached_data(prepared.cache),
        "timings": detached_data(prepared.timings),
        "save": {"ok": True, "project_sha256": digest},
    }
    if spec.epr_request is not None:
        result["surface_analysis_scope"] = _SURFACE_ANALYSIS_SCOPE
    return result


def solve_and_export_epr(
    prepared: PreparedEprHfss,
    resources: AedtResources | None = None,
    resource_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the explicit setup once and export native final/history evidence."""

    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec):
        raise TypeError("body-first solve/export requires HfssEprSpec")
    timings = detached_data(prepared.timings)
    started = time.perf_counter()
    solved = analyze_with_resources(
        prepared.app,
        spec.run_control.setup_name,
        prepared.request.workspace,
        resources,
        resource_evidence if resource_evidence is not None else {},
    )
    timings["solve_seconds"] = round(time.perf_counter() - started, 6)
    if not solved:
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR solved project save failed")
    timings["post_solve_save_seconds"] = round(time.perf_counter() - started, 6)
    run_dir = prepared.request.workspace
    output_dir = run_dir / "results/epr"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs, final_modes = _export_eigenmode(prepared.app, run_dir, output_dir, spec)
    timings["final_export_seconds"] = round(time.perf_counter() - started, 6)
    convergence_path = output_dir / "adaptive-convergence.prop"
    started = time.perf_counter()
    returned = prepared.app.export_convergence(
        spec.run_control.setup_name,
        variations="",
        output_file=str(convergence_path),
    )
    if (
        Path(returned).resolve() != convergence_path.resolve()
        or not convergence_path.is_file()
    ):
        raise RuntimeError("HFSS adaptive convergence export is missing")
    outputs[convergence_path.relative_to(run_dir).as_posix()] = file_sha256(
        convergence_path
    )
    convergence = read_hfss_convergence(run_dir, spec)
    timings["convergence_export_seconds"] = round(time.perf_counter() - started, 6)
    saved_field_evidence = (
        _saved_field_evidence(prepared.app, spec, convergence)
        if spec.epr_request is not None
        else None
    )
    cache_items = prepared.cache["items"] if spec.epr_request is not None else []
    persisted_cache_items = (
        prepared.cache["serialized_readback"]["items"]
        if spec.epr_request is not None
        else []
    )
    started = time.perf_counter()
    history = _adaptive_mode_history(
        prepared.app, spec, cache_items, persisted_cache_items
    )
    history_path = output_dir / "adaptive-mode-history.json"
    write_json(history_path, history)
    outputs[history_path.relative_to(run_dir).as_posix()] = file_sha256(history_path)
    adaptive_result: EprResult | None = None
    if spec.epr_request is not None:
        adaptive_result = _adaptive_epr_result(
            spec,
            history,
            convergence,
            native_surface_references=prepared.cache.get(
                "native_surface_references", []
            ),
        )
        adaptive_result_path = output_dir / "adaptive-epr-result.json"
        write_json(adaptive_result_path, adaptive_result.to_payload())
        outputs[adaptive_result_path.relative_to(run_dir).as_posix()] = file_sha256(
            adaptive_result_path
        )
    timings["history_export_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("HFSS EPR final project save failed")
    timings["final_save_seconds"] = round(time.perf_counter() - started, 6)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    benchmark_relative, benchmark_digest, benchmark = export_simulation_benchmark(
        prepared.app, run_dir, spec.run_control.setup_name
    )
    outputs[benchmark_relative] = benchmark_digest
    return {
        "workflow_status": "completed",
        "solver_invoked": True,
        "outputs": outputs,
        "benchmark": benchmark,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "expressions": detached_data(prepared.expressions),
        "cache": detached_data(prepared.cache),
        **(
            {"surface_analysis_scope": _SURFACE_ANALYSIS_SCOPE}
            if spec.epr_request is not None
            else {}
        ),
        "timings": timings,
        "convergence": convergence,
        "result_readback": {
            "final_modes": final_modes,
            "adaptive_history": history,
            "adaptive_epr_result": (
                None if adaptive_result is None else adaptive_result.to_payload()
            ),
        },
        "save": {"ok": True, "project_sha256": outputs[relative]},
        **(
            {"saved_field_evidence": saved_field_evidence}
            if saved_field_evidence is not None
            else {}
        ),
    }


def _saved_field_evidence(
    app: Any, spec: HfssEprSpec, convergence: dict[str, Any]
) -> dict[str, Any]:
    """Bind the live completed setup's LastAdaptive field and variation identity."""

    expected_solution = f"{spec.run_control.setup_name} : LastAdaptive"
    solutions = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if expected_solution not in solutions:
        raise RuntimeError(
            f"completed EPR solve lacks saved Fields solution {expected_solution!r}: "
            f"{solutions!r}"
        )
    variation = detached_data(app.available_variations.nominal_values)
    if not isinstance(variation, dict) or any(
        not isinstance(key, str) for key in variation
    ):
        raise RuntimeError("completed EPR physical variation readback is invalid")
    final_pass = convergence.get("final_pass")
    if type(final_pass) is not int or final_pass <= 0:
        raise RuntimeError("completed EPR convergence pass identity is invalid")
    return {
        "saved_fields": True,
        "solver_last_completed_pass": final_pass,
        "saved_fields_pass": final_pass,
        "fields_solution": expected_solution,
        "physical_variation": json.dumps(
            variation, sort_keys=True, separators=(",", ":")
        ),
    }


def _evaluate_mode_integrals(
    app: Any,
    spec: HfssEprAnalysisSpec,
    expressions: list[dict[str, Any]],
    *,
    mode: int,
    frequency_hz: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pp_values = {
        f"scgsim_epr_pp_mode_{index}": "1" if index == mode else "0"
        for index in range(1, spec.run_control.num_modes + 1)
    }
    observed = install_variables(app, pp_values)
    if observed != pp_values:
        raise RuntimeError("EPR mode postprocessing assignment readback differs")
    solution = f"{spec.run_control.setup_name} : LastAdaptive"
    raw = _empty_raw_integrals()
    raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
    evidence: list[dict[str, Any]] = []
    for expression in expressions:
        if expression.get("cache_only"):
            continue
        identity = expression["identity"]
        selection = identity["selection"]
        purpose = str(identity["purpose"])
        target = _integral_target(purpose, selection)
        scalar = parse_native_scalar(
            app.post.fields_calculator.evaluate(
                expression["name"],
                setup=solution,
                intrinsics={"Phase": "0deg", **pp_values},
            )
        )
        value, item_evidence = _canonical_integral(
            scalar, target=target, purpose=purpose
        )
        _store_integral(
            raw,
            target=target,
            value=value,
            expression_identity={
                "name": expression["name"],
                "identity_sha256": identity["sha256"],
                "purpose": purpose,
            },
        )
        evidence.append(
            {
                "name": expression["name"],
                "identity_sha256": identity["sha256"],
                "purpose": purpose,
                **item_evidence,
            }
        )
    return raw, evidence


def analyze_saved_epr(
    Hfss: Any,
    run_dir: Path,
    spec: HfssEprAnalysisSpec,
    inset_plan: dict[tuple[str, float], dict[str, Any]],
) -> dict[str, Any]:
    """Analyze an immutable saved-field cohort through a disposable project copy."""

    if not isinstance(spec, HfssEprAnalysisSpec):
        raise TypeError("saved EPR analysis requires HfssEprAnalysisSpec")
    timings: dict[str, Any] = {}
    started = time.perf_counter()
    work_root = Path(run_dir) / "analysis-work"
    if work_root.exists():
        raise FileExistsError("saved EPR analysis workcopy already exists")
    work_root.mkdir(parents=True)
    for member in spec.saved_solution.members:
        relative = Path(str(member["path"]))
        source = spec.saved_solution.root / relative
        if (
            source.is_symlink()
            or source.resolve() != source
            or not source.is_file()
            or source.stat().st_size != member["bytes"]
            or file_sha256(source) != member["sha256"]
        ):
            raise RuntimeError(
                f"saved EPR member changed before workcopy: {relative.as_posix()}"
            )
        target = work_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    timings["workcopy_seconds"] = round(time.perf_counter() - started, 6)
    relative_project = spec.saved_solution.project_path.relative_to(
        spec.saved_solution.root
    )
    project_path = work_root / relative_project
    if not project_path.is_file():
        raise RuntimeError("saved EPR analysis project copy is missing")
    started = time.perf_counter()
    app = Hfss(
        project=str(project_path),
        design=spec.design_name,
        solution_type="Eigenmode",
        new_desktop=False,
        close_on_exit=False,
    )
    timings["native_reopen_seconds"] = round(time.perf_counter() - started, 6)
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("saved EPR analysis did not bind AEDT 2024.2")
    if spec.run_control.setup_name not in app.setup_names:
        raise RuntimeError("saved EPR analysis setup is missing")
    observed_variation = detached_data(app.available_variations.nominal_values)
    observed_variation_identity = json.dumps(
        observed_variation, sort_keys=True, separators=(",", ":")
    )
    if (
        observed_variation_identity
        != spec.saved_solution.identity["physical_variation"]
    ):
        raise RuntimeError("saved EPR physical variation identity differs")
    expected_fields_solution = f"{spec.run_control.setup_name} : LastAdaptive"
    available_fields = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if expected_fields_solution not in available_fields:
        raise RuntimeError("saved EPR field solution is unavailable on the workcopy")
    started = time.perf_counter()
    geometry = bind_saved_planar_geometry(app, spec.geometry)
    _read_saved_junction_lines(
        app, project_path, spec.design_name, spec.geometry, geometry
    )
    timings["geometry_rebind_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    expressions, authoring, phase_timings = _author_epr_expressions(
        app,
        work_root,
        spec,
        geometry,
        inset_plan,
        namespace=f"analysis_{spec.saved_solution.content_sha256[:16]}",
    )
    timings["expression_authoring_seconds"] = round(time.perf_counter() - started, 6)
    timings["expression_authoring_phases"] = phase_timings
    cache = {
        "schema_version": "scgsim.aedt.epr-cache-request.v1",
        "status": "not_modified_for_saved_field_analysis",
        **authoring,
    }
    output_dir = Path(run_dir) / "results/epr"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    outputs, final_modes = _export_eigenmode(app, Path(run_dir), output_dir, spec)
    timings["final_export_seconds"] = round(time.perf_counter() - started, 6)
    frequencies = final_modes["eigenmodes"]["frequencies_ghz"]
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    rows: list[dict[str, Any]] = []
    raw_evidence: list[dict[str, Any]] = []
    mode_timings: list[dict[str, Any]] = []
    for mode in selected_modes:
        mode_started = time.perf_counter()
        raw, evidence = _evaluate_mode_integrals(
            app,
            spec,
            expressions,
            mode=mode,
            frequency_hz=float(frequencies[mode - 1]) * 1e9,
        )
        rows.append(
            {
                "mode": mode,
                "status": "complete",
                "raw_integrals": raw,
                "raw_integral_evidence": evidence,
                **combine_epr_mode(spec.geometry, raw, request=spec.epr_request),
            }
        )
        raw_evidence.append({"mode": mode, "expressions": evidence})
        mode_timings.append(
            {
                "mode": mode,
                "seconds": round(time.perf_counter() - mode_started, 6),
            }
        )
    timings["mode_integrals"] = mode_timings
    result_provenance = {
        "model_source_sha256": spec.geometry.model_sha256,
        "analysis_source_sha256": spec.geometry.source_sha256,
        "surface_analysis_scope": _SURFACE_ANALYSIS_SCOPE,
        "saved_solution_content_sha256": spec.saved_solution.content_sha256,
        "saved_solution_identity": detached(spec.saved_solution.identity),
        "requested_modes": list(selected_modes),
        "request": spec.epr_request.to_payload(),
        "raw_integral_evidence": raw_evidence,
    }
    result = EprResult(
        result_kind="saved_field",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        _legacy_payload=spec._legacy_payload,
        provenance=result_provenance,
    )
    result_path = output_dir / "epr-result.json"
    started = time.perf_counter()
    write_json(result_path, result.to_payload())
    outputs[result_path.relative_to(run_dir).as_posix()] = file_sha256(result_path)
    timings["result_write_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("saved EPR analysis workcopy save failed")
    timings["workcopy_save_seconds"] = round(time.perf_counter() - started, 6)
    project_relative = project_path.relative_to(run_dir).as_posix()
    outputs[project_relative] = file_sha256(project_path)
    native_references = cache.get("native_surface_references", [])
    if native_references:
        native_references = verify_saved_native_surface_references(
            project_path, spec.design_name, native_references
        )
        cache["native_surface_references"] = native_references
        for row in rows:
            _attach_saved_native_surface_membership(
                row["raw_integrals"], row, native_references
            )
        result = EprResult(
            result_kind="saved_field",
            setup_name=spec.run_control.setup_name,
            rows=tuple(rows),
            _legacy_payload=spec._legacy_payload,
            provenance=result_provenance,
        )
        started = time.perf_counter()
        write_json(result_path, result.to_payload())
        outputs[result_path.relative_to(run_dir).as_posix()] = file_sha256(result_path)
        timings["result_write_seconds"] = round(
            timings["result_write_seconds"] + time.perf_counter() - started, 6
        )
    return {
        "workflow_status": "epr_analysis_completed",
        "solver_invoked": False,
        "outputs": outputs,
        "connected": {
            "aedt_version": app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": project_relative,
        "geometry": geometry,
        "setup": _read_setup(app, spec),
        "expressions": expressions,
        "cache": cache,
        "result": result.to_payload(),
        "timings": timings,
    }
