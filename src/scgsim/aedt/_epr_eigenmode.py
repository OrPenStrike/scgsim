"""Native Eigenmode preparation and EPR orchestration inside run.py ownership."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._epr_fields import (
    author_named_expression,
    compile_mask_region_union_operations,
    field_integral_operations,
    install_variables,
    junction_voltage_operations,
    mask_region_union_variables,
    parse_native_scalar,
)
from ._epr_geometry import bind_saved_planar_geometry, prepare_native_planar_geometry
from ._epr_models import EprResult, detached
from ._epr_results import combine_epr_mode
from ._hfss_convergence import read_hfss_convergence
from ._hfss_runtime import _export_eigenmode
from ._native_common import (
    BoundAedtRequest,
    detached_data,
    pyaedt_version,
    saved_setup_properties,
)
from .spec import REQUIRED_AEDT_VERSION, HfssEprAnalysisSpec, HfssEprSpec
from .util import file_sha256, write_json


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


def prepare_epr_hfss(
    Hfss: Any, run_dir: Path, spec: HfssEprSpec
) -> PreparedEprHfss:
    """Create, save, and read back one body-first no-solve Eigenmode model."""

    request = BoundAedtRequest.bind(run_dir, spec)
    bound = request.parse()
    if not isinstance(bound, HfssEprSpec):
        raise TypeError("bound EPR request did not retain its schema")
    project_path = request.workspace / f"{bound.project_name}.aedt"
    timings: dict[str, Any] = {}
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
        expressions, cache = _prepare_expressions_and_cache(
            app, request.workspace, bound, geometry
        )
    timings["epr_authoring_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS EPR project was not saved after preparation")
    timings["save_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    setup = _read_setup(app, bound)
    if bound.epr_request is not None:
        cache["serialized_readback"] = _read_cache(app, bound, cache["items"])
    timings["readback_seconds"] = round(time.perf_counter() - started, 6)
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
    relative = prepared.project_path.relative_to(prepared.request.workspace).as_posix()
    digest = file_sha256(prepared.project_path)
    return {
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


def solve_and_export_epr(prepared: PreparedEprHfss) -> dict[str, Any]:
    """Run the explicit setup once and export native final/history evidence."""

    spec = prepared.request.parse()
    if not isinstance(spec, HfssEprSpec):
        raise TypeError("body-first solve/export requires HfssEprSpec")
    timings = detached_data(prepared.timings)
    started = time.perf_counter()
    solved = prepared.app.analyze_setup(
        name=spec.run_control.setup_name,
        cores=2,
        tasks=1,
        gpus=0,
        use_auto_settings=False,
        blocking=True,
    )
    timings["solve_seconds"] = round(time.perf_counter() - started, 6)
    if not solved:
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
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
    if Path(returned).resolve() != convergence_path.resolve() or not convergence_path.is_file():
        raise RuntimeError("HFSS adaptive convergence export is missing")
    outputs[convergence_path.relative_to(run_dir).as_posix()] = file_sha256(
        convergence_path
    )
    convergence = read_hfss_convergence(run_dir, spec)
    timings["convergence_export_seconds"] = round(
        time.perf_counter() - started, 6
    )
    saved_field_evidence = (
        _saved_field_evidence(prepared.app, spec, convergence)
        if spec.epr_request is not None
        else None
    )
    cache_items = (
        prepared.cache["items"] if spec.epr_request is not None else []
    )
    persisted_cache_items = (
        prepared.cache["serialized_readback"]["items"]
        if spec.epr_request is not None else []
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
        adaptive_result = _adaptive_epr_result(spec, history, convergence)
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
    return {
        "workflow_status": "completed",
        "solver_invoked": True,
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": pyaedt_version(),
        },
        "project": relative,
        "geometry": detached_data(prepared.geometry),
        "setup": detached_data(prepared.setup),
        "expressions": detached_data(prepared.expressions),
        "cache": detached_data(prepared.cache),
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
    variation = detached_data(app.available_variations.nominal_w_values_dict)
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


def _canonical_integral(
    scalar: dict[str, Any], *, purpose: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize only the dimensions of the six owned CLC integral recipes."""

    expected_unit = _purpose_unit(purpose)
    native_unit = scalar["unit"]
    suffix = "" if native_unit is None else str(native_unit).strip()
    number = float(scalar["value"])
    if not math.isfinite(number):
        raise RuntimeError("native integral value is nonfinite")
    # A blank suffix is not native unit verification: the scale is the SI
    # dimension of this exact E/H/geometry integral recipe, never a generic
    # assumption about arbitrary calculator values.
    if suffix and re.sub(r"\s+", "", suffix) != expected_unit:
        raise RuntimeError(
            f"native integral unit differs: expected {expected_unit!r}, got {native_unit!r}"
        )
    canonical = {"value": number, "unit": expected_unit}
    evidence = {
        "raw": scalar.get("raw"),
        "native_value": number,
        "native_unit": native_unit,
        "canonical_unit": expected_unit,
        "scale_factor": 1.0,
        "scale_basis": (
            f"{purpose}:field_integral_operations_or_"
            "junction_voltage_operations_SI_dimensional_recipe"
        ),
        "unit_binding": (
            "native_suffix" if suffix else "recipe_basis_native_suffix_absent"
        ),
    }
    return canonical, evidence


def _store_integral(
    raw: dict[str, Any], *, purpose: str, selection: dict[str, Any], value: dict[str, Any]
) -> None:
    if purpose.startswith("effective_volume_"):
        raw["effective_domain_volumes_m3"][selection["semantic_id"]] = value
    elif purpose.startswith("electric_volume_"):
        raw["electric_domain_integrals_v2_m"][selection["semantic_id"]] = value
    elif purpose.startswith("magnetic_energy_"):
        raw["magnetic_domain_integrals_a2_m"][selection["semantic_id"]] = value
        raw["relative_permeability"][selection["semantic_id"]] = {
            "value": selection["relative_permeability"],
            "unit": "1",
        }
    elif purpose.startswith("electric_normal_") or purpose.startswith(
        "electric_tangential_"
    ):
        key = f"{selection['binding_id']}@{float(selection['margin_um']):.17g}"
        record = raw["surface_integrals_v2"].setdefault(key, {})
        component = "normal" if purpose.startswith("electric_normal_") else "tangential"
        record[component] = value
    elif purpose.startswith("masked_area_"):
        key = f"{selection['binding_id']}@{float(selection['margin_um']):.17g}"
        raw["masked_areas_m2"][key] = value
    elif purpose.startswith("junction_voltage_real_") or purpose.startswith(
        "junction_voltage_imag_"
    ):
        record = raw["junction_integrals_v_m"].setdefault(
            selection["junction_id"], {}
        )
        component = "real" if purpose.startswith("junction_voltage_real_") else "imag"
        record[component] = value
    else:
        raise RuntimeError(f"unknown EPR expression purpose {purpose!r}")


def _purpose_unit(purpose: str) -> str:
    if purpose.startswith("effective_volume_"):
        return "m^3"
    if purpose.startswith("electric_volume_"):
        return "V^2*m"
    if purpose.startswith("magnetic_energy_"):
        return "A^2*m"
    if purpose.startswith(("electric_normal_", "electric_tangential_")):
        return "V^2"
    if purpose.startswith("masked_area_"):
        return "m^2"
    if purpose.startswith(("junction_voltage_real_", "junction_voltage_imag_")):
        return "V*m"
    raise RuntimeError(f"unknown EPR expression purpose {purpose!r}")


def _empty_raw_integrals() -> dict[str, Any]:
    """Return the canonical recombination payload before native values are added."""

    return {
        "effective_domain_volumes_m3": {},
        "electric_domain_integrals_v2_m": {},
        "magnetic_domain_integrals_a2_m": {},
        "relative_permeability": {},
        "surface_integrals_v2": {},
        "masked_areas_m2": {},
        "junction_integrals_v_m": {},
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
        identity = expression["identity"]
        selection = identity["selection"]
        purpose = str(identity["purpose"])
        scalar = parse_native_scalar(
            app.post.fields_calculator.evaluate(
                expression["name"],
                setup=solution,
                intrinsics={"Phase": "0deg", **pp_values},
            )
        )
        value, item_evidence = _canonical_integral(
            scalar, purpose=purpose
        )
        _store_integral(raw, purpose=purpose, selection=selection, value=value)
        evidence.append(
            {
                "name": expression["name"],
                "identity_sha256": identity["sha256"],
                "purpose": purpose,
                **item_evidence,
            }
        )
    return raw, evidence


def analyze_saved_epr(Hfss: Any, run_dir: Path, spec: HfssEprAnalysisSpec) -> dict[str, Any]:
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
    observed_variation = detached_data(app.available_variations.nominal_w_values_dict)
    observed_variation_identity = json.dumps(
        observed_variation, sort_keys=True, separators=(",", ":")
    )
    if observed_variation_identity != spec.saved_solution.identity["physical_variation"]:
        raise RuntimeError("saved EPR physical variation identity differs")
    expected_fields_solution = f"{spec.run_control.setup_name} : LastAdaptive"
    available_fields = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if expected_fields_solution not in available_fields:
        raise RuntimeError("saved EPR field solution is unavailable on the workcopy")
    started = time.perf_counter()
    geometry = bind_saved_planar_geometry(app, spec.geometry)
    timings["geometry_rebind_seconds"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    expressions, authoring = _author_epr_expressions(
        app,
        work_root,
        spec,
        geometry,
        namespace=f"analysis_{spec.saved_solution.content_sha256[:16]}",
    )
    timings["expression_authoring_seconds"] = round(
        time.perf_counter() - started, 6
    )
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
    result = EprResult(
        result_kind="saved_field",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        provenance={
            "model_source_sha256": spec.geometry.model_sha256,
            "analysis_source_sha256": spec.geometry.source_sha256,
            "saved_solution_content_sha256": spec.saved_solution.content_sha256,
            "saved_solution_identity": detached(spec.saved_solution.identity),
            "requested_modes": list(selected_modes),
            "request": spec.epr_request.to_payload(),
            "raw_integral_evidence": raw_evidence,
        },
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


def _solution_trace(data: Any, expression: str) -> dict[str, Any]:
    """Detach one native trace while retaining its coordinate columns."""

    real, imaginary = data.full_matrix_real_imag
    variation_keys = tuple(data.variations[0]) if data.variations else ()
    intrinsic_keys = tuple(data.intrinsics_by_variation(0)) if data.variations else ()
    if any(
        tuple(variation) != variation_keys
        or tuple(data.intrinsics_by_variation(index)) != intrinsic_keys
        for index, variation in enumerate(data.variations)
    ):
        raise RuntimeError("native adaptive trace coordinate columns differ by variation")
    columns = [*variation_keys, *intrinsic_keys, "value"]
    real_rows = real[expression].tolist()
    imaginary_rows = imaginary[expression].tolist()
    if any(
        len(row) != len(columns)
        for row in (*real_rows, *imaginary_rows)
    ):
        raise RuntimeError("native adaptive trace coordinate width differs")
    return {
        "columns": columns,
        "real_rows": real_rows,
        "imaginary_rows": imaginary_rows,
        "unit": data.units_data.get(expression),
    }


def _trace_by_pass(trace: dict[str, Any]) -> dict[int, float]:
    columns = trace["columns"]
    if "Pass" not in columns or columns[-1] != "value":
        raise RuntimeError("native adaptive trace lacks explicit Pass/value columns")
    pass_index = columns.index("Pass")
    result: dict[int, float] = {}
    for row in trace["real_rows"]:
        pass_number = float(row[pass_index])
        if not pass_number.is_integer() or pass_number <= 0:
            raise RuntimeError("native adaptive trace Pass coordinate is invalid")
        pass_value = int(pass_number)
        value = float(row[-1])
        if not math.isfinite(value):
            raise RuntimeError("native adaptive trace value is nonfinite")
        if pass_value in result:
            raise RuntimeError("native adaptive trace repeats one Pass coordinate")
        result[pass_value] = value
    return result


def _trace_for_cache_context(
    trace: dict[str, Any], item: dict[str, Any], query: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select the actual one-hot rows before interpreting Pass coordinates."""

    expected = {
        name: float(value)
        for name, value in re.findall(
            r"(scgsim_epr_pp_mode_\d+)='([01])'", item["intrinsics"]
        )
    }
    columns = trace["columns"]
    present = {name for name in expected if name in columns}
    if not present:
        if "Phase" in columns and any(
            float(row[columns.index("Phase")]) != 0.0
            for row in trace["real_rows"]
        ):
            return trace, {
                "status": "mismatch",
                "reason": "native phase coordinate differs from scoped query",
                "expected_postprocessing_variables": expected,
            }
        status = (
            "verified_persisted_scoped_query"
            if query.get("persisted_binding") == "exact"
            and query.get("scope") == "one_hot_one_quantity"
            and query.get("returned_expressions") == [query.get("quantity")]
            and query.get("quantity_binding") in {"title", "unique_expression"}
            else "not_reported"
        )
        return trace, {
            "status": status,
            "expected_postprocessing_variables": expected,
            "evidence": (
                "persisted_item_and_scoped_native_query"
                if status.startswith("verified") else None
            ),
        }
    if present != set(expected):
        return trace, {
            "status": "partial",
            "expected_postprocessing_variables": expected,
            "reported_postprocessing_variables": sorted(present),
        }
    if len(trace["real_rows"]) != len(trace["imaginary_rows"]):
        return trace, {
            "status": "mismatch",
            "reason": "native real/imaginary coordinate counts differ",
            "expected_postprocessing_variables": expected,
        }
    selected_real: list[list[float]] = []
    selected_imaginary: list[list[float]] = []
    for real, imaginary in zip(trace["real_rows"], trace["imaginary_rows"]):
        if real[:-1] != imaginary[:-1]:
            return trace, {
                "status": "mismatch",
                "reason": "native real/imaginary coordinates differ",
                "expected_postprocessing_variables": expected,
            }
        if "Phase" in columns and float(real[columns.index("Phase")]) != 0.0:
            continue
        if all(
            float(real[columns.index(name)]) == value
            for name, value in expected.items()
        ):
            selected_real.append(real)
            selected_imaginary.append(imaginary)
    if not selected_real:
        return trace, {
            "status": "mismatch",
            "reason": "no native rows match the persisted one-hot context",
            "expected_postprocessing_variables": expected,
        }
    return {**trace, "real_rows": selected_real, "imaginary_rows": selected_imaginary}, {
        "status": "verified",
        "expected_postprocessing_variables": expected,
        "excluded_context_rows": len(trace["real_rows"]) - len(selected_real),
    }


def _adaptive_epr_result(
    spec: HfssEprSpec,
    history: dict[str, Any],
    convergence: dict[str, Any],
) -> EprResult:
    frequency_traces = history["frequency_traces"]
    cache = history["cache_integral_traces"]
    trace_by_title = cache["traces"]
    query_by_title = cache.get("queries", {})
    rows: list[dict[str, Any]] = []
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request is None or spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    for mode in selected_modes:
        frequency_trace = frequency_traces[f"Mode({mode})"]
        frequencies = _trace_by_pass(frequency_trace)
        mode_items = [item for item in cache["items"] if item["mode"] == mode]
        item_values: dict[str, dict[int, float]] = {}
        item_contexts: dict[str, dict[str, Any]] = {}
        for item in mode_items:
            title = item["title"]
            if title not in trace_by_title:
                continue
            scoped_trace, context = _trace_for_cache_context(
                trace_by_title[title], item, query_by_title.get(title, {})
            )
            item_contexts[title] = context
            if not context["status"].startswith("verified"):
                continue
            try:
                item_values[title] = _trace_by_pass(scoped_trace)
            except RuntimeError as exc:
                item_contexts[title] = {
                    **context,
                    "status": "invalid_pass_axis",
                    "error": str(exc),
                }
        pass_ids = sorted(
            set(frequencies).union(
                *(set(values) for values in item_values.values())
            )
        )
        for pass_id in pass_ids:
            missing = [
                item["title"]
                for item in mode_items
                if pass_id not in item_values.get(item["title"], {})
            ]
            invalid_context = [
                title
                for title, status in item_contexts.items()
                if not status["status"].startswith("verified")
            ]
            row: dict[str, Any] = {
                "mode": mode,
                "native_pass": pass_id,
                "status": "partial",
                "missing_cache_titles": missing,
                "unverified_cache_context_titles": invalid_context,
                "frequency_native_unit": frequency_trace["unit"],
            }
            raw = _empty_raw_integrals()
            integral_evidence: list[dict[str, Any]] = []
            unit_mismatches: list[dict[str, Any]] = []
            for item in mode_items:
                title = item["title"]
                values = item_values.get(title, {})
                if pass_id not in values:
                    continue
                native_unit = trace_by_title[title].get("unit")
                context = item_contexts[title]
                try:
                    value, unit_evidence = _canonical_integral(
                        {"value": values[pass_id], "unit": native_unit},
                        purpose=item["purpose"],
                    )
                except RuntimeError as exc:
                    unit_mismatches.append(
                        {
                            "title": title,
                            "native_unit": native_unit,
                            "expected_unit": _purpose_unit(item["purpose"]),
                            "error": str(exc),
                        }
                    )
                    continue
                integral_evidence.append(
                    {
                        "title": title,
                        "purpose": item["purpose"],
                        "context": context,
                        **unit_evidence,
                    }
                )
                _store_integral(
                    raw,
                    purpose=item["purpose"],
                    selection=item["selection"],
                    value=value,
                )
            row["raw_integrals"] = raw
            row["raw_integral_evidence"] = integral_evidence
            if unit_mismatches:
                row["unit_mismatches"] = unit_mismatches
            frequency = frequencies.get(pass_id)
            if frequency is None:
                row["missing_frequency"] = True
                rows.append(row)
                continue
            if frequency_trace["unit"] == "GHz":
                frequency_hz = frequency * 1e9
            elif frequency_trace["unit"] == "Hz":
                frequency_hz = frequency
            else:
                row["frequency_native_value"] = frequency
                row["frequency_unit_status"] = "unverified"
                rows.append(row)
                continue
            raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
            if missing or invalid_context or unit_mismatches:
                row["frequency_hz"] = frequency_hz
                rows.append(row)
                continue
            try:
                row.update(
                    combine_epr_mode(
                        spec.geometry, raw, request=spec.epr_request
                    )
                )
                row["status"] = "complete"
            except (TypeError, ValueError, RuntimeError) as exc:
                row["combination_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return EprResult(
        result_kind="adaptive_history",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        provenance={
            "model_source_sha256": spec.geometry.model_sha256,
            "analysis_source_sha256": spec.geometry.source_sha256,
            "solver_last_completed_pass": convergence["final_pass"],
            "requested_modes": list(selected_modes),
            "cache_status": cache["status"],
            "report_inventory": history["report_inventory"],
            "join_identity": [
                "setup",
                "native_pass",
                "mode",
                "cache_title",
                "intrinsics",
            ],
        },
    )


def _adaptive_mode_history(
    app: Any,
    spec: HfssEprSpec,
    cache_items: list[dict[str, Any]],
    persisted_cache_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return raw frequency and cache traces without positional joins."""

    category = "Eigenmode Parameters"
    solution = f"{spec.run_control.setup_name} : AdaptivePass"
    frequency_solutions = [str(item) for item in app.post.available_report_solutions(category) or ()]
    field_solutions = [str(item) for item in app.post.available_report_solutions("Fields") or ()]
    if solution not in frequency_solutions:
        raise RuntimeError(
            f"native adaptive solution is unavailable for {category!r}: {frequency_solutions!r}"
        )
    quantities: list[str] = []
    for quantity_category in app.post.available_quantities_categories(
        category, solution=solution
    ) or ():
        for quantity in app.post.available_report_quantities(
            category,
            solution=solution,
            quantities_category=quantity_category,
        ) or ():
            if str(quantity).startswith("Mode(") and quantity not in quantities:
                quantities.append(str(quantity))
    expected = [f"Mode({index})" for index in range(1, spec.run_control.num_modes + 1)]
    if len(quantities) != len(expected) or set(quantities) != set(expected):
        raise RuntimeError(
            f"native adaptive Eigenmode quantities differ: {quantities!r}"
        )
    data = app.post.get_solution_data_per_variation(
        category, solution, [], {"Pass": "All"}, quantities
    )
    if not data:
        raise RuntimeError("native adaptive Eigenmode history is unavailable")
    data.primary_sweep = "Pass"
    rows = {quantity: _solution_trace(data, quantity) for quantity in expected}
    cache_titles = [item["title"] for item in cache_items]
    cache_observation: dict[str, Any] = {
        "status": "not_requested" if not cache_titles else "unavailable",
        "requested_titles": cache_titles,
        "items": cache_items,
        "traces": {},
        "queries": {},
    }
    if cache_titles:
        try:
            if solution not in field_solutions:
                raise RuntimeError(
                    f"native adaptive solution is unavailable for Fields: {field_solutions!r}"
                )
            field_categories = [
                str(item)
                for item in app.post.available_quantities_categories(
                    "Fields", solution=solution
                )
                or ()
            ]
            field_quantities: dict[str, list[str]] = {}
            for quantity_category in field_categories:
                field_quantities[quantity_category] = [
                    str(item)
                    for item in app.post.available_report_quantities(
                        "Fields",
                        solution=solution,
                        quantities_category=quantity_category,
                    )
                    or ()
                ]
            cache_observation["available_quantity_categories"] = field_categories
            cache_observation["available_quantities"] = field_quantities
            persisted_by_title = {
                item["title"]: item for item in persisted_cache_items
            }
            expression_counts = {
                expression: sum(
                    item["expression"] == expression for item in cache_items
                )
                for expression in {item["expression"] for item in cache_items}
            }
            if len(persisted_by_title) != len(persisted_cache_items):
                raise RuntimeError("persisted EPR cache titles are not unique")
            for item in cache_items:
                title = item["title"]
                query: dict[str, Any] = {
                    "persisted_item": persisted_by_title.get(title),
                    "scope": "one_hot_one_quantity",
                }
                cache_observation["queries"][title] = query
                persisted = persisted_by_title.get(title)
                if persisted != {
                    "title": title,
                    "expression": item["expression"],
                    "intrinsics": item["intrinsics"],
                }:
                    query["error"] = "persisted cache item expression/context differs"
                    continue
                query["persisted_binding"] = "exact"
                if item["intrinsics"] != _cache_intrinsics(
                    spec.run_control.num_modes, item["mode"]
                ):
                    query["error"] = "cache item is not the complete one-hot context"
                    continue
                matches = [
                    (category_name, quantity)
                    for category_name, names in field_quantities.items()
                    for quantity in names
                    if quantity in {persisted["title"], persisted["expression"]}
                ]
                query["inventory_matches"] = matches
                if len(matches) != 1:
                    query["error"] = "native quantity binding is missing or ambiguous"
                    continue
                quantity = matches[0][1]
                query["quantity"] = quantity
                query["quantity_binding"] = (
                    "title" if quantity == persisted["title"]
                    else "unique_expression"
                    if expression_counts[quantity] == 1
                    else "shared_expression_requires_reported_axes"
                )
                sweeps = {"Pass": "All", "Phase": "0deg"}
                sweeps.update(
                    {
                        f"scgsim_epr_pp_mode_{index}": (
                            "1" if index == item["mode"] else "0"
                        )
                        for index in range(1, spec.run_control.num_modes + 1)
                    }
                )
                query["sweeps"] = sweeps
                try:
                    cached = app.post.get_solution_data_per_variation(
                        "Fields", solution, [], sweeps, [quantity]
                    )
                    if not cached:
                        raise RuntimeError("native scoped cache report is unavailable")
                    returned = list(cached.expressions)
                    query["returned_expressions"] = returned
                    if returned != [quantity]:
                        raise RuntimeError("native scoped cache quantity differs")
                    cached.primary_sweep = "Pass"
                    cache_observation["traces"][title] = _solution_trace(
                        cached, quantity
                    )
                    query["status"] = "reported"
                except Exception as exc:  # noqa: BLE001 -- retain per-item partial history.
                    query["error"] = f"{type(exc).__name__}: {exc}"
            returned_titles = sorted(cache_observation["traces"])
            cache_observation["returned_titles"] = returned_titles
            cache_observation["missing_titles"] = sorted(
                set(cache_titles) - set(returned_titles)
            )
            cache_observation["unexpected_titles"] = []
            cache_observation["status"] = (
                "complete"
                if len(returned_titles) == len(cache_titles)
                else "partial"
            )
        except Exception as exc:  # noqa: BLE001 -- partial native history is evidence.
            cache_observation["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": "scgsim.aedt.epr-adaptive-history.v1",
        "category": category,
        "solution": solution,
        "primary_sweep": data.primary_sweep,
        "pass_axis": data.primary_sweep_values.tolist(),
        "pass_unit": data.units_sweeps.get("Pass"),
        "frequency_traces": rows,
        "cache_integral_traces": cache_observation,
        "report_inventory": {
            "eigenmode_parameter_solutions": frequency_solutions,
            "field_solutions": field_solutions,
        },
        "frequency_unit_status": "native_reported_only",
    }


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_intrinsics(mode_count: int, selected_mode: int) -> str:
    assignments = ["Phase='0deg'"]
    assignments.extend(
        f"scgsim_epr_pp_mode_{index}='{'1' if index == selected_mode else '0'}'"
        for index in range(1, mode_count + 1)
    )
    return " ".join(assignments)


def _cache_items(
    expressions: list[dict[str, Any]],
    mode_count: int,
    selected_modes: tuple[int, ...],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for mode in selected_modes:
        intrinsics = _cache_intrinsics(mode_count, mode)
        for expression in expressions:
            identity = {
                "name": expression["name"],
                "identity_sha256": expression["identity"]["sha256"],
                "intrinsics": intrinsics,
                "mode": mode,
            }
            items.append(
                {
                    "title": f"scgsim_epr_cache_m{mode}_{_identity_sha256(identity)[:16]}",
                    "expression": expression["name"],
                    "intrinsics": intrinsics,
                    "is_convergence": False,
                    "mode": mode,
                    "expression_identity_sha256": expression["identity"]["sha256"],
                    "purpose": expression["identity"]["purpose"],
                    "selection": detached_data(expression["identity"]["selection"]),
                }
            )
    if len({item["title"] for item in items}) != len(items):
        raise RuntimeError("compiled EPR cache titles are not unique")
    return items


def _native_expression_cache(items: list[dict[str, Any]]) -> list[Any]:
    result: list[Any] = ["NAME:ExpressionCache"]
    for item in items:
        result.append(
            [
                "NAME:CacheItem",
                "Title:=",
                item["title"],
                "Expression:=",
                item["expression"],
                "Intrinsics:=",
                item["intrinsics"],
                "IsConvergence:=",
                False,
                "UseRelativeConvergence:=",
                0,
                "MaxConvergenceDelta:=",
                1,
                "MaxConvergeValue:=",
                "1",
                "ReportType:=",
                "Fields",
                ["NAME:ExpressionContext"],
            ]
        )
    return result


def _author_epr_expressions(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
    *,
    namespace: str = "prepared",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR expression preparation requires an analysis request")
    solution = f"{spec.run_control.setup_name} : LastAdaptive"
    evidence_dir = run_dir / "metadata/epr_expressions"
    expressions: list[dict[str, Any]] = []
    pp_values = {
        f"scgsim_epr_pp_mode_{index}": "0"
        for index in range(1, spec.run_control.num_modes + 1)
    }
    pp_observed = install_variables(app, pp_values)
    source_assignment = {
        str(index): (f"scgsim_epr_pp_mode_{index}", "0deg")
        for index in range(1, spec.run_control.num_modes + 1)
    }
    if app.edit_sources(source_assignment, eigenmode_stored_energy=False) is not True:
        raise RuntimeError("native Eigenmode PP source-vector installation failed")
    objects = native_geometry["objects"]
    material_readback = native_geometry["material_readback"]
    for item in objects:
        if item["kind"] != "solution_domain":
            continue
        for quantity in ("effective_volume", "electric_volume", "magnetic_energy"):
            operations = field_integral_operations(
                quantity=quantity, selection_name=item["object_name"]
            )
            expressions.append(
                author_named_expression(
                    app,
                    purpose=f"{quantity}_{item['semantic_id']}",
                    operations=operations,
                    solution=solution,
                    phase_degrees=0.0,
                    dependencies=pp_observed,
                    selection={
                        "kind": "volume",
                        "semantic_id": item["semantic_id"],
                        "object_name": item["object_name"],
                        "relative_permeability": material_readback[
                            item["material_id"]
                        ]["relative_permeability"],
                    },
                    evidence_dir=evidence_dir,
                    namespace=namespace,
                )
            )
    bindings = {
        str(item["binding_id"]): item for item in spec.geometry.surface_bindings
    }
    selected_contributions = (
        {item.contribution_id for item in spec.geometry.contributions}
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    for selection in native_geometry["surface_selections"]:
        binding = bindings[selection["binding_id"]]
        if selection["contribution_id"] not in selected_contributions:
            continue
        plane = binding["mask_plane"]
        for margin in binding["margins_um"]:
            support = binding["mask_support"]
            support_regions = support["support_regions"]
            attribution_regions = support["attribution_regions"]
            mask_id = (
                f"{selection['binding_id']}_m_"
                f"{_identity_sha256({'margin_um': float(margin)})[:12]}"
            )
            variables = mask_region_union_variables(
                mask_id,
                support_regions,
                attribution_regions,
                margin_um=float(margin),
                plane_origin_um=plane["origin_um"],
                plane_u=plane["u"],
                plane_v=plane["v"],
            )
            observed = install_variables(app, variables)
            dependencies = {**pp_observed, **observed}
            mask = compile_mask_region_union_operations(
                patch_id=mask_id,
                support_regions=support_regions,
                attribution_regions=attribution_regions,
                margin_um=float(margin),
                plane_origin_um=plane["origin_um"],
                plane_u=plane["u"],
                plane_v=plane["v"],
            )
            for quantity in ("electric_normal", "electric_tangential"):
                operations = field_integral_operations(
                    quantity=quantity,
                    selection_name=selection["selection_name"],
                    mask_operations=mask,
                    adjacent_side=selection["adjacent_side"],
                    normal_vector=selection["native_normal"],
                )
                expressions.append(
                    author_named_expression(
                        app,
                        purpose=(
                            f"{quantity}_{selection['binding_id']}_"
                            f"{float(margin):.17g}um"
                        ),
                        operations=operations,
                        solution=solution,
                        phase_degrees=0.0,
                        dependencies=dependencies,
                        selection={**selection, "margin_um": float(margin)},
                        evidence_dir=evidence_dir,
                        namespace=namespace,
                        adjacent_selection_name=(
                            selection["selection_name"]
                            if selection["adjacent_side"] else None
                        ),
                    )
                )
            area_operations = field_integral_operations(
                quantity="masked_area",
                selection_name=selection["selection_name"],
                mask_operations=mask,
                adjacent_side=selection["adjacent_side"],
            )
            expressions.append(
                author_named_expression(
                    app,
                    purpose=(
                        f"masked_area_{selection['binding_id']}_"
                        f"{float(margin):.17g}um"
                    ),
                    operations=area_operations,
                    solution=solution,
                    phase_degrees=0.0,
                    dependencies=dependencies,
                    selection={**selection, "margin_um": float(margin)},
                    evidence_dir=evidence_dir,
                    namespace=namespace,
                )
            )
    junction_specs = {item.junction_id: item for item in spec.geometry.junctions}
    selected_junctions = (
        set(junction_specs)
        if request.junction_ids is None
        else set(request.junction_ids)
    )
    for selection in native_geometry["junctions"]:
        junction = junction_specs[selection["junction_id"]]
        # Every model junction contributes capacitive energy to the common
        # normalization.  junction_ids controls only reported junction rows.
        if junction.junction_id not in selected_junctions:
            report_selected = False
        else:
            report_selected = True
        for quantity in ("junction_voltage_real", "junction_voltage_imag"):
            operations = junction_voltage_operations(
                quantity=quantity,
                selection_name=selection["object_name"],
                direction_xy=junction.direction_xy,
            )
            expressions.append(
                author_named_expression(
                    app,
                    purpose=f"{quantity}_{junction.junction_id}",
                    operations=operations,
                    solution=solution,
                    phase_degrees=0.0,
                    dependencies=pp_observed,
                        selection={
                        "kind": "junction_sheet",
                        "junction_id": junction.junction_id,
                        "object_name": selection["object_name"],
                            "direction_xy": list(junction.direction_xy),
                            "report_selected": report_selected,
                    },
                    evidence_dir=evidence_dir,
                    namespace=namespace,
                )
            )
    return expressions, {
        "source_assignment": source_assignment,
        "postprocessing_variables": pp_observed,
    }


def _submit_epr_cache(
    app: Any,
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    authoring: dict[str, Any],
) -> dict[str, Any]:
    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR cache submission requires an analysis request")
    selected_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if request.mode_indices is None
        else request.mode_indices
    )
    items = _cache_items(expressions, spec.run_control.num_modes, selected_modes)
    setup = app.get_setup(spec.run_control.setup_name)
    before = detached_data(setup.props)
    properties = copy.deepcopy(before)
    properties.pop("ExpressionCache", None)
    properties["UseCacheFor"] = ["Pass"]
    raw_args = setup._setup_dict_to_arg(
        name=spec.run_control.setup_name, props=properties
    )
    raw_cache = _native_expression_cache(items)
    raw_args.append(raw_cache)
    setup.omodule.EditSetup(spec.run_control.setup_name, raw_args)
    return {
        "schema_version": "scgsim.aedt.epr-cache-request.v1",
        "setup_name": spec.run_control.setup_name,
        "use_cache_for": ["Pass"],
        "items": items,
        "source_assignment": authoring["source_assignment"],
        "postprocessing_variables": authoring["postprocessing_variables"],
        "native_readback": {
            "status": "NOT_VERIFIED_NATIVE_READBACK",
            "reason": (
                "AEDT 2024.2 serialized setup omits IsConvergence and no "
                "verified direct complete cache getter is available"
            ),
        },
        "raw_cache_args_sha256": _identity_sha256(raw_cache),
    }


def _prepare_expressions_and_cache(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expressions, authoring = _author_epr_expressions(
        app,
        run_dir,
        spec,
        native_geometry,
    )
    return expressions, _submit_epr_cache(app, spec, expressions, authoring)


def _create_setup(app: Any, spec: HfssEprSpec) -> None:
    if app.setup_names:
        raise RuntimeError("new EPR design must not inherit a setup")
    setup = app.create_setup(spec.run_control.setup_name)
    if setup is None:
        raise RuntimeError("HFSS EPR setup creation failed")
    setup.props["MinimumFrequency"] = f"{spec.run_control.minimum_frequency_ghz:g}GHz"
    setup.props["NumModes"] = spec.run_control.num_modes
    setup.props["MaxDeltaFreq"] = spec.run_control.maximum_delta_frequency_percent
    setup.props["MaximumPasses"] = spec.run_control.maximum_passes
    setup.props["MinimumPasses"] = spec.run_control.minimum_passes
    setup.props["MinimumConvergedPasses"] = spec.run_control.minimum_converged_passes
    setup.props["PercentRefinement"] = spec.run_control.percent_refinement
    setup.props["SaveAnyFields"] = True
    setup.props["SaveRadFieldsOnly"] = False
    if not setup.update():
        raise RuntimeError("HFSS EPR setup update failed")


def _read_setup(app: Any, spec: HfssEprSpec) -> dict[str, Any]:
    raw = saved_setup_properties(app, spec.run_control.setup_name)
    observed = {
        "minimum_frequency": raw.get("MinimumFrequency"),
        "num_modes": raw.get("NumModes"),
        "maximum_delta_frequency_percent": raw.get("MaxDeltaFreq"),
        "maximum_passes": raw.get("MaximumPasses"),
        "minimum_passes": raw.get("MinimumPasses"),
        "minimum_converged_passes": raw.get("MinimumConvergedPasses"),
        "percent_refinement": raw.get("PercentRefinement"),
    }
    expected = {
        "minimum_frequency": f"{spec.run_control.minimum_frequency_ghz:g}GHz",
        "num_modes": spec.run_control.num_modes,
        "maximum_delta_frequency_percent": (
            spec.run_control.maximum_delta_frequency_percent
        ),
        "maximum_passes": spec.run_control.maximum_passes,
        "minimum_passes": spec.run_control.minimum_passes,
        "minimum_converged_passes": spec.run_control.minimum_converged_passes,
        "percent_refinement": spec.run_control.percent_refinement,
    }
    if observed != expected:
        raise RuntimeError(f"HFSS EPR saved setup readback mismatch: {observed!r}")
    saved_field_properties = {
        "requested": {
            "SaveAnyFields": True,
            "SaveRadFieldsOnly": False,
        },
        "serialized": {
            key: raw[key]
            for key in ("SaveAnyFields", "SaveRadFieldsOnly")
            if key in raw
        },
    }
    saved_field_properties["status"] = (
        "verified_serialized"
        if saved_field_properties["serialized"]
        == saved_field_properties["requested"]
        else "verification_deferred_to_completed_saved_field_inventory"
    )
    return {
        "name": spec.run_control.setup_name,
        "native": observed,
        "saved_fields": saved_field_properties,
    }


def _read_cache(
    app: Any, spec: HfssEprSpec, submitted: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = saved_setup_properties(app, spec.run_control.setup_name)
    use_cache_for = raw.get("UseCacheFor")
    if isinstance(use_cache_for, str):
        use_cache_for = [use_cache_for]
    expression_cache = raw.get("ExpressionCache")
    if not isinstance(expression_cache, dict):
        raise RuntimeError("saved EPR setup lacks ExpressionCache")
    raw_items = expression_cache.get("CacheItem")
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raise RuntimeError("saved EPR ExpressionCache items are invalid")
    persisted = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise RuntimeError("saved EPR cache item is invalid")
        persisted.append(
            {
                "title": item.get("Title"),
                "expression": item.get("Expression"),
                "intrinsics": str(item.get("Intrinsics", "")).replace("\\'", "'"),
            }
        )
    expected = [
        {
            "title": item["title"],
            "expression": item["expression"],
            "intrinsics": item["intrinsics"],
        }
        for item in submitted
    ]
    if use_cache_for != ["Pass"] or persisted != expected:
        raise RuntimeError("saved EPR cache persisted-subset readback mismatch")
    return {
        "use_cache_for": use_cache_for,
        "items": persisted,
        "convergence_fields": "NOT_VERIFIED_SERIALIZED_OMISSION",
    }


__all__ = [
    "PreparedEprHfss",
    "analyze_saved_epr",
    "prepare_epr_hfss",
    "prepared_epr_result",
    "solve_and_export_epr",
]
