"""Native expression/cache preparation and saved-trace interpretation."""

from __future__ import annotations


import copy

import hashlib

import json

import math

import re

import time

from collections.abc import Mapping

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Literal

from scgsim.aedt.epr.analysis import (
    _electric_domain_energy_coefficient,
    _epr_domain_context,
    _junction_capacitive_energy_coefficient,
    _surface_energy_coefficients,
    combine_epr_mode,
)

from scgsim.aedt.epr.fields import (
    compile_named_expression,
    field_integral_operations,
    install_variables,
    junction_voltage_operations,
    load_compiled_expressions,
)

from scgsim.aedt.epr.geometry import plan_inset_sheet_names

from scgsim.aedt.epr.models import (
    EprResult,
    ExpressionCacheConvergence,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    _freeze,
    detached,
    surface_evaluations,
)

from scgsim.aedt.epr.native import bind_inset_surface_selections

from scgsim.aedt.epr.selection import surface_integral_groups

from scgsim.aedt.runtime.native.common import detached_data, saved_setup_properties

from scgsim.aedt.specs.hfss import HfssEprAnalysisSpec, HfssEprSpec


_SURFACE_ANALYSIS_SCOPE = {
    "included_field_sides": ["top", "bottom"],
    "excluded_field_sides": ["sidewall"],
    "excluded_sidewall_energy": "not_evaluated_not_zero",
    "q2d_sidewall_correction": "not_applied",
}

IntegralKind = Literal[
    "effective_volume",
    "electric_volume",
    "magnetic_energy",
    "electric_normal",
    "electric_tangential",
    "masked_area",
    "junction_voltage_real",
    "junction_voltage_imag",
]

_INTEGRAL_KINDS: tuple[IntegralKind, ...] = (
    "effective_volume",
    "electric_volume",
    "magnetic_energy",
    "electric_normal",
    "electric_tangential",
    "masked_area",
    "junction_voltage_real",
    "junction_voltage_imag",
)


@dataclass(frozen=True, slots=True)
class IntegralTarget:
    """Private, immutable interpretation of one persisted expression binding."""

    kind: IntegralKind
    scope_id: str
    selection: Mapping[str, Any]


def _integral_target(purpose: str, selection: Mapping[str, Any]) -> IntegralTarget:
    """Decode the legacy identity once; never add these fields to its wire form."""

    kind = next(
        (item for item in _INTEGRAL_KINDS if purpose.startswith(f"{item}_")),
        None,
    )
    if kind is None:
        raise RuntimeError(f"unknown EPR expression purpose {purpose!r}")
    if not isinstance(selection, Mapping):
        raise TypeError("EPR expression selection must be a mapping")
    if kind in {"effective_volume", "electric_volume", "magnetic_energy"}:
        selection_kind, key = "volume", "semantic_id"
    elif kind in {"electric_normal", "electric_tangential", "masked_area"}:
        selection_kind, key = "surface_group", "group_id"
    else:
        selection_kind, key = "junction_sheet", "junction_id"
    scope_id = selection.get(key)
    if (
        selection.get("kind") != selection_kind
        or not isinstance(scope_id, str)
        or not scope_id
    ):
        raise RuntimeError(f"{kind} requires a {selection_kind} selection with {key}")
    if kind == "magnetic_energy":
        permeability = selection.get("relative_permeability")
        if (
            isinstance(permeability, bool)
            or not isinstance(permeability, (int, float))
            or not math.isfinite(permeability)
            or permeability <= 0
        ):
            raise RuntimeError(
                "magnetic energy requires positive relative permeability"
            )
    return IntegralTarget(kind, scope_id, _freeze(selection))


def _canonical_integral(
    scalar: dict[str, Any], *, target: IntegralTarget, purpose: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize only the dimensions of the six owned CLC integral recipes."""

    expected_unit = _purpose_unit(target.kind)
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
    raw: dict[str, Any], *, target: IntegralTarget, value: dict[str, Any]
) -> None:
    if target.kind == "effective_volume":
        raw["effective_domain_volumes_m3"][target.scope_id] = value
    elif target.kind == "electric_volume":
        raw["electric_domain_integrals_v2_m"][target.scope_id] = value
    elif target.kind == "magnetic_energy":
        raw["magnetic_domain_integrals_a2_m"][target.scope_id] = value
        raw["relative_permeability"][target.scope_id] = {
            "value": target.selection["relative_permeability"],
            "unit": "1",
        }
    elif target.kind in {"electric_normal", "electric_tangential"}:
        key = target.scope_id
        selection = detached(target.selection)
        if (
            key in raw["surface_group_provenance"]
            and raw["surface_group_provenance"][key] != selection
        ):
            raise RuntimeError("surface group expression provenance differs")
        raw["surface_group_provenance"][key] = selection
        record = raw["surface_integrals_v2"].setdefault(key, {})
        component = "normal" if target.kind == "electric_normal" else "tangential"
        record[component] = value
    elif target.kind == "masked_area":
        key = target.scope_id
        selection = detached(target.selection)
        if (
            key in raw["surface_group_provenance"]
            and raw["surface_group_provenance"][key] != selection
        ):
            raise RuntimeError("surface group expression provenance differs")
        raw["surface_group_provenance"][key] = selection
        raw["masked_areas_m2"][key] = value
    elif target.kind in {"junction_voltage_real", "junction_voltage_imag"}:
        record = raw["junction_integrals_v_m"].setdefault(target.scope_id, {})
        component = "real" if target.kind == "junction_voltage_real" else "imag"
        record[component] = value
    else:
        raise AssertionError(f"unsupported integral kind {target.kind!r}")


def _purpose_unit(kind: IntegralKind) -> str:
    if kind == "effective_volume":
        return "m^3"
    if kind == "electric_volume":
        return "V^2*m"
    if kind == "magnetic_energy":
        return "A^2*m"
    if kind in {"electric_normal", "electric_tangential"}:
        return "V^2"
    if kind == "masked_area":
        return "m^2"
    if kind in {"junction_voltage_real", "junction_voltage_imag"}:
        return "V*m"
    raise AssertionError(f"unsupported integral kind {kind!r}")


def _empty_raw_integrals() -> dict[str, Any]:
    """Return the canonical recombination payload before native values are added."""

    return {
        "effective_domain_volumes_m3": {},
        "electric_domain_integrals_v2_m": {},
        "magnetic_domain_integrals_a2_m": {},
        "relative_permeability": {},
        "surface_integrals_v2": {},
        "masked_areas_m2": {},
        "surface_granularity": "owner_interface_margin.v1",
        "surface_group_provenance": {},
        "junction_integrals_v_m": {},
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
        raise RuntimeError(
            "native adaptive trace coordinate columns differ by variation"
        )
    columns = [*variation_keys, *intrinsic_keys, "value"]
    real_rows = real[expression].tolist()
    imaginary_rows = imaginary[expression].tolist()
    if any(len(row) != len(columns) for row in (*real_rows, *imaginary_rows)):
        raise RuntimeError("native adaptive trace coordinate width differs")
    return {
        "columns": columns,
        "real_rows": real_rows,
        "imaginary_rows": imaginary_rows,
        "unit": data.units_data.get(expression),
    }


def _frequency_solution_trace(
    data: Any, expression: str, *, category: str, solution: str
) -> dict[str, Any]:
    """Retain raw frequency and optional explicit native SI evidence."""

    trace = _solution_trace(data, expression)
    if trace["unit"] != "":
        return trace
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "query": {
            "category": category,
            "solution": solution,
            "expression": expression,
            "sweeps": {"Pass": "All"},
            "si_value": True,
            "real_method": "GetRealDataValues",
            "imaginary_method": "GetImagDataValues",
        },
        "unit": "Hz",
        "variations": [],
    }
    trace["si_evidence"] = evidence
    # SI evidence is optional; failure must retain the original partial trace.
    try:
        import numpy as np

        if len(data._original_data) != len(data.variations):
            raise RuntimeError("native SI frequency variation count differs")
        real_rows: list[list[Any]] = []
        imaginary_rows: list[list[Any]] = []
        for index, native in enumerate(data._original_data):
            variation = data.variations[index]
            observed: dict[str, Any] = {
                "index": index,
                "coordinates": dict(variation),
                "intrinsics": {
                    name: values.tolist()
                    for name, values in data.intrinsics_by_variation(index).items()
                },
            }
            evidence["variations"].append(observed)
            observed["real_values"] = list(native.GetRealDataValues(expression, True))
            observed["imaginary_values"] = list(
                native.GetImagDataValues(expression, True)
            )
            real_rows.extend(
                data._full_keys(
                    variation, np.asarray(observed["real_values"], dtype=float), index
                ).tolist()
            )
            imaginary_rows.extend(
                data._full_keys(
                    variation,
                    np.asarray(observed["imaginary_values"], dtype=float),
                    index,
                ).tolist()
            )
        coordinates = [row[:-1] for row in trace["real_rows"]]
        if not (
            coordinates
            == [row[:-1] for row in real_rows]
            == [row[:-1] for row in imaginary_rows]
            == [row[:-1] for row in trace["imaginary_rows"]]
        ):
            raise RuntimeError("native SI frequency coordinates differ from raw trace")
        evidence.update(
            status="available",
            columns=trace["columns"],
            real_rows=real_rows,
            imaginary_rows=imaginary_rows,
        )
    except Exception as exc:  # noqa: BLE001 -- retain optional native SI failure.
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return trace


def _trace_by_pass(trace: dict[str, Any], *, final_pass: int) -> dict[int, float]:
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
        if pass_value > final_pass:
            if math.isfinite(value):
                raise RuntimeError(
                    "native adaptive trace has a finite value beyond the completed pass"
                )
            continue
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
            float(row[columns.index("Phase")]) != 0.0 for row in trace["real_rows"]
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
            and query.get("quantity_binding")
            in {"title", "unique_expression", "native_expression_cache_title"}
            else "not_reported"
        )
        return trace, {
            "status": status,
            "expected_postprocessing_variables": expected,
            "evidence": (
                "persisted_item_and_scoped_native_query"
                if status.startswith("verified")
                else None
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
    return {
        **trace,
        "real_rows": selected_real,
        "imaginary_rows": selected_imaginary,
    }, {
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
    final_pass = convergence["final_pass"]
    if type(final_pass) is not int or final_pass <= 0:
        raise RuntimeError("native completed pass identity is invalid")
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
        frequencies = _trace_by_pass(frequency_trace, final_pass=final_pass)
        mode_items = [
            item
            for item in cache["items"]
            if item["mode"] == mode and not item.get("cache_only")
        ]
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
                item_values[title] = _trace_by_pass(scoped_trace, final_pass=final_pass)
            except RuntimeError as exc:
                item_contexts[title] = {
                    **context,
                    "status": "invalid_pass_axis",
                    "error": str(exc),
                }
        pass_ids = range(1, final_pass + 1)
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
                target = _integral_target(item["purpose"], item["selection"])
                try:
                    value, unit_evidence = _canonical_integral(
                        {"value": values[pass_id], "unit": native_unit},
                        target=target,
                        purpose=item["purpose"],
                    )
                except RuntimeError as exc:
                    unit_mismatches.append(
                        {
                            "title": title,
                            "native_unit": native_unit,
                            "expected_unit": _purpose_unit(target.kind),
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
                    target=target,
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
                si = (
                    frequency_trace.get("si_evidence")
                    if frequency_trace["unit"] == ""
                    else None
                )
                if si is None:
                    rows.append(row)
                    continue
                row["frequency_unit_evidence"] = {
                    key: value
                    for key, value in si.items()
                    if key not in {"variations", "real_rows", "imaginary_rows"}
                }
                if si["status"] != "available":
                    rows.append(row)
                    continue
                pass_index = si["columns"].index("Pass")
                si_rows = {
                    int(real_row[pass_index]): (real_row, imaginary_row)
                    for real_row, imaginary_row in zip(
                        si["real_rows"], si["imaginary_rows"]
                    )
                }
                if pass_id not in si_rows:
                    row["frequency_unit_evidence"].update(
                        status="unavailable",
                        error="native SI frequency Pass is unavailable",
                    )
                    rows.append(row)
                    continue
                real_row, imaginary_row = si_rows[pass_id]
                row["frequency_unit_evidence"].update(
                    real_row=real_row, imaginary_row=imaginary_row
                )
                frequency_hz = real_row[-1]
                row["frequency_unit_status"] = "native_si"
            raw["frequency_hz"] = {"value": frequency_hz, "unit": "Hz"}
            if missing or invalid_context or unit_mismatches:
                row["frequency_hz"] = frequency_hz
                rows.append(row)
                continue
            try:
                row.update(
                    combine_epr_mode(spec.geometry, raw, request=spec.epr_request)
                )
                row["status"] = "complete"
            except (TypeError, ValueError, RuntimeError) as exc:
                row["combination_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return EprResult(
        result_kind="adaptive_history",
        setup_name=spec.run_control.setup_name,
        rows=tuple(rows),
        _legacy_payload=spec._legacy_payload,
        provenance={
            "model_source_sha256": spec.geometry.model_sha256,
            "analysis_source_sha256": spec.geometry.source_sha256,
            "surface_analysis_scope": _SURFACE_ANALYSIS_SCOPE,
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
    frequency_solutions = [
        str(item) for item in app.post.available_report_solutions(category) or ()
    ]
    field_solutions = [
        str(item) for item in app.post.available_report_solutions("Fields") or ()
    ]
    if solution not in frequency_solutions:
        raise RuntimeError(
            f"native adaptive solution is unavailable for {category!r}: {frequency_solutions!r}"
        )
    quantities: list[str] = []
    for quantity_category in (
        app.post.available_quantities_categories(category, solution=solution) or ()
    ):
        for quantity in (
            app.post.available_report_quantities(
                category,
                solution=solution,
                quantities_category=quantity_category,
            )
            or ()
        ):
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
    rows = {
        quantity: _frequency_solution_trace(
            data, quantity, category=category, solution=solution
        )
        for quantity in expected
    }
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
            if solution not in frequency_solutions:
                raise RuntimeError(
                    f"native adaptive solution is unavailable for {category}: {frequency_solutions!r}"
                )
            cache_categories = [
                str(item)
                for item in app.post.available_quantities_categories(
                    category, solution=solution
                )
                or ()
            ]
            cache_quantities = (
                [
                    str(item)
                    for item in app.post.available_report_quantities(
                        category,
                        solution=solution,
                        quantities_category="Expression Cache",
                    )
                    or ()
                ]
                if "Expression Cache" in cache_categories
                else []
            )
            cache_observation["available_quantity_categories"] = cache_categories
            cache_observation["available_quantities"] = {
                "Expression Cache": cache_quantities
            }
            persisted_by_title = {item["title"]: item for item in persisted_cache_items}
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
                expected_quantity = f"ExprCache({persisted['title']})"
                matches = [
                    ("Expression Cache", quantity)
                    for quantity in cache_quantities
                    if quantity == expected_quantity
                ]
                query["inventory_matches"] = matches
                if len(matches) != 1:
                    query["error"] = "native quantity binding is missing or ambiguous"
                    continue
                quantity = matches[0][1]
                query["quantity"] = quantity
                query["quantity_binding"] = "native_expression_cache_title"
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
                        category, solution, [], sweeps, [quantity]
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
                "complete" if len(returned_titles) == len(cache_titles) else "partial"
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


def _expression_name(
    expressions: list[dict[str, Any]],
    *,
    purpose: str,
    selection_kind: str,
    selection_id_key: str,
    selection_id: str,
) -> str:
    matches = [
        item
        for item in expressions
        if item["identity"].get("purpose") == purpose
        and item["identity"].get("selection", {}).get("kind") == selection_kind
        and item["identity"].get("selection", {}).get(selection_id_key) == selection_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"compiled EPR expression binding is missing or ambiguous: {purpose!r}"
        )
    return str(matches[0]["name"])


def _weighted_native_terms(terms: list[tuple[str, float]]) -> list[str]:
    operations: list[str] = []
    for name, coefficient in terms:
        if coefficient == 0.0:
            continue
        has_prior_term = bool(operations)
        operations.extend(
            (
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            )
        )
        if has_prior_term:
            operations.append("Operation('+')")
    if not operations:
        raise RuntimeError("normalized expression has no weighted terms")
    return operations


def _native_term_sum(terms: list[list[str]]) -> list[str]:
    if not terms:
        raise RuntimeError("normalized expression has no energy terms")
    operations: list[str] = []
    for term in terms:
        has_prior_term = bool(operations)
        operations.extend(term)
        if has_prior_term:
            operations.append("Operation('+')")
    return operations


def _normalized_surface_total_operations(
    spec: HfssEprSpec,
    expressions: list[dict[str, Any]],
    target: NormalizedSurfaceEprTotal,
) -> list[str]:
    if spec.epr_request is None:
        raise RuntimeError("normalized surface convergence requires an EPR request")
    groups = [
        group
        for group in surface_integral_groups(spec.geometry, spec.epr_request)
        if group["interface_kind"] == target.interface_kind
        and group.get("evaluation_kind", "requested_margin") == target.evaluation_kind
        and float(group["margin_um"]) == target.margin_um
    ]
    if not groups:
        raise RuntimeError("normalized surface convergence target has no owner groups")

    numerator_terms: list[tuple[str, float]] = []
    for group in groups:
        normal_coefficient, tangential_coefficient = _surface_energy_coefficients(
            group["interface_kind"],
            float(group["film_thickness_m"]),
            float(group["film_relative_permittivity"]),
            float(group["substrate_relative_permittivity"]),
        )
        group_id = str(group["group_id"])
        if normal_coefficient != 0.0:
            numerator_terms.append(
                (
                    _expression_name(
                        expressions,
                        purpose=f"electric_normal_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    normal_coefficient,
                )
            )
        if tangential_coefficient != 0.0:
            numerator_terms.append(
                (
                    _expression_name(
                        expressions,
                        purpose=f"electric_tangential_{group_id}",
                        selection_kind="surface_group",
                        selection_id_key="group_id",
                        selection_id=group_id,
                    ),
                    tangential_coefficient,
                )
            )
    numerator_operations = _weighted_native_terms(numerator_terms)

    _, materials, domains = _epr_domain_context(spec.geometry)
    denominator_terms: list[list[str]] = []
    for domain_id, domain in domains.items():
        material = materials[domain["material_id"]]
        coefficient = _electric_domain_energy_coefficient(
            float(material["permittivity"])
        )
        name = _expression_name(
            expressions,
            purpose=f"electric_volume_{domain_id}",
            selection_kind="volume",
            selection_id_key="semantic_id",
            selection_id=domain_id,
        )
        denominator_terms.append(
            [
                f"NameOfExpression('{name}')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    for junction in spec.geometry.junctions:
        coefficient = _junction_capacitive_energy_coefficient(
            junction.capacitance_f, junction.width_um
        )
        if coefficient == 0.0:
            continue
        real_name = _expression_name(
            expressions,
            purpose=f"junction_voltage_real_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        imag_name = _expression_name(
            expressions,
            purpose=f"junction_voltage_imag_{junction.junction_id}",
            selection_kind="junction_sheet",
            selection_id_key="junction_id",
            selection_id=junction.junction_id,
        )
        denominator_terms.append(
            [
                f"NameOfExpression('{real_name}')",
                f"NameOfExpression('{real_name}')",
                "Operation('*')",
                f"NameOfExpression('{imag_name}')",
                f"NameOfExpression('{imag_name}')",
                "Operation('*')",
                "Operation('+')",
                f"Scalar_Constant({coefficient:.17g})",
                "Operation('*')",
            ]
        )
    denominator_operations = _native_term_sum(denominator_terms)
    return [
        *numerator_operations,
        *denominator_operations,
        "Operation('/')",
    ]


def _cache_items(
    expressions: list[dict[str, Any]],
    mode_count: int,
    selected_modes: tuple[int, ...],
    convergence: ExpressionCacheConvergence | None = None,
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
            item = items[-1]
            if expression.get("cache_only"):
                item["cache_only"] = True
            if (
                expression.get("cache_only")
                and convergence is not None
                and mode == convergence.mode
            ):
                item["is_convergence"] = True
                item["use_relative_convergence"] = convergence.use_relative_convergence
                item["criterion"] = convergence.criterion
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
                item.get("is_convergence", False),
                "UseRelativeConvergence:=",
                int(item.get("use_relative_convergence", False)),
                "MaxConvergenceDelta:=",
                item.get("criterion", 1),
                "MaxConvergeValue:=",
                str(item.get("criterion", 1)),
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
    inset_plan: dict[tuple[str, float], dict[str, Any]],
    *,
    namespace: str = "prepared",
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    request = spec.epr_request
    if request is None:
        raise RuntimeError("EPR expression preparation requires an analysis request")
    solution = f"{spec.run_control.setup_name} : LastAdaptive"
    evidence_dir = run_dir / "metadata/epr_expressions"
    expressions: list[dict[str, Any]] = []
    authoring_started = time.perf_counter()
    phase_seconds = {
        "pp_sources_seconds": 0.0,
        "sheet_binding_seconds": 0.0,
        "ordinary_compile_seconds": 0.0,
        "adjacent_compile_seconds": 0.0,
        "native_definition_readback_seconds": 0.0,
    }
    expression_counts = {"ordinary": 0, "adjacent": 0}
    expression_stage_seconds = {
        "volume": 0.0,
        "constituent": 0.0,
        "grouped": 0.0,
        "junction": 0.0,
    }
    compiled: list[tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]] = []

    def _timed_author(stage: str, **kwargs: Any) -> dict[str, Any]:
        adjacent = kwargs.get("adjacent_selection_name") is not None
        started = time.perf_counter()
        result = compile_named_expression(**kwargs)
        elapsed = time.perf_counter() - started
        kind = "adjacent" if adjacent else "ordinary"
        phase_seconds[f"{kind}_compile_seconds"] += elapsed
        expression_stage_seconds[stage] = (
            expression_stage_seconds.get(stage, 0.0) + elapsed
        )
        expression_counts[kind] += 1
        compiled.append(result)
        return result[0]

    started = time.perf_counter()
    # Keep a nonzero modal source selected when saving the project.
    pp_values = {
        f"scgsim_epr_pp_mode_{index}": "1" if index == 1 else "0"
        for index in range(1, spec.run_control.num_modes + 1)
    }
    pp_observed = install_variables(app, pp_values)
    source_assignment = {
        str(index): (f"scgsim_epr_pp_mode_{index}", "0deg")
        for index in range(1, spec.run_control.num_modes + 1)
    }
    if app.edit_sources(source_assignment, eigenmode_stored_energy=False) is not True:
        raise RuntimeError("native Eigenmode PP source-vector installation failed")
    phase_seconds["pp_sources_seconds"] = time.perf_counter() - started
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
                _timed_author(
                    "volume",
                    purpose=f"{quantity}_{item['semantic_id']}",
                    operations=operations,
                    solution=solution,
                    phase_degrees=0.0,
                    dependencies=pp_observed,
                    selection={
                        "kind": "volume",
                        "semantic_id": item["semantic_id"],
                        "object_name": item["object_name"],
                        "relative_permeability": material_readback[item["material_id"]][
                            "relative_permeability"
                        ],
                    },
                    namespace=namespace,
                )
            )
    bindings = {
        str(item["binding_id"]): item for item in spec.geometry.surface_bindings
    }
    groups = surface_integral_groups(spec.geometry, request)
    native_by_binding = {
        item["binding_id"]: item for item in native_geometry["surface_selections"]
    }
    if len(native_by_binding) != len(native_geometry["surface_selections"]):
        raise RuntimeError("native EPR surface binding IDs are not unique")
    group_by_member = {
        (member["binding_id"], group["evaluation_kind"], group["margin_um"]): group
        for group in groups
        for member in group["members"]
    }
    grouped_constituents: dict[tuple[str, str], list[dict[str, Any]]] = {}
    member_components: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    # Reuse native sheet facts only within this authoring session; side stays per binding.
    sheet_facts: dict[str, dict[str, Any]] = {}
    sheet_names = plan_inset_sheet_names(spec.geometry.surface_bindings, inset_plan)
    selected_contributions = (
        {item.contribution_id for item in spec.geometry.contributions}
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    for selection in native_geometry["surface_selections"]:
        binding = bindings[selection["binding_id"]]
        if (
            selection["contribution_id"] not in selected_contributions
            or binding["contribution"]["classification"] == "MM"
        ):
            continue
        for evaluation_kind, margin in surface_evaluations(
            binding["margins_um"],
            policy=spec.geometry.source.get("surface_evaluation_policy"),
        ):
            margin_um = float(margin)
            member_key = (selection["binding_id"], evaluation_kind, margin_um)
            group = group_by_member[member_key]
            started = time.perf_counter()
            components = bind_inset_surface_selections(
                app,
                binding,
                selection,
                margin_um,
                inset_plan,
                sheet_facts,
                sheet_names,
            )
            phase_seconds["sheet_binding_seconds"] += time.perf_counter() - started
            member_components[member_key] = components
            for component in components:
                component_selection = {
                    **selection,
                    **component,
                    "margin_um": margin_um,
                    "evaluation_kind": evaluation_kind,
                    "integration_geometry": "klayout_complement_minkowski.v1",
                }
                for quantity in (
                    "electric_normal",
                    "electric_tangential",
                    "masked_area",
                ):
                    operations = field_integral_operations(
                        quantity=quantity,
                        selection_name=component["selection_name"],
                        adjacent_side=component["adjacent_side"],
                        normal_vector=component["native_normal"],
                    )
                    authored = _timed_author(
                        "constituent",
                        purpose=(
                            f"{quantity}_{selection['binding_id']}_"
                            f"{evaluation_kind}_{margin_um:.17g}um_"
                            f"component_{component['component_index']}"
                        ),
                        operations=operations,
                        solution=solution,
                        phase_degrees=0.0,
                        dependencies=pp_observed,
                        selection=component_selection,
                        namespace=namespace,
                        adjacent_selection_name=(
                            component["selection_name"]
                            if component["adjacent_side"] and quantity != "masked_area"
                            else None
                        ),
                    )
                    grouped_constituents.setdefault(
                        (group["group_id"], quantity), []
                    ).append(authored)
    for group in groups:
        native_members = [
            {
                "binding_id": member["binding_id"],
                "source_selection_name": native_by_binding[member["binding_id"]][
                    "selection_name"
                ],
                "components": member_components[
                    (member["binding_id"], group["evaluation_kind"], group["margin_um"])
                ],
                "geometry_empty": not member_components[
                    (member["binding_id"], group["evaluation_kind"], group["margin_um"])
                ],
                "effective_domain_id": native_by_binding[member["binding_id"]][
                    "effective_domain_id"
                ],
            }
            for member in group["members"]
        ]
        for quantity in ("electric_normal", "electric_tangential", "masked_area"):
            members = grouped_constituents.get((group["group_id"], quantity), [])
            if len(members) != sum(len(item["components"]) for item in native_members):
                raise RuntimeError("grouped EPR expression lacks an inset component")
            members.sort(key=lambda item: item["identity"]["selection"]["binding_id"])
            operations: list[str] = []
            for member in members:
                operations.append(f"NameOfExpression('{member['name']}')")
                if len(operations) > 1:
                    operations.append("Operation('+')")
            if not operations:
                # Every source member was geometrically empty at this margin.
                operations = ["Scalar_Constant(0)"]
            dependencies = dict(pp_observed)
            for member in members:
                dependencies.update(member["identity"]["dependencies"])
            expressions.append(
                _timed_author(
                    "grouped",
                    purpose=f"{quantity}_{group['group_id']}",
                    operations=operations,
                    solution=solution,
                    phase_degrees=0.0,
                    dependencies=dependencies,
                    selection={
                        "kind": "surface_group",
                        **group,
                        "integration_geometry": "klayout_complement_minkowski.v1",
                        "polygon_approximation": {
                            "method": "klayout_complement_minkowski.v1",
                            "points_per_circle": 128,
                            "source_dbu_um": spec.geometry.source["source_dbu_um"],
                            "klayout_version": bindings[
                                group["members"][0]["binding_id"]
                            ]["mask_support"]["klayout_version"],
                        },
                        "native_member_selections": native_members,
                        "geometry_empty": not members,
                    },
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
                _timed_author(
                    "junction",
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
                    namespace=namespace,
                )
            )
    convergence = spec.expression_convergence if isinstance(spec, HfssEprSpec) else None
    if convergence is not None:
        target = convergence.target
        if isinstance(target, NormalizedSurfaceEprTotal):
            purpose = "normalized_surface_epr_total"
            selection = {
                "kind": "normalized_surface_total",
                "interface_kind": target.interface_kind,
                "evaluation_kind": target.evaluation_kind,
                "margin_um": target.margin_um,
            }
            operations = _normalized_surface_total_operations(spec, expressions, target)
        elif isinstance(target, NativeExpressionDefinition):
            purpose = "custom_convergence"
            selection = {
                "kind": "custom_convergence",
                "definition_name": target.name,
            }
            operations = list(target.operations)
        else:
            raise TypeError("unsupported expression-cache convergence target")
        authored = _timed_author(
            "convergence",
            purpose=purpose,
            operations=operations,
            solution=solution,
            phase_degrees=0.0,
            dependencies=pp_observed,
            selection=selection,
            namespace=namespace,
        )
        authored["cache_only"] = True
        expressions.append(authored)
    batch = load_compiled_expressions(app, compiled, evidence_dir)
    for key in (
        "collision_check_seconds",
        "evidence_write_seconds",
        "batch_import_seconds",
        "postload_readback_seconds",
        "native_definition_readback_seconds",
    ):
        phase_seconds[key] = batch[key]
    authoring_seconds = time.perf_counter() - authoring_started
    other_seconds = max(0.0, authoring_seconds - sum(phase_seconds.values()))
    return (
        expressions,
        {
            "source_assignment": source_assignment,
            "postprocessing_variables": pp_observed,
            "native_expression_readback": batch["native_expression_readback"],
        },
        {
            **{key: round(value, 6) for key, value in phase_seconds.items()},
            "other_seconds": round(other_seconds, 6),
            "ordinary_expression_count": expression_counts["ordinary"],
            "adjacent_expression_count": expression_counts["adjacent"],
            "expression_stage_compile_seconds": {
                key: round(value, 6) for key, value in expression_stage_seconds.items()
            },
            "batch_library_sha256": batch["library_sha256"],
            "native_definition_count": batch["native_definition_count"],
            "unique_inset_sheet_count": len(sheet_facts),
        },
    )


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
    items = _cache_items(
        expressions,
        spec.run_control.num_modes,
        selected_modes,
        spec.expression_convergence,
    )
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
            "status": "PENDING_SAVED_SETUP_READBACK",
            "reason": (
                "submitted cache settings are not saved-setup readback; "
                "saved properties are inspected after the project save"
            ),
        },
        "raw_cache_args_sha256": _identity_sha256(raw_cache),
    }


def _prepare_expressions_and_cache(
    app: Any,
    run_dir: Path,
    spec: HfssEprSpec,
    native_geometry: dict[str, Any],
    inset_plan: dict[tuple[str, float], dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if inset_plan is None:
        raise RuntimeError("EPR inset geometry must be precomputed before Desktop")
    expressions, authoring, phase_timings = _author_epr_expressions(
        app,
        run_dir,
        spec,
        native_geometry,
        inset_plan,
    )
    started = time.perf_counter()
    cache = _submit_epr_cache(app, spec, expressions, authoring)
    phase_timings["cache_seconds"] = round(time.perf_counter() - started, 6)
    return expressions, cache, phase_timings


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
    # EPR needs fields in the adaptive solve and in the sealed analysis copy.
    setup.props["SaveAnyFields"] = True
    setup.props["SaveRadFieldsOnly"] = False
    if not setup.update():
        raise RuntimeError("HFSS EPR setup update failed")


def _read_setup(app: Any, spec: HfssEprSpec | HfssEprAnalysisSpec) -> dict[str, Any]:
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
    creating_setup = isinstance(spec, HfssEprSpec)
    if creating_setup:
        observed.update(
            {
                "save_any_fields": raw.get("SaveAnyFields"),
                "save_radiated_fields_only": raw.get("SaveRadFieldsOnly"),
            }
        )
        expected.update(
            {
                "save_any_fields": True,
                "save_radiated_fields_only": False,
            }
        )
    if observed != expected:
        raise RuntimeError(f"HFSS EPR saved setup readback mismatch: {observed!r}")
    saved_field_properties = {
        "basis": (
            "AEDT 2024.2 Eigenmode SaveAnyFields/SaveRadFieldsOnly setup readback"
            if creating_setup
            else "existing saved-field analysis does not re-author setup field policy"
        ),
        "serialized": {
            key: raw[key]
            for key in ("SaveAnyFields", "SaveRadFieldsOnly")
            if key in raw
        },
        "status": (
            "verified" if creating_setup else "not_reauthored_existing_saved_solution"
        ),
    }
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
    convergence_field_names = (
        "IsConvergence",
        "UseRelativeConvergence",
        "MaxConvergenceDelta",
        "MaxConvergeValue",
    )
    convergence_fields = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise RuntimeError("saved EPR cache item is invalid")
        convergence_fields.append(
            {
                "title": item.get("Title"),
                "values": {
                    name: item[name] for name in convergence_field_names if name in item
                },
                "unavailable": [
                    name for name in convergence_field_names if name not in item
                ],
            }
        )
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
        "convergence_fields": convergence_fields,
    }
