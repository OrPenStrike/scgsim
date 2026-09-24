"""Pure SI energy and participation calculations for verified HFSS integrals."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pathlib import Path

from ._epr_models import (
    EprAnalysisRequest,
    EprResult,
    PreparedPlanarGeometry,
    SavedSolution,
    detached,
)
from .util import file_sha256, read_json, write_json

_EPSILON_0_F_PER_M = 8.8541878128e-12
_MU_0_H_PER_M = 1.25663706212e-6


def _finite(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _exact_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _quantity(value: Any, name: str, unit: str, *, nonnegative: bool = True) -> float:
    item = _exact_mapping(value, name)
    if set(item) != {"value", "unit"} or item.get("unit") != unit:
        raise ValueError(f"{name} must use canonical unit {unit!r}")
    return _finite(item.get("value"), name, nonnegative=nonnegative)


def resolve_saved_solution(manifest_path: str | Path) -> SavedSolution:
    """Verify one explicitly sealed saved-field cohort without opening AEDT."""

    requested_manifest = Path(manifest_path).expanduser().absolute()
    manifest_source = requested_manifest.resolve()
    if (
        requested_manifest != manifest_source
        or not manifest_source.is_file()
        or manifest_source.is_symlink()
    ):
        raise FileNotFoundError(f"saved-solution manifest is missing: {manifest_source}")
    base = manifest_source.parent
    manifest = _exact_mapping(read_json(manifest_source), "saved-solution manifest")
    expected = {
        "schema_version",
        "project",
        "results",
        "members",
        "content_sha256",
        "identity",
        "receipt",
        "receipt_sha256",
    }
    if (
        set(manifest) != expected
        or manifest.get("schema_version")
        != "scgsim.aedt.saved-solution-manifest.v1"
    ):
        raise ValueError("saved-solution manifest members are not canonical")
    payload = {
        "schema_version": "scgsim.aedt.saved-solution.v1",
        "project": manifest["project"],
        "results": manifest["results"],
        "members": manifest["members"],
        "content_sha256": manifest["content_sha256"],
        "identity": manifest["identity"],
    }
    saved = SavedSolution.from_payload(base, payload)
    if saved.project_path.is_symlink() or saved.result_path.is_symlink():
        raise RuntimeError("saved-solution project/results cannot be symlinks")
    if (
        saved.project_path.resolve() != saved.project_path
        or saved.result_path.resolve() != saved.result_path
    ):
        raise RuntimeError("saved-solution project/results traverse a symlink")
    if not saved.project_path.is_file() or not saved.result_path.is_dir():
        raise FileNotFoundError("saved-solution project or result directory is missing")
    receipt_relative = _safe_relative(manifest["receipt"], "receipt")
    receipt = base / receipt_relative
    if (
        not receipt.is_file()
        or receipt.is_symlink()
        or file_sha256(receipt) != manifest["receipt_sha256"]
    ):
        raise RuntimeError("saved-solution receipt hash mismatch")
    receipt_payload = _exact_mapping(read_json(receipt), "saved-solution receipt")
    if receipt_payload.get("status") != "completed" or receipt_payload.get(
        "saved_fields"
    ) is not True:
        raise RuntimeError("saved-solution receipt does not attest completed fields")
    if receipt_payload.get("identity") != detached(saved.identity):
        raise RuntimeError("saved-solution receipt identity mismatch")
    observed_members: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for member in saved.members:
        item = _exact_mapping(member, "saved-solution member")
        if set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("saved-solution member is not canonical")
        relative = _safe_relative(item["path"], "saved-solution member.path")
        path = base / relative
        if relative.as_posix() in declared_paths:
            raise ValueError("saved-solution member paths must be unique")
        declared_paths.add(relative.as_posix())
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve() != path
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"saved-solution member mismatch: {relative}")
        observed_members.append(dict(item))
    actual_paths = {saved.project_path.relative_to(base).as_posix()}
    for path in saved.result_path.rglob("*"):
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError(f"saved solution contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(base).as_posix())
    if declared_paths != actual_paths:
        raise RuntimeError("saved-solution manifest does not cover the exact result inventory")
    digest = hashlib.sha256(
        json.dumps(
            observed_members, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if digest != saved.content_sha256:
        raise RuntimeError("saved-solution content digest mismatch")
    return saved


def seal_saved_solution(
    run_dir: str | Path,
    *,
    project_name: str,
    design_name: str,
    setup_name: str,
    model_source_sha256: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a released project and its complete saved-result inventory."""

    root = Path(run_dir).resolve()
    project = root / f"{project_name}.aedt"
    results = root / f"{project_name}.aedtresults"
    if project.is_symlink() or project.resolve() != project or not project.is_file():
        raise RuntimeError("released saved-solution project is unavailable")
    if results.is_symlink() or results.resolve() != results or not results.is_dir():
        raise RuntimeError("released saved-solution result directory is unavailable")
    evidence = _exact_mapping(evidence, "saved-field evidence")
    expected_evidence = {
        "saved_fields",
        "solver_last_completed_pass",
        "saved_fields_pass",
        "fields_solution",
        "physical_variation",
    }
    if set(evidence) != expected_evidence or evidence.get("saved_fields") is not True:
        raise ValueError("saved-field evidence is not canonical")
    for name in ("solver_last_completed_pass", "saved_fields_pass"):
        if type(evidence.get(name)) is not int or evidence[name] <= 0:
            raise ValueError(f"saved-field evidence {name} is invalid")
    if not isinstance(evidence.get("fields_solution"), str) or not evidence[
        "fields_solution"
    ]:
        raise ValueError("saved-field evidence solution identity is invalid")
    if not isinstance(evidence.get("physical_variation"), str):
        raise TypeError("saved-field physical variation must be text")

    inventory = [project]
    for path in sorted(results.rglob("*")):
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError(f"saved solution contains a symlink: {path}")
        if path.is_file():
            inventory.append(path)
    if len(inventory) == 1:
        raise RuntimeError("saved-solution result inventory is empty")
    members = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in inventory
    ]
    content_sha256 = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        "model_source_sha256": model_source_sha256,
        "project_name": project_name,
        "design_name": design_name,
        "setup_name": setup_name,
        "physical_variation": evidence["physical_variation"],
        "solver_last_completed_pass": evidence["solver_last_completed_pass"],
        "saved_fields_pass": evidence["saved_fields_pass"],
        "saved_fields": True,
    }
    receipt_path = root / "metadata/saved-solution-receipt.json"
    manifest_path = root / "saved-solution-manifest.json"
    receipt_payload = {
        "schema_version": "scgsim.aedt.saved-solution-receipt.v1",
        "status": "completed",
        "saved_fields": True,
        "identity": identity,
        "fields_solution": evidence["fields_solution"],
        "content_sha256": content_sha256,
    }
    write_json(receipt_path, receipt_payload)
    manifest_payload = {
        "schema_version": "scgsim.aedt.saved-solution-manifest.v1",
        "project": project.relative_to(root).as_posix(),
        "results": results.relative_to(root).as_posix(),
        "members": members,
        "content_sha256": content_sha256,
        "identity": identity,
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
    }
    write_json(manifest_path, manifest_payload)
    return {
        "status": "complete",
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
        "content_sha256": content_sha256,
        "identity": identity,
        "member_count": len(members),
    }


def _safe_relative(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return relative


def resolve_epr_result(path: str | Path) -> EprResult:
    """Load one strict offline EPR result without filling incomplete rows."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"EPR result is missing: {source}")
    return EprResult.from_payload(read_json(source))


def plot_epr_result(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    show_convergence: bool = True,
) -> Any:
    """Render the exact selected pass plus an optional separate convergence view."""

    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    if mode is not None and (type(mode) is not int or mode <= 0):
        raise ValueError("mode must be a positive integer or None")
    if native_pass is not None and (type(native_pass) is not int or native_pass <= 0):
        raise ValueError("native_pass must be a positive integer or None")
    if native_pass is not None:
        if result.result_kind != "adaptive_history":
            raise ValueError("native_pass selection requires adaptive-history results")
    if type(show_convergence) is not bool:
        raise TypeError("show_convergence must be bool")
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    convergence_view = show_convergence and result.result_kind == "adaptive_history"
    figure = make_subplots(
        rows=2 if convergence_view else 1,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=(
            ("Selected-pass participation", "Pass convergence")
            if convergence_view
            else ("Participation",)
        ),
    )
    unavailable: list[str] = []
    requested_modes_raw = result.provenance.get(
        "requested_modes", sorted({int(row["mode"]) for row in result.rows})
    )
    known_modes = {
        int(item) for item in requested_modes_raw if type(item) is int and item > 0
    }
    if mode is not None and mode not in known_modes:
        raise ValueError(f"mode {mode} was not requested by this EPR result")
    requested_modes = [
        int(item)
        for item in requested_modes_raw
        if type(item) is int and item > 0 and (mode is None or item == mode)
    ]
    target_pass = native_pass
    if result.result_kind == "adaptive_history" and target_pass is None:
        observed_last = result.provenance.get("solver_last_completed_pass")
        if type(observed_last) is not int or observed_last <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        target_pass = observed_last
    selected = [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (
            result.result_kind != "adaptive_history"
            or row["native_pass"] == target_pass
        )
    ]
    selected_by_mode = {int(row["mode"]): row for row in selected}
    for requested_mode in requested_modes:
        if requested_mode not in selected_by_mode:
            unavailable.append(
                f"mode {requested_mode}: requested pass {target_pass} is missing"
                if target_pass is not None
                else f"mode {requested_mode}: result row is missing"
            )
    for row in selected:
        row_mode = row["mode"]
        pass_label = (
            "" if "native_pass" not in row else f", pass {row['native_pass']:g}"
        )
        if row.get("status") == "partial" or "normalization_energy_j" not in row:
            unavailable.append(
                f"mode {row_mode}{pass_label}: "
                f"{row.get('combination_error', 'incomplete native evidence')}"
            )
            continue
        labels: list[str] = []
        values: list[float] = []
        for surface in row.get("surface_contributions", ()):
            labels.append(
                f"surface:{surface['contribution_id']}@{surface['margin_um']:g}um"
            )
            values.append(float(surface["participation"]))
        for junction in row.get("junctions", ()):
            labels.extend(
                [
                    f"junction-L:{junction['junction_id']}",
                    f"junction-C:{junction['junction_id']}",
                ]
            )
            values.extend(
                [
                    float(junction["inductive_participation"]),
                    float(junction["capacitive_participation"]),
                ]
            )
        figure.add_trace(
            go.Bar(name=f"mode {row_mode}{pass_label}", x=labels, y=values),
            row=1,
            col=1,
        )
    if convergence_view:
        last_pass = result.provenance.get("solver_last_completed_pass")
        if type(last_pass) is not int or last_pass <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        pass_axis = list(range(1, last_pass + 1))
        for row_mode in requested_modes:
            complete = sorted(
                (
                    row
                    for row in result.rows
                    if row["mode"] == row_mode and row["status"] == "complete"
                ),
                key=lambda item: item["native_pass"],
            )
            series: dict[str, dict[int, float]] = {}
            frequency_by_pass: dict[int, float] = {}
            for row in complete:
                native = int(row["native_pass"])
                frequency_by_pass[native] = float(row["frequency_hz"])
                series.setdefault("magnetic+inductive / normalization", {})[
                    native
                ] = (
                    float(row["magnetic_energy_balance_j"])
                    / float(row["normalization_energy_j"])
                )
                for surface in row["surface_contributions"]:
                    label = (
                        f"surface:{surface['contribution_id']}@"
                        f"{surface['margin_um']:g}um"
                    )
                    series.setdefault(label, {})[native] = float(
                        surface["participation"]
                    )
                for junction in row["junctions"]:
                    series.setdefault(
                        f"junction-L:{junction['junction_id']}", {}
                    )[native] = float(junction["inductive_participation"])
                    series.setdefault(
                        f"junction-C:{junction['junction_id']}", {}
                    )[native] = float(junction["capacitive_participation"])
            for label, values in sorted(series.items()):
                figure.add_trace(
                    go.Scatter(
                        name=f"mode {row_mode} {label}",
                        x=pass_axis,
                        y=[values.get(item) for item in pass_axis],
                        mode="lines+markers",
                        connectgaps=False,
                    ),
                    row=2,
                    col=1,
                )
            figure.add_trace(
                go.Scatter(
                    name=f"mode {row_mode} frequency",
                    x=pass_axis,
                    y=[frequency_by_pass.get(item) for item in pass_axis],
                    mode="lines+markers",
                    connectgaps=False,
                    yaxis="y3",
                ),
                row=2,
                col=1,
            )
    figure.update_layout(
        barmode="group",
        title=f"AEDT EPR — {result.result_kind.replace('_', ' ')}",
        meta={
            "schema_version": "scgsim.aedt.epr-plot.v1",
            "result_kind": result.result_kind,
            "selected_mode": mode,
            "selected_native_pass": target_pass,
            "show_convergence": show_convergence,
            "unavailable": unavailable,
        },
    )
    figure.update_xaxes(title_text="Contribution", row=1, col=1)
    figure.update_yaxes(title_text="Participation", row=1, col=1)
    if convergence_view:
        figure.update_xaxes(title_text="Adaptive pass", row=2, col=1)
        figure.update_yaxes(title_text="Participation / energy-balance ratio", row=2, col=1)
        figure.update_layout(
            yaxis3={
                "title": "Frequency (Hz)",
                "overlaying": "y2",
                "side": "right",
                "showgrid": False,
            }
        )
    if unavailable:
        figure.add_annotation(
            text="<br>".join(unavailable),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
        )
    return figure


def combine_epr_mode(
    prepared: PreparedPlanarGeometry,
    raw: Mapping[str, Any],
    *,
    request: EprAnalysisRequest | None = None,
) -> dict[str, Any]:
    """Combine one complete pass/mode raw-integral record without fallback."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    if request is not None and not isinstance(request, EprAnalysisRequest):
        raise TypeError("request must be EprAnalysisRequest or None")
    raw = _exact_mapping(raw, "raw EPR mode")
    frequency_hz = _quantity(raw.get("frequency_hz"), "frequency_hz", "Hz")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be > 0")
    electric = _exact_mapping(
        raw.get("electric_domain_integrals_v2_m"),
        "electric_domain_integrals_v2_m",
    )
    effective_volumes = _exact_mapping(
        raw.get("effective_domain_volumes_m3"), "effective_domain_volumes_m3"
    )
    magnetic = _exact_mapping(
        raw.get("magnetic_domain_integrals_a2_m"),
        "magnetic_domain_integrals_a2_m",
    )
    relative_permeability = _exact_mapping(
        raw.get("relative_permeability"), "relative_permeability"
    )
    surface = _exact_mapping(
        raw.get("surface_integrals_v2"), "surface_integrals_v2"
    )
    masked_areas = _exact_mapping(raw.get("masked_areas_m2"), "masked_areas_m2")
    junction = _exact_mapping(
        raw.get("junction_integrals_v_m"), "junction_integrals_v_m"
    )

    source = detached(prepared.source)
    materials = _exact_mapping(source["materials"], "prepared materials")
    domains = {
        item["semantic_id"]: item for item in source["solution_regions"]
    }
    if (
        set(electric) != set(domains)
        or set(magnetic) != set(domains)
        or set(relative_permeability) != set(domains)
    ):
        raise ValueError(
            "electric and magnetic domain integrals must cover every prepared solution domain"
        )
    if set(effective_volumes) != set(domains):
        raise ValueError("effective volume integrals must cover every prepared solution domain")
    electric_energy_j = 0.0
    magnetic_energy_j = 0.0
    electric_rows: list[dict[str, Any]] = []
    for domain_id, domain in domains.items():
        material = _exact_mapping(
            materials.get(domain["material_id"]), f"material {domain['material_id']!r}"
        )
        if "permittivity" not in material:
            raise ValueError(
                f"material {domain['material_id']!r} lacks relative permittivity"
            )
        epsilon_r = _finite(material["permittivity"], "relative permittivity")
        if epsilon_r <= 0.0:
            raise ValueError("relative permittivity must be > 0")
        integral = _quantity(
            electric[domain_id], f"electric integral {domain_id!r}", "V^2*m"
        )
        magnetic_integral = _quantity(
            magnetic[domain_id], f"magnetic integral {domain_id!r}", "A^2*m"
        )
        permeability_r = _quantity(
            relative_permeability[domain_id],
            f"relative permeability {domain_id!r}",
            "1",
        )
        if permeability_r <= 0.0:
            raise ValueError("relative permeability must be > 0")
        effective_volume = _quantity(
            effective_volumes[domain_id],
            f"effective volume {domain_id!r}",
            "m^3",
        )
        energy = _EPSILON_0_F_PER_M * epsilon_r * integral / 2.0
        magnetic_energy = _MU_0_H_PER_M * permeability_r * magnetic_integral / 2.0
        electric_energy_j += energy
        magnetic_energy_j += magnetic_energy
        electric_rows.append(
            {
                "domain_id": domain_id,
                "material_id": domain["material_id"],
                "relative_permittivity": epsilon_r,
                "native_relative_permeability": permeability_r,
                "effective_volume_m3": effective_volume,
                "integral_v2_m": integral,
                "energy_j": energy,
                "magnetic_h_conj_dot_h_integral_a2_m": magnetic_integral,
                "magnetic_energy_j": magnetic_energy,
            }
        )

    junction_by_id = {item.junction_id: item for item in prepared.junctions}
    if set(junction) != set(junction_by_id):
        raise ValueError("junction integrals must cover every prepared junction")
    angular_frequency = 2.0 * math.pi * frequency_hz
    capacitive_energy_j = 0.0
    junction_rows: list[dict[str, Any]] = []
    for junction_id, spec in junction_by_id.items():
        value = _exact_mapping(junction[junction_id], f"junction {junction_id!r}")
        real = _quantity(
            value.get("real"), "junction real integral", "V*m", nonnegative=False
        )
        imag = _quantity(
            value.get("imag"), "junction imaginary integral", "V*m", nonnegative=False
        )
        voltage = complex(real, imag) / (spec.width_um * 1e-6)
        voltage_squared = abs(voltage) ** 2
        inductive = voltage_squared / (
            2.0 * angular_frequency**2 * spec.inductance_h
        )
        capacitive = spec.capacitance_f * voltage_squared / 2.0
        capacitive_energy_j += capacitive
        current = voltage / complex(0.0, angular_frequency * spec.inductance_h)
        junction_rows.append(
            {
                "junction_id": junction_id,
                "source_polygon_id": spec.source_polygon_id,
                "terminal_a_net": spec.terminal_a_net,
                "terminal_b_net": spec.terminal_b_net,
                "direction_xy": list(spec.direction_xy),
                "voltage_real_v": voltage.real,
                "voltage_imag_v": voltage.imag,
                "current_real_a": current.real,
                "current_imag_a": current.imag,
                "inductive_energy_j": inductive,
                "capacitive_energy_j": capacitive,
            }
        )

    model_inductive_energy_j = sum(
        float(row["inductive_energy_j"]) for row in junction_rows
    )

    normalization_j = electric_energy_j + capacitive_energy_j
    if not math.isfinite(normalization_j) or normalization_j <= 0.0:
        raise ValueError("EPR normalization energy must be finite and > 0")
    selected_junctions = (
        set(junction_by_id)
        if request is None or request.junction_ids is None
        else set(request.junction_ids)
    )
    junction_rows = [
        {
            **row,
            "inductive_participation": row["inductive_energy_j"] / normalization_j,
            "capacitive_participation": row["capacitive_energy_j"] / normalization_j,
        }
        for row in junction_rows
        if row["junction_id"] in selected_junctions
    ]

    bindings: dict[str, list[Mapping[str, Any]]] = {}
    for item in prepared.surface_bindings:
        contribution = _exact_mapping(item.get("contribution"), "surface binding contribution")
        contribution_id = contribution.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            raise ValueError("surface binding contribution id is invalid")
        bindings.setdefault(contribution_id, []).append(item)
    specs_by_id = {item.contribution_id: item for item in prepared.contributions}
    selected_surfaces = (
        set(specs_by_id)
        if request is None or request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    expected_surface_keys = {
        f"{binding['binding_id']}@{margin:.17g}"
        for binding in prepared.surface_bindings
        if binding["contribution"]["contribution_id"] in selected_surfaces
        for margin in specs_by_id[
            binding["contribution"]["contribution_id"]
        ].margins_um
        if specs_by_id[
            binding["contribution"]["contribution_id"]
        ].interface_kind
        in {"MA", "MS", "SA"}
    }
    if set(surface) != expected_surface_keys:
        raise ValueError("surface integrals do not match prepared contribution margins")
    if set(masked_areas) != expected_surface_keys:
        raise ValueError("masked areas do not match prepared contribution margins")
    surface_rows: list[dict[str, Any]] = []
    for spec in prepared.contributions:
        if spec.interface_kind == "MM" or spec.contribution_id not in selected_surfaces:
            continue
        local_bindings = bindings.get(spec.contribution_id)
        if not local_bindings:
            raise ValueError(f"missing prepared binding for {spec.contribution_id!r}")
        epsilon_values: set[float] = set()
        for binding in local_bindings:
            contribution = _exact_mapping(
                binding.get("contribution"), "surface binding contribution"
            )
            if (
                contribution.get("classification") != spec.interface_kind
                or contribution.get("side") != spec.field_side
            ):
                raise ValueError(
                    f"prepared binding contradicts contribution {spec.contribution_id!r}"
                )
            material_key = (
                "substrate_material"
                if spec.interface_kind == "SA"
                else "effective_material"
            )
            material = _exact_mapping(
                binding.get(material_key), "surface binding material"
            )
            if "permittivity" not in material:
                raise ValueError(
                    "field-side material lacks explicit relative permittivity"
                )
            epsilon_values.add(
                _finite(
                    material["permittivity"],
                    "field-side relative permittivity",
                )
            )
        if len(epsilon_values) != 1:
            raise ValueError(
                f"contribution {spec.contribution_id!r} spans inconsistent field-side materials"
            )
        epsilon_s = epsilon_values.pop()
        for margin in spec.margins_um:
            normal = 0.0
            tangential = 0.0
            binding_ids: list[str] = []
            for binding in local_bindings:
                binding_id = binding.get("binding_id")
                if not isinstance(binding_id, str) or not binding_id:
                    raise ValueError("surface binding id is invalid")
                key = f"{binding_id}@{margin:.17g}"
                values = _exact_mapping(surface[key], f"surface integral {key!r}")
                normal += _quantity(
                    values.get("normal"), "normal surface integral", "V^2"
                )
                tangential += _quantity(
                    values.get("tangential"), "tangential surface integral", "V^2"
                )
                binding_ids.append(binding_id)
            if spec.interface_kind == "MA":
                energy = (
                    _EPSILON_0_F_PER_M
                    * spec.film_thickness_m
                    * normal
                    / (2.0 * spec.film_relative_permittivity)
                )
            elif spec.interface_kind == "MS":
                energy = (
                    _EPSILON_0_F_PER_M
                    * spec.film_thickness_m
                    * epsilon_s**2
                    * normal
                    / (2.0 * spec.film_relative_permittivity)
                )
            else:
                energy = (
                    _EPSILON_0_F_PER_M
                    * spec.film_thickness_m
                    * (
                        spec.film_relative_permittivity * tangential
                        + normal / spec.film_relative_permittivity
                    )
                    / 2.0
                )
            surface_rows.append(
                {
                    "contribution_id": spec.contribution_id,
                    "interface_kind": spec.interface_kind,
                    "field_side": spec.field_side,
                    "binding_ids": binding_ids,
                    "margin_um": margin,
                    "normal_integral_v2": normal,
                    "tangential_integral_v2": tangential,
                    "masked_area_m2": _quantity(
                        masked_areas[
                            f"{binding_id}@{margin:.17g}"
                        ]
                        if len(local_bindings) == 1
                        else {
                            "value": sum(
                                _quantity(
                                    masked_areas[f"{item['binding_id']}@{margin:.17g}"],
                                    "masked area",
                                    "m^2",
                                )
                                for item in local_bindings
                            ),
                            "unit": "m^2",
                        },
                        "masked area",
                        "m^2",
                    ),
                    "energy_j": energy,
                    "participation": energy / normalization_j,
                }
            )

    return {
        "schema_version": "scgsim.aedt.epr-mode-energy.v1",
        "frequency_hz": frequency_hz,
        "electric_energy_j": electric_energy_j,
        "magnetic_energy_j": magnetic_energy_j,
        "junction_inductive_energy_j": model_inductive_energy_j,
        "magnetic_energy_balance_j": magnetic_energy_j + model_inductive_energy_j,
        "junction_capacitive_energy_j": capacitive_energy_j,
        "normalization_energy_j": normalization_j,
        "electric_domains": electric_rows,
        "surface_contributions": surface_rows,
        "junctions": junction_rows,
    }


__all__ = [
    "combine_epr_mode",
    "plot_epr_result",
    "resolve_epr_result",
    "resolve_saved_solution",
]
