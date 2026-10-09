"""Owned AEDT 2024.2/PyAEDT 1.3 Eigenmode setup construction and readback.

Ordinary Eigenmode and EPR share this base. EPR owns its explicit cache extension;
Driven and sweep properties never enter this payload through mutable setup caches.
"""

from __future__ import annotations

from typing import Any

from scgsim.aedt.runtime.native.common import saved_setup_properties
from scgsim.aedt.specs.common import REQUIRED_AEDT_VERSION
from scgsim.aedt.specs.hfss import EigenmodeRunControl


def _control_fields(control: EigenmodeRunControl) -> dict[str, Any]:
    return {
        "MinimumFrequency": f"{control.minimum_frequency_ghz:g}GHz",
        "NumModes": control.num_modes,
        "MaxDeltaFreq": control.maximum_delta_frequency_percent,
        "MaximumPasses": control.maximum_passes,
        "MinimumPasses": control.minimum_passes,
        "MinimumConvergedPasses": control.minimum_converged_passes,
        "PercentRefinement": control.percent_refinement,
    }


def eigenmode_setup_payload(control: EigenmodeRunControl) -> dict[str, Any]:
    """Return a fresh complete pinned HFSSEigen payload, not live setup props.

    PyAEDT 1.3 setup_templates.HFSSEigen/index2 plus MeshLink supplies these
    defaults; none of its <=2024.2 template overlays replaces index2.
    """
    return {
        **_control_fields(control),
        "ConvergeOnRealFreq": False,
        "IsEnabled": True,
        "MeshLink": {"ImportMesh": False},
        "BasisOrder": 1,
        "DoLambdaRefine": True,
        "DoMaterialLambda": True,
        "SetLambdaTarget": False,
        "Target": 0.2,
        "UseMaxTetIncrease": False,
        "Name": control.setup_name,
    }


def _require_eigenmode(app: Any) -> None:
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("Eigenmode setup requires the owned AEDT 2024.2 desktop")
    if app.solution_type != "Eigenmode":
        raise RuntimeError("Eigenmode setup authoring requires an Eigenmode design")


def _require_setup(setup: Any, control: EigenmodeRunControl) -> None:
    if (
        setup is None
        or setup is False
        or setup.setuptype != 2
        or setup.name != control.setup_name
    ):
        raise RuntimeError("native setup did not retain the requested HFSSEigen identity")


def create_eigenmode_setup(app: Any, control: EigenmodeRunControl) -> None:
    """Create one explicit Eigenmode setup with supported kwargs batching."""
    _require_eigenmode(app)
    if app.setup_names:
        raise RuntimeError("new Eigenmode design must not inherit a setup")
    # create_setup batches keyword edits with auto_update=False internally.
    payload = eigenmode_setup_payload(control)
    payload.pop("Name")
    setup = app.create_setup(
        name=control.setup_name, setup_type="HFSSEigen", **payload
    )
    _require_setup(setup, control)


def read_eigenmode_setup(app: Any, control: EigenmodeRunControl) -> dict[str, Any]:
    """Preserve the existing saved seven-control projection; never re-author."""
    raw = saved_setup_properties(app, control.setup_name)
    keys = {
        "minimum_frequency": "MinimumFrequency",
        "num_modes": "NumModes",
        "maximum_delta_frequency_percent": "MaxDeltaFreq",
        "maximum_passes": "MaximumPasses",
        "minimum_passes": "MinimumPasses",
        "minimum_converged_passes": "MinimumConvergedPasses",
        "percent_refinement": "PercentRefinement",
    }
    fields = _control_fields(control)
    native = {key: raw.get(field) for key, field in keys.items()}
    expected = {key: fields[field] for key, field in keys.items()}
    if native != expected:
        raise RuntimeError(f"HFSS Eigenmode saved setup readback mismatch: {native!r}")
    return {"name": control.setup_name, "native": native}


def submit_eigenmode_cache(
    app: Any,
    control: EigenmodeRunControl,
    *,
    expression_cache: list[Any],
    use_cache_for: list[str],
) -> None:
    """Edit the canonical Eigenmode base with only the EPR-owned extension."""
    _require_eigenmode(app)
    setup = app.get_setup(control.setup_name)
    _require_setup(setup, control)
    properties = eigenmode_setup_payload(control)
    if not expression_cache or expression_cache[0] != "NAME:ExpressionCache":
        raise ValueError("EPR cache extension must be a native ExpressionCache block")
    properties["UseCacheFor"] = list(use_cache_for)
    args = setup._setup_dict_to_arg(name=control.setup_name, props=properties)
    args.append(expression_cache)
    setup.omodule.EditSetup(control.setup_name, args)
