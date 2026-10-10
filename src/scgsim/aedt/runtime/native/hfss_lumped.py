"""Native assignments on source-bound support Sheets.

The caller owns source/effective geometry and Entity-to-object bindings. This
module never constructs a closest-edge electrode or chooses a reference Net.
Generic RLC boundaries remain separate from Josephson-junction EPR records.
"""

from __future__ import annotations

from decimal import Decimal
import re
from typing import Any


def assign_terminal_lumped(
    app: Any,
    sheet: Any,
    *,
    name: str,
    reference_objects: list[str],
    impedance_ohm: float,
    renormalize: bool,
    deembed_um: float,
) -> dict[str, Any]:
    """Use the pinned Terminal AutoIdentify route on one already-bound Sheet."""
    if app.solution_type != "Terminal":
        raise RuntimeError("source-bound Lumped ports require DrivenTerminal")
    if deembed_um != 0:
        raise ValueError("native Lumped Terminal supports zero deembedding only")
    if len(sheet.faces) != 1:
        raise RuntimeError(f"Lumped support is not one native face: {sheet.name!r}")
    before = set(app.oboundary.GetExcitationsOfType("Terminal"))
    boundary = app.lumped_port(
        sheet.name,
        reference=reference_objects,
        create_port_sheet=False,
        auto_identify=True,
        impedance=impedance_ohm,
        name=name,
        renormalize=renormalize,
        deembed=bool(deembed_um),
        terminals_rename=False,
    )
    if boundary is False or boundary is None:
        raise RuntimeError(f"native Lumped Terminal assignment failed: {name!r}")
    terminals = sorted(set(app.oboundary.GetExcitationsOfType("Terminal")) - before)
    if len(terminals) != 1:
        raise RuntimeError(f"native Lumped Terminal inventory differs: {terminals!r}")
    return {
        "boundary": boundary.name,
        "terminal_excitation": terminals[0],
        "native_terminal_names": terminals,
        "object_name": sheet.name,
        "native_object_id": int(sheet.id),
        "face_id": int(sheet.faces[0].id),
        "assignment_api": "lumped_port(auto_identify=True, create_port_sheet=False)",
    }


def assign_rlc(
    app: Any,
    sheet: Any,
    *,
    name: str,
    contact_points_um: list[list[float]],
    topology: str,
    resistance_ohm: float | None,
    inductance_h: float | None,
    capacitance_f: float | None,
) -> dict[str, Any]:
    """Assign exactly the authored enabled R/L/C branches on a bound support."""
    native_topology = {"series": "Serial", "parallel": "Parallel"}[topology]
    # The convenience writer rejects all-falsy values before native assignment.
    # Author the same native boundary once, with None alone disabling a branch.
    props = {
        "Objects": [sheet.name],
        "CurrentLine": {
            "Start": [f"{value}um" for value in contact_points_um[0]],
            "End": [f"{value}um" for value in contact_points_um[1]],
        },
        "RLC Type": native_topology,
    }
    for enabled, value_key, value, unit in (
        ("UseResist", "Resistance", resistance_ohm, "ohm"),
        ("UseInduct", "Inductance", inductance_h, "H"),
        ("UseCap", "Capacitance", capacitance_f, "F"),
    ):
        props[enabled] = value is not None
        if value is not None:
            props[value_key] = f"{value}{unit}"
    boundary = app._create_boundary(name, props, "Lumped RLC")
    if boundary is False or boundary is None:
        raise RuntimeError(f"native generic RLC assignment failed: {name!r}")
    return rlc_record(
        sheet,
        name=boundary.name,
        contact_points_um=contact_points_um,
        topology=topology,
        resistance_ohm=resistance_ohm,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
    )


def rlc_record(
    sheet: Any,
    *,
    name: str,
    contact_points_um: list[list[float]],
    topology: str,
    resistance_ohm: float | None,
    inductance_h: float | None,
    capacitance_f: float | None,
) -> dict[str, Any]:
    """The same authored request ledger is used on construction and saved rebind."""
    return {
        "boundary": name,
        "object_name": sheet.name,
        "native_object_id": int(sheet.id),
        "native_face_ids": [int(face.id) for face in sheet.faces],
        "requested": {
            "topology": topology,
            "native_topology": {"series": "Serial", "parallel": "Parallel"}[topology],
            "contact_points_um": contact_points_um,
            "resistance_ohm": resistance_ohm,
            "inductance_h": inductance_h,
            "capacitance_f": capacitance_f,
        },
    }


def read_rlc_parameters(app: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Retain live native enable/value evidence; disabled branches stay disabled."""
    from ansys.aedt.core.modules.boundary.common import BoundaryObject

    live = BoundaryObject(app, record["boundary"], auto_update=False)._child_object
    if live is None:
        raise RuntimeError("native generic RLC property object is unavailable")
    names = list(live.GetPropNames())
    values = {name: live.GetPropValue(name) for name in names}
    record["native_properties"] = values
    from scgsim.aedt.runtime.native.common import _resolve_native_assignment

    raw_ids = [
        int(value) for value in app.oboundary.GetBoundaryAssignment(record["boundary"])
    ]
    typed, covered_faces, covered_objects = _resolve_native_assignment(app, raw_ids)
    record["assignment"] = {
        "raw_ids": raw_ids,
        "typed": typed,
        "covered_faces": sorted(covered_faces),
        "covered_objects": sorted(covered_objects),
    }
    if covered_objects != {record["object_name"]} or covered_faces != set(
        record["native_face_ids"]
    ):
        raise RuntimeError("native generic RLC assignment differs from its support")
    requested = record["requested"]
    record["normalized_values_si"] = {}
    if values.get("RLC Type") != requested["native_topology"]:
        raise RuntimeError("native generic RLC topology differs")
    for key, enable, prop in (
        ("resistance_ohm", "Use Resist", "Resistance"),
        ("inductance_h", "Use Induct", "Inductance"),
        ("capacitance_f", "Use Cap", "Capacitance"),
    ):
        raw_enabled = values.get(enable)
        if raw_enabled in (True, "true", "True", 1):
            enabled = True
        elif raw_enabled in (False, "false", "False", 0):
            enabled = False
        else:
            raise RuntimeError(f"native generic RLC enable unavailable: {enable}")
        if enabled != (requested[key] is not None):
            raise RuntimeError(f"native generic RLC enable differs: {enable}")
        if enabled:
            observed = rlc_scalar_si(values.get(prop), key)
            record["normalized_values_si"][key] = str(observed)
            if observed != Decimal(str(requested[key])):
                raise RuntimeError(f"native generic RLC value differs: {prop}")
    return record


def rlc_scalar_si(value: Any, quantity: str) -> Decimal:
    """Shared exact live/offline RLC unit decoding; raw evidence is retained."""
    units = {
        "resistance_ohm": {
            "uOhm": "1e-6",
            "mOhm": "1e-3",
            "ohm": "1",
            "kOhm": "1e3",
            "megohm": "1e6",
            "GOhm": "1e9",
        },
        "inductance_h": {
            "fH": "1e-15",
            "pH": "1e-12",
            "nH": "1e-9",
            "uH": "1e-6",
            "mH": "1e-3",
            "H": "1",
        },
        "capacitance_f": {
            "fF": "1e-15",
            "pF": "1e-12",
            "nF": "1e-9",
            "uF": "1e-6",
            "mF": "1e-3",
            "F": "1",
        },
    }
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([A-Za-z]+)\s*",
        str(value),
    )
    if match is None or match[2] not in units[quantity]:
        raise RuntimeError(f"native generic RLC scalar is unavailable: {value!r}")
    return Decimal(match[1]) * Decimal(units[quantity][match[2]])
