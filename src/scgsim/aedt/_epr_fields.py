"""HFSS field-expression authoring and raw integral evaluation for EPR.

This module owns one bound field context at a time.  It does not select modes,
iterate adaptive passes, own Desktop, or derive participation ratios.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_native_scalar(value: Any) -> dict[str, Any]:
    """Parse one native scalar without treating false/missing as numeric zero."""

    if value is False or value is None or isinstance(value, bool):
        raise RuntimeError("HFSS field calculator returned no scalar value")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
        return {"value": number, "unit": "", "raw": value}
    text = str(value).strip()
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([^\s]*)",
        text,
    )
    if match is None:
        raise RuntimeError(f"HFSS field calculator scalar is invalid: {value!r}")
    number = float(match.group(1))
    if not math.isfinite(number):
        raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
    return {"value": number, "unit": match.group(2), "raw": text}


def _constant(value: float) -> list[str]:
    return [f"Scalar_Constant({value:.17g})"]


def _binary(left: Sequence[str], right: Sequence[str], operation: str) -> list[str]:
    return [*left, *right, f"Operation('{operation}')"]


def _square(value: Sequence[str]) -> list[str]:
    return [*value, "Scalar_Constant(2)", "Operation('Pow')"]




def _same_postprocessing_value(requested: str, actual: str) -> bool:
    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value

    requested_number, requested_unit = decompose_variable_value(requested)
    if requested == actual:
        return not (
            isinstance(requested_number, (int, float))
            and not math.isfinite(requested_number)
        ) and not re.fullmatch(
            r"[+-]?(?:nan|inf(?:inity)?)[A-Za-z]*", requested.strip(), re.I
        )
    actual_number, actual_unit = decompose_variable_value(actual)
    return (
        isinstance(requested_number, (int, float))
        and not isinstance(requested_number, bool)
        and isinstance(actual_number, (int, float))
        and not isinstance(actual_number, bool)
        and math.isfinite(requested_number)
        and math.isfinite(actual_number)
        and requested_unit == actual_unit
        and requested_number == actual_number
    )


def _validated_postprocessing_readback(
    values: Mapping[str, str], postprocessing: Mapping[str, Any]
) -> dict[str, str]:
    """Preserve exact expressions; accept rewritten finite literals only at exact parity."""

    observed: dict[str, str] = {}
    for name, requested in values.items():
        if name not in postprocessing:
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} is missing from readback"
            )
        actual = str(postprocessing[name])
        if not _same_postprocessing_value(requested, actual):
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} differs from requested value"
            )
        observed[name] = actual
    return observed


def install_variables(app: Any, values: Mapping[str, str]) -> dict[str, str]:
    manager = app.variable_manager
    existing_names = set(manager.design_variable_names)
    existing_postprocessing = manager.post_processing_variables
    new_props: list[Any] = ["NAME:NewProps"]
    for name, value in values.items():
        if not _same_postprocessing_value(value, value):
            raise RuntimeError(f"HFSS postprocessing variable {name!r} is nonfinite")
        if (
            name in existing_postprocessing
            and existing_postprocessing[name].sweep is False
            and _same_postprocessing_value(value, str(existing_postprocessing[name]))
        ):
            continue
        if name in existing_names or name.startswith("$"):
            if not manager.set_variable(
                name,
                expression=value,
                sweep=False,
                overwrite=True,
                is_post_processing=True,
            ):
                raise RuntimeError(f"HFSS postprocessing variable failed: {name!r}")
            continue
        new_props.append(
            [
                "NAME:" + name,
                "PropType:=", "PostProcessingVariableProp",
                "UserDef:=", True,
                "Value:=", value,
                "Description:=", "",
                "ReadOnly:=", False,
                "Hidden:=", False,
                "Sweep:=", False,
            ]
        )
    if len(new_props) > 1:
        changed = app.odesign.ChangeProperty(
            [
                "NAME:AllTabs",
                [
                    "NAME:LocalVariableTab",
                    ["NAME:PropServers", "LocalVariables"],
                    new_props,
                ],
            ]
        )
        if changed is False:
            raise RuntimeError("HFSS postprocessing variable batch creation failed")
    return _validated_postprocessing_readback(values, manager.post_processing_variables)




def field_integral_operations(
    *,
    quantity: str,
    selection_name: str,
    adjacent_side: bool = False,
    normal_vector: Sequence[float] | None = None,
) -> list[str]:
    """Return the documented HFSS stack for one raw integral quantity."""

    if quantity == "effective_volume":
        value = _constant(1.0)
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "electric_volume":
        value = [
            "Fundamental_Quantity('E')",
            "Operation('Conj')",
            "Fundamental_Quantity('E')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "magnetic_energy":
        value = [
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Conj')",
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"electric_normal", "electric_tangential", "masked_area"}:
        # Area is independent of which field side is sampled.
        enter_surface = (
            "EnterAdjacentSurf"
            if adjacent_side and quantity != "masked_area"
            else "EnterSurface"
        )
        if quantity == "masked_area":
            value = _constant(1.0)
            terminal = [
                f"{enter_surface}('{selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ]
            return [*value, *terminal]
        if normal_vector is None or len(normal_vector) != 3 or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in normal_vector
        ):
            raise ValueError("surface field integral requires a finite native normal")
        length = math.sqrt(sum(float(item) ** 2 for item in normal_vector))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("surface field integral requires a finite nonzero native normal")
        vector = "Vector_Constant(" + ", ".join(
            f"{float(item) / length:.17g}" for item in normal_vector
        ) + ")"
        normal_real = [
            "Fundamental_Quantity('E')",
            "Operation('Real')",
            vector,
            "Operation('Dot')",
        ]
        normal_imag = [
            "Fundamental_Quantity('E')",
            "Operation('Imag')",
            vector,
            "Operation('Dot')",
        ]
        normal_square = _binary(_square(normal_real), _square(normal_imag), "+")
        if quantity == "electric_normal":
            value = normal_square
        else:
            full_real = ["Fundamental_Quantity('E')", "Operation('Real')"]
            full_imag = ["Fundamental_Quantity('E')", "Operation('Imag')"]
            full_square = _binary(
                [*full_real, *full_real, "Operation('Dot')"],
                [*full_imag, *full_imag, "Operation('Dot')"],
                "+",
            )
            value = _binary(full_square, normal_square, "-")
        # This records side intent in the expression identity.  The adjacent
        # expression is authored through the native reporter, not this CLC
        # spelling, whose saved-file form is not established by PyAEDT.
        terminal = [
            f"{enter_surface}('{selection_name}')",
            "Operation('SurfaceValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError(
            "junction voltage requires an explicit directed vector; use "
            "junction_voltage_operations"
        )
    else:
        raise ValueError(f"unsupported HFSS EPR field quantity {quantity!r}")
    if adjacent_side and quantity not in {"electric_normal", "electric_tangential"}:
        raise ValueError("adjacent_side is valid only for surface field quantities")
    return [*value, *terminal]


def junction_voltage_operations(
    *, quantity: str, selection_name: str, direction_xy: Sequence[float]
) -> list[str]:
    """Integrate Re/Im(E) dot the declared in-plane terminal direction."""

    if quantity not in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError("junction voltage quantity must be real or imaginary")
    if len(direction_xy) != 2:
        raise ValueError("junction direction requires two values")
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("junction direction must be finite and nonzero")
    component = "Real" if quantity.endswith("real") else "Imag"
    return [
        "Fundamental_Quantity('E')",
        f"Operation('{component}')",
        f"Vector_Constant({dx / norm:.17g}, {dy / norm:.17g}, 0)",
        "Operation('Dot')",
        f"EnterSurface('{selection_name}')",
        "Operation('SurfaceValue')",
        "Operation('Integrate')",
    ]


def create_verified_face_list(
    app: Any, *, name: str, face_ids: Sequence[int]
) -> dict[str, Any]:
    ids = [int(value) for value in face_ids]
    if not ids or len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise ValueError("face list requires distinct positive native face ids")
    if name in {str(item.name) for item in app.modeler.user_lists}:
        raise RuntimeError(f"HFSS face-list name already exists: {name!r}")
    item = app.modeler.create_face_list(ids, name=name)
    if item is False or item is None or item.name != name or item.props.get("Type") != "Face":
        raise RuntimeError(f"HFSS face-list creation failed: {name!r}")
    native_id = app.modeler.get_entitylist_id(name)
    if type(native_id) is not int or native_id <= 0 or native_id != item.props.get("ID"):
        raise RuntimeError(f"HFSS face-list readback failed: {name!r}")
    members = item.props.get("EntityList")
    if isinstance(members, (int, str)):
        members = [members]
    if [int(value) for value in members or ()] != ids:
        raise RuntimeError(f"HFSS face-list membership differs: {name!r}")
    return {"name": name, "native_id": native_id, "face_ids": ids}


def evaluate_named_expression(
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Author and evaluate one immutable named expression in a bound context."""

    authored = author_named_expression(
        app,
        purpose=purpose,
        operations=operations,
        solution=solution,
        phase_degrees=phase_degrees,
        dependencies=dependencies,
        selection=selection,
        evidence_dir=evidence_dir,
    )
    name = authored["name"]
    raw = app.post.fields_calculator.evaluate(
        name,
        setup=solution,
        intrinsics={"Phase": f"{phase_degrees:g}deg"},
    )
    return {
        **authored,
        "scalar": parse_native_scalar(raw),
    }


def author_named_expression(
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
    namespace: str = "prepared",
    adjacent_selection_name: str | None = None,
) -> dict[str, Any]:
    """Author one immutable expression through the same local batch compiler."""

    compiled = compile_named_expression(
        purpose=purpose,
        operations=operations,
        solution=solution,
        phase_degrees=phase_degrees,
        dependencies=dependencies,
        selection=selection,
        namespace=namespace,
        adjacent_selection_name=adjacent_selection_name,
    )
    load_compiled_expressions(app, [compiled], evidence_dir)
    return compiled[0]


def _clc_block(name: str, operations: Sequence[str], expression: str | None = None) -> str:
    lines = ["$begin 'Named_Expression'", f"\tName('{name}')"]
    if expression is not None:
        lines.append(f"\tExpression('{expression}')")
    lines.extend(f"\t{operation}" for operation in operations)
    lines.append("$end 'Named_Expression'")
    return "\n".join(lines) + "\n"


def compile_named_expression(
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    namespace: str = "prepared",
    adjacent_selection_name: str | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]:
    """Compile one immutable field definition without an AEDT call."""

    if solution != str(solution).strip() or not solution:
        raise ValueError("solution must be non-empty canonical text")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be non-empty text")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", namespace):
        raise ValueError("expression namespace must be a safe native identifier")
    if isinstance(operations, (str, bytes)) or any(
        not isinstance(item, str) or not item or "\n" in item or "\0" in item
        for item in operations
    ):
        raise ValueError("expression operations must be canonical text entries")
    for key in ("object_name", "selection_name"):
        value = selection.get(key)
        if value is not None and (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
        ):
            raise ValueError(f"expression selection {key} is not a safe native name")
    identity = {
        "schema_version": "scgsim.aedt.epr-expression.v1",
        "namespace": namespace,
        "purpose": purpose,
        "operations": list(operations),
        "solution": solution,
        "phase_degrees": float(phase_degrees),
        "dependencies": dict(sorted(dependencies.items())),
        "selection": dict(selection),
    }
    identity["sha256"] = _digest(identity)
    name = f"scgsim_epr_{identity['sha256'][:24]}"
    definitions: list[tuple[str, str]] = []
    evidence: dict[str, bytes] = {}
    if adjacent_selection_name is None:
        block = _clc_block(name, operations)
        definitions.append((name, block))
        evidence[f"{name}.clc"] = block.encode("utf-8")
    else:
        if (
            selection.get("selection_name") != adjacent_selection_name
            or list(operations[-3:])
            != [
                f"EnterAdjacentSurf('{adjacent_selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ]
        ):
            raise ValueError("adjacent surface expression selection is inconsistent")
        integrand_name = f"{name}_integrand"
        integrand_block = _clc_block(integrand_name, operations[:-3])
        final_block = _clc_block(
            name,
            [
                f"NameOfExpression('{integrand_name}')",
                f"EnterAdjacentSurface('{adjacent_selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ],
            expression=(
                f"Integrate(AdjacentSurface({adjacent_selection_name}), "
                f"{integrand_name})"
            ),
        )
        definitions.extend(((integrand_name, integrand_block), (name, final_block)))
        evidence[f"{integrand_name}.clc"] = integrand_block.encode("utf-8")
        evidence[f"{name}.clc"] = (integrand_block + final_block).encode("utf-8")
    record = {
        "name": name,
        "identity": identity,
        "definition_sha256": hashlib.sha256(evidence[f"{name}.clc"]).hexdigest(),
        "definition_status": "authored_and_loaded",
        "solution": solution,
        "phase_degrees": float(phase_degrees),
    }
    return record, definitions, evidence


def load_compiled_expressions(
    app: Any,
    compiled: Sequence[tuple[dict[str, Any], list[tuple[str, str]], dict[str, bytes]]],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Load dependency-ordered local definitions once, then verify every name."""

    definitions = [definition for _, blocks, _ in compiled for definition in blocks]
    names = [name for name, _ in definitions]
    if not names or len(names) != len(set(names)):
        raise RuntimeError("EPR batch definitions are empty or repeat a native name")
    calculator = app.post.fields_calculator
    started = time.perf_counter()
    existing = [name for name in names if calculator.is_expression_defined(name)]
    if existing:
        raise RuntimeError(f"HFSS expressions already exist: {existing!r}")
    collision_check_seconds = time.perf_counter() - started
    started = time.perf_counter()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    files = {filename: data for _, _, evidence in compiled for filename, data in evidence.items()}
    if len(files) != sum(len(evidence) for _, _, evidence in compiled):
        raise RuntimeError("EPR evidence filenames repeat")
    for filename, data in files.items():
        (evidence_dir / filename).write_bytes(data)
    library = evidence_dir / "scgsim_epr_batch.clc"
    library_bytes = "".join(block for _, block in definitions).encode("utf-8")
    library.write_bytes(library_bytes)
    evidence_write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    try:
        result = calculator.ofieldsreporter.LoadNamedExpressions(
            str(library), "Fields", names
        )
    except Exception as exc:
        try:
            present = [name for name in names if calculator.is_expression_defined(name)]
            progress = f"{len(present)}/{len(names)} definitions became present"
        except Exception as readback_exc:
            progress = f"{len(names)} requested definitions; partial readback unavailable: {readback_exc!r}"
        raise RuntimeError(
            f"HFSS batch named-expression load raised after {progress}"
        ) from exc
    batch_import_seconds = time.perf_counter() - started
    started = time.perf_counter()
    present = [name for name in names if calculator.is_expression_defined(name)]
    if result is False or len(present) != len(names):
        present_set = set(present)
        missing = [name for name in names if name not in present_set]
        raise RuntimeError(
            f"HFSS batch named-expression load was partial or failed: "
            f"{len(present)}/{len(names)} present; missing={missing!r}; returned={result!r}"
        )
    return {
        "library_sha256": hashlib.sha256(library_bytes).hexdigest(),
        "native_definition_count": len(names),
        "main_expression_count": len(compiled),
        "collision_check_seconds": round(collision_check_seconds, 6),
        "evidence_write_seconds": round(evidence_write_seconds, 6),
        "batch_import_seconds": round(batch_import_seconds, 6),
        "postload_readback_seconds": round(time.perf_counter() - started, 6),
    }


__all__ = [
    "author_named_expression",
    "create_verified_face_list",
    "evaluate_named_expression",
    "field_integral_operations",
    "install_variables",
    "parse_native_scalar",
    "junction_voltage_operations",
]
