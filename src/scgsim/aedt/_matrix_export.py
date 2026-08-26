"""Strict parsing of solver-native AEDT matrix exports."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from .spec import Q2dSpec


def parse_matrix_export(
    path: Path,
    solver: str,
    problem: str,
    frequency_ghz: float,
    titles: dict[str, str],
    *,
    expected_setup: str | None = None,
    expected_unit_line: str | None = None,
    expected_labels: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{solver} {problem} matrix export is missing or empty")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    required_lines = {
        f"Problem Type:  {problem}",
        f"Frequency:  {frequency_ghz:g}GHz",
    }
    strict = any(
        value is not None
        for value in (expected_setup, expected_unit_line, expected_labels)
    )
    if any(line not in lines for line in required_lines):
        raise RuntimeError(f"{solver} {problem} matrix header is invalid")
    if strict:
        required_lines.add("Reduce Matrix:  Original")
        if expected_setup is not None:
            required_lines.add(expected_setup)
        if any(lines.count(line) != 1 for line in required_lines):
            raise RuntimeError(f"{solver} {problem} matrix header is invalid")
    unit_line = next((line for line in lines if " Units:" in line), None)
    if unit_line is None or (
        expected_unit_line is not None and unit_line != expected_unit_line
    ):
        raise RuntimeError(f"{solver} {problem} matrix units are invalid")
    units = dict(re.findall(r"([CGRL]) Units:([^,]+)", unit_line))
    result: list[dict[str, Any]] = []
    labels_by_quantity: dict[str, list[str]] = {}
    for title, quantity in titles.items():
        if title not in lines or (strict and lines.count(title) != 1):
            raise RuntimeError(f"{solver} export must contain one exact {title!r}")
        title_index = lines.index(title)
        header_index = title_index + 1
        while header_index < len(lines) and not lines[header_index].strip():
            header_index += 1
        header = next(csv.reader([lines[header_index]]))
        labels = [value.strip() for value in header[1:] if value.strip()]
        if (
            not labels
            or len(set(labels)) != len(labels)
            or (expected_labels is not None and labels != expected_labels)
        ):
            raise RuntimeError(f"{solver} {title} labels are invalid")
        labels_by_quantity[quantity] = labels
        for row_index, row_name in enumerate(labels, start=header_index + 1):
            values = next(csv.reader([lines[row_index]]))
            if values[0].strip() != row_name or len(values[1:]) != len(labels):
                raise RuntimeError(f"{solver} {title} matrix is not square")
            for column, raw in zip(labels, values[1:], strict=True):
                value = float(raw)
                if not math.isfinite(value):
                    raise RuntimeError(f"{solver} {title} contains a non-finite value")
                result.append(
                    {
                        "problem_type": problem,
                        "quantity": quantity,
                        "row": row_name,
                        "column": column,
                        "value": value,
                        "unit": units[quantity],
                    }
                )
        block_end = header_index + 1 + len(labels)
        if strict and block_end < len(lines) and lines[block_end].strip():
            raise RuntimeError(f"{solver} {title} contains unexpected matrix rows")
    return result, {
        "problem_type": problem,
        "frequency_ghz": frequency_ghz,
        "quantities": list(titles.values()),
        "labels": labels_by_quantity,
        "rows": len(result),
        "bytes": path.stat().st_size,
    }


def read_q2d_rlgc_matrix(
    path: Path, spec: Q2dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Q2D combined CG/RL matrix export is missing or empty")
    expected_unit_line = (
        "C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter"
    )
    if path.read_text(encoding="utf-8", errors="strict").splitlines()[:5] != [
        f"{spec.run_control.setup_name}:LastAdaptive",
        "Problem Type:  CG, RL",
        expected_unit_line,
        "Reduce Matrix:  Original",
        f"Frequency:  {spec.run_control.frequency_ghz:g}GHz",
    ]:
        raise RuntimeError("Q2D combined CG/RL matrix header is invalid")
    labels = [
        conductor.name
        for conductor in spec.conductors
        if conductor.conductor_type == "SignalLine"
    ]
    return parse_matrix_export(
        path,
        "Q2D",
        "CG, RL",
        spec.run_control.frequency_ghz,
        {
            "Capacitance Matrix": "C",
            "Conductance Matrix": "G",
            "Inductance Matrix": "L",
            "Resistance Matrix": "R",
        },
        expected_setup=f"{spec.run_control.setup_name}:LastAdaptive",
        expected_unit_line=expected_unit_line,
        expected_labels=labels,
    )


__all__ = ["parse_matrix_export", "read_q2d_rlgc_matrix"]
