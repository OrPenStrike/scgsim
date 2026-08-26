"""Accepted Q2D one-file native RLGC export contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scgsim.aedt import (
    MatrixRunControl,
    PdkMaterial,
    Q2dConductorSpec,
    Q2dRectangleSpec,
    Q2dSpec,
    resolve_results,
)
from scgsim.aedt._matrix_export import parse_matrix_export, read_q2d_rlgc_matrix
from scgsim.aedt.run import _export_q2d
from scgsim.aedt.util import file_sha256, write_json


def _spec(project_name: str = "PublicQ2dRun") -> Q2dSpec:
    return Q2dSpec(
        project_name=project_name,
        design_name="Q2DDesign1",
        materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        },
        vacuum_material_id="vacuum",
        rectangles=(
            Q2dRectangleSpec("Substrate", (0.0, -10.0), (20.0, 10.0), "silicon"),
            Q2dRectangleSpec("SignalObject", (8.0, 0.0), (4.0, 0.2), "metal"),
            Q2dRectangleSpec("GroundObject", (0.0, 0.0), (4.0, 0.2), "metal"),
        ),
        conductors=(
            Q2dConductorSpec("Signal", "SignalLine", ("SignalObject",), 0.2),
            Q2dConductorSpec("Ground", "ReferenceGround", ("GroundObject",), 0.2),
        ),
        run_control=MatrixRunControl("Setup1", 6.0, 20, 0.1),
        region_padding_um=(10.0, 10.0, 10.0, 10.0),
    )


def _native_matrix() -> str:
    return """Setup1:LastAdaptive
Problem Type:  CG, RL
C Units:pF/meter, G Units:mho/meter, R Units:ohm/meter, L Units:nH/meter
Reduce Matrix:  Original
Frequency:  6GHz

Capacitance Matrix
,Signal,
Signal,1.0

Capacitance Matrix Coupling Coefficient
,Signal,
Signal,1.0

Spice Capacitance Matrix
,Signal,
Signal,1.0

Conductance Matrix
,Signal,
Signal,2.0

Inductance Matrix
,Signal,
Signal,3.0

Resistance Matrix
,Signal,
Signal,4.0

"""


class _FakeQ2d:
    def __init__(self, *, succeeds: bool = True) -> None:
        self.succeeds = succeeds
        self.call: dict[str, object] | None = None

    def export_matrix_data(self, **kwargs: object) -> bool:
        self.call = kwargs
        if self.succeeds:
            Path(str(kwargs["file_name"])).write_text(
                _native_matrix(), encoding="utf-8"
            )
        return self.succeeds


class Q2dNativeExportTests(unittest.TestCase):
    def test_export_uses_one_public_combined_native_file(self) -> None:
        spec = _spec()
        app = _FakeQ2d()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, readback = _export_q2d(app, root, spec)

            self.assertEqual(
                app.call,
                {
                    "file_name": str(root / "results/q2d/rlgc_matrix.csv"),
                    "problem_type": "CG, RL",
                    "variations": "",
                    "setup": "Setup1",
                    "sweep": "LastAdaptive",
                    "reduce_matrix": "Original",
                    "r_unit": "ohm",
                    "l_unit": "nH",
                    "c_unit": "pF",
                    "g_unit": "mho",
                    "freq": "6e+09",
                    "matrix_type": "Maxwell, Spice, Couple",
                    "export_ac_dc_res": False,
                    "precision": 15,
                    "field_width": 20,
                    "use_sci_notation": True,
                    "length_setting": "Distributed",
                    "length": "1meter",
                },
            )
            self.assertEqual(
                {path.name for path in (root / "results/q2d").iterdir()},
                {"rlgc_matrix.csv"},
            )
            self.assertEqual(set(hashes), {"results/q2d/rlgc_matrix.csv"})
            self.assertEqual(readback["matrices"]["primary_rows"], 4)

    def test_export_fails_when_public_api_returns_false(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(RuntimeError, "combined CG/RL"),
        ):
            _export_q2d(_FakeQ2d(succeeds=False), Path(temporary), _spec())

    def test_native_primary_blocks_fail_closed(self) -> None:
        spec = _spec()
        replacements = {
            "wrong setup": ("Setup1:LastAdaptive", "Setup2:LastAdaptive"),
            "wrong frequency": ("Frequency:  6GHz", "Frequency:  7GHz"),
            "wrong units": ("C Units:pF/meter", "C Units:fF/meter"),
            "unexpected row": (
                "Signal,1.0\n\n",
                "Signal,1.0\nGhost,999\n\n",
            ),
            "wrong conductor order": (",Signal,\nSignal,1.0", ",Other,\nOther,1.0"),
            "nonfinite value": ("Signal,1.0", "Signal,nan"),
            "repeated primary block": (
                "\nConductance Matrix\n",
                "\nCapacitance Matrix\n,Signal,\nSignal,1.0\n\nConductance Matrix\n",
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rlgc_matrix.csv"
            for label, (old, new) in replacements.items():
                with self.subTest(label=label):
                    path.write_text(_native_matrix().replace(old, new, 1))
                    with self.assertRaises(RuntimeError):
                        read_q2d_rlgc_matrix(path, spec)

    def test_q3d_parser_keeps_its_existing_native_shape(self) -> None:
        native = """Problem Type:  C
C Units:pF, G Units:mho
Frequency:  6GHz

Capacitance Matrix
,Signal,
Signal,1.0

Conductance Matrix
,Signal,
Signal,2.0

"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "c_matrix.csv"
            path.write_text(native, encoding="utf-8")
            rows, summary = parse_matrix_export(
                path,
                "Q3D",
                "C",
                6.0,
                {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["quantities"], ["C", "G"])

    def test_resolver_binds_one_native_file_and_rejects_superseded_files(
        self,
    ) -> None:
        spec = _spec()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = root / "aedt_spec.json"
            project_path = root / f"{spec.project_name}.aedt"
            matrix_path = root / "results/q2d/rlgc_matrix.csv"
            receipt_path = root / "metadata/aedt_run_receipt.json"
            write_json(spec_path, spec.to_payload())
            project_path.write_text("synthetic project", encoding="utf-8")
            matrix_path.parent.mkdir(parents=True)
            matrix_path.write_text(_native_matrix(), encoding="utf-8")
            write_json(
                receipt_path,
                {
                    "schema_version": "scgsim.aedt.receipt.v1",
                    "status": "completed",
                    "mode": "q2d",
                    "source": {
                        "spec": "aedt_spec.json",
                        "spec_sha256": file_sha256(spec_path),
                    },
                    "project": project_path.name,
                    "save": {
                        "ok": True,
                        "project_sha256": file_sha256(project_path),
                    },
                    "release": {"ok": True},
                    "convergence": {"cg": {}, "rl": {}},
                    "outputs": {
                        project_path.name: file_sha256(project_path),
                        "results/q2d/rlgc_matrix.csv": file_sha256(matrix_path),
                    },
                },
            )
            with patch("scgsim.aedt.resolve._validate_readback"):
                result = resolve_results(root)
                self.assertEqual(result.primary_csv, matrix_path)
                before = sorted(path.relative_to(root) for path in root.rglob("*"))
                physics = result.physics_results()
                self.assertEqual(
                    [row["quantity"] for row in physics], ["C", "G", "L", "R"]
                )
                self.assertEqual(
                    sorted(path.relative_to(root) for path in root.rglob("*")), before
                )
                for name in ("cg_matrix.csv", "rl_matrix.csv", "matrices.csv"):
                    legacy = matrix_path.with_name(name)
                    legacy.touch()
                    with self.assertRaisesRegex(RuntimeError, "superseded"):
                        resolve_results(root)
                    legacy.unlink()

                matrix_path.write_text(_native_matrix().replace("1.0", "1.5", 1))
                with self.assertRaisesRegex(RuntimeError, "hash"):
                    resolve_results(root)


if __name__ == "__main__":
    unittest.main()
