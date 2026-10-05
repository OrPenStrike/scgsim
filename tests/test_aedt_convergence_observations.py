"""Accepted native convergence observations using generic synthetic text only.

Files exercise real parsers, not native solving or completed-run resolution.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scgsim.aedt._hfss_convergence import read_hfss_convergence
from scgsim.aedt._q2d_convergence import read_q2d_convergence, read_q3d_convergence
from scgsim.aedt.spec import (
    EigenmodeRunControl, FrequencySweepSpec, HfssDrivenSpec, HfssEigenmodeSpec,
    HfssRunControl, LayerImport, MatrixRunControl, ModalPort, ObjectBinding,
    PdkMaterial, Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec, Q3dNetSpec, Q3dSpec,
)


def _gds_inputs():
    return dict(
        gds_path="synthetic.gds", project_name="Synthetic", design_name="Design",
        materials={"vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
                   "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
                   "metal": PdkMaterial("metal", "superconductor", True, None)},
        vacuum_material_id="vacuum",
        layer_imports=(LayerImport(1, 0, "SignalLayer", 0, 1),
                       LayerImport(2, 0, "GroundLayer", 0, 1),
                       LayerImport(3, 0, "SubstrateLayer", -10, 0)),
        object_bindings=(ObjectBinding("Signal", 1, "signal", "metal"),
                         ObjectBinding("Ground", 2, "ground", "metal"),
                         ObjectBinding("Substrate", 3, "substrate", "silicon")),
        region_padding_um=(0, 0, 10, 10, 10, 10),
    )


def _hfss_spec(eigenmode=False):
    inputs = _gds_inputs()
    if eigenmode:
        return HfssEigenmodeSpec(**inputs, run_control=EigenmodeRunControl(
            "Setup", 1, 1, 10, .1, minimum_converged_passes=2))
    return HfssDrivenSpec(**inputs, mode="modal", ports=(
        ModalPort(1, "P1", "-X", ((0, 0, 0), (0, 1, 0))),
        ModalPort(2, "P2", "+X", ((10, 0, 0), (10, 1, 0)))),
        run_control=HfssRunControl("Setup", "Sweep", FrequencySweepSpec(1, 2),
                                   maximum_passes=10, maximum_delta_s=.1))


def _write_driven(root, spec, *, converged=True, final_pass=2, delta=.2, setup="Setup"):
    results = root / "Synthetic.aedtresults"
    profiles = results / "Design.results"
    profiles.mkdir(parents=True)
    asol = results / "Design.asol"
    asol.write_text(f"SimSetupName='{setup}'\nP( File='Adaptive.profile')\n")
    profile = profiles / "Adaptive.profile"
    status = "Adaptive Passes converged" if converged else "Adaptive Passes did not converge"
    profile.write_text("\n".join([
        *(f"Name='Adaptive Pass {i}'" for i in range(1, final_pass + 1)),
        f"\\'Max Mag. Delta S\\', {delta},", status, "\\'Max solved tets\\', 100,", ""]))
    return {"asol": asol, "profile": profile}


def _write_eigenmode(root, *, status="Yes", current=.2, consecutive=2,
                     table_final=None, target=.1, malformed=False):
    path = root / "results" / "eigenmode" / "adaptive-convergence.prop"
    path.parent.mkdir(parents=True)
    table_final = current if table_final is None else table_final
    text = (f"Setup: Setup\nCompleted: 2\nMaximum: 10\nMinimum: 1\n"
            f"Target Consecutive Passes: 2\nCriterion: Max Delta Freq. %\n"
            f"Target: {target}\nCurrent: {current}\n"
            f"Current Consecutive Passes: {consecutive}\nConverged: {status}\n"
            f"Pass Number|Solved Elements|Max Delta Freq. %|\n"
            f"1| 100| N/A|\n2| 200| {table_final}|\n")
    if malformed:
        text = text.replace("2| 200|", "invalid row|")
    path.write_text(text)
    return {"export_convergence": path}


def _matrix_spec(q3d=False):
    control = MatrixRunControl("Setup", 6, 3, 1)
    if q3d:
        return Q3dSpec(**_gds_inputs(), run_control=control, nets=(
            Q3dNetSpec("Signal", "Signal", ("Signal",), "Signal", "+X", "Signal", "-X"),
            Q3dNetSpec("Ground", "Ground", ("Ground",))))
    inputs = _gds_inputs()
    return Q2dSpec(project_name=inputs["project_name"], design_name=inputs["design_name"],
                   materials=inputs["materials"], vacuum_material_id="vacuum",
                   rectangles=(Q2dRectangleSpec("Substrate", (0, -10), (20, 10), "silicon"),
                               Q2dRectangleSpec("Signal", (8, 0), (4, .2), "metal"),
                               Q2dRectangleSpec("Ground", (0, 0), (4, .2), "metal")),
                   conductors=(Q2dConductorSpec("Signal", "SignalLine", ("Signal",), .2),
                               Q2dConductorSpec("Ground", "ReferenceGround", ("Ground",), .2)),
                   run_control=control, region_padding_um=(10, 10, 10, 10))


def _write_matrix(root, *, q3d=False, converged=True, delta=2, final_pass=2,
                  target=1, maximum=3, ambiguous=False):
    results = root / "Synthetic.aedtresults"
    profiles = results / "Design.results"
    profiles.mkdir(parents=True)
    names = ("CapConv", "ACRLConv") if q3d else ("CGConv", "RLConv")
    blocks, statuses = [], []
    for i, name in enumerate(names, 1):
        fields = f"p={final_pass}, tri=1200, "
        fields += f"delta={delta}" if q3d else f"de={delta}, ee=0.25"
        blocks.append(f"$begin '{i}'\nConvSetupName='{name}'\nConvTarget='{target}'\n"
                      f"MaxPasses='{maximum}'\nc({fields})\n$end '{i}'\n")
        status = "Adaptive Passes converged" if converged else "Adaptive Passes did not converge"
        if ambiguous:
            status += "\n" + ("Adaptive Passes did not converge" if converged
                               else "Adaptive Passes converged")
        statuses.append(f"$begin '{i}'\n{status}\n$end '{i}'\n")
    asol = results / "Design.asol"
    profile = profiles / "Adaptive.profile"
    asol.write_text("PRF(1) File='Adaptive.profile'\n" + "".join(blocks))
    profile.write_text("".join(statuses))
    return {"asol": asol, "profile": profile}


class ConvergenceObservationTests(unittest.TestCase):
    def assert_sources(self, root, value, sources):
        for key, path in sources.items():
            self.assertEqual(value["sources"][key], {
                "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def test_driven_native_yes_above_target_and_no_before_maximum(self):
        for converged in (True, False):
            with self.subTest(converged=converged), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec = _hfss_spec()
                sources = _write_driven(root, spec, converged=converged)
                value = read_hfss_convergence(root, spec)
                self.assertEqual((value["converged"], value["final_delta"], value["target"],
                                  value["final_pass"], value["unit"]),
                                 (converged, .2, .1, 2, "ratio"))
                self.assertEqual(value["stop_reason"], "Adaptive Passes converged" if converged
                                 else "Adaptive Passes did not converge")
                self.assert_sources(root, value, sources)

    def test_eigenmode_three_native_observations_remain_observations(self):
        for status, current, consecutive in (("Yes", .2, 2), ("No", .2, 2), ("Yes", .05, 1)):
            with self.subTest(status=status, current=current, consecutive=consecutive), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sources = _write_eigenmode(root, status=status, current=current, consecutive=consecutive)
                value = read_hfss_convergence(root, _hfss_spec(True))
                self.assertEqual((value["converged"], value["final_delta"], value["target"],
                                  value["final_pass"], value["unit"]),
                                 (status == "Yes", current, .1, 2, "percent"))
                self.assertEqual(value["final_solved_element_count"], 200)
                self.assertEqual(value["stop_reason"], f"Converged : {status}")
                self.assert_sources(root, value, sources)
                # Native count/history remain in the source, not a fabricated field.
                self.assertIn(f"Current Consecutive Passes: {consecutive}",
                              sources["export_convergence"].read_text())

    def test_hfss_identity_format_and_self_consistency_errors_remain(self):
        cases = (({"target": .3}, "target does not match"),
                 ({"table_final": .3}, "final convergence delta does not match"),
                 ({"malformed": True}, "pass row is malformed"),
                 ({"current": float("nan")}, "value is not finite"))
        for fields, error in cases:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _write_eigenmode(root, **fields)
                with self.assertRaisesRegex(RuntimeError, error):
                    read_hfss_convergence(root, _hfss_spec(True))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_driven(root, _hfss_spec(), setup="Other")
            with self.assertRaisesRegex(RuntimeError, "setup identity"):
                read_hfss_convergence(root, _hfss_spec())

    def test_matrix_native_yes_above_target_and_no_before_maximum(self):
        for q3d in (False, True):
            for converged in (True, False):
                with self.subTest(q3d=q3d, converged=converged), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    sources = _write_matrix(root, q3d=q3d, converged=converged)
                    reader = read_q3d_convergence if q3d else read_q2d_convergence
                    value = reader(root, _matrix_spec(q3d))
                    for key in (("capacitance", "ac_rl") if q3d else ("cg", "rl")):
                        record = value[key]
                        self.assertEqual((record["converged"], record["target_percent"],
                                          record["final_matrix_delta_percent"], record["final_pass"],
                                          record["final_triangle_count"]), (converged, 1, 2, 2, 1200))
                        self.assertEqual(record["stop_reason"], "Adaptive Passes converged" if converged
                                         else "Adaptive Passes did not converge")
                        if not q3d:
                            self.assertEqual(record["final_error_percent"], .25)
                    self.assert_sources(root, value, sources)

    def test_matrix_controls_ambiguous_status_and_missing_artifact_remain_errors(self):
        for q3d, fields, error in ((False, {"target": 2}, "target does not match"),
                                  (True, {"maximum": 4}, "maximum passes do not match"),
                                  (True, {"ambiguous": True}, "ambiguous convergence status")):
            with self.subTest(q3d=q3d, fields=fields), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _write_matrix(root, q3d=q3d, **fields)
                reader = read_q3d_convergence if q3d else read_q2d_convergence
                with self.assertRaisesRegex(RuntimeError, error):
                    reader(root, _matrix_spec(q3d))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "evidence is missing"):
                read_q2d_convergence(Path(temporary), _matrix_spec())


if __name__ == "__main__":
    unittest.main()
