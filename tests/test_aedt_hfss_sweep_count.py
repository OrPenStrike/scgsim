"""Accepted caller-owned sweep counts; transport data below is synthetic.

No fixture represents a completed native run. Complete and partial synthetic
readbacks exercise the real validator without bypassing native identity checks.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from scgsim.aedt import _hfss_runtime as runtime
from scgsim.aedt._hfss_convergence import read_hfss_convergence
from scgsim.aedt.resolve import _validate_readback
from scgsim.aedt.spec import (
    FrequencySweepSpec, HfssDrivenSpec, HfssRunControl, LayerImport,
    ModalPort, ObjectBinding, PdkMaterial, TerminalPort,
)


def _spec(points=3, mode="modal"):
    ports = tuple(
        ModalPort(i, f"P{i}", side, ((x, 0, 0), (x, 1, 0)))
        if mode == "modal" else TerminalPort(i, f"P{i}", side, ("Ground",))
        for i, side, x in ((1, "-X", 0), (2, "+X", 10))
    )
    return HfssDrivenSpec(
        mode=mode, gds_path="synthetic.gds", project_name="Synthetic",
        design_name="Design", materials={
            "vacuum": PdkMaterial("vacuum", "vacuum", False, "Vacuum"),
            "silicon": PdkMaterial("silicon", "dielectric", False, "Silicon"),
            "metal": PdkMaterial("metal", "superconductor", True, None),
        }, vacuum_material_id="vacuum",
        layer_imports=(LayerImport(1, 0, "SignalLayer", 0, 1),
                       LayerImport(2, 0, "GroundLayer", 0, 1),
                       LayerImport(3, 0, "SubstrateLayer", -10, 0)),
        object_bindings=(ObjectBinding("Signal", 1, "signal", "metal"),
                         ObjectBinding("Ground", 2, "ground", "metal"),
                         ObjectBinding("Substrate", 3, "substrate", "silicon")),
        ports=ports, run_control=HfssRunControl(
            "Setup", "Sweep", FrequencySweepSpec(1, 2, points=points)),
        region_padding_um=(0, 0, 10, 10, 10, 10),
    )


class _SolutionData:
    """Double only for PyAEDT's returned solution-data transport."""
    primary_sweep = "Freq"
    units_sweeps = {"Freq": "GHz"}

    def __init__(self, count, short_expression=False):
        self.frequencies = [1 + i / (count - 1) for i in range(count)]
        self.count = count - int(short_expression)

    def get_expression_data(self, expression, component):
        return self.frequencies, [i + (0 if component == "real" else .5)
                                  for i in range(self.count)]


class _NativeTransport:
    """Synthetic native-call boundary; no Desktop, solver or fake service."""
    setup_names = ()

    def __init__(self, count):
        self.count = count
        self.sweep_kwargs = None
        self.query = None
        self.export_ok = True
        self.readback_count = count
        self.sweep_error = None
        self.post = SimpleNamespace(get_solution_data=self.get_solution_data)

    def create_setup(self, name):
        return SimpleNamespace(props={}, enable_adaptive_setup_broadband=lambda *a, **k: True,
                               update=lambda: True)

    def create_linear_count_sweep(self, *args, **kwargs):
        self.sweep_kwargs = kwargs
        if self.sweep_error:
            raise self.sweep_error
        return SimpleNamespace(props={"Type": "Fast", "RangeStart": "1.0GHz",
                                      "RangeEnd": "2.0GHz", "RangeCount": self.readback_count})

    def export_touchstone(self, **kwargs):
        if not self.export_ok:
            return False
        data = _SolutionData(self.count)
        Path(kwargs["output_file"]).write_text(
            "! Port[1] = P1\n! Port[2] = P2\n# GHz S RI R 50\n" +
            "".join(f"{f} 1 0 2 0 3 0 4 0\n" for f in data.frequencies))
        return True

    def get_solution_data(self, **kwargs):
        self.query = kwargs
        return _SolutionData(self.count)


class HfssSweepCountTests(unittest.TestCase):
    def test_complete_synthetic_strict_readback_accepts_authored_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = _spec(5.0)
            transport = _NativeTransport(5)
            _, readback = runtime._export(transport, root, spec,
                                         [{"modal_excitation": "P1"},
                                          {"modal_excitation": "P2"}])
            ports = []
            for port in spec.ports:
                line = [list(point) for point in port.integration_line_um]
                ports.append({
                    "index": port.index, "boundary": port.name,
                    "modal_excitation": port.name, "face_id": port.index,
                    "face_center_um": [0, 0, 0],
                    "requested": {"integration_line_um": line, "modes": 1,
                                  "renormalize": False, "deembed_um": 0.0,
                                  "characteristic_impedance": "Zpi"},
                    "native": {
                        "excitation_names": [p.name for p in spec.ports[:port.index]],
                        "boundary_properties": {"Deembed": False, "Name": port.name,
                                                "Num Modes": "1", "Renorm All Modes": False,
                                                "Type": "Wave Port"},
                        "saved_boundary": {"bound_type": "Wave Port", "wave_port_type": "Modal",
                                           "faces": [port.index], "num_modes": 1, "deembed": False,
                                           "mode_number": 1, "use_integration_line": True,
                                           "characteristic_impedance": "Zpi",
                                           "integration_line_um": line},
                    },
                })
            results = root / "Synthetic.aedtresults"
            profiles = results / "Design.results"
            profiles.mkdir(parents=True)
            (results / "Design.asol").write_text(
                "SimSetupName='Setup'\nP( File='Adaptive.profile')\n")
            (profiles / "Adaptive.profile").write_text(
                "Name='Adaptive Pass 1'\n\\'Max Mag. Delta S\\', 0.01,\n"
                "Adaptive Passes converged\n\\'Max solved tets\\', 100,\n")
            # Isolated synthetic diagnostic text; never a protected workspace log.
            log = root / "batch.log"
            log.write_text("Synthetic unittest transport evidence; no native execution.\n")
            receipt = {
                "connected": {"aedt_version": spec.aedt_version, "pyaedt_version": spec.pyaedt_version},
                "ports": ports, "result_readback": readback,
                "setup": {"name": "Setup", "native": {
                    "solve_type": "Broadband", "low_frequency": "1GHz", "high_frequency": "2GHz",
                    "maximum_delta_s": .02, "maximum_passes": 99, "minimum_passes": 1,
                    "minimum_converged_passes": 1, "percent_refinement": 30.0}},
                "convergence": read_hfss_convergence(root, spec),
                "diagnostics": {"batch_log": "batch.log", "present": True,
                                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                                "physics_warnings": []},
            }
            self.assertIsNone(_validate_readback(root, receipt, spec))
            for key in ("modal_s", "touchstone"):
                with self.subTest(count_mismatch=key):
                    bad = deepcopy(receipt)
                    bad["result_readback"][key]["records"] = 20000
                    with self.assertRaisesRegex(RuntimeError, "frequency readback|Touchstone readback"):
                        _validate_readback(root, bad, spec)

    def test_default_and_historical_payload_identity(self):
        self.assertEqual(FrequencySweepSpec(1, 2).points, 20000)
        # These canonical hashes freeze historical authored numeric representation.
        for value, digest in (
            (20000, "4fbbe08ff714f18363f8d8fa77951ea9e9d99d5a750aa25ba88a271855c1c086"),
            (20000.0, "a580178fbd641852eaa5f89c504aa1c4ad1c16b4168a32dd4afa79781057e901"),
        ):
            with self.subTest(points=value, type=type(value)):
                payload = FrequencySweepSpec(1, 2, points=value).to_payload()
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
                self.assertIs(type(FrequencySweepSpec(**payload).points), type(value))

    def test_whole_numeric_count_is_lossless_and_fractional_is_not_truncated(self):
        for value in (7, 7.0):
            self.assertEqual(FrequencySweepSpec(1, 2, points=value).points, value)
        for value in (7.5, True, "7"):
            with self.subTest(points=value), self.assertRaisesRegex(ValueError, "whole-number"):
                FrequencySweepSpec(1, 2, points=value)

    def test_authored_count_flows_through_setup_export_and_strict_readback(self):
        for mode, points in (("modal", 3), ("terminal", 5.0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec = _spec(points, mode)
                transport = _NativeTransport(int(points))
                runtime._setup(transport, spec)
                self.assertEqual(transport.sweep_kwargs["num_of_freq_points"], int(points))
                self.assertIs(type(transport.sweep_kwargs["num_of_freq_points"]), int)
                key = "modal_excitation" if mode == "modal" else "terminal_excitation"
                ports = [{"index": i, key: f"P{i}", "unresolved_native": True}
                         for i in (1, 2)]
                hashes, readback = runtime._export(transport, root, spec, ports)
                prefix = "S" if mode == "modal" else "St"
                self.assertEqual(transport.query["expressions"],
                                 [f"{prefix}({a},{b})" for a in ("P1", "P2")
                                  for b in ("P1", "P2")])
                csv_path = root / "results" / mode / f"{mode}_{prefix.lower()}.csv"
                with csv_path.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), int(points))
                self.assertEqual((float(rows[0]["frequency_ghz"]),
                                  float(rows[-1]["frequency_ghz"])), (1, 2))
                for relative, digest in hashes.items():
                    self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)
                self.assertEqual(readback["touchstone"]["records"], int(points))
                # Partial transport evidence is never marked completed: reaching the
                # untouched port check proves count acceptance without bypassing it.
                receipt = {"connected": {"aedt_version": spec.aedt_version,
                                         "pyaedt_version": spec.pyaedt_version},
                           "ports": ports, "result_readback": readback}
                error = ("modal request does not match spec" if mode == "modal"
                         else "unresolved terminal native evidence")
                with self.assertRaisesRegex(RuntimeError, error):
                    _validate_readback(root, receipt, spec)
                result_key = f"{mode}_{prefix.lower()}"
                for label, mutate, message in (
                    ("csv count", lambda r: r["result_readback"][result_key].update(records=9), "frequency readback"),
                    ("touchstone count", lambda r: r["result_readback"]["touchstone"].update(records=9), "Touchstone readback"),
                    ("endpoint", lambda r: r["result_readback"][result_key].update(first_frequency_ghz=1.1), "frequency readback"),
                    ("port order", lambda r: r["result_readback"]["touchstone"].update(port_order=["P2", "P1"]), "Touchstone readback"),
                    ("identity", lambda r: r["connected"].update(pyaedt_version="other"), "version identity"),
                ):
                    with self.subTest(mismatch=label):
                        bad = deepcopy(receipt)
                        mutate(bad)
                        with self.assertRaisesRegex(RuntimeError, message):
                            _validate_readback(root, bad, spec)

    def test_native_sweep_mismatch_and_exception_are_not_hidden(self):
        transport = _NativeTransport(3)
        transport.readback_count = 4
        with self.assertRaisesRegex(RuntimeError, "sweep readback mismatch"):
            runtime._setup(transport, _spec())
        failure = RuntimeError("synthetic native transport failure")
        transport.sweep_error = failure
        with self.assertRaises(RuntimeError) as caught:
            runtime._setup(transport, _spec())
        self.assertIs(caught.exception, failure)

    def test_export_failure_and_expression_mismatch_remain_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = _NativeTransport(3)
            transport.export_ok = False
            with self.assertRaisesRegex(RuntimeError, "Touchstone export failed"):
                runtime._export(transport, root, _spec(),
                                [{"modal_excitation": "P1"}, {"modal_excitation": "P2"}])
            with self.assertRaisesRegex(RuntimeError, "expression length mismatch"):
                runtime._write_complex_csv(_SolutionData(3, True), ["S(P1,P1)"],
                                           root / "short.csv", suffix="", spec=_spec())


if __name__ == "__main__":
    unittest.main()
