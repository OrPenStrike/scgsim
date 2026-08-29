"""Accepted Palace report-family and notebook display contract."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scgsim import palace
from scgsim.palace import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceResultSelection,
    PalaceReturnedReceipt,
    PalaceTrustReport,
    ParsedTable,
    PhysicsQuantitiesReport,
    ResolvedPalaceResult,
    SimulationBenchmarkReport,
    inspect_run_trustworthiness,
)
from scgsim.palace.report import (
    AmrPassSnapshot,
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    _read_surface_epr,
    _surface_ranking_figure,
)
from scgsim.palace.returned_receipt import _iteration_output_paths


class _Html:
    def __init__(self, data: str) -> None:
        self.data = data


@contextmanager
def _captured_notebook_display():
    displayed: list[object] = []
    package = types.ModuleType("IPython")
    module = types.ModuleType("IPython.display")
    module.HTML = _Html
    module.display = displayed.append
    package.display = module
    with patch.dict(
        sys.modules,
        {"IPython": package, "IPython.display": module},
    ):
        yield displayed


def _surface_snapshot(pass_index: int, source: str) -> SurfaceEprSeriesSnapshot:
    record = SurfaceEprRecord(
        index=1,
        interface_type="MA",
        surface_id="SURFACE_1",
        face_kind="interface",
        owner_semantic_ids=("OWNER_1",),
        net_id="GROUND",
        equipotential_id="GROUND",
        source_provenance={"authority": "generic-test"},
        participation=0.25 + 0.01 * pass_index,
        quality_factor=math.inf,
        loss_tangent=0.0,
    )
    return SurfaceEprSeriesSnapshot(
        pass_index=pass_index,
        source=source,
        series_index=1,
        series_kind="mode",
        records=(record,),
        quality_factor_total=None,
        t1_seconds=None,
        loss_status="unavailable_nonfinite",
    )


def _trust_report(
    *, completeness: str = "complete", latest_source: str = "final"
) -> PalaceTrustReport:
    sources = (
        ("iteration01", "final") if completeness == "complete" else ("iteration01",)
    )
    passes = tuple(
        AmrPassSnapshot(
            pass_index=index,
            source=source,
            path=Path(source),
            frequencies_ghz=(5.0,),
            eig_columns={"Re{f} (GHz)": (5.0,)},
            port_epr=None,
            capacitance_matrix_f=None,
            surface_epr=(_surface_snapshot(index, source),),
            error_indicators={"Norm": 0.1},
            error_norm=0.1,
            degrees_of_freedom=100,
            mesh_elements=50,
            elapsed_total_s=1.0 + index,
            peak_node_memory_mb=10.0,
        )
        for index, source in enumerate(sources)
    )
    selection = PalaceResultSelection(
        final_snapshot_status="readable" if completeness == "complete" else "missing",
        selected_source=latest_source,
        selected_path=Path(latest_source),
        selected_pass_index=passes[-1].pass_index,
        reason=(
            "final_snapshot"
            if completeness == "complete"
            else "latest_complete_iteration_after_failed_attempt"
        ),
        integrity="receipt_bound"
        if completeness == "complete"
        else "observed_unsealed",
    )
    return PalaceTrustReport(
        run_dir=Path("run"),
        problem="Eigenmode",
        route="A",
        profile="ltlab-slurm",
        completeness=completeness,
        latest_source=latest_source,
        selection=selection,
        failure=None,
        identity={},
        passes=passes,
        amr_tolerance=0.02,
        amr_max_passes=1,
        durations={},
        cost={},
        provenance={},
    )


def _file_record(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        return {"path": relative, "bytes": None, "sha256": None, "present": False}
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "present": True,
    }


def _failed_eigenmode_run(
    root: Path,
    *,
    solver_exit_code: int = 137,
    log_text: str = "solver killed\n",
    include_iteration: bool = True,
    seal_iteration: bool = False,
) -> tuple[str, dict[str, object]]:
    metadata = root / "metadata"
    results = root / "results" / "palace"
    logs = root / "logs"
    metadata.mkdir(parents=True)
    results.mkdir(parents=True)
    logs.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (results / "eig.csv").write_text("m,Re{f} (GHz)\n", encoding="utf-8")
    (logs / "palace-1.log").write_text(log_text, encoding="utf-8")
    if include_iteration:
        iteration = results / "iteration07"
        iteration.mkdir()
        (iteration / "eig.csv").write_text(
            "m,Re{f} (GHz)\n1,5.0\n",
            encoding="utf-8",
        )

    handoff_id = "generic-handoff-id"
    input_hashes = [_file_record(root, "config.json")]
    (metadata / "palace_handoff_metadata.json").write_text(
        json.dumps(
            {
                "problem": "Eigenmode",
                "route": "A",
                "profile": "direct-local",
                "handoff_id": handoff_id,
                "status": "prepared",
                "hashes": input_hashes,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths = [
        "results/palace/eig.csv",
        "results/palace/port-EPR.csv",
        "results/palace/port-I.csv",
        "results/palace/port-V.csv",
        "results/palace/domain-E.csv",
        "results/palace/surface-Q.csv",
        "results/palace/error-indicators.csv",
        "results/palace/palace.json",
        "logs/palace-1.log",
    ]
    if seal_iteration:
        output_paths.append("results/palace/iteration07/eig.csv")
    output_files = [_file_record(root, relative) for relative in output_paths]
    receipt: dict[str, object] = {
        "schema": "palace-returned-run-receipt.v1",
        "schema_version": 1,
        "handoff_id": handoff_id,
        "route": "A",
        "problem": "Eigenmode",
        "status": "failed",
        "exit_code": solver_exit_code,
        "solver_exit_code": solver_exit_code,
        "tee_exit_code": 0,
        "identity_verified": False,
        "input_hashes": input_hashes,
        "output_files": output_files,
        "log": _file_record(root, "logs/palace-1.log"),
    }
    (metadata / "palace_returned_run_receipt.json").write_text(
        json.dumps(receipt) + "\n",
        encoding="utf-8",
    )
    return handoff_id, receipt


def _resolved_result() -> ResolvedPalaceResult:
    tables = {
        "eig": ParsedTable(
            name="eig",
            path=Path("eig.csv"),
            headers=("m", "Re{f} (GHz)"),
            rows=({"m": 1, "Re{f} (GHz)": 5.0},),
        ),
        "error-indicators": ParsedTable(
            name="error-indicators",
            path=Path("error-indicators.csv"),
            headers=("Norm",),
            rows=({"Norm": 0.1},),
        ),
    }
    receipt = PalaceReturnedReceipt(
        schema="palace-returned-run-receipt.v1",
        schema_version=1,
        handoff_id="handoff",
        status="completed",
        exit_code=0,
        solver_exit_code=0,
        tee_exit_code=0,
        route="A",
        problem="Eigenmode",
        identity_verified=True,
        timestamp_utc="2026-08-20T00:00:00Z",
        job_identity={},
        input_hashes=(),
        output_files=(),
        log=None,
        solver_identity={},
    )
    return ResolvedPalaceResult(
        run_dir=Path("run"),
        problem="Eigenmode",
        route="A",
        has_returned_outputs=True,
        status="completed",
        performance=PalacePerformance(counts={}, durations={}),
        cost=PalaceCost(
            problem_degrees_of_freedom=100,
            mesh_elements=50,
            mpi_size=1,
            openmp_threads=1,
            peak_memory_megabytes={},
            peak_node_memory_megabytes={},
            linear_solver={},
            git_tag="test",
        ),
        tables=tables,
        returned_receipt=receipt,
        provenance=PalaceProvenance(
            handoff_metadata={},
            run_metadata={},
            resource_record={},
            handoff_archive_manifest={},
            index_map={},
            mesh_manifest={},
            config={},
            palace_json={},
            returned_receipt={},
        ),
    )


class PalaceReportUxTests(unittest.TestCase):
    def test_public_methods_and_aggregate_order(self) -> None:
        result = _resolved_result()
        original_tables = result.tables
        trust = _trust_report().show_run_trustworthiness(
            theme="dark",
            show_details=True,
        )
        with (
            patch(
                "scgsim.palace.report._show_run_trustworthiness",
                return_value=trust,
            ) as show_trust,
            _captured_notebook_display() as displayed,
        ):
            returned = result.show_all_results(
                theme="dark",
                ranking_limit=10,
                show_details=True,
            )

        self.assertIsNone(returned)
        self.assertEqual(len(displayed), 3)
        self.assertIs(displayed[0], trust)
        benchmark = displayed[1]
        self.assertIsInstance(benchmark, SimulationBenchmarkReport)
        self.assertIs(benchmark.trust, trust)
        self.assertTrue(benchmark.show_details)
        physics = displayed[2]
        self.assertIsInstance(physics, PhysicsQuantitiesReport)
        self.assertIs(physics.trust, trust)
        self.assertEqual(physics.ranking_limit, 10)
        show_trust.assert_called_once_with(
            result,
            theme="dark",
            show_details=True,
        )
        self.assertIs(result.tables, original_tables)
        self.assertIn("eig", result.tables)

        parameters = inspect.signature(result.show_all_results).parameters
        self.assertEqual(tuple(parameters), ("theme", "ranking_limit", "show_details"))
        self.assertFalse(hasattr(result, "show_surface_epr_physics"))
        self.assertFalse(hasattr(palace, "NativeTabularSummary"))

    def test_partial_report_supports_the_same_display_surface(self) -> None:
        partial = _trust_report(
            completeness="partial",
            latest_source="iteration01",
        )
        trust = partial.show_run_trustworthiness(theme="dark", show_details=True)
        benchmark = SimulationBenchmarkReport(
            trust,
            {"performance_metadata": {"completeness": "partial"}},
            True,
        )
        physics = PhysicsQuantitiesReport(trust, 5)
        with (
            patch.object(
                PalaceTrustReport,
                "show_run_trustworthiness",
                return_value=trust,
            ) as show_trust,
            patch.object(
                PalaceTrustReport,
                "show_simulation_benchmark",
                return_value=benchmark,
            ) as show_benchmark,
            patch.object(
                PalaceTrustReport,
                "show_physics_quantities",
                return_value=physics,
            ) as show_physics,
            _captured_notebook_display() as displayed,
        ):
            returned = partial.show_all_results(
                theme="dark",
                ranking_limit=5,
                show_details=True,
            )

        self.assertIsNone(returned)
        self.assertEqual(displayed, [trust, benchmark, physics])
        show_trust.assert_called_once_with(theme="dark", show_details=True)
        show_benchmark.assert_called_once_with(show_details=True)
        show_physics.assert_called_once_with(theme="dark", ranking_limit=5)
        self.assertEqual(
            tuple(inspect.signature(partial.show_all_results).parameters),
            ("theme", "ranking_limit", "show_details"),
        )
        self.assertEqual(
            tuple(inspect.signature(partial.show_run_trustworthiness).parameters),
            ("theme", "show_details"),
        )
        self.assertEqual(
            tuple(inspect.signature(partial.show_physics_quantities).parameters),
            ("theme", "ranking_limit"),
        )
        metadata = partial.show_simulation_benchmark().data["performance_metadata"]
        self.assertEqual(metadata["completeness"], "partial")
        self.assertEqual(metadata["latest_source"], "iteration01")

    def test_details_are_opt_in_and_machine_data_remain_available(self) -> None:
        trust = _trust_report()
        with (
            patch.object(
                PalaceTrustReport, "_convergence_items", return_value=["numerical"]
            ),
            patch.object(
                PalaceTrustReport, "_surface_convergence_items", return_value=[]
            ),
            _captured_notebook_display() as displayed,
        ):
            trust.show_run_trustworthiness()._ipython_display_()
        html_output = "".join(
            item.data for item in displayed if isinstance(item, _Html)
        )
        self.assertNotIn("<h3>Provenance</h3>", html_output)

        with (
            patch.object(
                PalaceTrustReport, "_convergence_items", return_value=["numerical"]
            ),
            patch.object(
                PalaceTrustReport, "_surface_convergence_items", return_value=[]
            ),
            _captured_notebook_display() as displayed,
        ):
            trust.show_run_trustworthiness(show_details=True)._ipython_display_()
        html_output = "".join(
            item.data for item in displayed if isinstance(item, _Html)
        )
        self.assertIn("<h3>Provenance</h3>", html_output)

        benchmark = trust.show_simulation_benchmark()
        self.assertIsInstance(benchmark, SimulationBenchmarkReport)
        self.assertIn("performance_metadata", benchmark.data)
        with (
            patch("scgsim.palace.report._show_figure"),
            _captured_notebook_display() as displayed,
        ):
            benchmark._ipython_display_()
        html_output = "".join(
            item.data for item in displayed if isinstance(item, _Html)
        )
        self.assertIn("Simulation Benchmark", html_output)
        self.assertNotIn("Benchmark metadata", html_output)

        with (
            patch("scgsim.palace.report._show_figure"),
            _captured_notebook_display() as displayed,
        ):
            trust.show_simulation_benchmark(show_details=True)._ipython_display_()
        html_output = "".join(
            item.data for item in displayed if isinstance(item, _Html)
        )
        self.assertIn("Benchmark metadata", html_output)

    def test_surface_ranking_reserves_one_row_per_two_line_label(self) -> None:
        records = tuple(
            SurfaceEprRecord(
                index=index,
                interface_type=("MA", "MS", "SA")[index % 3],
                surface_id=f"SURFACE_{index}",
                face_kind="interface",
                owner_semantic_ids=(f"OWNER_{index}",),
                net_id="GROUND",
                equipotential_id="GROUND",
                source_provenance={"authority": "generic-test"},
                participation=1.0 / index,
                quality_factor=math.inf,
                loss_tangent=0.0,
            )
            for index in range(1, 21)
        )
        snapshot = SurfaceEprSeriesSnapshot(
            pass_index=6,
            source="iteration07",
            series_index=1,
            series_kind="mode",
            records=records,
            quality_factor_total=None,
            t1_seconds=None,
            loss_status="unavailable_nonfinite",
        )

        figure = _surface_ranking_figure(snapshot, 20, "light")

        self.assertIsNotNone(figure)
        self.assertEqual(figure.layout.height, 1110)
        self.assertEqual(figure.layout.bargap, 0.38)

    def test_trust_and_physics_render_separate_families(self) -> None:
        trust = _trust_report()
        total = object()
        percentage = object()
        ranking = object()
        trust_figures: list[object] = []
        with (
            patch.object(
                PalaceTrustReport,
                "_convergence_items",
                return_value=["numerical"],
            ),
            patch("scgsim.palace.report._surface_total_figure", return_value=total),
            patch(
                "scgsim.palace.report._surface_percentage_figure",
                return_value=percentage,
            ),
            patch(
                "scgsim.palace.report._surface_ranking_figure",
                side_effect=AssertionError("trust must not render physics ranking"),
            ),
            patch("scgsim.palace.report._show_figure", trust_figures.append),
            _captured_notebook_display(),
        ):
            trust._ipython_display_()
        self.assertEqual(trust_figures, [total, percentage])

        physics_figures: list[object] = []
        physics = trust.show_physics_quantities(ranking_limit=1)
        with (
            patch(
                "scgsim.palace.report._surface_total_figure",
                side_effect=AssertionError("physics must not redraw convergence"),
            ),
            patch(
                "scgsim.palace.report._surface_percentage_figure",
                side_effect=AssertionError("physics must not redraw convergence"),
            ),
            patch(
                "scgsim.palace.report._surface_ranking_figure",
                return_value=ranking,
            ),
            patch("scgsim.palace.report._show_figure", physics_figures.append),
            _captured_notebook_display() as displayed,
        ):
            physics._ipython_display_()
        self.assertEqual(physics_figures, [ranking])
        html_output = "".join(
            item.data for item in displayed if isinstance(item, _Html)
        )
        self.assertIn("unavailable / non-finite", html_output)

    def test_failed_parent_keeps_last_complete_iteration_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            handoff_id, receipt = _failed_eigenmode_run(root)
            trust = inspect_run_trustworthiness(root)

        self.assertEqual(trust.completeness, "partial")
        self.assertEqual(trust.latest_source, "iteration07")
        self.assertEqual(trust.selection.final_snapshot_status, "unreadable")
        self.assertEqual(
            trust.selection.reason,
            "latest_complete_iteration_after_failed_attempt",
        )
        self.assertEqual(trust.selection.integrity, "observed_unsealed")
        self.assertEqual(trust.selection.selected_pass_index, 6)
        self.assertEqual(
            trust.selection.selected_path,
            Path("results/palace/iteration07"),
        )
        self.assertFalse(trust.selection.selected_path.is_absolute())
        self.assertIsNotNone(trust.failure)
        self.assertEqual(trust.failure.category, "signal_killed")
        self.assertEqual(len(trust.passes), 1)
        self.assertEqual(trust.passes[0].source, "iteration07")
        self.assertEqual(trust.passes[0].path.name, "iteration07")
        self.assertEqual(trust.passes[0].frequencies_ghz, (5.0,))
        self.assertEqual(trust.identity["handoff_id"], handoff_id)
        self.assertEqual(trust.identity["receipt"], "failed / exit 137")
        self.assertEqual(trust.provenance["returned_receipt"], receipt)
        identity_html = trust._identity_html()
        self.assertIn("Run failed", identity_html)
        self.assertIn("Fallback result selected", identity_html)
        self.assertNotIn(str(root), identity_html)
        self.assertIsInstance(trust.show_run_trustworthiness(), PalaceTrustReport)
        self.assertIsInstance(trust.show_physics_quantities(), PhysicsQuantitiesReport)
        benchmark = trust.show_simulation_benchmark().data["performance_metadata"]
        self.assertEqual(benchmark["completeness"], "partial")
        self.assertEqual(benchmark["selection"]["selected_source"], "iteration07")
        self.assertEqual(
            benchmark["selection"]["selected_path"],
            "results/palace/iteration07",
        )
        self.assertNotIn(str(root), json.dumps(benchmark))
        self.assertEqual(benchmark["failure"]["category"], "signal_killed")
        benchmark_surface = trust.show_simulation_benchmark().data
        self.assertEqual(
            benchmark_surface["selected_snapshot"]["source"], "iteration07"
        )
        self.assertIn("attempted_run", benchmark_surface)

    def test_hash_verified_slurm_oom_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _failed_eigenmode_run(
                root,
                log_text=(
                    "slurmstepd: error: Detected 1 oom_kill event in StepId=1.0\n"
                ),
            )
            trust = inspect_run_trustworthiness(root)

        self.assertIsNotNone(trust.failure)
        self.assertEqual(trust.failure.category, "out_of_memory")
        self.assertEqual(trust.failure.evidence[-1]["marker"], "slurm_out_of_memory")

    def test_non_oom_solver_failure_uses_the_same_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _failed_eigenmode_run(root, solver_exit_code=2)
            trust = inspect_run_trustworthiness(root)

        self.assertEqual(trust.selection.selected_source, "iteration07")
        self.assertIsNotNone(trust.failure)
        self.assertEqual(trust.failure.category, "solver_error")

    def test_no_complete_iteration_keeps_reports_available(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _failed_eigenmode_run(root, include_iteration=False)
            trust = inspect_run_trustworthiness(root)

        self.assertEqual(trust.selection.reason, "no_complete_snapshot")
        self.assertEqual(trust.selection.integrity, "unavailable")
        self.assertEqual(trust.latest_source, "none")
        self.assertEqual(trust.show_physics_quantities().snapshots, ())
        with _captured_notebook_display() as displayed:
            self.assertIsNone(trust.show_all_results())
        self.assertEqual(len(displayed), 3)

    def test_sealed_iteration_is_receipt_bound_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _failed_eigenmode_run(root, seal_iteration=True)
            trust = inspect_run_trustworthiness(root)
            self.assertEqual(trust.selection.integrity, "receipt_bound")

            (root / "results/palace/iteration07/eig.csv").write_text(
                "m,Re{f} (GHz)\n1,6.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                inspect_run_trustworthiness(root)

    def test_recorder_collects_report_relevant_iteration_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            iteration = root / "results/palace/iteration07"
            iteration.mkdir(parents=True)
            (iteration / "eig.csv").write_text("m,Re{f} (GHz)\n1,5\n")
            (iteration / "surface-Q.csv").write_text("m,p_surf[1]\n1,0.1\n")
            (iteration / "ignored.txt").write_text("ignored\n")

            paths = _iteration_output_paths(root, "Eigenmode")

        self.assertEqual(
            paths,
            [
                "results/palace/iteration07/eig.csv",
                "results/palace/iteration07/surface-Q.csv",
            ],
        )

    def test_zero_loss_tangent_keeps_participation_and_marks_loss_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "surface-Q.csv"
            path.write_text(
                "m,p_surf[1],Q_surf[1]\n1,0.25,+inf\n",
                encoding="utf-8",
            )
            snapshots = _read_surface_epr(
                path,
                problem="Eigenmode",
                pass_index=0,
                source="iteration01",
                frequencies_ghz=(5.0,),
                expected_rows=1,
                bindings=(
                    {
                        "index": 1,
                        "interface_type": "MA",
                        "surface_id": "SURFACE_1",
                        "face_kind": "interface",
                        "owner_semantic_ids": ("OWNER_1",),
                        "net_id": "GROUND",
                        "equipotential_id": "GROUND",
                        "source_provenance": {"authority": "generic-test"},
                        "loss_tangent": 0.0,
                    },
                ),
            )

        self.assertIsNotNone(snapshots)
        snapshot = snapshots[0]
        self.assertEqual(snapshot.records[0].participation, 0.25)
        self.assertTrue(math.isinf(snapshot.records[0].quality_factor))
        self.assertEqual(snapshot.loss_status, "unavailable_nonfinite")
        self.assertIsNone(snapshot.quality_factor_total)
        self.assertIsNone(snapshot.t1_seconds)


if __name__ == "__main__":
    unittest.main()
