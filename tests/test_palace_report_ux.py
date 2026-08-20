"""Accepted Palace report-family and notebook display contract."""

from __future__ import annotations

import inspect
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
    PalaceReturnedReceipt,
    PalaceTrustReport,
    ParsedTable,
    PhysicsQuantitiesReport,
    ResolvedPalaceResult,
    inspect_run_trustworthiness,
)
from scgsim.palace.report import (
    AmrPassSnapshot,
    SurfaceEprRecord,
    SurfaceEprSeriesSnapshot,
    _read_surface_epr,
)


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


def _trust_report() -> PalaceTrustReport:
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
        for index, source in enumerate(("iteration01", "final"))
    )
    return PalaceTrustReport(
        run_dir=Path("run"),
        problem="Eigenmode",
        route="A",
        profile="ltlab-slurm",
        completeness="complete",
        latest_source="final",
        identity={},
        passes=passes,
        amr_tolerance=0.02,
        amr_max_passes=1,
        durations={},
        cost={},
        provenance={},
    )


def _resolved_result() -> ResolvedPalaceResult:
    tables = {
        "eig": ParsedTable(
            name="eig",
            path=Path("eig.csv"),
            headers=("m", "Re{f} (GHz)"),
            rows=({"m": 1, "Re{f} (GHz)": 5.0},),
        )
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
        trust = _trust_report()
        benchmark = {"cost": {}, "performance": {}, "resources": {}}
        with (
            patch(
                "scgsim.palace.report._show_run_trustworthiness",
                return_value=trust,
            ) as show_trust,
            patch(
                "scgsim.palace.report._show_simulation_benchmark",
                return_value=benchmark,
            ) as show_benchmark,
            _captured_notebook_display() as displayed,
        ):
            returned = result.show_all_results(theme="dark", ranking_limit=10)

        self.assertIsNone(returned)
        self.assertEqual(len(displayed), 3)
        self.assertIs(displayed[0], trust)
        self.assertIs(displayed[1], benchmark)
        physics = displayed[2]
        self.assertIsInstance(physics, PhysicsQuantitiesReport)
        self.assertIs(physics.trust, trust)
        self.assertEqual(physics.ranking_limit, 10)
        show_trust.assert_called_once_with(result, theme="dark")
        show_benchmark.assert_called_once_with(result)
        self.assertIs(result.tables, original_tables)
        self.assertIn("eig", result.tables)

        parameters = inspect.signature(result.show_all_results).parameters
        self.assertEqual(tuple(parameters), ("theme", "ranking_limit"))
        self.assertFalse(hasattr(result, "show_surface_epr_physics"))
        self.assertFalse(hasattr(palace, "NativeTabularSummary"))

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

    def test_header_only_parent_keeps_readable_eigenmode_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metadata = root / "metadata"
            results = root / "results" / "palace"
            iteration = results / "iteration01"
            metadata.mkdir()
            iteration.mkdir(parents=True)
            (metadata / "palace_handoff_metadata.json").write_text(
                '{"problem":"Eigenmode","route":"A","profile":"ltlab-slurm"}\n',
                encoding="utf-8",
            )
            (iteration / "eig.csv").write_text(
                "m,Re{f} (GHz)\n1,5.0\n",
                encoding="utf-8",
            )
            (results / "eig.csv").write_text(
                "m,Re{f} (GHz)\n",
                encoding="utf-8",
            )

            trust = inspect_run_trustworthiness(root)

        self.assertEqual(trust.completeness, "partial")
        self.assertEqual(trust.latest_source, "iteration01")
        self.assertEqual(len(trust.passes), 1)
        self.assertEqual(trust.passes[0].frequencies_ghz, (5.0,))

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
