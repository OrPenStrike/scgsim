"""Accepted native eig.csv Q=+inf behavior and single-snapshot display."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output

from scgsim.palace import inspect_run_trustworthiness
from scgsim.palace.results.resolve import _read_csv_table, _validate_table


_EIG_HEADER = (
    "m,Re{f} (GHz),Im{f} (GHz),Q,Error (Bkwd.),Error (Abs.)\n"
)
_EIG_CONFIG = {"Solver": {"Eigenmode": {"N": 1}}}
_FINITE_INDEX_COUNTS: dict[str, int] = {}


def _validate_eig(tmp_path: Path, row: str) -> dict[str, object]:
    path = tmp_path / "eig.csv"
    path.write_text(_EIG_HEADER + row + "\n", encoding="utf-8")
    table = _read_csv_table(path)
    _validate_table("Eigenmode", table, _FINITE_INDEX_COUNTS, _EIG_CONFIG)
    return table.rows[0]


class NativeEigenmodeQInfinityTests(unittest.TestCase):
    def test_positive_infinity_is_retained_and_finite_q_remains_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            infinite = _validate_eig(root, "1,5.0,0.0,+inf,0.0,0.0")
            self.assertEqual(infinite["Q"], math.inf)
            self.assertIsInstance(infinite["Q"], float)

            finite = _validate_eig(root, "1,5.0,0.0,125.0,0.0,0.0")
            self.assertEqual(finite["Q"], 125.0)

    def test_nan_negative_infinity_and_other_column_infinity_still_fail(self) -> None:
        rejected_rows = (
            ("Q NaN", "1,5.0,0.0,NaN,0.0,0.0"),
            ("Q negative infinity", "1,5.0,0.0,-inf,0.0,0.0"),
            ("frequency positive infinity", "1,+inf,0.0,10.0,0.0,0.0"),
            ("error NaN", "1,5.0,0.0,10.0,0.0,NaN"),
            ("error negative infinity", "1,5.0,0.0,10.0,-inf,0.0"),
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for label, row in rejected_rows:
                with self.subTest(value=label):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        _validate_eig(root, row)

    def test_single_snapshot_rich_display_emits_status_and_infinite_q_notice(self) -> None:
        """Capture real IPython display for a synthetic, unsealed partial report."""

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metadata = root / "metadata"
            results = root / "results" / "palace"
            metadata.mkdir(parents=True)
            results.mkdir(parents=True)
            (metadata / "palace_handoff_metadata.json").write_text(
                json.dumps(
                    {
                        "problem": "Eigenmode",
                        "route": "A",
                        "profile": "synthetic-stabilization-test-only",
                        "route_a_thin_film": None,
                    }
                ),
                encoding="utf-8",
            )
            (results / "eig.csv").write_text(
                _EIG_HEADER + "1,5.0,0.0,+inf,0.0,0.0\n",
                encoding="utf-8",
            )

            # Build through the public inspector; this is not a mocked report.
            report = inspect_run_trustworthiness(root)
            self.assertEqual(report.completeness, "partial")
            self.assertEqual(report.selection.integrity, "observed_unsealed")
            self.assertEqual(len(report.passes), 1)
            self.assertEqual(report.passes[0].eig_columns["Q"], (math.inf,))

            # Use the installed IPython publisher and formatter. Restore its
            # MIME settings so this focused test cannot affect later tests.
            shell = InteractiveShell.instance()
            prior_active_types = list(shell.display_formatter.active_types)
            shell.display_formatter.active_types = ["text/html", "text/plain"]
            try:
                with capture_output() as captured:
                    display(report)
            finally:
                shell.display_formatter.active_types = prior_active_types

            html_payloads = [
                output.data.get("text/html")
                for output in captured.outputs
                if isinstance(output.data.get("text/html"), str)
            ]
            emitted_html = "\n".join(html_payloads)
            self.assertIn("AMR stopped after the initial solve", emitted_html)
            self.assertIn(
                "A trend plot is omitted because a single pass cannot show adaptation.",
                emitted_html,
            )
            self.assertIn(
                "Native-reported Q includes +inf; a finite-Q convergence view is unavailable.",
                emitted_html,
            )
            self.assertNotIn("Q vs AMR pass", emitted_html)


if __name__ == "__main__":
    unittest.main()
