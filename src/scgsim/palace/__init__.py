"""Palace workflow contracts, offline diagnostics, and handoff utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior. Mesh-quality warnings describe the
stored mesh and are not workflow failures or acceptance thresholds.
"""

from scgsim.palace.mesh.quality import MeshQualityReport, check_mesh_quality

from .preparation.eigenmode import EigenmodeSim
from .results.epr import (
    PalaceEprResult,
    epr_result,
    reanalyze_epr,
    resolve_epr_result,
    show_epr,
)
from .preparation.electrostatic import ElectrostaticSim
from .presentation.reports import (
    PalaceTrustReport,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
    inspect_run_trustworthiness,
)
from .results.report_data import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)
from .results.resolve import (
    PalaceCost,
    PalacePerformance,
    PalaceProvenance,
    PalaceReturnedReceipt,
    ParsedTable,
    ResolvedPalaceResult,
    resolve_palace_result,
)

__all__ = [
    "EigenmodeSim",
    "PalaceEprResult",
    "ElectrostaticSim",
    "MeshQualityReport",
    "PalaceCost",
    "PalaceFailureDiagnosis",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceResultSelection",
    "PalaceReturnedReceipt",
    "PalaceTrustReport",
    "ParsedTable",
    "PassCostRecord",
    "PhysicsQuantitiesReport",
    "ResolvedPalaceResult",
    "SimulationBenchmarkReport",
    "SurfaceMaskEprRecord",
    "SurfaceMaskEprSeriesSnapshot",
    "check_mesh_quality",
    "inspect_run_trustworthiness",
    "resolve_palace_result",
    "epr_result",
    "reanalyze_epr",
    "resolve_epr_result",
    "show_epr",
]
