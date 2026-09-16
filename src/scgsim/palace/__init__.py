"""Palace workflow contracts, offline diagnostics, and handoff utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior. Mesh-quality warnings describe the
stored mesh and are not workflow failures or acceptance thresholds.
"""

from scgsim._mesh_quality import MeshQualityReport, check_mesh_quality

from .eigenmode import EigenmodeSim
from .electrostatic import ElectrostaticSim
from .report import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PalaceTrustReport,
    PassCostRecord,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
    inspect_run_trustworthiness,
)
from .resolve import (
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
]
