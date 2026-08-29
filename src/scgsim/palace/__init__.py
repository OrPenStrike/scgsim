"""Palace workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

from .eigenmode import EigenmodeSim
from .electrostatic import ElectrostaticSim
from .report import (
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PalaceTrustReport,
    PassCostRecord,
    PhysicsQuantitiesReport,
    SimulationBenchmarkReport,
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
    "inspect_run_trustworthiness",
    "resolve_palace_result",
]
