"""Palace workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

from .eigenmode import EigenmodeSim
from .electrostatic import ElectrostaticSim
from .report import NativeTabularSummary
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
    "NativeTabularSummary",
    "PalaceCost",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceReturnedReceipt",
    "ParsedTable",
    "ResolvedPalaceResult",
    "resolve_palace_result",
]
