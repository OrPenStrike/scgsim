"""AEDT-driven workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

from .handoff import HandoffPlan, prepare_handoff
from .resolve import ResolvedRun, resolve_results
from .run import main as run_main
from .spec import (
    FrequencySweepSpec,
    HfssDrivenMode,
    HfssDrivenSpec,
    HfssRunControl,
    LayerImport,
    LengthMeshSpec,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
)

__all__ = [
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssRunControl",
    "LayerImport",
    "LengthMeshSpec",
    "ModalPort",
    "ObjectBinding",
    "PdkMaterial",
    "ResolvedRun",
    "TerminalPort",
    "prepare_handoff",
    "resolve_results",
    "run_main",
]
