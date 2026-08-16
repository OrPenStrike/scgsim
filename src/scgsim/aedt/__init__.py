"""AEDT-driven workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

from .handoff import HandoffPlan, prepare_handoff
from .resolve import ResolvedRun, resolve_results
from .run import main as run_main
from .spec import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenMode,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssRunControl,
    HfssSpec,
    LayerImport,
    LengthMeshSpec,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
    parse_hfss_spec,
)

__all__ = [
    "EigenmodeRunControl",
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssEigenmodeSpec",
    "HfssRunControl",
    "HfssSpec",
    "LayerImport",
    "LengthMeshSpec",
    "ModalPort",
    "ObjectBinding",
    "PdkMaterial",
    "ResolvedRun",
    "TerminalPort",
    "parse_hfss_spec",
    "prepare_handoff",
    "resolve_results",
    "run_main",
]
