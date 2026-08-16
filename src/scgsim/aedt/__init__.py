"""AEDT-driven workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: all public entrypoints are fail-closed and reject incomplete
state without silently changing behavior.
"""

from .handoff import HandoffPlan, prepare_handoff
from .resolve import ResolvedRun, resolve_results
from .spec import (
    AedtSpec,
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
    Q3dNetSpec,
    Q3dRunControl,
    Q3dSpec,
    TerminalPort,
    parse_aedt_spec,
)

__all__ = [
    "AedtSpec",
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
    "Q3dNetSpec",
    "Q3dRunControl",
    "Q3dSpec",
    "ResolvedRun",
    "TerminalPort",
    "parse_aedt_spec",
    "prepare_handoff",
    "resolve_results",
]
