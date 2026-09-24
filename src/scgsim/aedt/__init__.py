"""AEDT-driven workflow contracts and manual handoff/runtime utilities.

Ownership: SCGSim Development.
Failure intent: public entrypoints preserve explicit partial adaptive-history
records, but reject corrupt or identity-inconsistent state without fallback.
"""

from ._epr_geometry import prepare_planar_geometry_input
from ._epr_models import (
    EprAnalysisRequest,
    EprResult,
    PlanarJunction,
    PreparedPlanarGeometry,
    SavedSolution,
    SurfaceEprSpec,
)
from ._epr_results import (
    combine_epr_mode,
    plot_epr_result,
    resolve_epr_result,
    resolve_saved_solution,
)
from .handoff import (
    analyze_epr,
    HandoffPlan,
    prepare_handoff,
    prepare_hfss_eigenmode_from_geometry,
)
from .resolve import ResolvedRun, resolve_results
from .spec import (
    AedtSpec,
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenMode,
    HfssDrivenSpec,
    HfssEprSpec,
    HfssEprAnalysisSpec,
    HfssEigenmodeSpec,
    HfssRunControl,
    HfssSpec,
    LayerImport,
    LengthMeshSpec,
    MatrixRunControl,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    Q2dConductorSpec,
    Q2dRectangleSpec,
    Q2dSpec,
    Q3dNetSpec,
    Q3dSpec,
    TerminalPort,
    parse_aedt_spec,
)

__all__ = [
    "AedtSpec",
    "EigenmodeRunControl",
    "FrequencySweepSpec",
    "EprResult",
    "EprAnalysisRequest",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssEprSpec",
    "HfssEprAnalysisSpec",
    "HfssEigenmodeSpec",
    "HfssRunControl",
    "HfssSpec",
    "LayerImport",
    "LengthMeshSpec",
    "MatrixRunControl",
    "ModalPort",
    "ObjectBinding",
    "PdkMaterial",
    "PlanarJunction",
    "PreparedPlanarGeometry",
    "Q2dConductorSpec",
    "Q2dRectangleSpec",
    "Q2dSpec",
    "Q3dNetSpec",
    "Q3dSpec",
    "ResolvedRun",
    "SavedSolution",
    "SurfaceEprSpec",
    "TerminalPort",
    "analyze_epr",
    "combine_epr_mode",
    "parse_aedt_spec",
    "plot_epr_result",
    "prepare_handoff",
    "prepare_hfss_eigenmode_from_geometry",
    "prepare_planar_geometry_input",
    "resolve_epr_result",
    "resolve_results",
    "resolve_saved_solution",
]
