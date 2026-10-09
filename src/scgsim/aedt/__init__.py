"""AEDT request, execution, analysis, and result APIs."""

from .eigenmode import EigenmodeSim
from .epr.analysis import combine_epr_mode, combine_surface_epr_mode, reanalyze_epr
from .epr.geometry import planar_junction_from_port, prepare_planar_geometry_input
from .epr.models import (
    EprAnalysisRequest,
    EprResult,
    ExpressionCacheConvergence,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    PlanarJunction,
    PreparedPlanarGeometry,
    SavedSolution,
    SurfaceEprSpec,
)
from .epr.workflow import analyze_epr
from .preparation.geometry import (
    prepare_hfss_eigenmode_from_geometry,
    prepare_q3d_from_geometry,
)
from .preparation.handoff import HandoffPlan, prepare_handoff
from .presentation.epr import plot_epr_result, show_epr
from .results.epr import resolve_epr_result, resolve_saved_solution
from .results.resolve import ResolvedRun, resolve_results
from .specs.common import (
    AedtResources,
    HfssDrivenMode,
    LayerImport,
    MatrixRunControl,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
)
from .specs.hfss import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenSpec,
    HfssRunControl,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    HfssSpec,
    LengthMeshSpec,
)
from .specs.parse import AedtSpec, parse_aedt_spec
from .specs.q2d import Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec
from .specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec

__all__ = [
    "AedtResources",
    "AedtSpec",
    "EigenmodeRunControl",
    "EigenmodeSim",
    "EprAnalysisRequest",
    "EprResult",
    "ExpressionCacheConvergence",
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssEigenmodeSpec",
    "HfssEprAnalysisSpec",
    "HfssEprSpec",
    "HfssRunControl",
    "HfssSpec",
    "LayerImport",
    "LengthMeshSpec",
    "MatrixRunControl",
    "ModalPort",
    "NativeExpressionDefinition",
    "NormalizedSurfaceEprTotal",
    "ObjectBinding",
    "PdkMaterial",
    "PhysicalLayerSpec",
    "PlanarJunction",
    "PreparedPlanarGeometry",
    "Q2dConductorSpec",
    "Q2dRectangleSpec",
    "Q2dSpec",
    "Q3dBodySpec",
    "Q3dNetSpec",
    "Q3dSpec",
    "ResolvedRun",
    "SavedSolution",
    "SurfaceEprSpec",
    "TerminalPort",
    "analyze_epr",
    "combine_epr_mode",
    "combine_surface_epr_mode",
    "parse_aedt_spec",
    "planar_junction_from_port",
    "plot_epr_result",
    "prepare_handoff",
    "prepare_hfss_eigenmode_from_geometry",
    "prepare_planar_geometry_input",
    "prepare_q3d_from_geometry",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_results",
    "resolve_saved_solution",
    "show_epr",
]

from .specs.modeling import PhysicalLayerSpec
