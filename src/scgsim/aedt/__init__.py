"""AEDT request, execution, analysis, and result APIs."""

from .eigenmode import EigenmodeSim
from .epr.analysis import combine_epr_mode, combine_surface_epr_mode, reanalyze_epr
from .epr.geometry import planar_junction_from_port
from .preparation.planar import prepare_planar_geometry_input
from scgsim.aedt.specs.epr import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    ExpressionCacheConvergenceSet,
    NativeExpressionDefinition,
    NormalizedSurfaceEprTotal,
    PlanarJunction,
    SurfaceEprSpec,
)
from scgsim.aedt.epr.models import EprResult
from scgsim.aedt.specs.epr import SavedSolution
from scgsim.aedt.specs.common import (
    LumpedRlc,
)
from scgsim.aedt.specs.geometry import (
    PreparedPlanarGeometry,
)
from .epr.workflow import analyze_epr
from .preparation.geometry import (
    prepare_hfss_eigenmode_from_geometry,
    prepare_hfss_driven_from_geometry,
    prepare_q3d_from_geometry,
)
from .preparation.handoff import HandoffPlan, prepare_handoff
from .presentation.epr import plot_epr_result, show_epr
from .results.epr import resolve_epr_result, resolve_saved_solution
from .results.resolve import ResolvedRun, resolve_results
from .results.prepared import PreparedModel, resolve_prepared_model
from .specs.common import (
    AedtResources,
    HfssDrivenMode,
    LayerImport,
    MatrixRunControl,
    ModalPort,
    ObjectBinding,
    PdkMaterial,
    TerminalPort,
    LumpedTerminalPort,
)
from .specs.hfss import (
    EigenmodeRunControl,
    FrequencySweepSpec,
    HfssDrivenSpec,
    HfssDrivenGeometrySpec,
    HfssRunControl,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    HfssSpec,
    LengthMeshSpec,
)
from .specs.parse import AedtSpec, parse_aedt_spec
from .specs.q2d import Q2dConductorSpec, Q2dRectangleSpec, Q2dSpec
from .specs.q3d import Q3dBodySpec, Q3dNetSpec, Q3dSpec, Q3dTerminalPair

__all__ = [
    "AedtResources",
    "AedtSpec",
    "EigenmodeRunControl",
    "EigenmodeSim",
    "EprAnalysisRequest",
    "EprResult",
    "ExpressionCacheConvergence",
    "ExpressionCacheConvergenceSet",
    "FrequencySweepSpec",
    "HandoffPlan",
    "HfssDrivenMode",
    "HfssDrivenSpec",
    "HfssDrivenGeometrySpec",
    "LumpedSupport",
    "LumpedRlc",
    "LumpedTerminalPort",
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
    "PreparedModel",
    "Q2dConductorSpec",
    "Q2dRectangleSpec",
    "Q2dSpec",
    "Q3dBodySpec",
    "Q3dNetSpec",
    "Q3dSpec",
    "Q3dTerminalPair",
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
    "prepare_hfss_driven_from_geometry",
    "prepare_planar_geometry_input",
    "prepare_q3d_from_geometry",
    "reanalyze_epr",
    "resolve_epr_result",
    "resolve_results",
    "resolve_prepared_model",
    "resolve_saved_solution",
    "show_epr",
]

from .specs.modeling import PhysicalLayerSpec

from scgsim.geometry.models.lumped import LumpedSupport

__all__ += [
    "NativeBatchPackage", "prepare_native_batch", "run_native_batch", "collect_native_batch",
]

def __getattr__(name: str):
    # Keep the standalone batch CLI unimported until a facade export is used.
    if name in {
        "NativeBatchPackage", "prepare_native_batch",
        "run_native_batch", "collect_native_batch",
    }:
        from importlib import import_module

        value = getattr(import_module(".batch", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
