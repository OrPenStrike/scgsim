"""Source-facing semantic geometry API; input ownership remains with caller/components."""

from scgsim.geometry.models.common import (
    Coordinate,
    CurveKindLiteral,
    CurveOrientationLiteral,
    PathInput,
    PolygonRing,
    RouteLiteral,
    Vector3D,
)
from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
    SourceCurveSpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.source.adapter import (
    build_gds_stack_geometry_input,
    build_gdsfactory_geometry_input,
)
from scgsim.geometry.source.curves import bind_source_curves
from scgsim.geometry.source.plan import GeometryPlan, GeometryPlanSnapshot
from scgsim.geometry.source.stack import build_component_stack
from scgsim.geometry.source.summary import InputSummary, summarize_geometry_input
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack

__all__ = [
    "BoundaryCurveChainSpec",
    "Coordinate",
    "CurveKindLiteral",
    "CurveOrientationLiteral",
    "GdsBoundaryReconstructionSpec",
    "GeometryBuildInput",
    "GeometryPlan",
    "GeometryPlanSnapshot",
    "InputSummary",
    "LayoutPolygonSpec",
    "LumpedSupport",
    "PathInput",
    "PolygonRing",
    "PortSheetOverlapRecord",
    "PortSheetRegionRecord",
    "RouteLiteral",
    "SemanticEntitySpec",
    "SourceCurveSpec",
    "VacuumRegionSpec",
    "Vector3D",
    "apply_vacuum_region_to_stack",
    "bind_source_curves",
    "build_component_stack",
    "build_gds_stack_geometry_input",
    "build_gdsfactory_geometry_input",
    "summarize_geometry_input",
]
