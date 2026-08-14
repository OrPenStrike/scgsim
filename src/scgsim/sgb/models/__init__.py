"""Stage-oriented model package for Semantic Geometry Builder.

The public import path stays `scgsim.sgb.models`. Internally the
records are grouped by pipeline handoff so future stages can grow without
turning one file into another catch-all.
"""

from scgsim.sgb.models.common import (
    DEFAULT_INTERFACE_SOLVER_USE,
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    ROUTE_ALLOWED_REPRESENTATIONS,
    RUN_METADATA_DIR,
    SEMANTIC_GEOMETRY_METADATA_DIR,
    ConductorPartRoleLiteral,
    ConductorRepresentationLiteral,
    Coordinate,
    CurveKindLiteral,
    CurveOrientationLiteral,
    DimensionLiteral,
    GmshDimTag,
    InterfaceKindLiteral,
    PathInput,
    PolygonRing,
    RouteLiteral,
    SolverUseLiteral,
    SurfaceLoopRoleLiteral,
    SurfaceOrientationLiteral,
    TagSourceKindLiteral,
    Vector3D,
)
from scgsim.sgb.models.construction import (
    ConstructionBodyPlanRecord,
    ConstructionPlanRecord,
    CutHostOperationRecord,
    RouteABConstructionPlanRecord,
    SurfacePartitionRecord,
)
from scgsim.sgb.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.sgb.models.regions import (
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
)
from scgsim.sgb.models.tags import (
    BackendEntityTagRecord,
    FinalPhysicalGroupRecord,
    StructuredFinalPhysicalGroupRecord,
    TagPlanRecord,
)
from scgsim.sgb.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    RouteABVolumePlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)

__all__ = [
    "DEFAULT_INTERFACE_SOLVER_USE",
    "HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES",
    "ROUTE_ALLOWED_REPRESENTATIONS",
    "RUN_METADATA_DIR",
    "SEMANTIC_GEOMETRY_METADATA_DIR",
    "BackendEntityTagRecord",
    "ConductorPartRoleLiteral",
    "ConductorRepresentationLiteral",
    "ConstructionBodyPlanRecord",
    "ConstructionPlanRecord",
    "Coordinate",
    "CurveKindLiteral",
    "CurveOrientationLiteral",
    "CurvePlanRecord",
    "CurveRefRecord",
    "CutHostOperationRecord",
    "DimensionLiteral",
    "FinalPhysicalGroupRecord",
    "GeometryBuildInput",
    "GmshDimTag",
    "InnerPecVoidShellRecord",
    "InterfaceKindLiteral",
    "InterfacePlanRecord",
    "LayoutPolygonSpec",
    "MMContactRecord",
    "PathInput",
    "PointPlanRecord",
    "PolygonRing",
    "PortSheetOverlapRecord",
    "PortSheetRegionRecord",
    "RouteABConstructionPlanRecord",
    "RouteABVolumePlanRecord",
    "RouteLiteral",
    "SemanticEntitySpec",
    "SolverUseLiteral",
    "StructuredFinalPhysicalGroupRecord",
    "SurfaceLoopRecord",
    "SurfaceLoopRoleLiteral",
    "SurfaceOrientationLiteral",
    "SurfacePartitionRecord",
    "SurfacePlanRecord",
    "SurfaceRefRecord",
    "TagPlanRecord",
    "TagSourceKindLiteral",
    "Vector3D",
    "VolumePlanRecord",
]
