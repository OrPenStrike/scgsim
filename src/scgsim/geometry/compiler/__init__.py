"""Direct exports of canonical geometry definitions; no compatibility wrappers."""

from scgsim.geometry.compiler.builder import SemanticGeometryBuilder
from scgsim.geometry.compiler.dispatch import build_route_construction_plan
from scgsim.geometry.compiler.validation import (
    validate_backend_tag_ledger,
    validate_curve_plan_coverage,
    validate_interface_surface_source_of_truth,
    validate_no_surface_overlap,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_deduplication,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_surface_use_counts,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)
from scgsim.geometry.construction.export import export_physical_group_records
from scgsim.geometry.models.common import (
    RUN_METADATA_DIR,
    SEMANTIC_GEOMETRY_METADATA_DIR,
    DimensionLiteral,
    GmshDimTag,
    InterfaceKindLiteral,
    SolverUseLiteral,
    SurfaceLoopRoleLiteral,
    SurfaceOrientationLiteral,
    TagSourceKindLiteral,
)
from scgsim.geometry.models.construction import (
    ConstructionBodyPlanRecord,
    ConstructionPlanRecord,
    CutHostOperationRecord,
    SurfacePartitionRecord,
)
from scgsim.geometry.models.tags import (
    BackendEntityTagRecord,
    FinalPhysicalGroupRecord,
    TagPlanRecord,
)
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.interfaces import (
    complete_interface_plan_from_surfaces,
    plan_conductor_contact_patches,
    plan_mm_contact_records,
    recognize_route_interfaces,
)
from scgsim.geometry.planning.surfaces import (
    plan_route_surfaces,
    plan_surface_contribution_patches,
    plan_surface_partitions,
)
from scgsim.geometry.planning.tags import plan_route_tags
from scgsim.geometry.planning.topology import plan_canonical_topology
from scgsim.geometry.planning.volumes import (
    plan_cut_host_operations,
    plan_route_construction_bodies,
    plan_route_volumes,
)
from scgsim.geometry.source.validation import validate_geometry_input

__all__ = [
    "BackendEntityTagRecord",
    "ConstructionBodyPlanRecord",
    "ConstructionPlanRecord",
    "CurvePlanRecord",
    "CurveRefRecord",
    "CutHostOperationRecord",
    "DimensionLiteral",
    "FinalPhysicalGroupRecord",
    "GmshDimTag",
    "InnerPecVoidShellRecord",
    "InterfaceKindLiteral",
    "InterfacePlanRecord",
    "MMContactRecord",
    "PointPlanRecord",
    "RUN_METADATA_DIR",
    "SEMANTIC_GEOMETRY_METADATA_DIR",
    "SemanticGeometryBuilder",
    "SolverUseLiteral",
    "SurfaceLoopRecord",
    "SurfaceLoopRoleLiteral",
    "SurfaceOrientationLiteral",
    "SurfacePartitionRecord",
    "SurfacePlanRecord",
    "SurfaceRefRecord",
    "TagPlanRecord",
    "TagSourceKindLiteral",
    "VolumePlanRecord",
    "build_route_construction_plan",
    "complete_interface_plan_from_surfaces",
    "export_physical_group_records",
    "plan_canonical_topology",
    "plan_conductor_contact_patches",
    "plan_cut_host_operations",
    "plan_mm_contact_records",
    "plan_route_construction_bodies",
    "plan_route_surfaces",
    "plan_route_tags",
    "plan_route_volumes",
    "plan_surface_contribution_patches",
    "plan_surface_partitions",
    "recognize_route_interfaces",
    "validate_backend_tag_ledger",
    "validate_curve_plan_coverage",
    "validate_geometry_input",
    "validate_interface_surface_source_of_truth",
    "validate_no_surface_overlap",
    "validate_route_operation_coverage",
    "validate_route_volume_surface_refs",
    "validate_selected_route",
    "validate_surface_deduplication",
    "validate_surface_partition_coverage",
    "validate_surface_sheet_interface_coverage",
    "validate_surface_use_counts",
    "validate_tag_plan_coverage",
    "validate_volume_surface_closure",
]
