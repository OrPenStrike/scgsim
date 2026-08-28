"""Headless, solver-metadata-bound geometry previews."""

from ._aedt import inspect_aedt_geometry
from ._preview import GeometryPreview, GeometryPreviewArtifact, inspect_palace_geometry

__all__ = [
    "GeometryPreview",
    "GeometryPreviewArtifact",
    "inspect_aedt_geometry",
    "inspect_palace_geometry",
]
