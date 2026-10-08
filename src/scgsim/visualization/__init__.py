"""Headless, solver-metadata-bound geometry previews."""

from .loaders.aedt import inspect_aedt_geometry
from .loaders.palace import inspect_palace_geometry
from .preview import GeometryPreview, GeometryPreviewArtifact

__all__ = [
    "GeometryPreview",
    "GeometryPreviewArtifact",
    "inspect_aedt_geometry",
    "inspect_palace_geometry",
]
