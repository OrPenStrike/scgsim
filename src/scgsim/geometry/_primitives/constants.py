"""Single existing topology precision/metadata vocabulary. Values and comparison semantics are preserved during relocation."""

from __future__ import annotations

_GEOMETRY_REF_METADATA_KEYS = (
    "plane",
    "contact_plane",
    "footprint",
    "outer_loop",
    "hole_loops",
    "loop_geometry_ref",
    "z_um",
    "z_min_um",
    "z_max_um",
    "thickness_um",
)


_INTERFACE_KIND_ORDER = ("MM", "SS", "AA", "MS", "MA", "SA")


_TOPOLOGY_EPS_UM = 1e-9
