"""Solver-independent, revision-bound semantic facts.

This internal package is standard-library only.  It classifies already-valid
plain facts and never imports SGB, Palace, AEDT, native backends, or callbacks.
"""

from .evidence import EvidenceResult, SemanticEvidenceFacade, SourcedPatch
from .interfaces import (
    conductor_solution_interface_kind,
    solution_interface_kind,
    solution_interface_owner_ids,
)
from .materials import MATERIAL_KINDS, is_supported_material_kind, is_vacuum_material_kind
from .route_a import (
    apply_thin_film_profile,
    derive_thin_film_facts,
    normalize_optional_profile,
)
from .snapshot import SemanticFactsSnapshot, canonical_sha256, create_snapshot

__all__ = [
    "EvidenceResult",
    "MATERIAL_KINDS",
    "SemanticEvidenceFacade",
    "SemanticFactsSnapshot",
    "SourcedPatch",
    "apply_thin_film_profile",
    "canonical_sha256",
    "conductor_solution_interface_kind",
    "create_snapshot",
    "derive_thin_film_facts",
    "is_supported_material_kind",
    "is_vacuum_material_kind",
    "normalize_optional_profile",
    "solution_interface_kind",
    "solution_interface_owner_ids",
]
