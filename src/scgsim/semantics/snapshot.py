"""Recursively immutable, detached and canonically hashed fact snapshots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

RevisionKind = Literal["declared_revision", "content_hash_fallback"]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"semantic facts require plain values, got {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def freeze_plain(value: Any) -> Any:
    """Detach and recursively freeze one plain-data value."""
    return _freeze(_plain(value))


def _detached(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detached(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticFactsSnapshot:
    source_revision: str
    source_revision_kind: RevisionKind
    geometry_revision: str
    geometry_revision_kind: RevisionKind
    projection_version: str
    canonical_sha256: str
    facts: Mapping[str, Any]

    def detached(self) -> dict[str, Any]:
        return {
            "source_revision": self.source_revision,
            "source_revision_kind": self.source_revision_kind,
            "geometry_revision": self.geometry_revision,
            "geometry_revision_kind": self.geometry_revision_kind,
            "projection_version": self.projection_version,
            "canonical_sha256": self.canonical_sha256,
            "facts": _detached(self.facts),
        }


def create_snapshot(
    facts: Mapping[str, Any],
    *,
    source_revision: str | None,
    geometry_revision: str | None,
    projection_version: str,
    source_revision_kind: RevisionKind | None = None,
    geometry_revision_kind: RevisionKind | None = None,
) -> SemanticFactsSnapshot:
    plain = _plain(facts)
    digest = canonical_sha256(plain)
    fallback = f"sha256:{digest}"
    return SemanticFactsSnapshot(
        source_revision=source_revision or fallback,
        source_revision_kind=(
            source_revision_kind
            or ("declared_revision" if source_revision else "content_hash_fallback")
        ),
        geometry_revision=geometry_revision or fallback,
        geometry_revision_kind=(
            geometry_revision_kind
            or ("declared_revision" if geometry_revision else "content_hash_fallback")
        ),
        projection_version=projection_version,
        canonical_sha256=digest,
        facts=freeze_plain(plain),
    )
