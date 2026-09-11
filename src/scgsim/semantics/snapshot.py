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
        plain: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "semantic fact mapping keys must be strings, "
                    f"got {type(key).__name__}"
                )
            plain[key] = _plain(item)
        return plain
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
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_detached(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_sha256_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")


def _validate_revision(name: str, value: str, kind: RevisionKind) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if kind not in ("declared_revision", "content_hash_fallback"):
        raise ValueError(f"{name}_kind must be declared_revision or content_hash_fallback")
    if kind == "content_hash_fallback":
        prefix, separator, digest = value.partition(":")
        if (
            prefix != "sha256"
            or separator != ":"
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"{name} marked content_hash_fallback must be sha256:<64 lowercase hex>"
            )


@dataclass(frozen=True, slots=True)
class SemanticSnapshotReference:
    """Immutable identity of the exact facts used for one classification."""

    source_revision: str
    source_revision_kind: RevisionKind
    geometry_revision: str
    geometry_revision_kind: RevisionKind
    projection_version: str
    canonical_sha256: str

    def __post_init__(self) -> None:
        _validate_revision(
            "source_revision", self.source_revision, self.source_revision_kind
        )
        _validate_revision(
            "geometry_revision", self.geometry_revision, self.geometry_revision_kind
        )
        if not isinstance(self.projection_version, str) or not self.projection_version:
            raise ValueError("projection_version must be a non-empty string")
        _validate_sha256_digest("canonical_sha256", self.canonical_sha256)

    def detached(self) -> dict[str, str]:
        return {
            "source_revision": self.source_revision,
            "source_revision_kind": self.source_revision_kind,
            "geometry_revision": self.geometry_revision,
            "geometry_revision_kind": self.geometry_revision_kind,
            "projection_version": self.projection_version,
            "canonical_sha256": self.canonical_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticFactsSnapshot:
    source_revision: str
    source_revision_kind: RevisionKind
    geometry_revision: str
    geometry_revision_kind: RevisionKind
    projection_version: str
    canonical_sha256: str
    facts: Mapping[str, Any]

    def __post_init__(self) -> None:
        plain = _plain(self.facts)
        digest = canonical_sha256(plain)
        _validate_sha256_digest("canonical_sha256", self.canonical_sha256)
        if self.canonical_sha256 != digest:
            raise ValueError("canonical_sha256 does not match semantic facts")
        _validate_revision(
            "source_revision", self.source_revision, self.source_revision_kind
        )
        _validate_revision(
            "geometry_revision", self.geometry_revision, self.geometry_revision_kind
        )
        if not isinstance(self.projection_version, str) or not self.projection_version:
            raise ValueError("projection_version must be a non-empty string")
        object.__setattr__(self, "facts", freeze_plain(plain))

    def reference(self) -> SemanticSnapshotReference:
        return SemanticSnapshotReference(
            source_revision=self.source_revision,
            source_revision_kind=self.source_revision_kind,
            geometry_revision=self.geometry_revision,
            geometry_revision_kind=self.geometry_revision_kind,
            projection_version=self.projection_version,
            canonical_sha256=self.canonical_sha256,
        )

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
    resolved_source_revision, resolved_source_kind = _resolve_revision(
        "source_revision", source_revision, source_revision_kind, fallback
    )
    resolved_geometry_revision, resolved_geometry_kind = _resolve_revision(
        "geometry_revision", geometry_revision, geometry_revision_kind, fallback
    )
    return SemanticFactsSnapshot(
        source_revision=resolved_source_revision,
        source_revision_kind=resolved_source_kind,
        geometry_revision=resolved_geometry_revision,
        geometry_revision_kind=resolved_geometry_kind,
        projection_version=projection_version,
        canonical_sha256=digest,
        facts=freeze_plain(plain),
    )


def _resolve_revision(
    name: str,
    value: str | None,
    kind: RevisionKind | None,
    fallback: str,
) -> tuple[str, RevisionKind]:
    if value is None:
        if kind not in (None, "content_hash_fallback"):
            raise ValueError(
                f"{name}_kind must be content_hash_fallback when {name} is None"
            )
        return fallback, "content_hash_fallback"
    return value, kind or "declared_revision"
