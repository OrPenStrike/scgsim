"""Streaming custody-preserving reader for ASCII Gmsh MSH 2.2 meshes."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, Mapping

import numpy as np


_PHYSICAL_NAME = re.compile(r'^\s*(\d+)\s+(\d+)\s+"(.*)"\s*$')
# Public MSH element codes, dimensions, and arities from the Gmsh MSH format
# reference. Unknown future codes remain unknown instead of acquiring a guessed
# dimension. https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
_ELEMENT_TYPES: Mapping[int, tuple[int, int]] = MappingProxyType(
    {
        1: (1, 2),
        2: (2, 3),
        3: (2, 4),
        4: (3, 4),
        5: (3, 8),
        6: (3, 6),
        7: (3, 5),
        8: (1, 3),
        9: (2, 6),
        10: (2, 9),
        11: (3, 10),
        12: (3, 27),
        13: (3, 18),
        14: (3, 14),
        15: (0, 1),
        16: (2, 8),
        17: (3, 20),
        18: (3, 15),
        19: (3, 13),
        20: (2, 9),
        21: (2, 10),
        22: (2, 12),
        23: (2, 15),
        24: (2, 15),
        25: (2, 21),
        26: (1, 4),
        27: (1, 5),
        28: (1, 6),
        29: (3, 20),
        30: (3, 35),
        31: (3, 56),
        92: (3, 64),
        93: (3, 125),
    }
)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class MeshData:
    """One detached, compact snapshot of a parsed MSH 2.2 file."""

    source_path: Path
    sha256: str
    bytes_read: int
    file_identity: FileIdentity
    node_ids: np.ndarray
    coordinates: np.ndarray
    element_ids: np.ndarray
    connectivity: np.ndarray
    node_indices: np.ndarray
    raw_tags: np.ndarray
    tag_offsets: np.ndarray
    physical_tags: np.ndarray
    elementary_tags: np.ndarray
    physical_names: Mapping[tuple[int, int], str]
    element_counts: Mapping[str, int]
    optional_sections: Mapping[str, int]


class _Lines:
    def __init__(self, handle: BinaryIO) -> None:
        self.handle = handle
        self.hasher = hashlib.sha256()
        self.bytes_read = 0
        self.line_number = 0

    def read(self, *, required: bool = False) -> bytes:
        line = self.handle.readline()
        if line:
            self.hasher.update(line)
            self.bytes_read += len(line)
            self.line_number += 1
        elif required:
            raise ValueError(f"truncated MSH file at line {self.line_number + 1}")
        return line

    def text(self, *, required: bool = False) -> str:
        raw = self.read(required=required)
        try:
            return raw.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"MSH 2.2 ASCII input contains non-ASCII bytes at line {self.line_number}"
            ) from error


def _identity(stat: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    value.flags.writeable = False
    return value


def _count(lines: _Lines, section: str) -> int:
    raw = lines.text(required=True)
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{section} count must be an integer") from error
    if value < 0:
        raise ValueError(f"{section} count must be non-negative")
    return value


def _expect(lines: _Lines, expected: str) -> None:
    actual = lines.text(required=True)
    if actual != expected:
        raise ValueError(f"expected {expected!r}, found {actual!r}")


def read_msh22(mesh_path: str | Path) -> MeshData:
    """Read one ASCII MSH 2.2 file sequentially and retain only compact arrays."""

    path = Path(mesh_path).expanduser().resolve()
    before = _identity(path.stat())
    node_ids: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    element_ids_seen: set[int] = set()
    tet_ids: list[int] = []
    tet_connectivity: list[tuple[int, int, int, int]] = []
    flat_tags: list[int] = []
    tag_offsets: list[tuple[int, int]] = []
    physical_tags: list[int] = []
    elementary_tags: list[int] = []
    physical_names: dict[tuple[int, int], str] = {}
    optional_sections: dict[str, int] = {}
    counts = {
        "all": 0,
        "tetrahedra": 0,
        "other_known_3d": 0,
        "ordinary_2d": 0,
        "known_other": 0,
        "unknown": 0,
    }
    seen: set[str] = set()

    with path.open("rb") as handle:
        lines = _Lines(handle)
        while True:
            section = lines.text()
            if section == "":
                if handle.tell() == before.size:
                    break
                continue
            if not section.startswith("$") or section.startswith("$End"):
                raise ValueError(
                    f"expected an MSH section delimiter at line {lines.line_number}"
                )
            name = section[1:]
            if not seen and name != "MeshFormat":
                raise ValueError("$MeshFormat must be the first MSH section")
            if name == "MeshFormat":
                if name in seen:
                    raise ValueError("duplicate $MeshFormat section")
                seen.add(name)
                descriptor = lines.text(required=True).split()
                if descriptor != ["2.2", "0", "8"]:
                    if len(descriptor) >= 2 and descriptor[1] == "1":
                        raise ValueError("binary MSH files are unsupported; expected ASCII MSH 2.2")
                    raise ValueError("unsupported mesh format; expected ASCII MSH 2.2 with size 8")
                _expect(lines, "$EndMeshFormat")
                continue
            if name == "PhysicalNames":
                if name in seen:
                    raise ValueError("duplicate $PhysicalNames section")
                seen.add(name)
                for _ in range(_count(lines, "PhysicalNames")):
                    raw = lines.text(required=True)
                    match = _PHYSICAL_NAME.fullmatch(raw)
                    if match is None:
                        raise ValueError(f"malformed PhysicalNames row: {raw!r}")
                    key = (int(match.group(1)), int(match.group(2)))
                    if key in physical_names:
                        raise ValueError(f"duplicate physical name for dimension/tag {key}")
                    physical_names[key] = match.group(3)
                _expect(lines, "$EndPhysicalNames")
                continue
            if name == "Nodes":
                if name in seen:
                    raise ValueError("duplicate $Nodes section")
                seen.add(name)
                identifiers: set[int] = set()
                for _ in range(_count(lines, "Nodes")):
                    fields = lines.text(required=True).split()
                    if len(fields) != 4:
                        raise ValueError("each MSH 2.2 node row must have four fields")
                    try:
                        node_id = int(fields[0])
                        point = tuple(float(value) for value in fields[1:])
                    except ValueError as error:
                        raise ValueError("malformed MSH 2.2 node row") from error
                    if node_id in identifiers:
                        raise ValueError(f"duplicate node ID {node_id}")
                    identifiers.add(node_id)
                    node_ids.append(node_id)
                    coordinates.append(point)  # type: ignore[arg-type]
                _expect(lines, "$EndNodes")
                continue
            if name == "Elements":
                if name in seen:
                    raise ValueError("duplicate $Elements section")
                if "Nodes" not in seen:
                    raise ValueError("$Nodes must precede $Elements")
                seen.add(name)
                for _ in range(_count(lines, "Elements")):
                    fields = lines.text(required=True).split()
                    if len(fields) < 3:
                        raise ValueError("malformed MSH 2.2 element row")
                    try:
                        element_id, element_type, tag_count = map(int, fields[:3])
                        values = [int(value) for value in fields[3:]]
                    except ValueError as error:
                        raise ValueError("malformed MSH 2.2 element row") from error
                    if element_id in element_ids_seen:
                        raise ValueError(f"duplicate element ID {element_id}")
                    element_ids_seen.add(element_id)
                    if tag_count < 0 or len(values) < tag_count:
                        raise ValueError(f"invalid tag count for element {element_id}")
                    tags = values[:tag_count]
                    connectivity = values[tag_count:]
                    counts["all"] += 1
                    known = _ELEMENT_TYPES.get(element_type)
                    if known is not None and len(connectivity) != known[1]:
                        raise ValueError(
                            f"Gmsh element type {element_type} element {element_id} "
                            f"must reference {known[1]} nodes"
                        )
                    if element_type == 4:
                        start = len(flat_tags)
                        flat_tags.extend(tags)
                        tag_offsets.append((start, len(tags)))
                        tet_ids.append(element_id)
                        tet_connectivity.append(tuple(connectivity))  # type: ignore[arg-type]
                        physical_tags.append(tags[0] if tags and tags[0] != 0 else 0)
                        elementary_tags.append(tags[1] if len(tags) > 1 else 0)
                        counts["tetrahedra"] += 1
                    elif known is not None and known[0] == 3:
                        counts["other_known_3d"] += 1
                    elif known is not None and known[0] == 2:
                        counts["ordinary_2d"] += 1
                    elif known is not None:
                        counts["known_other"] += 1
                    else:
                        counts["unknown"] += 1
                _expect(lines, "$EndElements")
                continue

            end = f"$End{name}"
            rows = 0
            while True:
                raw = lines.text(required=True)
                if raw == end:
                    break
                if raw.startswith("$End") or raw.startswith("$"):
                    raise ValueError(f"invalid delimiter inside optional ${name} section")
                rows += 1
            optional_sections[name] = optional_sections.get(name, 0) + rows

        digest = lines.hasher.hexdigest()
        bytes_read = lines.bytes_read

    after = _identity(path.stat())
    if after != before:
        raise RuntimeError("mesh file identity changed while it was being read")
    if bytes_read != before.size:
        raise RuntimeError("mesh file byte count changed while it was being read")
    missing = {"MeshFormat", "Nodes", "Elements"} - seen
    if missing:
        raise ValueError(f"missing required MSH sections: {', '.join(sorted(missing))}")

    ids_array = np.asarray(node_ids, dtype=np.int64)
    coords_array = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
    element_array = np.asarray(tet_ids, dtype=np.int64)
    connectivity_array = np.asarray(tet_connectivity, dtype=np.int64).reshape((-1, 4))
    index_by_id = {int(node_id): index for index, node_id in enumerate(node_ids)}
    node_index_array = np.full(connectivity_array.shape, -1, dtype=np.int64)
    for row in range(connectivity_array.shape[0]):
        for column in range(4):
            node_index_array[row, column] = index_by_id.get(
                int(connectivity_array[row, column]), -1
            )

    return MeshData(
        source_path=path,
        sha256=digest,
        bytes_read=bytes_read,
        file_identity=before,
        node_ids=_readonly(ids_array),
        coordinates=_readonly(coords_array),
        element_ids=_readonly(element_array),
        connectivity=_readonly(connectivity_array),
        node_indices=_readonly(node_index_array),
        raw_tags=_readonly(np.asarray(flat_tags, dtype=np.int64)),
        tag_offsets=_readonly(np.asarray(tag_offsets, dtype=np.int64).reshape((-1, 2))),
        physical_tags=_readonly(np.asarray(physical_tags, dtype=np.int64)),
        elementary_tags=_readonly(np.asarray(elementary_tags, dtype=np.int64)),
        physical_names=MappingProxyType(dict(physical_names)),
        element_counts=MappingProxyType(dict(counts)),
        optional_sections=MappingProxyType(dict(optional_sections)),
    )
