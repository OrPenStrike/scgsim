"""Q2D request contract and payload codec."""

from __future__ import annotations


from collections.abc import Mapping

from dataclasses import dataclass

from typing import Any, Literal

from scgsim.aedt.specs.common import (
    LOCKED_PYAEDT,
    MatrixRunControl,
    OFFICIAL_PYAEDT_SOURCE_URL,
    PdkMaterial,
    Q2D_SCHEMA_VERSION,
    REQUIRED_AEDT_VERSION,
    _normalize_common_spec,
    _number,
    _padding_2d,
    _project_name_from_payload,
    _text,
)


@dataclass(frozen=True)
class Q2dRectangleSpec:
    """One explicit native Q2D cross-section rectangle."""

    name: str
    origin_um: tuple[float, float]
    size_um: tuple[float, float]
    material_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "rectangle.name"))
        if len(self.origin_um) != 2 or len(self.size_um) != 2:
            raise ValueError("Q2D rectangle origin_um and size_um require x,y pairs")
        origin = tuple(
            _number(value, "rectangle.origin_um") for value in self.origin_um
        )
        size = tuple(_number(value, "rectangle.size_um") for value in self.size_um)
        if any(value <= 0 for value in size):
            raise ValueError("Q2D rectangle size_um values must be > 0")
        object.__setattr__(self, "origin_um", origin)
        object.__setattr__(self, "size_um", size)
        object.__setattr__(self, "material_id", _text(self.material_id, "material_id"))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "origin_um": list(self.origin_um),
            "size_um": list(self.size_um),
            "material_id": self.material_id,
        }


@dataclass(frozen=True)
class Q2dConductorSpec:
    """One exact Q2D signal or the single reference-ground group."""

    name: str
    conductor_type: Literal["SignalLine", "ReferenceGround"]
    object_names: tuple[str, ...]
    thickness_um: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "conductor.name"))
        if self.conductor_type not in {"SignalLine", "ReferenceGround"}:
            raise ValueError("Q2D conductor_type must be SignalLine or ReferenceGround")
        objects = tuple(
            _text(value, "conductor.object_name") for value in self.object_names
        )
        if not objects or len(set(objects)) != len(objects):
            raise ValueError("Q2D conductor object_names must be nonempty and unique")
        thickness = _number(self.thickness_um, "conductor.thickness_um")
        if thickness <= 0:
            raise ValueError("Q2D conductor thickness_um must be > 0")
        object.__setattr__(self, "object_names", objects)
        object.__setattr__(self, "thickness_um", thickness)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "conductor_type": self.conductor_type,
            "object_names": list(self.object_names),
            "thickness_um": self.thickness_um,
        }


@dataclass(frozen=True)
class Q2dSpec:
    """One explicit native Q2D cross-section matrix extraction."""

    project_name: str
    design_name: str
    materials: Mapping[str, PdkMaterial]
    vacuum_material_id: str
    rectangles: tuple[Q2dRectangleSpec, ...]
    conductors: tuple[Q2dConductorSpec, ...]
    run_control: MatrixRunControl
    region_padding_um: tuple[float, float, float, float]
    aedt_version: str = REQUIRED_AEDT_VERSION
    pyaedt_version: str = LOCKED_PYAEDT

    @property
    def mode(self) -> Literal["q2d"]:
        return "q2d"

    def __post_init__(self) -> None:
        materials = _normalize_common_spec(self)
        rectangles = tuple(self.rectangles)
        if (
            not rectangles
            or len({item.name for item in rectangles}) != len(rectangles)
            or any(item.material_id not in materials for item in rectangles)
        ):
            raise ValueError(
                "Q2D rectangles must be unique and use declared PDK materials"
            )
        if any(materials[item.material_id].kind == "vacuum" for item in rectangles):
            raise ValueError("Q2D vacuum is owned by the Region, not a rectangle")
        conductors = tuple(self.conductors)
        if (
            not conductors
            or len({item.name for item in conductors}) != len(conductors)
            or sum(item.conductor_type == "ReferenceGround" for item in conductors) != 1
            or not any(item.conductor_type == "SignalLine" for item in conductors)
        ):
            raise ValueError(
                "Q2D requires SignalLine conductors and one ReferenceGround"
            )
        owners = {
            object_name: conductor
            for conductor in conductors
            for object_name in conductor.object_names
        }
        if len(owners) != sum(len(item.object_names) for item in conductors):
            raise ValueError(
                "Q2D conductor rectangles must belong to exactly one group"
            )
        superconductors = {
            item.name
            for item in rectangles
            if materials[item.material_id].is_superconducting
        }
        if set(owners) != superconductors:
            raise ValueError(
                "Q2D conductor groups must cover every superconducting rectangle exactly"
            )
        object.__setattr__(self, "rectangles", rectangles)
        object.__setattr__(self, "conductors", conductors)
        object.__setattr__(
            self, "region_padding_um", _padding_2d(self.region_padding_um)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": Q2D_SCHEMA_VERSION,
            "mode": self.mode,
            "aedt": {"requested_version": self.aedt_version},
            "pyaedt": {
                "locked_version": self.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "project": {"name": self.project_name, "design": self.design_name},
            "materials": {
                material_id: item.to_payload()
                for material_id, item in self.materials.items()
            },
            "vacuum_material_id": self.vacuum_material_id,
            "rectangles": [item.to_payload() for item in self.rectangles],
            "conductors": [item.to_payload() for item in self.conductors],
            "run_control": self.run_control.to_payload(),
            "region_padding_um": list(self.region_padding_um),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Q2dSpec:
        if payload.get("schema_version") != Q2D_SCHEMA_VERSION:
            raise ValueError("unsupported Q2D schema")
        raw_materials = payload.get("materials")
        if not isinstance(raw_materials, dict):
            raise TypeError("materials must be a JSON object")
        materials = {
            material_id: PdkMaterial(**item)
            for material_id, item in raw_materials.items()
        }
        run = payload.get("run_control")
        if not isinstance(run, dict):
            raise TypeError("run_control must be a JSON object")
        return cls(
            project_name=_project_name_from_payload(
                payload.get("project", {}).get("name")
            ),
            design_name=_text(
                payload.get("project", {}).get("design"), "project.design"
            ),
            materials=materials,
            vacuum_material_id=_text(
                payload.get("vacuum_material_id"), "vacuum_material_id"
            ),
            rectangles=tuple(
                Q2dRectangleSpec(**item) for item in payload.get("rectangles", ())
            ),
            conductors=tuple(
                Q2dConductorSpec(**item) for item in payload.get("conductors", ())
            ),
            run_control=MatrixRunControl(**run),
            region_padding_um=tuple(payload.get("region_padding_um", ())),  # type: ignore[arg-type]
            aedt_version=_text(
                payload.get("aedt", {}).get("requested_version"),
                "aedt.requested_version",
            ),
            pyaedt_version=_text(
                payload.get("pyaedt", {}).get("locked_version"), "pyaedt.locked_version"
            ),
        )
