"""Explicit physical-layer declarations and one source-to-effective AEDT Z map.

Layer identities and film/contact roles belong to the source author. This
module never classifies a layer from a Net, material name or geometric size.
The detached mapping is shared by conductor, domain, port and EPR preparation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

Modeling = Literal["solid", "thin_film"]
LayerModelKind = Literal["stack_film", "local_film", "retained_solid"]
ContactSide = Literal["lower", "upper"]


def _modeling(value: Any) -> Modeling:
    if value not in {"solid", "thin_film"}:
        raise ValueError("modeling must be explicitly 'solid' or 'thin_film'")
    return value


@dataclass(frozen=True)
class PhysicalLayerSpec:
    """One declared physical level in the source's common Z coordinate system.

    Multiple Entities, occurrences and polygons may bind this same identity.
    Its physical interval is removed once, not once per object. Local films
    declare their actual contact side and never compress the stack.
    source_thickness_um retains an independently authored physical fact;
    endpoint spans remain the coordinate-map authority.
    """

    physical_layer_id: str
    kind: LayerModelKind
    z_min_um: float
    z_max_um: float
    contact_side: ContactSide | None = None
    source_thickness_um: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.physical_layer_id, str) or not self.physical_layer_id:
            raise ValueError("physical_layer_id must be explicit nonempty text")
        if self.kind not in {"stack_film", "local_film", "retained_solid"}:
            raise ValueError("physical layer kind is invalid")
        low, high = float(self.z_min_um), float(self.z_max_um)
        if high <= low:
            raise ValueError("a physical layer requires a positive source interval")
        object.__setattr__(self, "z_min_um", low)
        object.__setattr__(self, "z_max_um", high)
        if self.source_thickness_um is not None:
            object.__setattr__(self, "source_thickness_um", float(self.source_thickness_um))
        if self.kind in {"stack_film", "local_film"}:
            if self.contact_side not in {"lower", "upper"}:
                raise ValueError("a film requires an explicit sheet contact side")
        elif self.contact_side is not None:
            raise ValueError("a retained solid has no sheet contact side")

    @property
    def physical_thickness_um(self) -> float:
        """Retain an authored thickness; endpoint-only inputs define their own span."""
        return (self.source_thickness_um if self.source_thickness_um is not None
                else self.z_max_um - self.z_min_um)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.source_thickness_um is None:
            payload.pop("source_thickness_um")
        return payload


@dataclass(frozen=True)
class EffectiveLayer:
    """Derived native representation; its immutable physical source is retained."""

    source: PhysicalLayerSpec
    representation: Literal["solid", "sheet"]
    z_min_um: float
    z_max_um: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "physical_layer_id": self.source.physical_layer_id,
            "representation": self.representation,
            "source_z_min_um": self.source.z_min_um,
            "source_z_max_um": self.source.z_max_um,
            "physical_thickness_um": self.source.physical_thickness_um,
            "effective_z_min_um": self.z_min_um,
            "effective_z_max_um": self.z_max_um,
            "contact_side": self.source.contact_side,
        }


class _ZMap:
    """The single preparation authority for declared stack-height removal."""

    def __init__(self, modeling: Modeling, layers: Sequence[PhysicalLayerSpec]):
        self.modeling = _modeling(modeling)
        self.layers: dict[str, PhysicalLayerSpec] = {}
        for layer in layers:
            if not isinstance(layer, PhysicalLayerSpec):
                raise TypeError("physical_layers must contain PhysicalLayerSpec records")
            previous = self.layers.setdefault(layer.physical_layer_id, layer)
            if previous != layer:
                raise ValueError("one physical layer identity has conflicting declarations")
        self.removed = tuple(sorted(
            (layer for layer in self.layers.values() if layer.kind == "stack_film"),
            key=lambda layer: layer.z_min_um,
        )) if modeling == "thin_film" else ()
        for lower, upper in zip(self.removed, self.removed[1:]):
            if lower.z_max_um > upper.z_min_um:
                raise ValueError("distinct stack-film declarations overlap in the source stack")

    def map_z(self, source_z_um: float) -> float:
        value = float(source_z_um)
        return value - sum(min(max(value - layer.z_min_um, 0.0),
                               layer.z_max_um - layer.z_min_um) for layer in self.removed)

    def layer(self, physical_layer_id: str) -> EffectiveLayer:
        source = self.layers[physical_layer_id]
        if self.modeling == "thin_film" and source.kind != "retained_solid":
            contact = source.z_min_um if source.contact_side == "lower" else source.z_max_um
            z = self.map_z(contact)
            return EffectiveLayer(source, "sheet", z, z)
        return EffectiveLayer(source, "solid", self.map_z(source.z_min_um),
                              self.map_z(source.z_max_um))

    def to_payload(self) -> dict[str, Any]:
        return {
            "modeling": self.modeling,
            "source_layers": [layer.to_payload() for layer in self.layers.values()],
            "removed_physical_layer_ids": [layer.physical_layer_id for layer in self.removed],
            "effective_layers": [self.layer(key).to_payload() for key in self.layers],
        }


def _request_modeling(spec: Any) -> _ZMap | None:
    """Normalize new requests; historical readers retain their recorded basis."""
    if spec.modeling is None and spec._historical_modeling:
        return None
    mapping = _ZMap(spec.modeling, spec.physical_layers)
    object.__setattr__(spec, "physical_layers", tuple(mapping.layers.values()))
    return mapping


def _modeling_payload(spec: Any) -> dict[str, Any]:
    if spec.modeling is None:
        return {}
    return {"modeling": spec.modeling,
            "physical_layers": [layer.to_payload() for layer in spec.physical_layers]}



def _require_source_film_interval(low: float, high: float, source: PhysicalLayerSpec) -> None:
    """A film reference binds the authored interval, not another sheet plane."""
    if low != source.z_min_um or high != source.z_max_um:
        raise ValueError(
            f"source film interval differs from physical layer {source.physical_layer_id!r}"
        )

def _effective_record(record: Any, mapping: _ZMap, *, retained_domain: bool = False) -> dict[str, Any]:
    """Source coordinates and physical thickness accompany native geometry."""
    low, high = record.z_min_um, record.z_max_um
    identity = record.physical_layer_id
    thickness = high - low
    if identity is None:
        if mapping.modeling == "thin_film" and not retained_domain and getattr(record, "physical_role", None) != "substrate":
            raise ValueError("Thin Film source objects require physical_layer_id")
        representation, effective_low, effective_high = "solid", mapping.map_z(low), mapping.map_z(high)
    else:
        effective = mapping.layer(identity)
        representation = effective.representation
        if effective.source.kind in {"stack_film", "local_film"}:
            thickness = effective.source.physical_thickness_um
        if representation == "sheet":
            _require_source_film_interval(low, high, effective.source)
            effective_low, effective_high = effective.z_min_um, effective.z_max_um
        else:
            effective_low, effective_high = mapping.map_z(low), mapping.map_z(high)
    return {**record.to_payload(), "physical_layer_id": identity,
            "physical_thickness_um": thickness,
            "representation": representation,
            "effective_z_min_um": effective_low,
            "effective_z_max_um": effective_high}


def _source_physical_layers(build_input: Any, stack: Any, modeling: Modeling
                            ) -> tuple[tuple[PhysicalLayerSpec, ...], dict[str, str]]:
    """Read explicit source level declarations and bind qualified Entities."""
    from scgsim.semantics.route_a import geometry_z_range

    records = {item["semantic_id"]: item for item in stack["layers"]}
    records.update(stack["solution_regions"])
    layers = []
    bindings = {}
    for entity in build_input.entities:
        record = records.get(entity.metadata.get("source_semantic_id", entity.semantic_id), {})
        declaration = entity.metadata.get("aedt_modeling", record.get("metadata", {}).get("aedt_modeling"))
        if declaration is None:
            if modeling == "thin_film" and entity.material_kind == "conductor":
                raise ValueError(f"Thin Film conductor {entity.semantic_id!r} lacks declared physical layer modeling")
            continue
        low, high = geometry_z_range(record.get("geometry", entity.geometry), entity.semantic_id)
        layer = PhysicalLayerSpec(
            z_min_um=low, z_max_um=high,
            source_thickness_um=record.get("geometry", entity.geometry).get("thickness_um"),
            **declaration)
        layers.append(layer)
        bindings[entity.semantic_id] = layer.physical_layer_id
    mapping = _ZMap(modeling, layers)
    return tuple(mapping.layers.values()), bindings


def _mapped_geometry(geometry: Any, mapping: _ZMap, effective: EffectiveLayer | None = None) -> dict[str, Any]:
    from scgsim.aedt.epr.models import detached
    from scgsim.semantics.route_a import geometry_z_range

    result = detached(geometry)
    low, high = geometry_z_range(geometry, "source geometry")
    if effective is not None and effective.representation == "sheet":
        _require_source_film_interval(low, high, effective.source)
    low, high = (effective.z_min_um, effective.z_max_um) if effective is not None else (mapping.map_z(low), mapping.map_z(high))
    result.update(z_min_um=low, z_max_um=high)
    if "z_um" in result:
        result["z_um"] = low
    if "thickness_um" in result:
        result["thickness_um"] = high - low
    return result


def _effective_geometry_input(build_input: Any, stack: Any, modeling: Modeling):
    """Derive one detached geometry/stack map without changing source facts."""
    from dataclasses import replace
    from scgsim.aedt.epr.models import detached, canonical_sha256

    layers, bindings = _source_physical_layers(build_input, stack, modeling)
    mapping = _ZMap(modeling, layers)
    entities = []
    effective_stack = detached(stack)
    records = {record["semantic_id"]: record for record in effective_stack["layers"]}
    records.update(effective_stack["solution_regions"])
    for entity in build_input.entities:
        identity = bindings.get(entity.semantic_id)
        layer = mapping.layer(identity) if identity is not None else None
        geometry = _mapped_geometry(entity.geometry, mapping, layer)
        representation = "sheet" if layer is not None and layer.representation == "sheet" else "solid"
        primitive_representation = ("surface_sheet" if representation == "sheet" else "cutout_boundary_shell") if entity.material_kind == "conductor" else "solution_domain"
        trace = {"physical_layer_id": identity, "representation": representation,
                 "physical_thickness_um": (layer.source.physical_thickness_um if layer is not None and layer.source.kind in {"stack_film", "local_film"} else entity.geometry.get("thickness_um", entity.geometry.get("z_max_um", 0)-entity.geometry.get("z_min_um", 0))),
                 "source_geometry": detached(entity.geometry),
                 "effective_z_min_um": geometry["z_min_um"],
                 "effective_z_max_um": geometry["z_max_um"]}
        metadata = {**entity.metadata, "aedt_effective_layer": trace}
        entities.append(replace(entity, geometry=geometry, metadata=metadata,
                                route_representations={**entity.route_representations, "_effective": primitive_representation}))
        record = records.get(entity.metadata.get("source_semantic_id", entity.semantic_id))
        if record is not None:
            record["geometry"] = geometry
            record["metadata"] = {**record.get("metadata", {}), "aedt_effective_layer": trace}
    trace = {**mapping.to_payload(), "source_entity_layers": bindings,
             "source_stack_sha256": canonical_sha256(detached(stack))}
    effective_stack["metadata"] = {**effective_stack.get("metadata", {}), "aedt_modeling": trace}
    return replace(build_input, entities=tuple(entities),
                   solution_regions=effective_stack["solution_regions"],
                   metadata={**build_input.metadata, "aedt_modeling": trace}), effective_stack


def _source_record_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Decode source fields; native lowering always comes from the authored map."""
    return {key: item for key, item in value.items()
            if not (key == "physical_layer_id" and item is None) and key not in {
        "representation", "effective_z_min_um", "effective_z_max_um",
        "physical_thickness_um", "native_import_layer"}}


def _verify_effective_records(actual, expected) -> None:
    if list(actual) != list(expected):
        raise ValueError("canonical native lowering records differ from authored modeling map")
