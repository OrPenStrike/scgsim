"""Surface-first OCC construction with explicit source-to-native face binding.

Stock shell/solid correction may replace faces while retaining canonical edge
incidence. Callers consume that correspondence; final shared-body topology is
still checked against the compiler's source-owned surface graph.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class _SurfaceFirstResult:
    volume_tag: int
    # Original face -> (result face, typed outer-wire basis parity).
    face_bindings: dict[int, tuple[int, int]]
    wire_parities: dict[int, dict[tuple[int, ...], int]]


def _wire_signature(loop: Sequence[int]) -> tuple[int, ...]:
    edges = tuple(abs(int(edge)) for edge in loop)
    # Native WireExplorer supplies cyclic incidence, not an outer-wire label.
    return min(sequence[i:] + sequence[:i]
               for sequence in (edges, edges[::-1]) for i in range(len(edges)))


def _face_boundary(gmsh: Any, tag: int) -> dict[tuple[int, ...], Counter]:
    """Complete per-wire directed incidence, without geometry matching."""
    directed = Counter(int(edge) for dim, edge in gmsh.model.getBoundary(
        [(2, tag)], combined=False, oriented=True) if dim == 1)
    wires = {}
    used = Counter()
    for loop in gmsh.model.occ.getCurveLoops(tag)[1]:
        signature = _wire_signature(loop)
        edges = Counter(abs(int(edge)) for edge in loop)
        if signature in wires or used.keys() & edges.keys():
            raise ValueError(f"native face wire incidence is ambiguous: face={tag}")
        wire = Counter({edge: count for edge, count in directed.items()
                        if abs(edge) in edges})
        if Counter(abs(edge) for edge in wire.elements()) != edges:
            raise ValueError(f"native face wire incidence is incomplete: face={tag}")
        wires[signature] = wire
        used.update(edges)
    if used != Counter(abs(edge) for edge in directed.elements()):
        raise ValueError(f"native face wire incidence is incomplete: face={tag}")
    return wires


def _surface_outer_wire_edges(surfaces, loops, curve_tags, surface_tags):
    """Bind the existing typed outer loop to current native edges."""
    loops_by_id = {loop.loop_id: loop for loop in loops}
    return {surface_tags[surface.surface_id]: tuple(
        curve_tags[ref.curve_id] for ref in loops_by_id[surface.outer_loop_ref].curve_refs)
        for surface in surfaces if surface.surface_id in surface_tags}


def _rebind_surface_loops(surfaces, loops, curve_tags, surface_tags, repaired):
    """Consume wire reversal only when every use of a canonical loop agrees."""
    uses = {}
    for surface in surfaces:
        if surface.construction_only:
            continue
        for loop_id in (surface.outer_loop_ref, *surface.hole_loop_refs):
            uses.setdefault(loop_id, []).append(surface.surface_id)
    result = []
    for loop in loops:
        signature = _wire_signature(tuple(curve_tags[ref.curve_id] for ref in loop.curve_refs))
        requirements = {}
        for sid in uses.get(loop.loop_id, ()):
            tag = surface_tags[sid]
            if tag in repaired.wire_parities:
                parities = repaired.wire_parities[tag]
                if signature not in parities:
                    raise ValueError(f"native repair lost source loop correspondence: {loop.loop_id}, surface={sid}")
                requirements[sid] = parities[signature]
            else:
                # This construction did not replace or reverse that incident face.
                requirements[sid] = 1
        if len(set(requirements.values())) > 1:
            raise ValueError(f"native repair has conflicting shared loop directions: "
                             f"loop={loop.loop_id}, surfaces={requirements}")
        if requirements and next(iter(requirements.values())) < 0:
            loop = replace(loop, curve_refs=tuple(replace(ref, orientation=-ref.orientation)
                                                  for ref in reversed(loop.curve_refs)))
        result.append(loop)
    return tuple(result)


def _rebind_volume_surface_refs(volumes, parities):
    """Express retained shell directions in the final typed outer-wire basis."""
    def rebind(ref):
        return ref if parities[ref.surface_id] > 0 else replace(
            ref, orientation='reversed' if ref.orientation == 'forward' else 'forward')
    return tuple(replace(volume,
        surface_refs=tuple(rebind(ref) for ref in volume.surface_refs),
        exterior_surface_refs=tuple(rebind(ref) for ref in volume.exterior_surface_refs),
        inner_pec_void_shells=tuple(replace(void, surface_refs=tuple(
            rebind(ref) for ref in void.surface_refs)) for void in volume.inner_pec_void_shells))
        for volume in volumes)


def add_surface_first_volume(gmsh: Any, shell_surfaces: Sequence[Sequence[int]], *,
                             outer_wire_edges: Mapping[int, Sequence[int]]) -> _SurfaceFirstResult:
    """Construct with stock correction and return unique canonical face bindings.

    Native errors propagate and the caller's AutoFix option is restored on
    failure. Complete wire/edge incidence binds repaired faces to their source.
    Wires may reverse independently; the typed outer loop defines face basis;
    ambiguous or changed boundaries cannot establish that correspondence.
    Final coowner bodies must still consume the same source-bound native faces.
    """
    gmsh.model.occ.synchronize()
    inputs = {tag: _face_boundary(gmsh, tag)
              for shell in shell_surfaces for tag in shell}
    previous = gmsh.option.getNumber("Geometry.OCCAutoFix")
    try:
        gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
        shell_tags = tuple(gmsh.model.occ.addSurfaceLoop(tags, sewing=False)
                           for tags in shell_surfaces)
        volume = gmsh.model.occ.addVolume(shell_tags)
    finally:
        gmsh.option.setNumber("Geometry.OCCAutoFix", previous)
    gmsh.model.occ.synchronize()
    boundary = tuple((dim, int(tag)) for dim, tag in gmsh.model.getBoundary(
        [(3, volume)], combined=False, oriented=True) if dim == 2)
    actual = {abs(tag): _face_boundary(gmsh, abs(tag)) for _, tag in boundary}
    bindings = {}
    wire_parities = {}
    for original, wires in inputs.items():
        candidates = [tag for tag, candidate_wires in actual.items()
                      if candidate_wires.keys() == wires.keys()]
        if len(candidates) != 1:
            raise ValueError(f"native repair face correspondence is not unique: "
                             f"input={original}, candidates={candidates}, volume={volume}")
        repaired = candidates[0]
        parities = {}
        for signature, directed in wires.items():
            repaired_directed = actual[repaired][signature]
            if directed == repaired_directed:
                parities[signature] = 1
            elif Counter({-edge: count for edge, count in directed.items()}) == repaired_directed:
                parities[signature] = -1
            else:
                raise ValueError(f"native repair changed directed wire incidence: "
                                 f"input={original}, result={repaired}, volume={volume}")
        outer = _wire_signature(outer_wire_edges[original])
        if outer not in parities:
            raise ValueError(f"native repair lost typed outer wire: input={original}, volume={volume}")
        bindings[original] = (repaired, parities[outer])
        wire_parities[original] = parities
    if len({tag for tag, _ in bindings.values()}) != len(inputs) or {
            tag for tag, _ in bindings.values()} != set(actual):
        raise ValueError(f"native repair boundary correspondence is incomplete: volume={volume}")
    return _SurfaceFirstResult(volume, bindings, wire_parities)
