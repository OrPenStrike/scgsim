"""Native mesh observations and detached summaries; no solver-DOF inference."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _corner_chord_statistics(
    edges: set[tuple[int, int]], points: Mapping[int, tuple[float, ...]]
) -> dict[str, Any]:
    """Measure unique primary-edge chords in Palace's micrometre coordinates."""
    lengths = [math.dist(points[a], points[b]) for a, b in edges]
    return dict(
        unit="um",
        interpretation="primary-corner chords, not curved edge lengths",
        count=len(lengths),
        minimum=min(lengths) if lengths else None,
        maximum=max(lengths) if lengths else None,
        mean=sum(lengths) / len(lengths) if lengths else None,
    )


def observe_native_mesh(groups: Mapping[str, Any]) -> dict[str, Any]:
    """Read final Gmsh connectivity before its native session is finalized."""
    import gmsh

    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = {
        int(tag): tuple(float(x) for x in coordinates[3 * i : 3 * i + 3])
        for i, tag in enumerate(node_tags)
    }
    types = []
    element_types, element_tags, _ = gmsh.model.mesh.getElements()
    for code, tags in zip(element_types, element_tags, strict=True):
        name, dimension, order, nodes, reference, primary = (
            gmsh.model.mesh.getElementProperties(code)
        )
        types.append(
            dict(
                code=int(code),
                name=name,
                dimension=int(dimension),
                order=int(order),
                nodes_per_element=int(nodes),
                primary_nodes=int(primary),
                count=len(tags),
                reference_coordinates=[float(x) for x in reference],
            )
        )
    regions = []
    for dim, attribute in gmsh.model.getPhysicalGroups():
        codes: dict[int, set[int]] = {}
        region_edges: set[tuple[int, int]] = set()
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, attribute):
            ts, ids, connectivity = gmsh.model.mesh.getElements(dim, int(entity))
            for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
                codes.setdefault(int(code), set()).update(int(t) for t in tags)
                if dim == 1:
                    _, _, _, count, _, _ = gmsh.model.mesh.getElementProperties(code)
                    endpoints = [
                        int(n)
                        for i in range(len(tags))
                        for n in nodes[i * count : i * count + 2]
                    ]
                elif dim > 1:
                    # Native edge topology avoids treating quad/hex diagonals as edges.
                    endpoints = gmsh.model.mesh.getElementEdgeNodes(
                        int(code), int(entity), primary=True
                    )
                else:
                    endpoints = ()
                region_edges.update(
                    tuple(sorted((int(endpoints[i]), int(endpoints[i + 1]))))
                    for i in range(0, len(endpoints), 2)
                )
        regions.append(
            dict(
                dimension=int(dim),
                attribute=int(attribute),
                name=gmsh.model.getPhysicalName(dim, attribute),
                element_count=sum(len(v) for v in codes.values()),
                element_types={str(k): len(v) for k, v in codes.items()},
                corner_chord_edge_lengths=_corner_chord_statistics(
                    region_edges, points
                ),
            )
        )
    edges: set[tuple[int, int]] = set()
    faces: set[tuple[int, int, int]] = set()
    tetrahedra: set[int] = set()
    vertices: set[int] = set()
    shared: dict[tuple[int, ...], tuple[int, ...]] = {}
    volumes = {int(t) for v in groups["volumes"].values() for t in v["tags"]}
    for volume in volumes:
        ts, ids, connectivity = gmsh.model.mesh.getElements(3, volume)
        for code, tags, nodes in zip(ts, ids, connectivity, strict=True):
            name, dim, order, count, _, primary = gmsh.model.mesh.getElementProperties(
                code
            )
            if "tetrahedron" not in name.lower():
                continue
            for i, tag in enumerate(tags):
                if int(tag) in tetrahedra:
                    continue
                tetrahedra.add(int(tag))
                corners = tuple(int(n) for n in nodes[i * count : i * count + primary])
                vertices.update(corners)
                edges.update(
                    tuple(sorted(e)) for e in itertools.combinations(corners, 2)
                )
                faces.update(
                    tuple(sorted(f)) for f in itertools.combinations(corners, 3)
                )
            full = gmsh.model.mesh.getElementFaceNodes(
                int(code), 3, volume, primary=False
            )
            corners = gmsh.model.mesh.getElementFaceNodes(
                int(code), 3, volume, primary=True
            )
            face_size = (order + 1) * (order + 2) // 2
            for i in range(len(corners) // 3):
                key = tuple(sorted(int(n) for n in corners[3 * i : 3 * i + 3]))
                value = tuple(
                    sorted(int(n) for n in full[face_size * i : face_size * (i + 1)])
                )
                if key in shared and shared[key] != value:
                    raise ValueError(
                        "shared tetrahedral face has different high-order nodes"
                    )
                shared[key] = value
    return dict(
        node_count=len(node_tags),
        element_count=sum(len(t) for t in element_tags),
        element_types=types,
        physical_regions=regions,
        tetrahedral_topology=dict(
            vertices=len(vertices),
            edges=len(edges),
            faces=len(faces),
            tetrahedra=len(tetrahedra),
        ),
        corner_chord_edge_lengths=_corner_chord_statistics(edges, points),
    )


def mesh_summary(
    manifest: Mapping[str, Any],
    *,
    problem: str,
    fem_order: int | None,
    mesh_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Detach recorded observations and a separately labeled FEM topology estimate."""
    observed = manifest.get("meshing")
    result: dict[str, Any] = dict(
        status="recorded" if observed is not None else "not_recorded",
        mesh_sha256=mesh_sha256,
        manifest_sha256=manifest_sha256,
        meshing=copy.deepcopy(observed),
        curve_source=copy.deepcopy(manifest.get("curve_source")),
        curved_arrangement=copy.deepcopy(manifest.get("curved_arrangement")),
        finite_element_order=fem_order,
        dof_estimate=dict(
            status="unavailable",
            actual_solver_dof=False,
            reason="H1 estimate not implemented"
            if problem == "Electrostatic"
            else "tetrahedral topology not recorded",
        ),
    )
    if problem == "Eigenmode" and observed is not None and fem_order is not None:
        topology = observed["statistics"]["tetrahedral_topology"]
        p = fem_order
        estimate = (
            p * topology["edges"]
            + p * (p - 1) * topology["faces"]
            + p * (p - 1) * (p - 2) * topology["tetrahedra"] // 2
        )
        result["dof_estimate"] = dict(
            status="available",
            space="ND",
            value=estimate,
            basis="tetrahedral primary-corner topology; before boundary constraints",
            actual_solver_dof=False,
        )
    return result


def summary_from_files(
    mesh_path: Path, manifest_path: Path, *, problem: str, fem_order: int
) -> dict[str, Any]:
    """Detach current facade artifacts without changing solver or mesh state."""
    payload = manifest_path.read_bytes()
    return mesh_summary(
        json.loads(payload),
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
    )
