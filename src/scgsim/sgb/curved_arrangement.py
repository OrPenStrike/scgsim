"""Compiler-owned OCC planar arrangement and exact surface-first Z sweep.

Source polygons remain diagnostic input. Native planar Boolean history owns
split curves and footprint membership before material interfaces are named.
The detached boundary XAO carries shared native edges/faces to volume lowering;
no surface is repaired by fitting polygon chords and no volumes are fragmented.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .models import (
    BoundaryCurveChainSpec, ConstructionPlanRecord, CurvePlanRecord, CurveRefRecord,
    InterfacePlanRecord, PointPlanRecord, RouteABConstructionPlanRecord,
    SourceCurveSpec, SurfaceLoopRecord, SurfacePlanRecord, SurfaceRefRecord,
    VolumePlanRecord, RouteABVolumePlanRecord, InnerPecVoidShellRecord, MMContactRecord,
)
from .native_construction import add_surface_first_volume
from .source_curves import boundary_binding


def _identity(prefix, value):
    return prefix + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def _edge_vertices(gmsh, edge):
    vertices = [abs(tag) for dim, tag in gmsh.model.getBoundary([(1, edge)], oriented=True) if dim == 0]
    return sorted(vertices, key=lambda tag: float(gmsh.model.getParametrization(
        1, edge, gmsh.model.getValue(0, tag, []))[0]))


def _face_loops(gmsh, occ, face):
    oriented = {abs(tag): tag for dim, tag in gmsh.model.getBoundary([(2, face)], oriented=True) if dim == 1}
    result = []
    for raw in occ.getCurveLoops(face)[1]:
        pending = [oriented[abs(int(tag))] for tag in raw]
        ordered = [pending.pop(0)]
        def endpoints(tag):
            vertices = _edge_vertices(gmsh, abs(tag))
            return vertices if tag > 0 else vertices[::-1]
        while pending:
            end = endpoints(ordered[-1])[-1]
            next_index = next((i for i, tag in enumerate(pending) if endpoints(tag)[0] == end), None)
            if next_index is None:
                raise ValueError("native face boundary does not form one oriented closed chain")
            ordered.append(pending.pop(next_index))
        result.append(tuple(ordered))
    return result


def _z_range(entity):
    geometry = entity.geometry
    lower = float(geometry.get("z_min_um", geometry.get("z_um", 0.0)))
    return lower, float(geometry.get("z_max_um", lower + float(geometry.get("thickness_um", 0.0))))


def _source_rings(build_input):
    """Use adapter-normalized Entity-owned logical polygons, including holes."""
    polygons = {p.polygon_id: p for p in build_input.polygons}
    result = {}
    for entity in build_input.entities:
        if entity.polygon_ids:
            result[entity.semantic_id] = [(pid, polygons[pid].exterior, polygons[pid].holes)
                                          for pid in entity.polygon_ids]
        else:
            outer = entity.geometry.get("outer_loop")
            if entity.metadata.get("is_auto_vacuum_region"):
                outer = entity.metadata.get("auto_vacuum_envelope_outer_loop", outer)
            if outer is None:
                bounds = entity.geometry.get("domain_bounds_um", entity.geometry.get("bounds_um"))
                if bounds is None:
                    raise ValueError(f"{entity.semantic_id} has no source planar footprint")
                outer = ((bounds['x_min_um'], bounds['y_min_um']), (bounds['x_max_um'], bounds['y_min_um']),
                         (bounds['x_max_um'], bounds['y_max_um']), (bounds['x_min_um'], bounds['y_max_um']))
            result[entity.semantic_id] = [(entity.semantic_id, outer,
                () if entity.metadata.get("is_auto_vacuum_region") else entity.geometry.get("hole_loops", ()))]
    return result


def _select_boundary(record, rings):
    import gdstk
    candidates = []
    for pid, outer, holes in rings.get(record.entity_id, ()):
        region = [gdstk.Polygon(outer)]
        if holes:
            region = gdstk.boolean(region, [gdstk.Polygon(h) for h in holes], 'not') or []
        if gdstk.inside([record.selector_point_um], region)[0]:
            candidates.append((pid, outer, holes))
    if len(candidates) != 1:
        raise ValueError(f"{record.boundary_id} selector must bind one Entity-owned logical polygon")
    pid, outer, holes = candidates[0]
    ring = outer if record.role == 'outer' else holes[record.hole_index]
    return (record.entity_id, pid, record.role, record.hole_index), tuple(tuple(float(x) for x in xy) for xy in ring)


def _circle_points(points):
    a,b,c=points
    ax,ay=a
    bx,by=b
    cx,cy=c
    determinant=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/determinant
    uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/determinant
    return (ux,uy), math.hypot(ax-ux,ay-uy)


def _reconstruct(record, ring):
    count=len(ring)
    segments=tuple(record.segment_indices)
    if not segments or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<count for i in segments):
        raise ValueError(f"{record.boundary_id} requires explicit original segment indices")
    if len(set(segments)) != len(segments) or any(b != (a+1)%count for a,b in zip(segments,segments[1:])):
        raise ValueError(f"{record.boundary_id} segments must form one ordered source chain")
    if record.closed != (len(segments)==count and (segments[-1]+1)%count==segments[0]):
        raise ValueError(f"{record.boundary_id} closure differs from selected source chain")
    vertex_indices=(*segments,(segments[-1]+1)%count)
    anchors=tuple(record.anchor_indices) or vertex_indices
    if any(i not in vertex_indices for i in anchors):
        raise ValueError(f"{record.boundary_id} anchors must belong to selected source chain")
    points=tuple(ring[i] for i in anchors)
    if not record.closed and (anchors[0]!=vertex_indices[0] or anchors[-1]!=vertex_indices[-1]):
        raise ValueError(f"{record.boundary_id} reconstruction must retain original chain endpoints")
    if record.kind=='circular_arc' and record.closed:
        if len(points)!=3:
            raise ValueError("closed circular reconstruction requires three explicit distinct anchors")
        center,radius=_circle_points(points)
        start=points[0]
        angle=math.atan2(start[1]-center[1],start[0]-center[0])
        area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(ring,(*ring[1:],ring[0])))
        direction=1 if area>0 else -1
        position=lambda t:(center[0]+radius*math.cos(t),center[1]+radius*math.sin(t))
        opposite=position(angle+direction*math.pi)
        curves=(SourceCurveSpec('circular_arc',(start,position(angle+direction*math.pi/2),opposite)),
                SourceCurveSpec('circular_arc',(opposite,position(angle+direction*3*math.pi/2),start)))
    else:
        curves=(SourceCurveSpec(record.kind,points,degree=record.degree,
                                parameter_interval=record.parameter_interval),)
    return segments,curves,{'requested':asdict(record),'original_ring_um':ring,
        'selected_segments':[{'index':i,'start_um':ring[i],'end_um':ring[(i+1)%count]} for i in segments],
        'selected_anchor_indices':anchors,'selected_anchor_points_um':points}


def _boundaries(build_input,rings):
    authored={}
    reconstructed=defaultdict(list)
    provenance=[]
    for raw in build_input.boundary_curves:
        record=boundary_binding(raw)
        key,ring=_select_boundary(record,rings)
        if key in authored:
            raise ValueError(f"multiple complete authored boundaries claim {key}")
        authored[key]=record.curves
        provenance.append({'source':asdict(record),'original_ring_um':ring,'operation':'authored'})
    for raw in build_input.boundary_reconstruction:
        record=boundary_binding(raw,reconstruction=True)
        key,ring=_select_boundary(record,rings)
        segments,curves,operation=_reconstruct(record,ring)
        reconstructed[key].append((segments,curves,record.boundary_id))
        provenance.append({'operation':'explicit_gds_reconstruction',**operation})
    result={}
    for entity_id,records in rings.items():
        for pid,outer,holes in records:
            for role,index,ring in [('outer',None,outer),*[('hole',i,h) for i,h in enumerate(holes)]]:
                key=(entity_id,pid,role,index)
                if key in authored:
                    if key in reconstructed:
                        raise ValueError(f"authored and reconstructed boundaries both claim {key}")
                    result[key]=authored[key]
                    continue
                operations=reconstructed.get(key,())
                claimed={}
                starts={}
                for segments,curves,bid in operations:
                    if any(i in claimed for i in segments):
                        raise ValueError(f"overlapping reconstruction segment claims on {key}")
                    for i in segments:
                        claimed[i]=bid
                    starts[segments[0]]=curves
                # Traverse from an operation start when it crosses original ring index zero.
                first=next((segments[0] for segments,_,_ in operations if 0 in segments),0)
                indices=[(first+i)%len(ring) for i in range(len(ring))]
                curves=[]
                for i in indices:
                    if i in starts:
                        curves.extend(starts[i])
                    elif i not in claimed:
                        curves.append(SourceCurveSpec('line_segment',(tuple(ring[i]),tuple(ring[(i+1)%len(ring)]))))
                result[key]=tuple(curves)
    return result,provenance


class _Kernel:
    def __init__(self,gmsh):
        self.gmsh=gmsh
        self.occ=gmsh.model.occ
        self.points={}
        self.source_edges={}

    def point(self,xyz):
        xyz=tuple(float(x) for x in xyz)
        if xyz not in self.points:
            self.points[xyz]=self.occ.addPoint(*xyz)
        return self.points[xyz]

    def line(self,a,b):
        points=[self.point(a),self.point(tuple(a[j]+(b[j]-a[j])/3 for j in range(3))),
                self.point(tuple(a[j]+2*(b[j]-a[j])/3 for j in range(3))),self.point(b)]
        return self.occ.addBSpline(points,degree=3,knots=[0,1],multiplicities=[4,4])

    def curve(self,spec,source):
        points=[(*xy,0.0) for xy in spec.points_um]
        if spec.kind=='line_segment':
            tags=[self.line(*points)]
        elif spec.kind=='circular_arc':
            center,radius=_circle_points(spec.points_um)
            cx,cy=center
            angles=[math.atan2(y-cy,x-cx) for x,y in spec.points_um]
            ccw=(angles[2]-angles[0])%(2*math.pi)
            mid=(angles[1]-angles[0])%(2*math.pi)
            sweep=ccw if mid<ccw else ccw-2*math.pi
            pieces=math.ceil(abs(sweep)/(math.pi/2))
            tags=[]
            for i in range(pieces):
                lo=angles[0]+sweep*i/pieces
                hi=angles[0]+sweep*(i+1)/pieces
                half=(hi-lo)/2
                w=math.cos(half)
                m=(lo+hi)/2
                q=[(cx+radius*math.cos(lo),cy+radius*math.sin(lo),0),
                   (cx+radius*math.cos(m)/w,cy+radius*math.sin(m)/w,0),
                   (cx+radius*math.cos(hi),cy+radius*math.sin(hi),0)]
                if i == 0:
                    q[0] = points[0]
                if i == pieces-1:
                    q[2] = points[-1]
                # Exact homogeneous degree elevation of a rational quadratic arc.
                h=[(*[v*weight for v in xyz],weight) for xyz,weight in zip(q,[1,w,1])]
                elevated=[h[0],tuple((h[0][j]+2*h[1][j])/3 for j in range(4)),
                          tuple((2*h[1][j]+h[2][j])/3 for j in range(4)),h[2]]
                tags.append(self.occ.addBSpline([self.point(tuple(v[j]/v[3] for j in range(3))) for v in elevated],
                    degree=3,weights=[v[3] for v in elevated],knots=[0,1],multiplicities=[4,4]))
        elif spec.kind=='interpolation_spline':
            tags=[self.occ.addSpline([self.point(p) for p in points])]
        elif spec.kind=='bspline':
            arguments={}
            if spec.degree is not None:
                arguments['degree']=spec.degree
            if spec.weights:
                arguments['weights']=list(spec.weights)
            if spec.knots:
                arguments['knots']=list(spec.knots)
            if spec.multiplicities:
                arguments['multiplicities']=list(spec.multiplicities)
            tags=[self.occ.addBSpline([self.point(p) for p in points],**arguments)]
        else:
            raise ValueError(f"unsupported reconstruction kind {spec.kind!r}")
        for index,tag in enumerate(tags):
            self.source_edges[tag]={'source':source,'geometry':asdict(spec),'native_piece_index':index}
        if spec.parameter_interval is not None:
            if len(tags)!=1:
                raise ValueError("explicit source parameter trimming needs one native curve")
            self.occ.synchronize()
            tag=tags[0]
            lo,hi=spec.parameter_interval
            points=[self.occ.addPoint(*self.gmsh.model.getValue(1,tag,[u])) for u in (lo,hi)]
            output,mapping=self.occ.fragment([(1,tag)],[(0,p) for p in points],removeObject=False)
            self.occ.synchronize()
            retained=[]
            for dim,candidate in mapping[0]:
                if dim!=1:
                    continue
                bounds=self.gmsh.model.getParametrizationBounds(1,candidate)
                xyz=self.gmsh.model.getValue(1,candidate,[(float(bounds[0][0])+float(bounds[1][0]))/2])
                parameter=float(self.gmsh.model.getParametrization(1,tag,xyz)[0])
                if lo<=parameter<=hi:
                    retained.append(candidate)
                self.source_edges[candidate]={**self.source_edges[tag],'requested_parameter_interval':spec.parameter_interval}
            tags=retained
        # A periodic edge has no pair of distinct topological endpoints. Split
        # its native parameter domain for the four-edge ruled wall construction;
        # this changes topology only, never the source spline geometry.
        self.occ.synchronize()
        open_tags = []
        for tag in tags:
            if len(_edge_vertices(self.gmsh, tag)) == 2:
                open_tags.append(tag)
                continue
            lower, upper = self.gmsh.model.getParametrizationBounds(1, tag)
            lo, hi = float(lower[0]), float(upper[0])
            split_points = [self.occ.addPoint(*self.gmsh.model.getValue(1, tag, [u]))
                            for u in (lo, (lo + hi) / 2)]
            _, history = self.occ.fragment([(1, tag)], [(0, p) for p in split_points],
                                           removeObject=False)
            self.occ.synchronize()
            for dim, piece in history[0]:
                if dim == 1:
                    open_tags.append(piece)
                    self.source_edges[piece] = {**self.source_edges[tag],
                                               'periodic_source_parameter_bounds': (lo, hi)}
        return open_tags

    def face(self,chains):
        loops=[]
        edges=[]
        for curves,source in chains:
            tags=[]
            for index,curve in enumerate(curves):
                tags.extend(self.curve(curve,{**source,'curve_index':index}))
            wire = self.occ.addCurveLoop(tags)
            orientation_face = self.occ.addPlaneSurface([wire])
            self.occ.synchronize()
            bounds = self.gmsh.model.getParametrizationBounds(2, orientation_face)
            normal = self.gmsh.model.getNormal(orientation_face, [float((a+b)/2) for a,b in zip(*bounds)])
            self.occ.remove([(2, orientation_face)], recursive=False)
            # OCC's plane-surface constructor subtracts subsequent wires in the
            # outer wire's plane basis. Normalize that basis, not authored order.
            if normal[2] < 0:
                wire = self.occ.addCurveLoop([-tag for tag in reversed(tags)])
            loops.append(wire)
            edges.extend(tags)
        return self.occ.addPlaneSurface(loops),edges


def _detached_edge(gmsh, edge, lifted_inverse, vertical_inverse,
                   curve_descriptors, native_edges, z_values):
    endpoints = [tuple(float(v) for v in gmsh.model.getValue(0, tag, []))
                 for tag in _edge_vertices(gmsh, edge)]
    if len(endpoints) != 2:
        raise ValueError('native arrangement edge lacks two detached topology endpoints')
    if edge in lifted_inverse:
        level, source = lifted_inverse[edge]
        geometry = {**curve_descriptors[source], 'z_um': z_values[level]}
        kinds = {record['geometry']['kind'] for record in native_edges[source]}
        kind = next(iter(kinds)) if len(kinds) == 1 else 'bspline'
    elif edge in vertical_inverse:
        geometry = {'kind': 'line_segment', 'endpoints_um': endpoints}
        kind = 'line_segment'
    else:
        raise ValueError('native sidewall construction replaced a planned shared boundary curve')
    cid = _identity('C__', geometry)
    pids = [_identity('P__', xyz) for xyz in endpoints]
    bounds = gmsh.model.getParametrizationBounds(1, edge)
    return CurvePlanRecord(cid, kind, *pids, geometry=geometry,
        parameter_interval=(float(bounds[0][0]), float(bounds[1][0]))), dict(zip(pids, endpoints))


def build_curved_construction_plan(build_input,*,route):
    """Compile source curves globally before naming shared material surfaces."""
    import gmsh
    from .planning import _prepare_auto_vacuum_solution_regions, plan_route_tags, _with_surface_contract_metadata
    from .validation import validate_selected_route
    prepared=_prepare_auto_vacuum_solution_regions(build_input,route=route,native_complement=True)
    validate_selected_route(prepared,route)
    rings=_source_rings(prepared)
    chains,operations=_boundaries(prepared,rings)
    entities={e.semantic_id:e for e in prepared.entities}
    initialized=bool(gmsh.isInitialized())
    if not initialized:
        gmsh.initialize()
    previous=gmsh.model.getCurrent()
    gmsh.model.add('scgsim-curved-arrangement')
    terminal = gmsh.option.getNumber('General.Terminal')
    gmsh.option.setNumber('General.Terminal',0)
    try:
        kernel=_Kernel(gmsh)
        occ=kernel.occ
        footprints=[]
        source_edges=[]
        for entity_id,records in rings.items():
            for pid,outer,holes in records:
                sources=[]
                for role,index in [('outer',None),*[('hole',i) for i in range(len(holes))]]:
                    key=(entity_id,pid,role,index)
                    sources.append((chains[key],{'entity_id':entity_id,'polygon_id':pid,'role':role,'hole_index':index}))
                face,edges=kernel.face(sources)
                footprints.append((face,entity_id))
                source_edges.extend(edges)
        for port in prepared.port_sheet_regions:
            sources=[]
            for role,index,ring in [('outer',None,port.exterior),*[('hole',i,h) for i,h in enumerate(port.holes)]]:
                curves=tuple(SourceCurveSpec('line_segment',(tuple(a),tuple(b))) for a,b in zip(ring,(*ring[1:],ring[0])))
                sources.append((curves,{'port_sheet_id':port.port_sheet_id,'role':role,'hole_index':index}))
            face,edges=kernel.face(sources)
            footprints.append((face,'PORT::'+port.port_sheet_id))
            source_edges.extend(edges)
        source_edges=list(dict.fromkeys(source_edges))
        occ.synchronize()
        source_footprints=[{'source_owner':owner,'area_um2':occ.getMass(2,face),'native_bounds_um':occ.getBoundingBox(2,face)} for face,owner in footprints]
        references={tag:occ.copy([(1,tag)])[0][1] for tag in source_edges}
        output,mapping=occ.fragment([(2,face) for face,_ in footprints],[(1,tag) for tag in source_edges])
        occ.synchronize()
        cells={tag for dim,tag in output if dim==2}
        membership=defaultdict(set)
        for (_,owner),mapped in zip(footprints,mapping[:len(footprints)]):
            for dim,tag in mapped:
                if dim==2:
                    membership[tag].add(owner)
        native_edges=defaultdict(list)
        for source,mapped in zip(source_edges,mapping[len(footprints):]):
            reference=references[source]
            for dim,tag in mapped:
                if dim!=1:
                    continue
                bounds=gmsh.model.getParametrizationBounds(1,tag)
                endpoint_values=[gmsh.model.getValue(1,tag,[float(u[0])]).tolist() for u in bounds]
                source_interval=[float(gmsh.model.getParametrization(1,reference,xyz)[0]) for xyz in endpoint_values]
                native_edges[tag].append({**kernel.source_edges[source],'source_parameter_interval':source_interval,
                                          'native_parameter_bounds':[float(u[0]) for u in bounds]})
        cell_loops={tag:_face_loops(gmsh,occ,tag) for tag in cells}
        edge_cells=defaultdict(set)
        for cell,loops in cell_loops.items():
            for loop in loops:
                for edge in loop:
                    edge_cells[abs(edge)].add(cell)
        # OCC may omit lower-dimensional history for an unchanged exterior
        # wire while replacing its native tags during a face Boolean. Recover
        # that history through native common geometry, never endpoint matching
        # or a caller-chosen geometric tolerance.
        for edge in edge_cells:
            if native_edges[edge]:
                continue
            for source, reference in references.items():
                existing = set(occ.getEntities())
                common, _ = occ.intersect([(1, edge)], [(1, reference)],
                                          removeObject=False, removeTool=False)
                occ.synchronize()
                for dim, common_edge in common:
                    if dim != 1:
                        continue
                    bounds = gmsh.model.getParametrizationBounds(1, edge)
                    common_bounds = gmsh.model.getParametrizationBounds(1, common_edge)
                    values = [gmsh.model.getValue(1, common_edge, [float(u[0])]) for u in common_bounds]
                    native_edges[edge].append({**kernel.source_edges[source],
                        'source_parameter_interval': [float(gmsh.model.getParametrization(1, reference, xyz)[0]) for xyz in values],
                        'arrangement_parameter_interval': [float(gmsh.model.getParametrization(1, edge, xyz)[0]) for xyz in values],
                        'native_parameter_bounds': [float(u[0]) for u in bounds],
                        'native_history_method': 'occ_common_geometry'})
                occ.remove([entity for entity in common if entity not in existing], recursive=False)
            if not native_edges[edge]:
                raise ValueError('native arrangement boundary has no source curve provenance')
        # Remove only scratch source/reference entities after the Boolean history has been detached.
        retained_edges=set(edge_cells)
        occ.remove([(2,t) for _,t in occ.getEntities(2) if t not in cells],recursive=False)
        occ.remove([(1,t) for _,t in occ.getEntities(1) if t not in retained_edges],recursive=False)
        occ.synchronize()
        curve_descriptors={edge:{'native_type':gmsh.model.getType(1,edge),'sources':native_edges[edge],
            'native_bounds':[float(u[0]) for u in gmsh.model.getParametrizationBounds(1,edge)]} for edge in retained_edges}
        cell_ids={cell:_identity('CELL__',{'curves':sorted(_identity('',curve_descriptors[abs(e)]) for loop in loops for e in loop),
                                                   'source_owners':sorted(membership[cell])}) for cell,loops in cell_loops.items()}
        # Route A face metal is a sheet; all other finite conductors remain excluded bodies.
        sheet_ids={eid for eid,e in entities.items() if route=='A' and e.material_kind=='conductor' and e.part_role=='face_metal'}
        ranges={eid:_z_range(e) for eid,e in entities.items()}
        # Native shared cells establish actual XY overlap; ownership uses the
        # same explicit attachment policy as the polygon conductor path.
        from .planning import _TOPOLOGY_EPS_UM, _validate_volumetric_overlap_ownership
        from .validation import _resolve_contact_pad_attachment
        from itertools import combinations
        conductors = tuple(sorted((e for e in entities.values()
                                   if e.material_kind == 'conductor'),
                                  key=lambda e: e.semantic_id))
        def finite_overlap(first, second):
            return any(first.semantic_id in membership[cell]
                       and second.semantic_id in membership[cell] for cell in cells)

        for conductor in conductors:
            if conductor.part_role == 'contact_pad':
                _resolve_contact_pad_attachment(conductor, conductors, finite_overlap=finite_overlap)
        for lower, upper in combinations(conductors, 2):
            lower_min, lower_max = ranges[lower.semantic_id]
            upper_min, upper_max = ranges[upper.semantic_id]
            if min(lower_max, upper_max) - max(lower_min, upper_min) <= _TOPOLOGY_EPS_UM:
                continue
            if finite_overlap(lower, upper):
                _validate_volumetric_overlap_ownership(
                    lower, upper, conductors, finite_overlap=finite_overlap)
        port_planes = {}
        for port in prepared.port_sheet_regions:
            host_ids = tuple(dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps))
            intervals = [ranges[eid] for eid in host_ids]
            if len(host_ids) != 2 or len(set(intervals)) != 1:
                raise ValueError("curved lumped-port hosts must share one authored Z range")
            lo, hi = intervals[0]
            port_planes[port.port_sheet_id] = lo if route == 'A' else (lo + hi) / 2
        z_values=sorted({z for eid,interval in ranges.items() for z in (interval[:1] if eid in sheet_ids else interval)} | set(port_planes.values()))
        occupied={}
        owners={}
        for slab,(lo,hi) in enumerate(zip(z_values,z_values[1:])):
            middle=(lo+hi)/2
            for cell in cells:
                candidates=[eid for eid in membership[cell] if eid in entities and eid not in sheet_ids and ranges[eid][0]<middle<ranges[eid][1]]
                metals=[eid for eid in candidates if entities[eid].material_kind=='conductor']
                if len({entities[eid].net_id for eid in metals})>1:
                    raise ValueError('native curved arrangement found overlapping conductors on different Nets')
                if metals:
                    occupied[slab,cell]=sorted(metals)
                    continue
                domains=[eid for eid in candidates if entities[eid].role=='solution_region']
                if domains:
                    owner=max(domains,key=lambda eid:(entities[eid].material_kind=='dielectric',entities[eid].priority,eid))
                    owners[slab,cell]=owner
        # Connected planar/slab cells form solution shells; no 3D Boolean repair.
        parents={key:key for key in owners}
        def find(key):
            while parents[key]!=key:
                parents[key]=parents[parents[key]]
                key=parents[key]
            return key
        def union(a,b):
            if a in owners and b in owners and owners[a]==owners[b]:
                parents[find(b)]=find(a)
        for slab,cell in owners:
            for edge in (abs(e) for loop in cell_loops[cell] for e in loop):
                for other in edge_cells[edge]:
                    union((slab,cell),(slab,other))
            if not any(eid in membership[cell] and ranges[eid][0] == z_values[slab]
                       for eid in sheet_ids):
                union((slab,cell),(slab-1,cell))
        components=defaultdict(list)
        for key in owners:
            components[find(key)].append(key)
        volume_ids={key:_identity('VOL__',{'owner':owners[key],'cells':sorted((slab,cell_ids[cell]) for slab,cell in members)})
                    for key,members in components.items()}
        volume_for={key:volume_ids[find(key)] for key in owners}
        # Sweep the entire shared curve complex together, strictly 1D -> 2D.
        # A native prism preserves trimmed rational/spline geometry and shares
        # the vertical edge of each source vertex across adjacent walls.
        lifted = {}
        lifted_vertices = {}
        vertical = {}
        swept_walls = {}
        for edge in retained_edges:
            tag = occ.copy([(1, edge)])[0][1]
            occ.translate([(1, tag)], 0, 0, z_values[0])
            lifted[0, edge] = tag
        _, copied_history = occ.fragment([(1, lifted[0, edge]) for edge in retained_edges], [])
        for edge, history in zip(retained_edges, copied_history):
            tags = [tag for dim, tag in history if dim == 1]
            if len(tags) != 1:
                raise ValueError('Z-copy changed an already arranged planar edge')
            lifted[0, edge] = tags[0]
        occ.synchronize()
        for edge in retained_edges:
            for source, copied in zip(_edge_vertices(gmsh, edge), _edge_vertices(gmsh, lifted[0, edge])):
                lifted_vertices[0, source] = copied
        for slab, (lo, hi) in enumerate(zip(z_values, z_values[1:])):
            source_by_tag = {lifted[slab, edge]: edge for edge in retained_edges}
            source_by_vertex = {tag: vertex for (level, vertex), tag in lifted_vertices.items() if level == slab}
            output = occ.extrude([(1, tag) for tag in source_by_tag], 0, 0, hi - lo)
            occ.synchronize()
            for dim, face in output:
                if dim != 2:
                    continue
                boundary = [abs(tag) for d, tag in gmsh.model.getBoundary([(2, face)], oriented=True) if d == 1]
                bottom_tags = [tag for tag in boundary if tag in source_by_tag]
                if len(bottom_tags) != 1:
                    raise ValueError('native edge sweep did not retain one canonical source edge per wall')
                edge = source_by_tag[bottom_tags[0]]
                swept_walls[slab, edge] = face
                for tag in boundary:
                    if tag == bottom_tags[0]:
                        continue
                    vertices = _edge_vertices(gmsh, tag)
                    lower_vertices = [v for v in vertices if v in source_by_vertex]
                    if not lower_vertices:
                        lifted[slab + 1, edge] = tag
                    elif len(lower_vertices) == 1:
                        lower = lower_vertices[0]
                        vertex = source_by_vertex[lower]
                        key = (slab, vertex)
                        if key in vertical and vertical[key] != tag:
                            raise ValueError('native collective sweep duplicated a shared vertical edge')
                        vertical[key] = tag
                        lifted_vertices[slab + 1, vertex] = next(v for v in vertices if v != lower)
                    else:
                        raise ValueError('native edge sweep returned an unexpected horizontal boundary')
                if (slab + 1, edge) not in lifted:
                    raise ValueError('native edge sweep lost the canonical upper edge')
            if any((slab, edge) not in swept_walls for edge in retained_edges):
                raise ValueError('native edge sweep did not cover the arranged curve complex')
        native_surfaces={}
        surface_data={}
        volume_refs=defaultdict(list)
        native_contacts = []
        contact_keys = set()

        def contact(tag, pair, key, normal, operation):
            pair = tuple(sorted(pair))
            if normal is not None and ranges[pair[0]][0] > ranges[pair[1]][0]:
                pair = pair[::-1]
            if pair[0] == pair[1] or (pair, key) in contact_keys:
                return
            if len({entities[eid].net_id for eid in pair}) > 1:
                raise ValueError('native curved contact joins conductors on different Nets')
            contact_keys.add((pair, key))
            native_contacts.append({'tag': tag, 'pair': pair, 'key': key,
                                    'normal': normal, 'operation': operation})

        def cap(cell, level):
            wires = []
            for index, loop in enumerate(cell_loops[cell]):
                basis = loop if index == 0 else tuple(-edge for edge in reversed(loop))
                wires.append(occ.addCurveLoop([
                    int(math.copysign(lifted[level, abs(edge)], edge)) for edge in basis]))
            return occ.addPlaneSurface(wires)

        def wall(edge, slab):
            return swept_walls[slab, edge]

        # Finite shared arrangement cells own contact normalization. Native
        # walls own lateral contacts; neither a Net label nor a polygon witness
        # creates a geometric contact.
        for cell in cells:
            metals = sorted(eid for eid in membership[cell] if eid in entities
                            and entities[eid].material_kind == 'conductor')
            for a, b in combinations(metals, 2):
                lo = max(ranges[a][0], ranges[b][0])
                hi = min(ranges[a][1], ranges[b][1])
                if lo > hi:
                    continue
                if (a in sheet_ids and ranges[a][0] != lo
                        or b in sheet_ids and ranges[b][0] != lo):
                    continue
                level = z_values.index(lo)
                contact(cap(cell, level), (a, b), ('cell', cell_ids[cell], lo),
                        (0., 0., 1.), 'native_planar_contact' if lo == hi else 'native_overlap_normalization')
        for slab in range(len(z_values) - 1):
            for edge, neighbors in edge_cells.items():
                if len(neighbors) != 2:
                    continue
                left, right = tuple(neighbors)
                pairs = {(a, b) for a in occupied.get((slab, left), ())
                         for b in occupied.get((slab, right), ()) if a != b}
                if pairs:
                    tag = wall(edge, slab)
                    for pair in pairs:
                        contact(tag, pair, ('edge', _identity('', curve_descriptors[edge]), slab),
                                None, 'native_shared_wall_contact')
        def surface(key,tag,adjacent,metal=(),port=None,normal=None,face_kind='interface'):
            sid=_identity('SURF__',key)
            native_surfaces[sid]=tag
            boundary_keys=[k for k in adjacent if k in owners]
            boundary_owners=tuple(owners[k] for k in boundary_keys)
            semantic_owners=tuple(dict.fromkeys((*metal,*boundary_owners)))
            kinds=[]
            if metal:
                kinds=list(dict.fromkeys('MS' if entities[owners[k]].material_kind=='dielectric' else 'MA' for k in boundary_keys))
            elif len(boundary_keys)==2:
                materials=[entities[owners[k]].material_kind for k in boundary_keys]
                kinds=['SS' if materials==['dielectric','dielectric'] else 'AA' if materials==['vacuum','vacuum'] else 'SA']
            else:
                kinds=['SA' if boundary_keys and entities[owners[boundary_keys[0]]].material_kind=='dielectric' else 'AA']
            prefix='_'.join(kinds)
            iid=(kinds[0]+'__'+ '__'.join(semantic_owners)+'__'+sid) if semantic_owners else None
            metadata={'owner_semantic_ids':semantic_owners,'boundary_volume_ids':boundary_owners,
                      'interface_kinds':tuple(kinds),'curved_arrangement_surface':True,
                      'physical_name':prefix+'__'+'__'.join(semantic_owners)+'__'+face_kind.upper(),
                      'exposed_surface_role':face_kind,
                      'representation':('surface_sheet' if metal and route=='A' else 'cutout_boundary_shell' if metal else 'solution_surface')}
            if port:
                hosts = tuple(dict.fromkeys(overlap.host_semantic_id for overlap in port.overlaps))
                domain_owners = tuple(dict.fromkeys(boundary_owners))
                if route == 'B':
                    domain_owners = domain_owners[:1]
                embedded = next(owner for owner in domain_owners if entities[owner].material_kind == 'vacuum')
                source = {**dict(port.metadata), 'source_layer':port.source_layer,
                          'source_polygon_id':port.source_polygon_id,
                          'overlaps':[asdict(overlap) for overlap in port.overlaps],
                          'host_polygon_ids':tuple(overlap.host_polygon_id for overlap in port.overlaps),
                          'route':route}
                if route == 'B':
                    metadata['embedded_volume_plan_ids'] = tuple(dict.fromkeys(
                        volume_for[key] for key in boundary_keys
                        if entities[owners[key]].material_kind == 'vacuum'))
                    if not metadata['embedded_volume_plan_ids']:
                        raise ValueError(f"{sid} active port has no final vacuum volume host")
                metadata.update({'physical_name':'LUMPED_PORT__'+port.metadata['source_name'],
                    'route':route,'representation':'lumped_port_sheet','interface_type':'lumped_port',
                    'face_kind':'sheet','net_id':None,'conductor_component_id':None,'equipotential_id':None,
                    'owner_semantic_ids':hosts,'boundary_volume_ids':domain_owners,
                    'source_provenance':source,'route_a_boundary_port':route=='A',
                    'embedded_surface':True,'embedded_volume_id':embedded,
                    'physical_attribute':{'port_index':port.metadata['port_index'],
                        'port_name':port.metadata['source_name'],'source_layer':port.source_layer,
                        'target_layer':port.metadata['target_layer'],'direction':port.metadata['direction'],
                        'embedded_volume_id':embedded,'owner_semantic_ids':hosts,
                        'owner_provenance':tuple({'semantic_id':host,'net_id':entities[host].net_id,
                            'equipotential_id':entities[host].metadata.get('equipotential_id'),
                            'conductor_component_id':'COMP__'+host} for host in hosts)}})
                semantic_owners = hosts
                iid = None
            surface_data[sid]={'owner':semantic_owners[0] if semantic_owners else port.port_sheet_id,
                              'interface':iid,'metadata':metadata,'normal':normal,'role':'lumped_port' if port else 'cutout_boundary_shell' if metal and route=='B' else 'interface' if len(semantic_owners)>1 else 'domain_boundary',
                              'face_kind':face_kind}
            return sid
        # Horizontal children are arrangement cells, hence port and metal footprints have already been partitioned.
        ports={p.port_sheet_id:p for p in prepared.port_sheet_regions}
        for level,z in enumerate(z_values):
            for cell in cells:
                below=(level-1,cell)
                above=(level,cell)
                adjacent=[key for key in (below,above) if key in owners]
                metals=set(occupied.get(below,()))|set(occupied.get(above,()))
                metals.update(eid for eid in membership[cell] if eid in sheet_ids and ranges[eid][0]==z)
                active_ports=[ports[eid[6:]] for eid in membership[cell] if eid.startswith('PORT::') and port_planes[eid[6:]]==z]
                if not adjacent or (len(adjacent)==2 and volume_for[below]==volume_for[above] and not metals and not active_ports):
                    continue
                tag=cap(cell, level)
                face_kind = ('interface' if route=='A' and any(eid in sheet_ids for eid in metals) else
                             'bottom' if metals and z==ranges[sorted(metals)[0]][0] else
                             'top' if metals else 'interface')
                sid=surface({'cell':cell_ids[cell],'z_um':z},tag,adjacent,tuple(sorted(metals)),active_ports[0] if active_ports and not metals else None,(0.,0.,1.),face_kind)
                if not (route == 'B' and active_ports and not metals):
                    for key in adjacent:
                        volume_refs[volume_for[key]].append(SurfaceRefRecord(sid,'forward' if key==below else 'reversed','boundary'))
        for slab in range(len(z_values)-1):
            for edge,neighbors in edge_cells.items():
                adjacent=[(slab,c) for c in neighbors if (slab,c) in owners]
                if not adjacent or (len(adjacent)==2 and volume_for[adjacent[0]]==volume_for[adjacent[1]]):
                    continue
                metals=tuple(sorted({eid for c in neighbors for eid in occupied.get((slab,c),())}))
                tag=wall(edge, slab)
                sid=surface({'edge':curve_descriptors[edge],'z_interval':[z_values[slab],z_values[slab+1]]},tag,adjacent,metals,face_kind='sidewall')
                for key in adjacent:
                    cell=key[1]
                    sign=next(e for loop in cell_loops[cell] for e in loop if abs(e)==edge)
                    volume_refs[volume_for[key]].append(SurfaceRefRecord(sid,'forward' if sign>0 else 'reversed','boundary'))
        occ.synchronize()
        # Native history and geometry determine curve identity; endpoints alone cannot identify a curve.
        used_edges={abs(t) for tag in native_surfaces.values() for _,t in gmsh.model.getBoundary([(2,tag)],oriented=True)}
        points={}
        curve_ids={}
        curve_records=[]
        point_curves=defaultdict(set)
        lifted_inverse={tag:(level,edge) for (level,edge),tag in lifted.items()}
        vertical_inverse={tag:key for key,tag in vertical.items()}
        for edge in sorted(used_edges):
            record, edge_points = _detached_edge(gmsh, edge, lifted_inverse, vertical_inverse,
                                                curve_descriptors, native_edges, z_values)
            curve_ids[edge] = record.curve_id
            points.update(edge_points)
            for pid in edge_points:
                point_curves[pid].add(record.curve_id)
            curve_records.append(record)
        loops=[]
        surfaces=[]
        interfaces=[]
        curve_surfaces=defaultdict(set)
        curve_owners=defaultdict(set)
        curve_volumes=defaultdict(set)
        curve_interfaces=defaultdict(set)
        for sid,tag in native_surfaces.items():
            data=surface_data[sid]
            native_loops=_face_loops(gmsh,occ,tag)
            loop_ids=[]
            for index,tags in enumerate(native_loops):
                lid=_identity('LOOP__',[sid,index])
                loop_ids.append(lid)
                refs=tuple(CurveRefRecord(curve_ids[abs(int(t))],1 if t>0 else -1,'boundary') for t in tags)
                loops.append(SurfaceLoopRecord(lid,refs,'outer' if index==0 else 'hole',sid))
                for ref in refs:
                    curve_surfaces[ref.curve_id].add(sid)
                    curve_owners[ref.curve_id].update(data['metadata']['owner_semantic_ids'])
                    curve_volumes[ref.curve_id].update(data['metadata']['boundary_volume_ids'])
                    if data['interface']:
                        curve_interfaces[ref.curve_id].add(data['interface'])
            geometry={'native_curved_face':True,'native_surface_type':gmsh.model.getType(2,tag),'arrangement_surface_id':sid,
                      'representation':data['metadata']['representation'],'shell_part':data['face_kind'],
                      'source_polygon_ids':tuple(pid for owner in data['metadata']['owner_semantic_ids'] if owner in entities for pid in entities[owner].polygon_ids)}
            surfaces.append(SurfacePlanRecord(sid,data['owner'],data['role'],geometry,loop_ids[0],tuple(loop_ids[1:]),
                interface_id=data['interface'],normal_hint=data['normal'],valid_routes=(route,),metadata=data['metadata']))
            if data['interface']:
                owners_tuple=data['metadata']['owner_semantic_ids']
                pair=(owners_tuple[0],owners_tuple[-1])
                interfaces.append(InterfacePlanRecord(data['interface'],data['metadata']['interface_kinds'][0],pair,
                    'native_curved_planar_arrangement_z_sweep',metadata={'curved_arrangement_surface_id':sid}))
        contact_parents = {eid: eid for row in native_contacts for eid in row['pair']}
        def contact_root(eid):
            while contact_parents[eid] != eid:
                eid = contact_parents[eid]
            return eid
        for row in native_contacts:
            a, b = row['pair']
            contact_parents[contact_root(b)] = contact_root(a)
        contact_members = defaultdict(set)
        for eid in contact_parents:
            contact_members[contact_root(eid)].add(eid)
        mm_contacts = []
        for row in native_contacts:
            tag = row['tag']
            a, b = row['pair']
            members = tuple(sorted(contact_members[contact_root(a)]))
            component_id = _identity('COMP__', members)
            equipotentials = {entities[eid].metadata.get('equipotential_id') for eid in members}
            equipotentials.discard(None)
            if len(equipotentials) > 1:
                raise ValueError('native curved contact has conflicting equipotential IDs')
            cid = _identity('MM__', [row['pair'], row['key']])
            contact_curves = {}
            contact_points = {}
            contact_loops = []
            for index, edges in enumerate(_face_loops(gmsh, occ, tag)):
                refs = []
                for edge in edges:
                    record, edge_points = _detached_edge(gmsh, abs(edge), lifted_inverse,
                        vertical_inverse, curve_descriptors, native_edges, z_values)
                    contact_curves[record.curve_id] = record
                    contact_points.update(edge_points)
                    refs.append(CurveRefRecord(record.curve_id, 1 if edge > 0 else -1, 'boundary'))
                contact_loops.append(SurfaceLoopRecord(
                    _identity('LOOP__', [cid, index]), tuple(refs),
                    'outer' if index == 0 else 'hole'))
            geometry = {'native_curved_face': any(c.curve_kind != 'line_segment' for c in contact_curves.values()),
                        'native_surface_type': gmsh.model.getType(2, tag),
                        'surface_method': ('planar_arrangement_cap' if row['normal'] is not None
                                           else 'native_collective_exact_edge_sweep'),
                        'sweep_direction': None if row['normal'] is not None else (0., 0., 1.),
                        'outer_loop_ref': contact_loops[0].loop_id,
                        'hole_loop_refs': tuple(loop.loop_id for loop in contact_loops[1:]),
                        'points': [asdict(PointPlanRecord(pid, xyz)) for pid, xyz in contact_points.items()],
                        'curves': [asdict(c) for c in contact_curves.values()],
                        'surface_loops': [asdict(loop) for loop in contact_loops]}
            polygon = None
            if row['normal'] is not None and not geometry['native_curved_face']:
                polygon = tuple(contact_points[
                    contact_curves[ref.curve_id].start_point_id if ref.orientation > 0
                    else contact_curves[ref.curve_id].end_point_id][:2]
                    for ref in contact_loops[0].curve_refs)
            mm_contacts.append(MMContactRecord(
                cid, a, b,
                a + ('__top' if row['operation'] == 'native_planar_contact' else '__' + row['operation']),
                b + ('__bottom' if row['operation'] == 'native_planar_contact' else '__' + row['operation']),
                entities[a].polygon_ids, entities[b].polygon_ids, polygon,
                occ.getMass(2, tag), row['normal'], component_id, entities[a].net_id,
                next(iter(equipotentials), None),
                layer_provenance={eid: entities[eid].metadata.get('source_layer_name') for eid in row['pair']},
                material_provenance={eid: entities[eid].material_id for eid in row['pair']},
                source_provenance={'route': route, 'recognition_rule': row['operation'],
                                   'hidden_solver_contact': True, 'source_entity_ids': row['pair']},
                geometry_ref=geometry))
        mm_contacts = tuple(mm_contacts)
        from .planning import _component_metadata
        component_by_entity = {
            eid: component['conductor_component_id']
            for component in _component_metadata(mm_contacts)
            for eid in component['members']
        }
        for index, surface_record in enumerate(surfaces):
            if surface_record.surface_role != 'lumped_port':
                continue
            attribute = surface_record.metadata['physical_attribute']
            surfaces[index] = replace(surface_record, metadata={
                **surface_record.metadata,
                'physical_attribute': {**attribute, 'owner_provenance': tuple(
                    {**owner, 'conductor_component_id': component_by_entity.get(
                        owner['semantic_id'], owner['conductor_component_id'])}
                    for owner in attribute['owner_provenance'])}})
        ordinary = {surface.surface_id: surface for surface in _with_surface_contract_metadata(
            prepared, route=route, interfaces=tuple(interfaces), mm_contacts=mm_contacts,
            surfaces=tuple(surface for surface in surfaces if surface.surface_role != 'lumped_port'))}
        surfaces = [ordinary.get(surface.surface_id, surface) for surface in surfaces]
        # Exact terminal identities come from shared arrangement curves, not from polygon chord tests.
        for port_surface in tuple(surface for surface in surfaces if surface.surface_role == 'lumped_port'):
            port_curves = {ref.curve_id for loop in loops if loop.surface_id == port_surface.surface_id for ref in loop.curve_refs}
            overlaps = []
            for overlap in port_surface.metadata['source_provenance']['overlaps']:
                owner = overlap['host_semantic_id']
                terminal_ids = tuple(sorted(cid for cid in port_curves if any(
                    surface.owner_semantic_id == owner and surface.surface_role != 'lumped_port'
                    and (route == 'A' or surface.geometry_ref.get('shell_part') == 'sidewall')
                    and surface.surface_id in curve_surfaces[cid] for surface in surfaces)))
                overlaps.append({**overlap, 'native_terminal_curve_ids':terminal_ids})
                if route == 'B':
                    for index, owner_surface in enumerate(surfaces):
                        if owner_surface.owner_semantic_id != owner or owner_surface.geometry_ref.get('shell_part') != 'sidewall':
                            continue
                        if any(owner_surface.surface_id in curve_surfaces[cid] for cid in terminal_ids):
                            binding={'port_surface_id':port_surface.surface_id,'overlap_id':overlap['overlap_id'],'host_semantic_id':owner}
                            surfaces[index]=replace(owner_surface, metadata={**owner_surface.metadata,
                                'route_b_port_sheet_sidewall_partition':True,
                                'route_b_port_sheet_bindings':(*owner_surface.metadata.get('route_b_port_sheet_bindings',()),binding)})
            index=next(i for i,surface in enumerate(surfaces) if surface.surface_id==port_surface.surface_id)
            surfaces[index]=replace(port_surface, metadata={**port_surface.metadata,
                'source_provenance':{**port_surface.metadata['source_provenance'],'overlaps':overlaps}})
        curves=tuple(replace(curve,used_by_surface_ids=tuple(sorted(curve_surfaces[curve.curve_id])),
                     owner_semantic_ids=tuple(sorted(curve_owners[curve.curve_id])),interface_ids=tuple(sorted(curve_interfaces[curve.curve_id])),
                     boundary_volume_ids=tuple(sorted(curve_volumes[curve.curve_id]))) for curve in curve_records)
        point_records=tuple(PointPlanRecord(pid,xyz,used_by_curve_ids=tuple(sorted(point_curves[pid]))) for pid,xyz in points.items())
        # Decompose disconnected boundary shells through their canonical edge
        # incidences. A floating PEC body is an inner shell, not another outer
        # face in the same native surface loop.
        surface_curves = defaultdict(set)
        for loop in loops:
            surface_curves[loop.surface_id].update(ref.curve_id for ref in loop.curve_refs)
        volumes = []
        for root, members in components.items():
            vid = volume_ids[root]
            refs = tuple(volume_refs[vid])
            pending = {ref.surface_id: ref for ref in refs}
            shells = []
            while pending:
                first = next(iter(pending))
                stack = [pending.pop(first)]
                connected = []
                while stack:
                    ref = stack.pop()
                    connected.append(ref)
                    neighbors = [sid for sid in pending
                                 if surface_curves[sid] & surface_curves[ref.surface_id]]
                    stack.extend(pending.pop(sid) for sid in neighbors)
                shells.append(tuple(connected))
            enclosed = []
            for index, shell in enumerate(shells):
                native_volume = add_surface_first_volume(
                    gmsh, [[native_surfaces[ref.surface_id] for ref in shell]])
                signed_mass = occ.getMass(3, native_volume)
                enclosed.append(abs(signed_mass))
                occ.synchronize()
                outward_sign = -1 if signed_mass < 0 else 1
                directions = {abs(tag): 'forward' if tag * outward_sign > 0 else 'reversed'
                              for dim, tag in gmsh.model.getBoundary([(3, native_volume)], oriented=True)
                              if dim == 2}
                shells[index] = tuple(replace(ref, orientation=directions[native_surfaces[ref.surface_id]])
                                      for ref in shell)
                occ.remove([(3, native_volume)], recursive=False)
            exterior_index = max(range(len(shells)), key=enclosed.__getitem__)
            inner = []
            for index, shell in enumerate(shells):
                if index == exterior_index:
                    continue
                metal_owners = tuple(sorted({eid for ref in shell
                    for eid in surface_data[ref.surface_id]['metadata']['owner_semantic_ids']
                    if entities[eid].material_kind == 'conductor'}))
                # A void boundary faces into the enclosed PEC, opposite to
                # the outward direction of its temporary finite solid.
                shell = tuple(replace(ref, orientation='reversed' if ref.orientation == 'forward' else 'forward')
                              for ref in shell)
                shells[index] = shell
                inner.append(InnerPecVoidShellRecord(
                    _identity('VOID__', [vid, sorted(ref.surface_id for ref in shell)]),
                    shell, _identity('COMP__', metal_owners), metal_owners))
            volumes.append(RouteABVolumePlanRecord(
                vid, owners[root], entities[owners[root]].material_id,
                tuple(ref for shell in shells for ref in shell),
                valid_routes=(route,), exterior_surface_refs=shells[exterior_index],
                inner_pec_void_shells=tuple(inner),
                metadata={'physical_name': entities[owners[root]].metadata.get('auto_vacuum_group_id', owners[root]),
                          'material_kind': entities[owners[root]].material_kind,
                          'curved_arrangement_cells': [(s, cell_ids[c]) for s, c in members],
                          'native_boundary_shell_enclosed_volumes_um3': enclosed}))
        volumes = tuple(volumes)
        # Preserve exact shared boundary shapes in one detached XAO, before any volume exists.
        active_faces=set(native_surfaces.values())
        occ.remove([(2,t) for _,t in occ.getEntities(2) if t not in active_faces],recursive=False)
        occ.remove([(1,t) for _,t in occ.getEntities(1) if t not in used_edges],recursive=False)
        occ.synchronize()
        active_points={abs(t) for edge in used_edges for _,t in gmsh.model.getBoundary([(1,edge)])}
        occ.remove([(0,t) for _,t in occ.getEntities(0) if t not in active_points],recursive=False)
        occ.synchronize()
        for edge,cid in curve_ids.items():
            physical=gmsh.model.addPhysicalGroup(1,[edge])
            gmsh.model.setPhysicalName(1,physical,'SGB_CURVE::'+cid)
        for sid,tag in native_surfaces.items():
            physical=gmsh.model.addPhysicalGroup(2,[tag])
            gmsh.model.setPhysicalName(2,physical,'SGB_SURFACE::'+sid)
        with TemporaryDirectory(prefix='scgsim-curved-boundaries-') as temporary:
            path=Path(temporary)/'boundaries.xao'
            gmsh.write(str(path))
            payload=path.read_bytes()
        metadata={'backend_strategy':'surface_plan_first_bottom_up_occ','curved_boundary_xao':base64.b64encode(payload).decode('ascii'),
                  'curved_boundary_xao_sha256':hashlib.sha256(payload).hexdigest(),
                  'curved_arrangement':{'native_gmsh_version':gmsh.__version__,'source_boundary_bindings':[asdict(r) for r in prepared.boundary_curves],
                  'source_footprints':source_footprints,'reconstruction_operations':operations,'planar_cells':[{'cell_id':cell_ids[c],'source_membership':sorted(membership[c])} for c in sorted(cells)],
                  'native_curve_intervals':[{'curve_id':curve_ids[e],'geometry':curve_descriptors[source]} for e,(level,source) in lifted_inverse.items() if e in curve_ids],
                  'z_levels_um':z_values,'shared_curves':len(curves),'shared_surfaces':len(surfaces),'solution_volumes':len(volumes),
                  'sidewall_method':'native_collective_exact_edge_sweep_reusing_canonical_edges'}}
        plan_type=RouteABConstructionPlanRecord if route in {'A','B'} else ConstructionPlanRecord
        plan = plan_type(route,interfaces=tuple(interfaces),points=point_records,curves=curves,surface_loops=tuple(loops),surfaces=tuple(surfaces),
                         volumes=volumes,tags=plan_route_tags(route=route,surfaces=tuple(surfaces),volumes=volumes),
                         metadata=metadata,port_sheet_regions=prepared.port_sheet_regions, mm_contacts=mm_contacts)
        from .validation import (
            validate_curve_plan_coverage, validate_volume_surface_closure,
            validate_tag_plan_coverage,
        )
        validate_curve_plan_coverage(points=plan.points, curves=plan.curves,
                                    surface_loops=plan.surface_loops, surfaces=plan.surfaces)
        validate_volume_surface_closure(volumes=plan.volumes, surfaces=plan.surfaces,
                                        surface_loops=plan.surface_loops)
        validate_tag_plan_coverage(surfaces=plan.surfaces, volumes=plan.volumes, tags=plan.tags)
        return plan
    finally:
        gmsh.model.remove()
        if previous:
            gmsh.model.setCurrent(previous)
        gmsh.option.setNumber('General.Terminal', terminal)
        if not initialized:
            gmsh.finalize()
