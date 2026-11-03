"""Outdoor scenes: only model what is really there.

Indoor BIM (Cloud2BIM) assumes a room: a floor, a ceiling and walls around it. Outdoors that
invents geometry, so instead:

  * ground  -> one IfcSlab following the scanned ground footprint (no rectangle, no roof)
  * facades -> an IfcWall per large vertical plane that is clearly a building face:
               long, tall, reaching down to the ground and densely covered with points.
               Cars, vans, hedges, poles and fences fail these tests.

Input: aligned points (metres, z-up, ground at z = 0). Output: room.ifc (IFC4).
"""
from __future__ import annotations

import numpy as np


def log(msg):
    print(f"[outdoor] {msg}", flush=True)


def find_facades(P, N, cams_xy, min_length=3.0, min_height=2.5, max_bottom=2.8,
                 min_cover=0.5, max_planes=30):
    """-> list of dicts: start, end (xy on the visible face), normal (towards cameras), bottom, top."""
    import open3d as o3d

    cand = np.nonzero((np.abs(N[:, 2]) < 0.2) & (P[:, 2] > 0.3))[0]
    if len(cand) < 2000:
        return []
    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[cand]))
    idx_map = cand
    facades = []
    for _ in range(max_planes):
        if len(rest.points) < 2000:
            break
        model, inl = rest.segment_plane(distance_threshold=0.06, ransac_n=3, num_iterations=3000)
        Q = np.asarray(rest.points)[inl]
        keep = np.ones(len(rest.points), bool)
        keep[inl] = False
        rest = rest.select_by_index(np.nonzero(keep)[0])
        n = np.array(model[:3]) / np.linalg.norm(model[:3])
        if abs(n[2]) > 0.1 or len(Q) < 1500:
            continue
        n = np.array([n[0], n[1], 0.0]) / np.linalg.norm(n[:2])
        u = np.array([-n[1], n[0], 0.0])                  # along the facade
        s = Q @ u
        z = Q[:, 2]
        # split the plane into connected stretches along u (two houses on one line, gaps)
        order = np.argsort(s)
        breaks = np.nonzero(np.diff(s[order]) > 0.8)[0]
        for seg in np.split(order, breaks + 1):
            if len(seg) < 1000:
                continue
            ss, zz = s[seg], z[seg]
            s0, s1 = np.percentile(ss, [1, 99])
            z0, z1 = np.percentile(zz, [1, 99])
            length, height = s1 - s0, z1 - z0             # visible part only
            # coverage of the (along, up) rectangle by 0.25 m cells
            gi = ((ss - s0) / 0.25).astype(int)
            gj = ((zz - z0) / 0.25).astype(int)
            ok = (gi >= 0) & (gj >= 0)
            cells = len(set(zip(gi[ok], gj[ok])))
            total = max(1, int(np.ceil(length / 0.25)) * int(np.ceil((z1 - z0) / 0.25)))
            cover = cells / total
            reason = None
            if length < min_length:
                reason = f"too short ({length:.1f} m)"
            elif height < min_height:
                reason = f"too low ({height:.1f} m)"
            elif z0 > max_bottom:
                # parked cars hide up to ~2.5 m of a facade; higher = balcony, sign, tree...
                reason = f"floating (visible part starts at {z0:.1f} m)"
            elif cover < min_cover:
                reason = f"sparse ({cover * 100:.0f}% covered)"
            if reason:
                log(f"  rejected vertical plane: {reason}")
                continue
            d = float(np.median(Q[seg] @ n))
            c = d * n
            # orient the normal towards the cameras (the street side)
            if np.median((cams_xy - c[:2]) @ n[:2]) < 0:
                n, u, d = -n, -u, -d
                s0, s1 = -s1, -s0
            start = (d * n + s0 * u)[:2]
            end = (d * n + s1 * u)[:2]
            facades.append({"start": start, "end": end, "normal": n[:2], "bottom": 0.0,
                            "visible_from": float(z0), "top": float(z1), "cover": float(cover),
                            "points": int(len(seg))})
    return merge_facades(facades)


def merge_facades(facades, max_angle=10, max_offset=0.6, max_gap=1.0):
    """RANSAC often returns one facade as several pieces: join pieces on the same line."""
    def line(f):
        n = np.r_[f["normal"], 0][:2]
        return n, float(np.mean([f["start"] @ n, f["end"] @ n]))

    merged = True
    while merged:
        merged = False
        for i in range(len(facades)):
            for j in range(i + 1, len(facades)):
                a, b = facades[i], facades[j]
                (na, da), (nb, db) = line(a), line(b)
                if na @ nb < np.cos(np.radians(max_angle)) or abs(da - db) > max_offset:
                    continue
                u = np.array([-na[1], na[0]])
                ia = sorted([a["start"] @ u, a["end"] @ u])
                ib = sorted([b["start"] @ u, b["end"] @ u])
                if max(ia[0], ib[0]) - min(ia[1], ib[1]) > max_gap:
                    continue
                lo, hi = min(ia[0], ib[0]), max(ia[1], ib[1])
                w = a["points"] / (a["points"] + b["points"])
                d = w * da + (1 - w) * db
                facades[i] = {"start": d * na + lo * u, "end": d * na + hi * u, "normal": na, "bottom": 0.0,
                              "visible_from": min(a["visible_from"], b["visible_from"]),
                              "top": max(a["top"], b["top"]),
                              "cover": w * a["cover"] + (1 - w) * b["cover"],
                              "points": a["points"] + b["points"]}
                del facades[j]
                merged = True
                break
            if merged:
                break
    for f in facades:
        L = np.linalg.norm(np.asarray(f["end"]) - np.asarray(f["start"]))
        note = f", lower {f['visible_from']:.1f} m occluded" if f["visible_from"] > 1.0 else ""
        log(f"  facade: {L:.1f} m long, {f['top']:.1f} m high, {f['cover'] * 100:.0f}% covered{note}")
    return facades


def ground_polygon(P, N, cell=0.25):
    """Footprint of the scanned ground as a (simplified) polygon."""
    from scipy.ndimage import binary_closing, binary_fill_holes, label
    from shapely.geometry import box
    from shapely.ops import unary_union

    g = P[(np.abs(N[:, 2]) > 0.85) & (np.abs(P[:, 2]) < 0.2)]
    if len(g) < 500:
        return None
    ij = np.floor(g[:, :2] / cell).astype(int)
    lo = ij.min(0)
    occ = np.zeros(ij.max(0) - lo + 1, bool)
    occ[tuple((ij - lo).T)] = True
    occ = binary_fill_holes(binary_closing(occ, np.ones((3, 3)), iterations=2))
    lab, n = label(occ)
    if n == 0:
        return None
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    cells = np.argwhere(big) + lo
    poly = unary_union([box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell) for i, j in cells])
    poly = poly.simplify(cell, preserve_topology=True)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def write_ifc(path, name, ground, facades, wall_thickness=0.2, slab_thickness=0.1):
    import ifcopenshell
    import ifcopenshell.api as api
    from ifcopenshell.util.shape_builder import ShapeBuilder

    f = api.run("project.create_file", version="IFC4")
    proj = api.run("root.create_entity", f, ifc_class="IfcProject", name=name)
    # metres (IfcOpenShell's default project unit is millimetres)
    api.run("unit.assign_unit", f, length={"is_metric": True, "raw": "METERS"})
    model = api.run("context.add_context", f, context_type="Model")
    body = api.run("context.add_context", f, context_type="Model", context_identifier="Body",
                   target_view="MODEL_VIEW", parent=model)
    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")   # no location (privacy)
    bld = api.run("root.create_entity", f, ifc_class="IfcBuilding", name=name)
    sto = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Ground")
    api.run("aggregate.assign_object", f, products=[site], relating_object=proj)
    api.run("aggregate.assign_object", f, products=[bld], relating_object=site)
    api.run("aggregate.assign_object", f, products=[sto], relating_object=bld)
    sb = ShapeBuilder(f)

    if ground is not None:
        slab = api.run("root.create_entity", f, ifc_class="IfcSlab", name="Ground")
        slab.PredefinedType = "BASESLAB"
        pts = [tuple(map(float, p)) for p in list(ground.exterior.coords)[:-1]]
        solid = sb.extrude(sb.profile(sb.polyline(pts, closed=True)), slab_thickness,
                           position=(0.0, 0.0, -slab_thickness))
        slab.Representation = f.createIfcProductDefinitionShape(None, None,
                                                                [sb.get_representation(body, [solid])])
        api.run("geometry.edit_object_placement", f, product=slab, matrix=np.eye(4))
        api.run("spatial.assign_container", f, products=[slab], relating_structure=sto)

    for k, fa in enumerate(facades, 1):
        a, b, n = np.array(fa["start"]), np.array(fa["end"]), np.array(fa["normal"])
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            continue
        ux = (b - a) / L
        # wall body sits behind the visible face (away from the street)
        origin = a - n * wall_thickness
        m = np.eye(4)
        m[:2, 0] = ux
        m[:2, 1] = n                         # local +y points from the back face to the visible face
        m[:2, 3] = origin
        w = api.run("root.create_entity", f, ifc_class="IfcWall", name=f"Facade {k}")
        rep = api.run("geometry.add_wall_representation", f, context=body, length=L,
                      height=float(fa["top"]), thickness=wall_thickness)
        api.run("geometry.assign_representation", f, product=w, representation=rep)
        api.run("geometry.edit_object_placement", f, product=w, matrix=m)
        api.run("spatial.assign_container", f, products=[w], relating_structure=sto)
        occl = (f"; lower {fa['visible_from']:.1f} m occluded (e.g. parked cars), extended to the ground"
                if fa["visible_from"] > 1.0 else "")
        w.Description = f"Scanned facade, {fa['cover'] * 100:.0f}% covered by points{occl}"
    f.write(str(path))
    return path


def build(P, cams_xy, out_ifc, name):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    N = np.asarray(pc.normals)
    ground = ground_polygon(P, N)
    facades = find_facades(P, N, cams_xy)
    log(f"ground: {ground.area:.0f} m²" if ground is not None else "no ground found")
    log(f"{len(facades)} facade(s) kept")
    write_ifc(out_ifc, name, ground, facades)
    return out_ifc
