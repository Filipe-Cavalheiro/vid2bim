#!/usr/bin/env python3
"""Dense room cloud -> aligned, metric, z-up cloud -> IFC (Cloud2BIM) -> FreeCAD + Blender.

Steps (all open source):
  1. align   : floor plane (Open3D RANSAC) -> z-up, floor at z=0;
               dominant wall direction -> walls parallel to x/y (Manhattan)
  2. bim     : Cloud2BIM (MIT, github.com/VaclavNezerka/Cloud2BIM) -> room.ifc
               (slabs, walls, door/window openings, spaces)
  3. freecad : open room.ifc headless in FreeCAD (BIM/IFC importer) -> room.FCStd
  4. meshes  : triangulate room.ifc with IfcOpenShell -> bim_meshes.json for Blender

Inputs : <scene>/dense.ply + dense.json (from dense_moge.py), <scene>/model_txt
Outputs: <scene>/bim/{aligned.ply, room.xyz, config.yaml, room.ifc, room.FCStd,
         bim_meshes.json} and <scene>/bim/align.json (COLMAP -> BIM transform)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOUD2BIM = HERE / "third_party" / "Cloud2BIM"


def log(msg):
    print(f"[room] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1. alignment
# --------------------------------------------------------------------------- #
def rotation_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def camera_up(model_dir):
    """Average 'up' of the handheld camera (-Y of COLMAP cameras) and camera centres (COLMAP units)."""
    import pycolmap
    rec = pycolmap.Reconstruction(model_dir)
    ims = [im for im in rec.images.values() if im.has_pose]
    downs = np.array([im.cam_from_world().rotation.matrix()[1] for im in ims])
    up = -downs.mean(0)
    return up / np.linalg.norm(up), np.array([im.projection_center() for im in ims])


def align(pcd, up_guess, cam_centers, max_tilt_deg=15):
    """Floor = lowest strong peak of horizontal-surface heights below the cameras
    (bed and desk tops are higher, walls are not horizontal), refined by a plane fit."""
    import open3d as o3d

    P = np.asarray(pcd.points)
    pc = o3d.geometry.PointCloud(pcd.points)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    N = np.asarray(pc.normals)
    h = P @ up_guess
    h_cam = float(np.median(cam_centers @ up_guess))
    cand = (np.abs(N @ up_guess) > 0.9) & (h < h_cam - 0.8)
    up, d_floor = up_guess, None
    if cand.sum() > 500:
        hist, edges = np.histogram(h[cand], bins=np.arange(h[cand].min(), h[cand].max() + 0.02, 0.02))
        strong = np.nonzero(hist >= 0.3 * hist.max())[0]
        floor_h = (edges[strong[0]] + edges[strong[0] + 1]) / 2
        near = np.nonzero(cand & (np.abs(h - floor_h) < 0.05))[0]
        sub = pcd.select_by_index(near)
        model, _ = sub.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=2000)
        n = np.array(model[:3]) / np.linalg.norm(model[:3])
        if n @ up_guess < 0:
            n = -n
        tilt = np.degrees(np.arccos(np.clip(n @ up_guess, -1, 1)))
        if tilt < max_tilt_deg:
            up = n
        d_floor = -float(np.median(P[near] @ up))
        log(f"floor {h_cam - floor_h:.2f} m below the cameras, tilt vs camera-up {tilt:.1f}°"
            + ("" if tilt < max_tilt_deg else " (too large, keeping camera up)"))
    else:
        log("no floor found below the cameras, using camera up")

    R1 = rotation_between(up, np.array([0, 0, 1.0]))
    Q = P @ R1.T

    # yaw: dominant direction of vertical surfaces (normals folded mod 90°)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    N = np.asarray(pc.normals)
    vert = np.abs(N[:, 2]) < 0.15
    ang = np.mod(np.arctan2(N[vert, 1], N[vert, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7) / 7, "valid")
    yaw = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    c, s = np.cos(-yaw), np.sin(-yaw)
    R2 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    R = R2 @ R1
    Q = P @ R.T

    # floor -> z = 0 (robust: low percentile of near-floor points), centre xy
    z0 = np.percentile(Q[:, 2], 1) if d_floor is None else float(-d_floor)
    xy0 = np.median(Q[:, :2], axis=0)
    t = -np.array([xy0[0], xy0[1], z0])
    return R, t


# --------------------------------------------------------------------------- #
# 2. Cloud2BIM
# --------------------------------------------------------------------------- #
C2B_RUNNER = r"""
import sys, types, runpy, os
sys.path.insert(0, {c2b!r})
sys.modules['e57'] = types.ModuleType('e57')          # only needed for .e57 input
import matplotlib; matplotlib.use('Agg')
import open3d as o3d
o3d.visualization.draw_geometries = lambda *a, **k: None   # headless
import plotly.graph_objects as go
# Cloud2BIM targets ifcopenshell 0.8.1, which accepted STEP-style ".T."/".F." booleans
import ifcopenshell
_ce = ifcopenshell.file.create_entity
_B = {{'.T.': True, '.F.': False}}
def _create_entity(self, *a, **kw):
    kw = {{k: (_B[v] if isinstance(v, str) and v in _B else v) for k, v in kw.items()}}
    return _ce(self, *a, **kw)
ifcopenshell.file.create_entity = _create_entity
go.Figure.show = lambda *a, **k: None
for d in ('images/pdf', 'images/wall_outputs_images', 'output_xyz', 'output_IFC'):
    os.makedirs(d, exist_ok=True)
sys.argv = ['cloud2entities.py', {cfg!r}]
runpy.run_path(os.path.join({c2b!r}, 'cloud2entities.py'), run_name='__main__')
"""


def run_cloud2bim(work: Path, xyz: Path, args) -> Path:
    if not (CLOUD2BIM / "cloud2entities.py").exists():
        sys.exit(f"Cloud2BIM not found in {CLOUD2BIM}. Clone it:\n"
                 f"  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "
                 f"https://github.com/VaclavNezerka/Cloud2BIM {CLOUD2BIM}")
    import yaml
    ifc = work / "room.ifc"
    cfg = {
        "e57_input": False, "xyz_files": [str(xyz)], "exterior_scan": False,
        "dilute": False, "dilution_factor": 1,
        "pc_resolution": args.pc_resolution, "grid_coefficient": args.grid_coefficient,
        "bfs_thickness": 0.3, "tfs_thickness": 0.3,
        "min_wall_length": args.min_wall_length, "min_wall_thickness": 0.05,
        "max_wall_thickness": 0.75, "exterior_walls_thickness": 0.2,
        "output_ifc": str(ifc),
        "ifc_project_name": work.parent.name, "ifc_project_long_name": "vid2bim room scan",
        "ifc_project_version": "1",
        "ifc_author_name": "vid2bim", "ifc_author_surname": "", "ifc_author_organization": "",
        "ifc_building_name": work.parent.name, "ifc_building_type": "Room",
        "ifc_building_phase": "Survey",
        # privacy: never write a real location into the model
        "ifc_site_latitude": [0, 0, 0], "ifc_site_longitude": [0, 0, 0], "ifc_site_elevation": 0.0,
        "material_for_objects": "Plaster",
    }
    cfg_path = work / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    runner = C2B_RUNNER.format(c2b=str(CLOUD2BIM), cfg=str(cfg_path))
    env = dict(os.environ, MPLBACKEND="Agg")
    log("running Cloud2BIM (slabs -> walls -> openings -> spaces)")
    r = subprocess.run([sys.executable, "-c", runner], cwd=work, env=env)
    if r.returncode != 0 or not ifc.exists():
        sys.exit("Cloud2BIM failed (see output above)")
    return ifc


# --------------------------------------------------------------------------- #
# 3. FreeCAD / 4. Blender meshes
# --------------------------------------------------------------------------- #
FREECAD_SCRIPT = r"""
import FreeCAD
ifc, out = {ifc!r}, {out!r}
doc = FreeCAD.newDocument("room")
# FreeCAD 1.x NativeIFC importer (the legacy importIFC breaks with ifcopenshell 0.8)
from nativeifc import ifc_import
ifc_import.insert(ifc, doc.Name, strategy=2, shapemode=0, switchwb=False, silent=True, singledoc=False)
# NativeIFC skips IfcFurnishingElement on import ("lazy loaded"): load furniture explicitly
from nativeifc import ifc_tools
prj = next(o for o in doc.Objects if getattr(o, "IfcClass", "") == "IfcProject")
ifcfile = ifc_tools.get_ifcfile(prj)
storeys = {{o.StepId: o for o in doc.Objects if getattr(o, "IfcClass", "") == "IfcBuildingStorey"}}
n = 0
for e in ifcfile.by_type("IfcFurnishingElement"):
    obj = ifc_tools.create_object(e, doc, ifcfile)
    for rel in e.ContainedInStructure:
        st = storeys.get(rel.RelatingStructure.id())
        if st is not None:
            st.addObject(obj)
    n += 1
print("[freecad] furniture loaded:", n)
doc.recompute()
doc.saveAs(out)
print("[freecad] objects:", len(doc.Objects), "->", out)
"""


def freecad_cmd():
    for c in ("freecadcmd", "FreeCADCmd", "freecad.cmd"):
        if shutil.which(c):
            return [c]
    if shutil.which("flatpak"):
        r = subprocess.run(["flatpak", "info", "org.freecad.FreeCAD"], capture_output=True)
        if r.returncode == 0:
            return ["flatpak", "run", "--command=FreeCADCmd", "org.freecad.FreeCAD"]
    return None


def to_freecad(ifc: Path) -> Path | None:
    cmd = freecad_cmd()
    if not cmd:
        log("FreeCAD not found - skipping .FCStd")
        return None
    out = ifc.with_suffix(".FCStd")
    script = ifc.parent / "freecad_import.py"
    script.write_text(FREECAD_SCRIPT.format(ifc=str(ifc), out=str(out)))
    subprocess.run(cmd + [str(script)])
    if out.exists():
        log(f"FreeCAD model: {out}")
        return out
    log("FreeCAD import failed (room.ifc is still valid - open it in FreeCAD manually)")
    return None


def ifc_meshes(ifc: Path) -> Path:
    import ifcopenshell
    import ifcopenshell.geom

    f = ifcopenshell.open(str(ifc))
    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    meshes = []
    for el in f.by_type("IfcProduct"):
        if not el.Representation or el.is_a("IfcOpeningElement") or el.is_a("IfcSpace"):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(st, el)
        except Exception:
            continue
        g = shape.geometry
        m = {
            "name": f"{el.is_a()}_{el.id()}" + (f"_{el.Name}" if getattr(el, "Name", None) else ""),
            "type": el.is_a(),
            "verts": np.round(np.array(g.verts), 5).tolist(),
            "faces": list(g.faces),
        }
        try:   # surface colour + material name (set by furniture.py)
            import ifcopenshell.util.element as ue
            if g.materials:
                d = g.materials[0].diffuse
                m["color"] = [float(d.r()), float(d.g()), float(d.b())] if hasattr(d, "r") else [float(x) for x in d][:3]
            mat = ue.get_material(el)
            if mat is not None and getattr(mat, "Name", None):
                m["material"] = mat.Name
        except Exception:
            pass
        meshes.append(m)
    out = ifc.parent / "bim_meshes.json"
    out.write_text(json.dumps({"meshes": meshes}))
    counts = {}
    for m in meshes:
        counts[m["type"]] = counts.get(m["type"], 0) + 1
    log(f"IFC elements: {counts}")
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="vid2bim scene folder with dense.ply")
    ap.add_argument("--voxel", type=float, default=0.02, help="cloud spacing fed to Cloud2BIM (m)")
    ap.add_argument("--pc-resolution", type=float, default=0.02)
    ap.add_argument("--grid-coefficient", type=float, default=5)
    ap.add_argument("--min-wall-length", type=float, default=0.5)
    ap.add_argument("--ceiling-height", type=float, default=2.6,
                    help="if the video did not capture the ceiling, add a synthetic one at this height (m)")
    ap.add_argument("--skip-bim", action="store_true", help="only align")
    a = ap.parse_args()

    import open3d as o3d

    o3d.utility.random.seed(0)  # deterministic RANSAC
    scene = Path(a.scene).resolve()
    work = scene / "bim"
    work.mkdir(exist_ok=True)
    dense = json.loads((scene / "dense.json").read_text())
    scale = dense["scale"]

    pcd = o3d.io.read_point_cloud(str(scene / "dense.ply"))
    pcd = pcd.voxel_down_sample(a.voxel)
    up, centers = camera_up(scene / "model_txt")
    R, t = align(pcd, up, centers * scale)
    P = np.asarray(pcd.points) @ R.T + t
    C = np.asarray(pcd.colors)

    # keep the room: between floor and a plausible ceiling, drop far outliers (views out of windows)
    zmax = np.percentile(P[:, 2], 99.5)
    keep = (P[:, 2] > -0.05) & (P[:, 2] < zmax + 0.05)
    P, C = P[keep], C[keep]
    log(f"aligned cloud: {len(P)} points, "
        f"x {np.ptp(P[:, 0]):.2f} m, y {np.ptp(P[:, 1]):.2f} m, height {np.percentile(P[:, 2], 99):.2f} m")

    aligned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    aligned.colors = o3d.utility.Vector3dVector(C)
    o3d.io.write_point_cloud(str(work / "aligned.ply"), aligned)

    # COLMAP world -> BIM (metres, z-up):  x_bim = R @ (scale * x_colmap) + t
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    (work / "align.json").write_text(json.dumps({"colmap_to_bim": T.tolist(), "scale": scale}, indent=2))

    np.savez_compressed(work / "aligned.npz", xyz=P.astype(np.float32), rgb=C.astype(np.float32))

    # Cloud2BIM needs a floor AND a ceiling slab. Phone videos often miss the ceiling:
    # then add a flat synthetic ceiling over the room footprint (marked grey, z = --ceiling-height).
    top = np.percentile(P[:, 2], 99.5)
    if top < 2.1:
        from scipy.ndimage import binary_closing, binary_fill_holes
        g = 0.02
        ij = np.floor(P[:, :2] / g).astype(int)
        lo = ij.min(0)
        occ = np.zeros(ij.max(0) - lo + 1, bool)
        occ[tuple((ij - lo).T)] = True
        occ = binary_fill_holes(binary_closing(occ, np.ones((25, 25)), iterations=2))
        cells = np.argwhere(occ) + lo
        ceil = np.c_[(cells + 0.5) * g, np.full(len(cells), a.ceiling_height)]
        log(f"⚠ no ceiling in the scan (top of cloud {top:.2f} m) - adding a synthetic ceiling at "
            f"{a.ceiling_height} m ({len(ceil)} pts); set the real height with --ceiling-height")
        P = np.vstack([P, ceil])
        C = np.vstack([C, np.full((len(ceil), 3), 0.5)])

    # furniture tops (desk, bed, shelves) look like extra floors to Cloud2BIM's slab finder:
    # drop horizontal surfaces between floor and ceiling (walls and floor/ceiling are kept)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    nz = np.abs(np.asarray(pc.normals)[:, 2])
    ztop = np.percentile(P[:, 2], 99.5)
    furn = (nz > 0.8) & (P[:, 2] > 0.12) & (P[:, 2] < ztop - 0.25)
    log(f"removed {furn.sum()} points on furniture tops")
    P, C = P[~furn], C[~furn]

    xyz = work / "room.xyz"
    with open(xyz, "w") as f:
        f.write("//X Y Z R G B\n")
        np.savetxt(f, np.c_[P, (C * 255).round()], fmt="%.3f %.3f %.3f %d %d %d")
    if a.skip_bim:
        return

    ifc = run_cloud2bim(work, xyz, a)
    log(f"IFC: {ifc}")
    ifc_meshes(ifc)
    to_freecad(ifc)


if __name__ == "__main__":
    main()
