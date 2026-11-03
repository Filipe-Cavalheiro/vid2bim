#!/usr/bin/env python3
"""Dense, metric, geometrically consistent room model from a vid2bim scene.

COLMAP gives accurate camera poses but sparse points; COLMAP multi-view stereo
(MVS) is consistent across frames but leaves holes on plain walls; MoGe-2 (MIT,
github.com/microsoft/MoGe) predicts dense metric depth for every pixel but each
frame has its own small scale/offset error. We combine all three:

  1. undistort N evenly spaced frames and run COLMAP PatchMatch stereo (CUDA)
  2. per frame, fit MoGe depth to the MVS depth with a robust scale + offset
     (thousands of pixels; falls back to the sparse COLMAP points)
  3. per pixel: MVS depth where it exists and agrees, aligned MoGe elsewhere
  4. fuse all frames in a TSDF volume (Open3D) -> one clean surface
  5. metres per COLMAP unit = median MoGe (metric) / COLMAP depth ratio

Outputs (scene folder):
  dense.ply       - fused point cloud, COLMAP orientation, METRES (COLMAP coords * scale)
  dense_mesh.ply  - TSDF mesh, same frame
  dense.json      - {"scale": metres_per_colmap_unit, ...}
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[dense] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# MoGe-2 helpers (also used by reconstruct.py for the focal length)
# --------------------------------------------------------------------------- #
_MODEL = {}


def load_moge(name="Ruicheng/moge-2-vitl-normal"):
    import torch
    from moge.model.v2 import MoGeModel
    if name not in _MODEL:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL[name] = MoGeModel.from_pretrained(name).to(dev).eval()
    return _MODEL[name]


def moge_depth(model, rgb, fov_x=None):
    """rgb uint8 HxWx3 -> (depth HxW metres, valid mask, normalised intrinsics)."""
    import torch
    dev = next(model.parameters()).device
    t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        out = model.infer(t, fov_x=fov_x, use_fp16=dev.type == "cuda")
    depth = out["depth"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy().astype(bool) & np.isfinite(depth) & (depth > 0)
    return depth, mask, out["intrinsics"].float().cpu().numpy()


def estimate_focal(frames, width=960):
    """Median focal length (pixels, at the frames' full resolution) predicted by MoGe-2."""
    import cv2
    model = load_moge()
    fx = []
    for fn in frames:
        bgr = cv2.imread(str(fn))
        W0 = bgr.shape[1]
        h = int(round(bgr.shape[0] * width / W0)) // 2 * 2
        rgb = cv2.cvtColor(cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _, _, K = moge_depth(model, rgb)
        fx.append(float(K[0, 0]) * W0)   # intrinsics are normalised by width
    return float(np.median(fx))


# --------------------------------------------------------------------------- #
def read_colmap_array(path):
    """COLMAP depth/normal map (.bin): 'w&h&c&' header + float32 column-major data."""
    with open(path, "rb") as f:
        header = b""
        amp = 0
        while amp < 3:
            ch = f.read(1)
            header += ch
            amp += ch == b"&"
        w, h, c = map(int, header.decode().strip("&").split("&"))
        data = np.fromfile(f, np.float32)
    return data.reshape((w, h, c), order="F").transpose(1, 0, 2).squeeze()


def edge_mask(depth, rel=0.03):
    """True where depth is smooth (drops flying pixels at depth discontinuities)."""
    ok = np.ones_like(depth, bool)
    dy = np.abs(np.diff(depth, axis=0)) / np.maximum(depth[:-1], 1e-6) > rel
    dx = np.abs(np.diff(depth, axis=1)) / np.maximum(depth[:, :-1], 1e-6) > rel
    ok[:-1] &= ~dy
    ok[1:] &= ~dy
    ok[:, :-1] &= ~dx
    ok[:, 1:] &= ~dx
    return ok


def robust_affine(x, y, iters=4):
    """Fit y ~ a*x + b robustly (iteratively reweighted by residual MAD)."""
    keep = np.ones(len(x), bool)
    a, b = np.median(y / x), 0.0
    for _ in range(iters):
        A = np.c_[x[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = np.abs(a * x + b - y)
        keep = r < max(3 * np.median(r[keep]), 1e-6)
        if keep.sum() < 50:
            break
    return float(a), float(b), keep.mean()


def pick_frames(rec, n, min_points=15):
    imgs = sorted((im for im in rec.images.values() if im.has_pose and im.num_points3D >= min_points),
                  key=lambda im: im.name)
    if len(imgs) > n:
        imgs = [imgs[i] for i in np.linspace(0, len(imgs) - 1, n).round().astype(int)]
    return imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="vid2bim scene folder (contains model_txt/ and images/)")
    ap.add_argument("--frames", type=int, default=100, help="frames used for MVS + MoGe + fusion")
    ap.add_argument("--size", type=int, default=960, help="working image width (VRAM / speed)")
    ap.add_argument("--max-depth", type=float, default=6.0, help="ignore depth beyond this (metres)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size (metres)")
    ap.add_argument("--mvs", action="store_true",
                    help="also run COLMAP PatchMatch stereo (slow: ~15 s/frame on a 6 GB GPU)")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth pixel")
    ap.add_argument("--neighbours", type=int, default=4, help="frames on each side used for the consistency check")
    ap.add_argument("--consistency", type=float, default=0.03, help="max relative depth disagreement")
    ap.add_argument("--smooth", type=int, default=7, help="frames in the running median of the per-frame scale")
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal", help="MoGe-2 checkpoint (MIT)")
    a = ap.parse_args()

    import cv2
    import open3d as o3d
    import pycolmap

    scene = Path(a.scene).resolve()
    rec = pycolmap.Reconstruction(scene / "model_txt")
    sel = pick_frames(rec, a.frames)
    log(f"{rec.num_reg_images()} registered frames, using {len(sel)}")

    # ---- 1. undistort + PatchMatch stereo ------------------------------------
    mvs = scene / "mvs"
    if mvs.exists():
        shutil.rmtree(mvs)
    # model with only the selected frames, so MVS picks its source views among them
    keep = {im.frame_id for im in sel}
    for fid in [im.frame_id for im in rec.images.values() if im.has_pose and im.frame_id not in keep]:
        rec.deregister_frame(fid)
    subset = scene / "mvs_subset"
    subset.mkdir(exist_ok=True)
    rec.write(subset)
    und = pycolmap.UndistortCameraOptions()
    und.max_image_size = a.size
    pycolmap.undistort_images(mvs, subset, scene / "images",
                              image_names=[im.name for im in sel], undistort_options=und,
                              num_patch_match_src_images=12)
    shutil.rmtree(subset)
    urec = pycolmap.Reconstruction(mvs / "sparse")
    have_mvs = False
    if a.mvs:
        try:
            log("COLMAP PatchMatch stereo (GPU)")
            pm = pycolmap.PatchMatchOptions()
            pm.max_image_size = a.size
            pm.geom_consistency = True
            pm.cache_size = 8.0
            pm.filter_min_ncc = 0.1
            pycolmap.patch_match_stereo(mvs, options=pm)
            have_mvs = True
        except Exception as e:  # no CUDA / out of memory
            log(f"MVS failed ({e}); continuing with MoGe aligned to sparse points")

    # ---- 2+3. per-frame aligned depth -----------------------------------------
    model = load_moge(a.model)
    frames, ratios = [], []
    for k, im in enumerate(sorted(urec.images.values(), key=lambda i: i.name)):
        cam = im.camera
        bgr = cv2.imread(str(mvs / "images" / im.name))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fx, fy, cx, cy = cam.params[:4] if len(cam.params) >= 4 else (cam.params[0],) * 2 + tuple(cam.params[1:3])
        d_moge, m_moge, _ = moge_depth(model, rgb, fov_x=float(np.degrees(2 * np.arctan(W / (2 * fx)))))
        m_moge &= edge_mask(d_moge)

        d_mvs = None
        dm_path = mvs / "stereo" / "depth_maps" / f"{im.name}.geometric.bin"
        if have_mvs and dm_path.exists():
            d_mvs = read_colmap_array(dm_path)
            if d_mvs.shape != d_moge.shape:
                d_mvs = cv2.resize(d_mvs, (W, H), interpolation=cv2.INTER_NEAREST)
        valid_mvs = (d_mvs > 0) & m_moge if d_mvs is not None else np.zeros_like(m_moge)

        if valid_mvs.sum() > 2000:            # scale only: MoGe depth is metric up to scale
            r = d_mvs[valid_mvs] / d_moge[valid_mvs]
            s, b = float(np.median(r)), 0.0
            inl = float(np.mean(np.abs(r / s - 1) < 0.05))
            src = f"MVS {valid_mvs.mean() * 100:.0f}% px"
        else:
            # fall back to this frame's sparse COLMAP points (scale only)
            R = im.cam_from_world().rotation.matrix()
            t = im.cam_from_world().translation
            p2d = [p for p in im.points2D if p.has_point3D()]
            if len(p2d) < 20:
                log(f"  {im.name}: no MVS and too few sparse points, skipped")
                continue
            uv = np.array([p.xy for p in p2d])
            X = np.array([urec.points3D[p.point3D_id].xyz for p in p2d])
            z = (X @ R.T + t)[:, 2]
            ui, vi = np.clip(uv[:, 0].astype(int), 0, W - 1), np.clip(uv[:, 1].astype(int), 0, H - 1)
            ok = m_moge[vi, ui] & (z > 0)
            if ok.sum() < 20:
                log(f"  {im.name}: not enough overlap, skipped")
                continue
            s, b, inl = float(np.median(z[ok] / d_moge[vi, ui][ok])), 0.0, 1.0
            src = f"{ok.sum()} sparse pts"
        if s <= 0:
            continue
        ratios.append(s)                                # COLMAP units per MoGe metre
        frames.append([im, d_moge, m_moge, d_mvs, rgb, (fx, fy, cx, cy), s])
        log(f"  [{k + 1}/{urec.num_images()}] {im.name}: scale {s:.3f} offset {b:+.3f} ({src}, {inl * 100:.0f}% inliers)")

    if not frames:
        raise SystemExit("no frames could be fused")
    # the true metres->COLMAP factor is one constant; per-frame values are MoGe's own
    # scale error + noise. Running median over neighbouring frames keeps the first, drops the second.
    raw = np.array([f[6] for f in frames])
    half = max(a.smooth // 2, 0)
    for i, f in enumerate(frames):
        f[6] = float(np.median(raw[max(0, i - half): i + half + 1]))
    log(f"per-frame scale: median {np.median(raw):.3f}, spread ±{np.median(np.abs(raw / np.median(raw) - 1)) * 100:.0f}%")
    m_per_unit = 1.0 / float(np.median(ratios))
    log(f"1 COLMAP unit = {m_per_unit:.3f} m")

    # ---- 4. multi-view consistent fusion in metres ------------------------------
    # (Open3D's TSDF integration returns nothing in the 0.20 / Python 3.14 build, so
    #  points are kept only where >= 2 neighbouring frames agree on the depth)
    views = []
    for im, d_moge, m_moge, d_mvs, rgb, (fx, fy, cx, cy), s in frames:
        depth = np.where(m_moge, s * d_moge, 0).astype(np.float32)        # COLMAP units
        if d_mvs is not None:                           # trust MVS where it agrees
            agree = (d_mvs > 0) & (np.abs(d_mvs - depth) < 0.05 * depth)
            depth[agree] = d_mvs[agree]
        depth *= m_per_unit                             # metres
        depth[depth > a.max_depth] = 0
        R = im.cam_from_world().rotation.matrix()
        t = im.cam_from_world().translation * m_per_unit
        views.append((depth, rgb, np.array([fx, fy, cx, cy]), R, t))

    def project(P, view):
        depth, _, (fx, fy, cx, cy), R, t = view
        Pc = P @ R.T + t
        z = Pc[:, 2]
        ok = z > 0.05
        u = np.full(len(P), -1, int)
        v = np.full(len(P), -1, int)
        u[ok] = np.round(fx * Pc[ok, 0] / z[ok] + cx).astype(int)
        v[ok] = np.round(fy * Pc[ok, 1] / z[ok] + cy).astype(int)
        H, W = depth.shape
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = np.zeros(len(P), np.float32)
        d[ok] = depth[v[ok], u[ok]]
        return z, d, ok & (d > 0)

    pts, cols = [], []
    for i, (depth, rgb, (fx, fy, cx, cy), R, t) in enumerate(views):
        vv, uu = np.nonzero(depth[::a.stride, ::a.stride] > 0)
        vv, uu = vv * a.stride, uu * a.stride
        z = depth[vv, uu]
        Pc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], 1)
        P = (Pc - t) @ R                                 # world, metres
        support = np.zeros(len(P), np.int32)
        for j in range(max(0, i - a.neighbours), min(len(views), i + a.neighbours + 1)):
            if j == i:
                continue
            zj, dj, okj = project(P, views[j])
            support += okj & (np.abs(zj - dj) < a.consistency * zj)
        keep = support >= 2
        pts.append(P[keep].astype(np.float32))
        cols.append(rgb[vv[keep], uu[keep]].astype(np.float32) / 255)
    P = np.concatenate(pts)
    C = np.concatenate(cols)
    log(f"{len(P)} consistent points from {len(views)} frames")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))
    pcd = pcd.voxel_down_sample(a.voxel / 2)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * a.voxel, max_nn=30))
    cams = np.array([-v[3].T @ v[4] for v in views])
    pcd.orient_normals_towards_camera_location(cams.mean(0))
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.08))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    o3d.io.write_point_cloud(str(scene / "dense.ply"), pcd)
    o3d.io.write_triangle_mesh(str(scene / "dense_mesh.ply"), mesh)
    (scene / "dense.json").write_text(json.dumps({
        "scale": m_per_unit, "frames_used": len(frames), "points": len(pcd.points),
        "mvs": have_mvs, "model": a.model,
        "note": "dense.ply / dense_mesh.ply = COLMAP world coordinates * scale (metres)",
    }, indent=2))
    log(f"wrote dense.ply ({len(pcd.points)} points) and dense_mesh.ply ({len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    main()
