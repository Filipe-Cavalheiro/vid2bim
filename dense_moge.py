#!/usr/bin/env python3
"""Dense, metric, geometrically consistent room model from a vid2blend scene.

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
    ap.add_argument("scene", help="vid2blend scene folder (contains model_txt/ and images/)")
    ap.add_argument("--frames", type=int, default=100, help="frames used for MVS + MoGe + fusion")
    ap.add_argument("--size", type=int, default=960, help="working image width (VRAM / speed)")
    ap.add_argument("--max-depth", type=float, default=6.0, help="ignore depth beyond this (metres)")
    ap.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size (metres)")
    ap.add_argument("--no-mvs", action="store_true", help="skip COLMAP MVS (MoGe aligned to sparse points only)")
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
    if not a.no_mvs:
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

        if valid_mvs.sum() > 2000:
            s, b, inl = robust_affine(d_moge[valid_mvs], d_mvs[valid_mvs])
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
        aligned = s * d_moge + b                        # COLMAP units
        depth = np.where(m_moge, aligned, 0).astype(np.float32)
        if d_mvs is not None:                           # trust MVS where it agrees with the fit
            agree = (d_mvs > 0) & (np.abs(d_mvs - aligned) < 0.05 * aligned)
            depth[agree] = d_mvs[agree]
        ratios.append(s)                                # COLMAP units per MoGe metre
        frames.append((im, depth, rgb, (fx, fy, cx, cy)))
        log(f"  [{k + 1}/{urec.num_images()}] {im.name}: scale {s:.3f} offset {b:+.3f} ({src}, {inl * 100:.0f}% inliers)")

    if not frames:
        raise SystemExit("no frames could be fused")
    m_per_unit = 1.0 / float(np.median(ratios))
    log(f"1 COLMAP unit = {m_per_unit:.3f} m")

    # ---- 4. TSDF fusion in metres -------------------------------------------
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=4 * a.voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for im, depth, rgb, (fx, fy, cx, cy) in frames:
        H, W = depth.shape
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        T = np.eye(4)
        T[:3, :3] = im.cam_from_world().rotation.matrix()
        T[:3, 3] = im.cam_from_world().translation * m_per_unit
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth * m_per_unit)),
            depth_scale=1.0, depth_trunc=a.max_depth, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, T)

    pcd = vol.extract_point_cloud()
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
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
