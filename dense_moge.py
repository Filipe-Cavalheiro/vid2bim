#!/usr/bin/env python3
"""Dense, metric point cloud from a vid2blend scene.

COLMAP gives accurate camera poses but only sparse points (and almost none on
plain walls). MoGe-2 (MIT, github.com/microsoft/MoGe) predicts a dense metric
depth map per frame. We fuse the two:

  1. run MoGe-2 on N evenly spaced registered frames (focal length from COLMAP)
  2. per frame, scale MoGe depth to COLMAP units using the frame's own sparse
     points (median ratio) -> geometrically consistent across frames
  3. back-project with the COLMAP pose, voxel-merge, remove outliers
  4. metres per COLMAP unit = median of MoGe metric depth / COLMAP depth

Output (in the scene folder):
  dense.ply   - fused cloud in COLMAP world coordinates, already in METRES
                (i.e. COLMAP coords * scale)
  dense.json  - {"scale": metres_per_colmap_unit, ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[dense] {msg}", flush=True)


_MODEL = {}


def load_moge(name="Ruicheng/moge-2-vitl-normal"):
    import torch
    from moge.model.v2 import MoGeModel
    if name not in _MODEL:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL[name] = MoGeModel.from_pretrained(name).to(dev).eval()
    return _MODEL[name]


def estimate_focal(frames, width=960):
    """Median focal length (pixels, at the frames' full resolution) predicted by MoGe-2."""
    import cv2
    import torch
    model = load_moge()
    dev = next(model.parameters()).device
    fx = []
    for fn in frames:
        bgr = cv2.imread(str(fn))
        W0 = bgr.shape[1]
        h = int(round(bgr.shape[0] * width / W0)) // 2 * 2
        rgb = cv2.cvtColor(cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            out = model.infer(t, use_fp16=dev.type == "cuda")
        fx.append(float(out["intrinsics"][0, 0]) * W0)   # intrinsics are normalised by width
    return float(np.median(fx))


def pick_images(rec, n):
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    if len(imgs) <= n:
        return imgs
    idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
    return [imgs[i] for i in idx]


def edge_mask(depth, rel=0.03):
    """True where depth is smooth (drops flying pixels at depth edges)."""
    d = depth
    ok = np.ones_like(d, bool)
    for axis in (0, 1):
        diff = np.abs(np.diff(d, axis=axis)) / np.maximum(d[:-1] if axis == 0 else d[:, :-1], 1e-6)
        bad = diff > rel
        if axis == 0:
            ok[:-1] &= ~bad
            ok[1:] &= ~bad
        else:
            ok[:, :-1] &= ~bad
            ok[:, 1:] &= ~bad
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="vid2blend scene folder (contains model_txt/ and images/)")
    ap.add_argument("--frames", type=int, default=60, help="number of frames to run MoGe on")
    ap.add_argument("--width", type=int, default=960, help="MoGe input width (VRAM/speed)")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth pixel")
    ap.add_argument("--max-depth", type=float, default=7.0, help="drop points farther than this (metres)")
    ap.add_argument("--voxel", type=float, default=0.01, help="voxel size for merging (metres)")
    ap.add_argument("--model", default="Ruicheng/moge-2-vitl-normal", help="MoGe-2 checkpoint (MIT)")
    a = ap.parse_args()

    import cv2
    import open3d as o3d
    import pycolmap
    import torch
    from moge.model.v2 import MoGeModel

    scene = Path(a.scene).resolve()
    rec = pycolmap.Reconstruction(scene / "model_txt")
    img_root = scene / "images"
    sel = pick_images(rec, a.frames)
    log(f"{rec.num_reg_images()} registered frames, using {len(sel)}")

    model = load_moge(a.model)
    dev = next(model.parameters()).device

    clouds, colors, ratios_metric = [], [], []
    for k, im in enumerate(sel):
        cam = im.camera
        bgr = cv2.imread(str(img_root / im.name))
        H0, W0 = bgr.shape[:2]
        s = a.width / W0
        W, H = a.width, int(round(H0 * s)) // 2 * 2
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        fx = cam.params[0]
        fov_x = float(np.degrees(2 * np.arctan(W0 / (2 * fx))))

        t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float() / 255
        with torch.no_grad():
            out = model.infer(t, fov_x=fov_x, use_fp16=dev.type == "cuda")
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool) & np.isfinite(depth)

        # scale: COLMAP depth of this frame's sparse points vs MoGe depth there
        R = im.cam_from_world().rotation.matrix()
        tr = im.cam_from_world().translation
        p2d = [p for p in im.points2D if p.has_point3D()]
        if len(p2d) < 20:
            log(f"  {im.name}: only {len(p2d)} sparse points, skipped")
            continue
        uv = np.array([p.xy for p in p2d]) * s
        X = np.array([rec.points3D[p.point3D_id].xyz for p in p2d])
        z_col = (X @ R.T + tr)[:, 2]
        ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
        vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
        z_moge = depth[vi, ui]
        good = mask[vi, ui] & (z_col > 0) & (z_moge > 0)
        if good.sum() < 20:
            log(f"  {im.name}: not enough overlap, skipped")
            continue
        r = z_col[good] / z_moge[good]          # COLMAP units per metre (this frame)
        ratio = float(np.median(r))
        spread = float(np.median(np.abs(r / ratio - 1)))
        ratios_metric.append(ratio)

        # back-project MoGe depth with COLMAP intrinsics (scaled) and pose
        keep = mask & edge_mask(depth) & (depth < a.max_depth)
        sub = np.zeros_like(keep)
        sub[:: a.stride, :: a.stride] = True
        keep &= sub
        v, u = np.nonzero(keep)
        z = depth[v, u] * ratio
        f = fx * s
        cx, cy = cam.params[1] * s, cam.params[2] * s
        pc = np.stack([(u + 0.5 - cx) / f * z, (v + 0.5 - cy) / f * z, z], 1)
        pw = (pc - tr) @ R                      # R^T (pc - t)
        clouds.append(pw.astype(np.float32))
        colors.append(rgb[v, u].astype(np.float32) / 255)
        log(f"  [{k + 1}/{len(sel)}] {im.name}: {len(pw)} pts, scale {ratio:.3f} (±{spread * 100:.0f}%)")

    if not clouds:
        raise SystemExit("no frames could be fused")
    unit_per_m = float(np.median(ratios_metric))
    m_per_unit = 1.0 / unit_per_m
    P = np.concatenate(clouds) * m_per_unit
    C = np.concatenate(colors)
    log(f"fused {len(P)} raw points; 1 COLMAP unit = {m_per_unit:.3f} m")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))
    pcd = pcd.voxel_down_sample(a.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(str(scene / "dense.ply"), pcd)
    (scene / "dense.json").write_text(json.dumps({
        "scale": m_per_unit,
        "frames_used": len(clouds),
        "points": len(pcd.points),
        "model": a.model,
        "note": "dense.ply = COLMAP world coordinates * scale (metres)",
    }, indent=2))
    log(f"wrote {scene / 'dense.ply'} ({len(pcd.points)} points)")


if __name__ == "__main__":
    main()
