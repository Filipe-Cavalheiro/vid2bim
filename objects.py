#!/usr/bin/env python3
"""Find furniture in a vid2bim scene and measure it in the BIM frame.

  1. Grounding DINO (Apache-2.0) finds objects in the frames from text prompts,
     SAM 2.1 (Apache-2.0) turns each box into a precise mask
  2. the fused mesh (dense_mesh.ply) is ray-cast from every camera; each visible
     vertex inside a mask gets a vote for that class
  3. vertices voted mostly for one class are clustered (DBSCAN) into instances
  4. per instance: oriented box (yaw snapped to the room axes when close), floor
     contact, dominant colours, best photo crop

Inputs : <scene>/dense_mesh.ply, dense.json, mvs/ (from dense_moge.py), bim/align.json
Outputs: <scene>/objects/objects.json, <id>.ply (points), <id>.jpg (best crop)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_CLASSES = ["bed", "wardrobe", "desk", "chair", "table", "bookshelf",
                   "chest of drawers", "nightstand", "sofa", "television"]


FLOOR_STANDING = {"bed", "wardrobe", "desk", "chair", "table", "bookshelf", "chest of drawers",
                  "nightstand", "sofa"}


def log(msg):
    print(f"[objects] {msg}", flush=True)


def load_models(dev):
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                              Sam2Model, Sam2Processor)
    det_id, sam_id = "IDEA-Research/grounding-dino-base", "facebook/sam2.1-hiera-base-plus"
    det_p = AutoProcessor.from_pretrained(det_id)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(dev).eval()
    sam_p = Sam2Processor.from_pretrained(sam_id)
    sam = Sam2Model.from_pretrained(sam_id).to(dev).eval()
    return det_p, det, sam_p, sam


def detect(rgb, classes, models, dev, box_thr, text_thr):
    """-> list of (class, score, box xyxy, mask HxW bool)."""
    import torch
    from PIL import Image
    det_p, det, sam_p, sam = models
    img = Image.fromarray(rgb)
    text = ". ".join(classes) + "."
    inp = det_p(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = det(**inp)
    res = det_p.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[img.size[::-1]])[0]
    labels = res.get("text_labels", res.get("labels"))
    dets = []
    for score, box, lab in zip(res["scores"], res["boxes"], labels):
        lab = str(lab).strip()
        cls = next((c for c in classes if c in lab or lab in c), None) if lab else None
        if cls:
            dets.append((cls, float(score), box.tolist()))
    if not dets:
        return []
    boxes = [[d[2] for d in dets]]
    sin = sam_p(images=img, input_boxes=boxes, return_tensors="pt").to(dev)
    with torch.no_grad():
        sout = sam(**sin, multimask_output=False)
    masks = sam_p.post_process_masks(sout.pred_masks.cpu(), sin["original_sizes"])[0]
    masks = masks[:, 0].numpy().astype(bool)
    return [(c, s, b, m) for (c, s, b), m in zip(dets, masks)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES),
                    help="comma separated object names to look for")
    ap.add_argument("--frames", type=int, default=50, help="frames to run detection on")
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--min-vertices", type=int, default=1500)
    ap.add_argument("--reuse-votes", action="store_true", help="skip detection, reuse objects/votes.npz")
    a = ap.parse_args()

    import cv2
    import open3d as o3d
    import pycolmap
    import torch
    from sklearn.cluster import DBSCAN

    scene = Path(a.scene).resolve()
    out_dir = scene / "objects"
    out_dir.mkdir(exist_ok=True)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    scale = json.loads((scene / "dense.json").read_text())["scale"]
    T = np.array(json.loads((scene / "bim" / "align.json").read_text())["colmap_to_bim"])
    R_bim, t_bim = T[:3, :3] / scale, T[:3, 3]           # metres-frame -> BIM frame

    mesh = o3d.io.read_triangle_mesh(str(scene / "dense_mesh.ply"))
    V = np.asarray(mesh.vertices)                       # metres, COLMAP orientation
    VC = np.asarray(mesh.vertex_colors)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    rays = o3d.t.geometry.RaycastingScene()
    rays.add_triangles(tmesh)
    log(f"mesh: {len(V)} vertices")

    urec = pycolmap.Reconstruction(scene / "mvs" / "sparse")
    ims = sorted(urec.images.values(), key=lambda i: i.name)
    ims = [ims[i] for i in sorted({int(x) for x in np.linspace(0, len(ims) - 1, a.frames)})]

    votes = defaultdict(lambda: np.zeros(len(V), np.int32))
    seen = np.zeros(len(V), np.int32)
    best_crop = {}                                      # class -> list of (area, frame, box)
    cache = out_dir / "votes.npz"
    if a.reuse_votes and cache.exists():
        z = np.load(cache, allow_pickle=True)
        for c, col in zip(z["names"], z["votes"].T):
            votes[str(c)] = col
        seen = z["seen"]
        best_crop = json.loads(str(z["crops"]))
        ims = []
        log("reusing cached detections")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = load_models(dev)

    for k, im in enumerate(ims):
        bgr = cv2.imread(str(scene / "mvs" / "images" / im.name))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = detect(rgb, classes, models, dev, a.box_threshold, a.text_threshold)

        # project vertices; visibility via ray-cast depth
        fx, fy, cx, cy = im.camera.params[:4]
        Rcw = im.cam_from_world().rotation.matrix()
        tcw = im.cam_from_world().translation * scale
        Pc = V @ Rcw.T + tcw
        z = Pc[:, 2]
        u = fx * Pc[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * Pc[:, 1] / np.maximum(z, 1e-6) + cy
        inb = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            continue
        C = -Rcw.T @ tcw                                 # camera centre
        d = V[idx] - C
        dist = np.linalg.norm(d, axis=1)
        r = o3d.core.Tensor(np.c_[np.tile(C, (len(idx), 1)), d / dist[:, None]].astype(np.float32))
        hit = rays.cast_rays(r)["t_hit"].numpy()
        vis = idx[np.abs(hit - dist) < 0.03 + 0.01 * dist]
        seen[vis] += 1
        ui, vi = u[vis].astype(int), v[vis].astype(int)
        for cls, score, box, mask in dets:
            inside = vis[mask[vi, ui]]
            votes[cls][inside] += 1
            area = float(mask.sum())
            best_crop.setdefault(cls, []).append((area, im.name, box, len(inside)))
        log(f"  [{k + 1}/{len(ims)}] {im.name}: " + (", ".join(f"{c} {s:.2f}" for c, s, _, _ in dets) or "-"))

    if ims:
        names_ = list(votes)
        np.savez_compressed(cache, names=np.array(names_), seen=seen,
                            votes=np.stack([votes[c] for c in names_], 1) if names_ else np.zeros((len(V), 0)),
                            crops=json.dumps(best_crop))

    # ---- instances --------------------------------------------------------------
    Vb = V @ R_bim.T + t_bim                            # BIM frame (metres, z-up, floor 0)
    from scipy.spatial import cKDTree

    names = list(votes)
    V_votes = np.stack([votes[c] for c in names], 1) if names else np.zeros((len(V), 0), np.int32)
    best = V_votes.argmax(1) if names else np.zeros(len(V), int)
    bestv = V_votes[np.arange(len(V)), best] if names else np.zeros(len(V))
    labelled = (bestv >= 2) & (bestv >= 0.4 * np.maximum(seen, 1))

    # 1. raw instances per class
    inst = []                                           # [class index, vertex ids]
    for ci in range(len(names)):
        sel = np.nonzero(labelled & (best == ci))[0]      # each vertex -> one class only
        if len(sel) < a.min_vertices:
            continue
        lab = DBSCAN(eps=0.12, min_samples=8).fit_predict(Vb[sel])
        sizes = {l: (lab == l).sum() for l in set(lab) - {-1}}
        if not sizes:
            continue
        biggest = max(sizes.values())
        for l, n in sizes.items():
            if n >= max(a.min_vertices, 0.15 * biggest):
                inst.append([ci, sel[lab == l]])

    # 2. one real object with two labels (bookshelf/wardrobe, desk/table): if most points of
    #    one instance lie within 6 cm of another's, merge them and keep the label with the
    #    most votes over the merged points
    from shapely.geometry import MultiPoint

    def footprint(v):
        P = Vb[v]
        return MultiPoint(P[:: max(1, len(P) // 3000), :2]).convex_hull, np.percentile(P[:, 2], [2, 98])

    merged = True
    while merged:
        merged = False
        inst.sort(key=lambda x: -len(x[1]))
        trees = [cKDTree(Vb[v]) for _, v in inst]
        feet = [footprint(v) for _, v in inst]
        for i in range(len(inst)):
            for j in range(i + 1, len(inst)):
                d, _ = trees[i].query(Vb[inst[j][1]], k=1, distance_upper_bound=0.06)
                (fi, zi), (fj, zj) = feet[i], feet[j]
                z_overlap = min(zi[1], zj[1]) - max(zi[0], zj[0]) > 0.3 * min(np.ptp(zi), np.ptp(zj))
                plan = fi.intersection(fj).area / max(min(fi.area, fj.area), 1e-6)
                if np.mean(np.isfinite(d)) > 0.5 or (plan > 0.6 and z_overlap):
                    vid = np.union1d(inst[i][1], inst[j][1])
                    ci = int(V_votes[vid].sum(0).argmax())
                    inst[i] = [ci, vid]
                    del inst[j]
                    merged = True
                    break
            if merged:
                break

    # 3. measure
    objects = []
    for ci, vid in sorted(inst, key=lambda x: -len(x[1])):
        cls = names[ci]
        P = Vb[vid]
        # yaw from the footprint's principal axis, snapped to room axes when within 10°
        xy = P[:, :2] - P[:, :2].mean(0)
        _, ev = np.linalg.eigh(np.cov(xy.T))
        yaw = float(np.arctan2(ev[1, 1], ev[0, 1]))
        snap = round(yaw / (np.pi / 2)) * (np.pi / 2)
        if abs(yaw - snap) < np.radians(10):
            yaw = snap
        c, s = np.cos(-yaw), np.sin(-yaw)
        loc = xy @ np.array([[c, s], [-s, c]])
        lo, hi = np.percentile(loc, 2, 0), np.percentile(loc, 98, 0)
        zlo, zhi = np.percentile(P[:, 2], [1, 98])
        if zhi < 0.1:                                   # below / in the floor: noise
            continue
        # furniture stands on the floor; the scan often misses its bottom (occluded)
        if zlo < 0.15 or (cls in FLOOR_STANDING and zlo < 0.8):
            zlo = 0.0
        mid_loc = (lo + hi) / 2
        cxy = P[:, :2].mean(0) + mid_loc @ np.array([[c, -s], [s, c]])
        col = VC[vid]                                   # dominant colours, largest share first
        k3 = min(3, len(col))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3)
        _, lbl, cen = cv2.kmeans(col.astype(np.float32), k3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(lbl.ravel(), minlength=k3)
        order = np.argsort(-share)
        oid = f"{cls.replace(' ', '_')}_{sum(o['class'] == cls for o in objects) + 1}"
        objects.append({
            "id": oid, "class": cls,
            "center": [float(cxy[0]), float(cxy[1]), float((zlo + zhi) / 2)],
            "size": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(zhi - zlo)],
            "yaw_deg": float(np.degrees(yaw)),
            "floor_z": float(zlo),
            "colors": [[float(x) for x in cen[i]] for i in order],
            "color_share": [float(x) for x in share[order] / len(col)],
            "n_vertices": int(len(vid)),
        })
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.colors = o3d.utility.Vector3dVector(col)
        o3d.io.write_point_cloud(str(out_dir / f"{oid}.ply"), pc)
        np.save(out_dir / f"{oid}_vertices.npy", vid)

    # best photo per object: the frame where its points project largest while staying in view
    all_ims = sorted(urec.images.values(), key=lambda i: i.name)
    for o in objects:
        vid = np.load(out_dir / f"{o['id']}_vertices.npy")
        Pm = V[vid[:: max(1, len(vid) // 2000)]]          # metres frame, subsampled
        best_ = None
        for im in all_ims:
            fx, fy, cx, cy = im.camera.params[:4]
            W, H = im.camera.width, im.camera.height
            Pc = Pm @ im.cam_from_world().rotation.matrix().T + im.cam_from_world().translation * scale
            ok = Pc[:, 2] > 0.1
            if ok.mean() < 0.95:
                continue
            u = fx * Pc[:, 0] / Pc[:, 2] + cx
            v = fy * Pc[:, 1] / Pc[:, 2] + cy
            x0, x1, y0, y1 = np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 2), np.percentile(v, 98)
            inside = np.mean((u >= 0) & (u < W) & (v >= 0) & (v < H))
            score = (x1 - x0) * (y1 - y0) * inside ** 4
            if best_ is None or score > best_[0]:
                best_ = (score, im.name, [x0, y0, x1, y1])
        if best_:
            img = cv2.imread(str(scene / "mvs" / "images" / best_[1]))
            Hh, Ww = img.shape[:2]
            x0, y0, x1, y1 = best_[2]
            m = 0.1 * max(x1 - x0, y1 - y0)
            x0, y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
            x1, y1 = int(min(x1 + m, Ww)), int(min(y1 + m, Hh))
            cv2.imwrite(str(out_dir / f"{o['id']}.jpg"), img[y0:y1, x0:x1])
            o["crop"] = f"{o['id']}.jpg"
            o["crop_frame"] = best_[1]

    (out_dir / "objects.json").write_text(json.dumps({"frame": "BIM (metres, z-up, floor z=0)",
                                                      "objects": objects}, indent=2))
    for o in objects:
        log(f"{o['id']}: {o['size'][0]:.2f} x {o['size'][1]:.2f} x {o['size'][2]:.2f} m at "
            f"({o['center'][0]:.2f}, {o['center'][1]:.2f}), yaw {o['yaw_deg']:.0f}°, {o['n_vertices']} verts")
    log(f"wrote {out_dir / 'objects.json'} ({len(objects)} objects)")


if __name__ == "__main__":
    main()
