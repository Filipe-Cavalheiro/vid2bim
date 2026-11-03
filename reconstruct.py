#!/usr/bin/env python3
"""vid2blend - turn one or more videos into a COLMAP sparse reconstruction
and a ready-to-open Blender scene (colored point cloud + animated cameras).

Unix port of polyfjord's batch_reconstruct.bat workflow:
    ffmpeg frames -> SIFT features -> sequential matching -> global/incremental mapper
plus a direct Blender export (no add-on required). The TXT model it writes is
also importable with SBCV's Blender Photogrammetry Importer add-on.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BLENDER_SCRIPT = HERE / "blender_build_scene.py"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".m2ts"}


def log(msg: str) -> None:
    print(f"[vid2blend] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[vid2blend] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_dir():
            videos += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            videos.append(p)
        else:
            die(f"input not found: {item}")
    if not videos:
        die("no video files found in the given inputs")
    return videos


def safe_name(path: Path) -> str:
    keep = "-_."
    return "".join(c if (c.isalnum() and c.isascii()) or c in keep else "_" for c in path.stem) or "video"


def probe_fps(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        num, den = out.split("/")
        return float(num) / float(den) if float(den) else 30.0
    except ValueError:
        return 30.0


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def extract_frames(video: Path, out_dir: Path, fps: float | None, max_size: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if max_size > 0:
        # downscale only if the longest side exceeds max_size
        filters.append(
            f"scale='if(gte(iw,ih),min(iw,{max_size}),-2)':'if(gte(iw,ih),-2,min(ih,{max_size}))'"
        )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-i", str(video)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    # passthrough avoids duplicated frames from variable-frame-rate phone videos
    cmd += ["-fps_mode", "cfr" if fps else "passthrough", "-qscale:v", "2",
            str(out_dir / "frame_%06d.jpg")]
    if subprocess.run(cmd).returncode != 0:
        die(f"ffmpeg failed on {video}")
    n = len(list(out_dir.glob("frame_*.jpg")))
    if n == 0:
        die(f"no frames extracted from {video}")
    return n


def cross_video_pairs(images_root: Path, groups: list[str], per_video: int) -> list[tuple[str, str]]:
    """Pairs of evenly spaced keyframes between every two videos, so separate
    clips of the same place end up in one shared coordinate system."""
    keyframes = {}
    for g in groups:
        frames = sorted(p.name for p in (images_root / g).glob("frame_*.jpg"))
        step = max(1, len(frames) // per_video)
        keyframes[g] = [f"{g}/{f}" for f in frames[::step][:per_video]]
    pairs = []
    for a, b in itertools.combinations(groups, 2):
        pairs += [(x, y) for x in keyframes[a] for y in keyframes[b]]
    return pairs


def clean_model(rec, min_points: int = 15) -> None:
    """Remove what makes a model look broken in Blender:
    frames registered with (almost) no 3D points - their poses are unreliable and the global
    mapper can put them millions of units away - and isolated 3D points far from the scene."""
    import numpy as np
    bad_frames = {im.frame_id for im in rec.images.values()
                  if im.has_pose and im.num_points3D < min_points}
    for fid in bad_frames:
        rec.deregister_frame(fid)
    if rec.num_points3D():
        ids = list(rec.point3D_ids())
        P = np.array([rec.points3D[i].xyz for i in ids])
        d = np.linalg.norm(P - np.median(P, axis=0), axis=1)
        limit = 3 * np.percentile(d, 95)
        far = [i for i, di in zip(ids, d) if di > limit]
        for i in far:
            rec.delete_point3D(i)
    else:
        far = []
    log(f"      cleaned model: dropped {len(bad_frames)} unreliable frames, {len(far)} far points")


def known_focal(images: Path, groups: list[str], args):
    """(focal_px, width, height) from --hfov, or estimated with MoGe-2 if installed, else None."""
    import cv2
    import numpy as np
    first = sorted((images / groups[0]).glob("frame_*.jpg"))[0]
    h, w = cv2.imread(str(first)).shape[:2]
    if args.hfov:
        f = w / 2 / math.tan(math.radians(args.hfov) / 2)
        log(f"      focal length from --hfov {args.hfov}°: {f:.0f} px (fixed)")
        return f, w, h
    if args.no_focal_estimate:
        return None
    try:
        from dense_moge import estimate_focal
    except ImportError:
        log("      MoGe not installed - COLMAP will estimate the focal length")
        return None
    frames = []
    for g in groups:
        fs = sorted((images / g).glob("frame_*.jpg"))
        frames += [fs[i] for i in sorted({int(x) for x in np.linspace(0, len(fs) - 1, 12)})]
    f = estimate_focal(frames)
    log(f"      focal length estimated by MoGe-2: {f:.0f} px, "
        f"hfov {math.degrees(2 * math.atan(w / 2 / f)):.1f}° (fixed; override with --hfov)")
    return f, w, h


def run_colmap(scene: Path, groups: list[str], args) -> Path | None:
    import pycolmap

    images = scene / "images"
    db = scene / "database.db"
    sparse = scene / "sparse"
    sparse.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()
    device = pycolmap.Device.cpu if args.cpu else pycolmap.Device.auto

    log("[2/4] feature extraction")
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = args.max_image_size
    ext.use_gpu = not args.cpu
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.camera_model
    focal = known_focal(images, groups, args)
    if focal:
        # one shared, fixed focal length: self-calibration from handheld video is unreliable
        w, h = focal[1], focal[2]
        reader.camera_params = f"{focal[0]},{w / 2},{h / 2}" + ",0" * (args.camera_model != "SIMPLE_PINHOLE")
    pycolmap.extract_features(
        db, images,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,  # one shared camera per video
        reader_options=reader, extraction_options=ext, device=device,
    )

    log(f"[3/4] sequential matching (overlap {args.overlap})")
    match = pycolmap.FeatureMatchingOptions()
    match.use_gpu = not args.cpu
    seq = pycolmap.SequentialPairingOptions()
    seq.overlap = args.overlap
    pycolmap.match_sequential(db, matching_options=match, pairing_options=seq, device=device)

    if len(groups) > 1:
        pairs = cross_video_pairs(images, groups, args.cross_keyframes)
        log(f"      cross-video matching ({len(pairs)} keyframe pairs)")
        pair_file = scene / "cross_pairs.txt"
        pair_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
        imp = pycolmap.ImportedPairingOptions()
        imp.match_list_path = str(pair_file)
        pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imp, device=device)

    if args.mapper == "global":
        log("[4/4] global mapping (GLOMAP)")
        gopts = pycolmap.GlobalPipelineOptions()
        gopts.mapper.bundle_adjustment.refine_focal_length = not focal
        recs = pycolmap.global_mapping(db, images, sparse, options=gopts)
    else:
        log("[4/4] incremental mapping (this is the slow part)")
        opts = pycolmap.IncrementalPipelineOptions()
        opts.num_threads = os.cpu_count() or -1
        opts.ba_refine_focal_length = not focal
        recs = pycolmap.incremental_mapping(db, images, sparse, options=opts)
    if not recs:
        log("✖ mapper produced no model (too little texture/parallax? try more frames or --overlap)")
        return None

    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    clean_model(best)
    total = sum(len(list((images / g).glob("*.jpg"))) for g in groups)
    log(f"      {len(recs)} model(s); using #{best_id}: "
        f"{best.num_reg_images()}/{total} frames registered, {best.num_points3D()} points")
    if len(recs) > 1:
        log("      (other models are kept in sparse/<n>; the scene broke apart somewhere)")

    txt = scene / "model_txt"
    txt.mkdir(exist_ok=True)
    best.write_text(txt)  # cameras.txt / images.txt / points3D.txt (add-on compatible)
    best.export_PLY(scene / "points.ply")
    return txt


def build_blend(scene: Path, txt: Path, meta_path: Path, blender: str) -> None:
    out = scene / f"{scene.name}.blend"
    cmd = [blender, "--background", "--factory-startup", "--python", str(BLENDER_SCRIPT), "--",
           "--model", str(txt), "--meta", str(meta_path), "--output", str(out)]
    bim = scene / "bim"
    for flag, f in (("--align", bim / "align.json"), ("--dense", bim / "aligned.npz"),
                    ("--bim-meshes", bim / "bim_meshes.json")):
        if f.exists():
            cmd += [flag, str(f)]
    if subprocess.run(cmd).returncode != 0 or not out.exists():
        log("✖ Blender export failed (the COLMAP model in model_txt/ is still usable)")
    else:
        log(f"✔ Blender scene: {out}")


def run_bim(scene: Path, args) -> None:
    """Dense metric cloud (MoGe-2) -> aligned room -> IFC (Cloud2BIM) -> FreeCAD."""
    py = sys.executable
    log("[bim 1/2] dense metric cloud (MoGe-2)")
    if subprocess.run([py, str(HERE / "dense_moge.py"), str(scene),
                       "--frames", str(args.dense_frames)]).returncode != 0:
        log("✖ dense step failed - skipping BIM")
        return
    log("[bim 2/2] align room, Cloud2BIM -> IFC, FreeCAD")
    if subprocess.run([py, str(HERE / "room_bim.py"), str(scene)]).returncode != 0:
        log("✖ BIM step failed (dense.ply / bim/aligned.ply are still usable)")


# --------------------------------------------------------------------------- #
def process_scene(scene: Path, videos: list[Path], args, blender: str | None) -> None:
    done_marker = scene / ".done"
    if done_marker.exists() and args.rebuild_blend and not args.force:
        if args.bim:
            run_bim(scene, args)
        if blender:
            build_blend(scene, scene / "model_txt", scene / "scene.json", blender)
        return
    if done_marker.exists() and not args.force:
        log(f"↻ skipping '{scene.name}' (already reconstructed; use --force to redo)")
        return
    if scene.exists():
        shutil.rmtree(scene)
    (scene / "images").mkdir(parents=True)

    t0 = time.time()
    meta = {"videos": []}
    groups = []
    for v in videos:
        g = safe_name(v)
        while g in groups:
            g += "_"
        groups.append(g)
        log(f"[1/4] extracting frames: {v.name}")
        n = extract_frames(v, scene / "images" / g, args.fps, args.max_image_size)
        fps = args.fps or probe_fps(v)
        log(f"      {n} frames @ {fps:g} fps")
        meta["videos"].append({"group": g, "source": v.name, "fps": fps, "frames": n})
    meta["images_dir"] = str((scene / "images").resolve())
    meta_path = scene / "scene.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    txt = run_colmap(scene, groups, args)
    if txt is None:
        return
    if args.bim:
        run_bim(scene, args)
    if blender:
        build_blend(scene, txt, meta_path, blender)
    if not args.keep_db:
        (scene / "database.db").unlink(missing_ok=True)
    done_marker.touch()
    log(f"✔ finished '{scene.name}' in {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Video(s) -> COLMAP sparse point cloud + camera track -> .blend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="video files and/or folders containing videos")
    ap.add_argument("-o", "--output", default=str(HERE / "scenes"), help="output folder")
    ap.add_argument("-m", "--merge", metavar="NAME",
                    help="reconstruct ALL videos together into one scene called NAME "
                         "(default: one separate scene per video)")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames per second to extract (default: every frame, like the original script)")
    ap.add_argument("--max-image-size", type=int, default=4096, help="max frame side in pixels")
    ap.add_argument("--overlap", type=int, default=15, help="sequential matcher overlap")
    ap.add_argument("--cross-keyframes", type=int, default=40,
                    help="with --merge: keyframes per video matched against every other video")
    ap.add_argument("--mapper", choices=["global", "incremental"], default="global",
                    help="global (GLOMAP) is faster and copes better with handheld pans; "
                         "incremental is the original COLMAP mapper")
    ap.add_argument("--hfov", type=float, default=None,
                    help="horizontal field of view of the video in degrees (fixes the focal length); "
                         "default: estimated with MoGe-2 when installed")
    ap.add_argument("--no-focal-estimate", action="store_true",
                    help="let COLMAP self-calibrate the focal length instead")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL",
                    help="COLMAP camera model (SIMPLE_RADIAL, OPENCV, PINHOLE, ...)")
    ap.add_argument("--cpu", action="store_true", help="disable CUDA")
    ap.add_argument("--blender", default=shutil.which("blender"),
                    help="Blender executable used to write the .blend")
    ap.add_argument("--no-blend", action="store_true", help="skip the .blend export")
    ap.add_argument("--keep-db", action="store_true", help="keep COLMAP database.db")
    ap.add_argument("--bim", action="store_true",
                    help="also build a BIM model of the room: dense cloud (MoGe-2) -> IFC (Cloud2BIM) "
                         "-> FreeCAD .FCStd, and add it to the .blend")
    ap.add_argument("--dense-frames", type=int, default=60, help="with --bim: frames used for the dense cloud")
    ap.add_argument("--rebuild-blend", action="store_true",
                    help="only re-create the .blend (and with --bim the BIM) of already reconstructed scenes")
    ap.add_argument("--force", action="store_true", help="redo scenes that already exist")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found - install it (e.g. sudo apt install ffmpeg)")
    blender = None if args.no_blend else args.blender
    if not args.no_blend and not blender:
        log("Blender not found - skipping .blend export (pass --blender /path/to/blender)")

    videos = collect_videos(args.inputs)
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    jobs = ([(out / args.merge, videos)] if args.merge
            else [(out / safe_name(v), [v]) for v in videos])
    for i, (scene, vids) in enumerate(jobs, 1):
        log(f"=== [{i}/{len(jobs)}] scene '{scene.name}' ({len(vids)} video(s)) ===")
        process_scene(scene, vids, args, blender)
    log(f"All jobs finished - results are in {out}")


if __name__ == "__main__":
    main()
