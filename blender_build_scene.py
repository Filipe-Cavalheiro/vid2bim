"""Build a .blend from a COLMAP TXT model. Run through Blender:

    blender --background --factory-startup --python blender_build_scene.py -- \
        --model scenes/x/model_txt --meta scenes/x/scene.json --output scenes/x/x.blend

Creates, under one "COLMAP" empty (rotate/scale that to level the scene):
  * PointCloud  - colored sparse points (geometry nodes, works in Solid/Eevee/Cycles)
  * Cam_<video> - one camera per video, animated along the solved track, with the
                  extracted frames as its background image sequence
  * Track_<video> - polyline of the camera path
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import bpy
from mathutils import Matrix, Quaternion, Vector


# --------------------------------------------------------------------------- #
# COLMAP TXT readers
# --------------------------------------------------------------------------- #
def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            el = line.split()
            cams[int(el[0])] = (el[1], int(el[2]), int(el[3]), [float(x) for x in el[4:]])
    return cams


def read_images(path):
    imgs = []
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    for line in lines[0::2]:  # every other line holds the 2D points
        el = line.split()
        if len(el) < 10:
            continue
        q = Quaternion([float(x) for x in el[1:5]])
        t = Vector([float(x) for x in el[5:8]])
        imgs.append((int(el[8]), " ".join(el[9:]), q, t))
    return imgs


def read_points(path):
    xyz, rgb = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            el = line.split()
            xyz.append((float(el[1]), float(el[2]), float(el[3])))
            rgb.append((int(el[4]) / 255, int(el[5]) / 255, int(el[6]) / 255))
    return xyz, rgb


# --------------------------------------------------------------------------- #
def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_point_cloud(xyz, rgb, parent, radius):
    mesh = bpy.data.meshes.new("PointCloud")
    mesh.vertices.add(len(xyz))
    mesh.vertices.foreach_set("co", [c for p in xyz for c in p])
    col = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    col.data.foreach_set("color", [v for c in rgb for v in (*map(srgb_to_linear, c), 1.0)])
    mesh.update()
    obj = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent

    # material: emit the point color so it looks like the photo under any lighting
    mat = bpy.data.materials.new("PointColor")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Col"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    # geometry nodes: put a tiny ico sphere on every vertex and realize it, so
    # the color attribute survives and Solid/Eevee/Cycles all display it
    ng = bpy.data.node_groups.new("PointsToSpheres", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    rs = ng.interface.new_socket("Point Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    rs.default_value, rs.min_value = radius, 0.0
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    ico = n.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 1
    inst = n.new("GeometryNodeInstanceOnPoints")
    realize = n.new("GeometryNodeRealizeInstances")
    setmat = n.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    l = ng.links
    l.new(gin.outputs[0], inst.inputs["Points"])
    l.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    l.new(gin.outputs[1], inst.inputs["Scale"])
    l.new(inst.outputs["Instances"], realize.inputs["Geometry"])
    l.new(realize.outputs["Geometry"], setmat.inputs["Geometry"])
    l.new(setmat.outputs["Geometry"], gout.inputs[0])
    mod = obj.modifiers.new("Points", "NODES")
    mod.node_group = ng
    return obj


def camera_intrinsics(cam_data, model, w, h, params):
    if model.startswith("SIMPLE") or model in ("RADIAL", "RADIAL_FISHEYE", "FOV"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    f = (fx + fy) / 2
    cam_data.sensor_width = cam_data.sensor_height = 36.0
    if w >= h:
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.lens = f * 36.0 / w
    else:
        cam_data.sensor_fit = "VERTICAL"
        cam_data.lens = f * 36.0 / h
    m = max(w, h)
    cam_data.shift_x = -(cx - w / 2) / m
    cam_data.shift_y = (cy - h / 2) / m


def set_fcurves(obj, data_path, frames, values_per_comp):
    """Fast keyframe insertion (thousands of frames) with linear interpolation."""
    action = obj.animation_data.action
    for i, values in enumerate(values_per_comp):
        fc = action.fcurves.new(data_path, index=i, action_group="Track")
        fc.keyframe_points.add(len(frames))
        fc.keyframe_points.foreach_set("co", [v for pair in zip(frames, values) for v in pair])
        fc.keyframe_points.foreach_set("interpolation", [1] * len(frames))  # LINEAR
        fc.update()


def frame_number(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else None


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--point-radius", type=float, default=None)
    a = ap.parse_args(argv)

    meta = {"videos": []}
    if a.meta and os.path.exists(a.meta):
        with open(a.meta) as f:
            meta = json.load(f)
    vinfo = {v["group"]: v for v in meta["videos"]}

    cams = read_cameras(os.path.join(a.model, "cameras.txt"))
    imgs = read_images(os.path.join(a.model, "images.txt"))
    xyz, rgb = read_points(os.path.join(a.model, "points3D.txt"))
    print(f"[blender] {len(cams)} cameras, {len(imgs)} images, {len(xyz)} points")

    # empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # COLMAP world: +X right, +Y down, +Z forward  ->  Blender Z-up
    root = bpy.data.objects.new("COLMAP", None)
    root.empty_display_type = "ARROWS"
    root.rotation_euler = (-math.pi / 2, 0, 0)
    scene.collection.objects.link(root)

    # robust scene size for a sensible default point radius
    centers = [(-(q.to_matrix().transposed() @ t)) for _, _, q, t in imgs]
    pts = [Vector(p) for p in xyz] or centers
    if pts:
        mid = sum(pts, Vector()) / len(pts)
        d = sorted((p - mid).length for p in pts)
        extent = d[int(len(d) * 0.9)] if d else 1.0
    else:
        extent = 1.0
    radius = a.point_radius or max(extent * 0.002, 1e-6)

    if xyz:
        make_point_cloud(xyz, rgb, root, radius)

    # group frames per video (images/<group>/frame_000001.jpg)
    per_group = defaultdict(list)
    for cam_id, name, q, t in imgs:
        group = os.path.dirname(name) or "video"
        per_group[group].append((frame_number(name), cam_id, name, q, t))

    flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))  # COLMAP cam -> Blender cam
    max_frame, fps_set = 1, False
    for group, entries in sorted(per_group.items()):
        entries.sort(key=lambda e: (e[0] is None, e[0] or 0, e[2]))
        info = vinfo.get(group, {})
        model, w, h, params = cams[entries[0][1]]

        cdata = bpy.data.cameras.new(f"Cam_{group}")
        camera_intrinsics(cdata, model, w, h, params)
        cdata.clip_start = extent * 0.001
        cdata.clip_end = extent * 100
        cobj = bpy.data.objects.new(f"Cam_{group}", cdata)
        scene.collection.objects.link(cobj)
        cobj.parent = root
        cobj.rotation_mode = "QUATERNION"

        frames, locs, rots = [], [], []
        prev_q = None
        for i, (fnum, _, _, q, t) in enumerate(entries):
            r_wc = q.to_matrix().transposed()
            c = -(r_wc @ t)
            rq = (r_wc @ flip).to_quaternion()
            if prev_q is not None and prev_q.dot(rq) < 0:  # keep rotations continuous
                rq.negate()
            prev_q = rq
            frames.append(float(fnum if fnum is not None else i + 1))
            locs.append(c)
            rots.append(rq)
        cobj.animation_data_create()
        cobj.animation_data.action = bpy.data.actions.new(f"Track_{group}")
        set_fcurves(cobj, "location", frames, [[v[k] for v in locs] for k in range(3)])
        set_fcurves(cobj, "rotation_quaternion", frames, [[q[k] for q in rots] for k in range(4)])
        max_frame = max(max_frame, int(max(frames)), info.get("frames", 0))

        # camera path as a polyline
        curve = bpy.data.curves.new(f"Track_{group}", "CURVE")
        curve.dimensions = "3D"
        spline = curve.splines.new("POLY")
        spline.points.add(len(locs) - 1)
        spline.points.foreach_set("co", [v for c in locs for v in (*c, 1.0)])
        tobj = bpy.data.objects.new(f"Track_{group}", curve)
        scene.collection.objects.link(tobj)
        tobj.parent = root

        # extracted frames as camera background (image sequence, frame-accurate)
        img_dir = os.path.join(meta.get("images_dir", ""), group)
        first = os.path.join(img_dir, "frame_000001.jpg")
        if os.path.exists(first):
            img = bpy.data.images.load(first)
            img.source = "SEQUENCE"
            cdata.show_background_images = True
            bg = cdata.background_images.new()
            bg.image = img
            bg.image_user.frame_duration = info.get("frames", max_frame)
            bg.image_user.frame_start = 1
            bg.image_user.use_auto_refresh = True
            bg.alpha = 0.6

        if scene.camera is None:
            scene.camera = cobj
            scene.render.resolution_x, scene.render.resolution_y = w, h
        if not fps_set and info.get("fps"):
            fps = info["fps"]
            scene.render.fps = max(1, round(fps))
            scene.render.fps_base = round(fps) / fps if fps else 1.0
            fps_set = True
        print(f"[blender] {group}: {len(entries)} solved frames")

    scene.view_settings.view_transform = "Standard"  # true photo colors
    scene.frame_start, scene.frame_end = 1, max_frame
    scene.frame_set(1)

    # viewport: solid mode with attribute colors, framed on the cloud
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "SOLID"
                        space.shading.color_type = "VERTEX"
                        space.clip_end = extent * 1000

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.output))
    print(f"[blender] saved {a.output}")


main()
