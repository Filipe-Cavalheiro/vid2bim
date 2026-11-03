#!/usr/bin/env python3
"""Measured objects -> parametric FreeCAD models -> placed in the house (IFC, FCStd, Blender).

  1. FreeCAD (headless) builds a template model per object class (bed = frame +
     mattress (+ headboard), table = top + 4 legs, bookshelf = sides + shelves, ...)
     sized by the measured box and placed at the measured position/yaw
     -> objects/furniture.FCStd (one Part object per item, editable)
  2. the shapes are tessellated and added to the house IFC as IfcFurniture with the
     measured main colour -> bim/house.ifc
  3. house.ifc -> FreeCAD (NativeIFC) -> bim/house.FCStd, and meshes for Blender

An agent can refine objects/furniture.FCStd afterwards (see agent/refine_objects.md);
re-run this script with --from-fcstd to push the refined shapes into the house.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from room_bim import freecad_cmd, ifc_meshes, to_freecad  # noqa: E402


def log(msg):
    print(f"[furniture] {msg}", flush=True)


# Runs inside FreeCAD. Templates are built in local coordinates: x = length, y = depth,
# z = up, origin at the centre of the footprint on the floor.
FREECAD_TEMPLATES = r'''
import json, math, FreeCAD, Part
from FreeCAD import Vector, Placement, Rotation

objs_path, out_fcstd, out_mesh, from_fcstd = {objs!r}, {fcstd!r}, {mesh!r}, {from_fcstd!r}

def box(l, w, h, x=0.0, y=0.0, z=0.0):
    """box centred on (x, y) in plan, from z to z+h"""
    return Part.makeBox(max(l, 1e-3), max(w, 1e-3), max(h, 1e-3), Vector(x - l / 2, y - w / 2, z))

def legs(L, W, h, t=0.05, inset=0.03):
    return [box(t, t, h, sx * (L / 2 - inset - t / 2), sy * (W / 2 - inset - t / 2))
            for sx in (-1, 1) for sy in (-1, 1)]

def bed(L, W, H):
    headboard = H > 0.75             # tall points -> headboard on the -x end (agent can flip it)
    top = min(H, 0.6) if headboard else H
    frame_h = 0.55 * top
    parts = [box(L, W, frame_h - 0.1, z=0.1)] + legs(L, W, 0.1, t=0.06)
    mattress = box(L - 0.04, W - 0.04, max(top - frame_h, 0.05), z=frame_h)
    try:
        mattress = mattress.makeFillet(min(0.03, (top - frame_h) / 3), mattress.Edges)
    except Exception:
        pass
    parts.append(mattress)
    if headboard:
        parts.append(box(0.06, W, H, x=-L / 2 + 0.03))
    return parts

def table(L, W, H, top=0.03):
    return [box(L, W, top, z=H - top)] + legs(L, W, H - top)

def chair(L, W, H):
    seat = min(0.46, 0.5 * H)
    return [box(L, W, 0.04, z=seat - 0.04)] + legs(L, W, seat - 0.04, t=0.035) + \
           [box(0.04, W, H - seat, x=-L / 2 + 0.02, z=seat)]

def shelf(L, W, H, t=0.02, pitch=0.35):
    parts = [box(t, W, H, x=-L / 2 + t / 2), box(t, W, H, x=L / 2 - t / 2), box(L, t, H, y=-W / 2 + t / 2)]
    n = max(2, int(H / pitch) + 1)
    parts += [box(L - 2 * t, W - t, t, y=t / 2, z=i * (H - t) / (n - 1)) for i in range(n)]
    return parts

def cabinet(L, W, H, fronts=1, drawers=0):
    parts = [box(L, W - 0.02, H - 0.05, y=-0.01, z=0.05), box(L - 0.04, W - 0.04, 0.05, y=-0.01)]
    if drawers:
        hh = (H - 0.07) / drawers
        parts += [box(L - 0.02, 0.02, hh - 0.01, y=W / 2 - 0.01, z=0.06 + i * hh) for i in range(drawers)]
    else:
        ww = L / fronts
        parts += [box(ww - 0.005, 0.02, H - 0.06, x=-L / 2 + ww * (i + 0.5), y=W / 2 - 0.01, z=0.055)
                  for i in range(fronts)]
    return parts

def sofa(L, W, H):
    seat = min(0.45, 0.5 * H)
    return [box(L, W, seat), box(L, 0.2, H - seat, y=-W / 2 + 0.1, z=seat),
            box(0.18, W, 0.62 * H - seat if 0.62 * H > seat else 0.1, x=-L / 2 + 0.09, z=seat),
            box(0.18, W, 0.62 * H - seat if 0.62 * H > seat else 0.1, x=L / 2 - 0.09, z=seat)]

def wheels(L, W, r, t=0.2):
    return [Part.makeCylinder(r, t, Vector(sx * (L / 2 - 1.6 * r), sy * W / 2 - (t if sy > 0 else 0), r),
                              Vector(0, 1, 0))
            for sx in (-1, 1) for sy in (-1, 1)]

def vehicle(L, W, H, cabin=0.55, wheel=0.32):
    r = min(wheel, 0.25 * H)
    body_h = 0.5 * H
    parts = [box(L, W - 0.04, body_h - r * 0.6, z=r * 0.6)]
    parts.append(box(cabin * L, W - 0.16, H - body_h, x=-0.05 * L, z=body_h))   # cabin, slightly rear
    return parts + wheels(L, W, r)

def two_wheeler(L, W, H):
    r = min(0.33, 0.3 * H)
    return [Part.makeCylinder(r, 0.08, Vector(sx * (L / 2 - r), -0.04, r), Vector(0, 1, 0)) for sx in (-1, 1)] + \
           [box(L - 2 * r, 0.12, 0.15, z=r), box(0.25, 0.12, H - r, x=L / 2 - r, z=r)]

TEMPLATES = {{
    "car": vehicle, "van": lambda L, W, H: vehicle(L, W, H, cabin=0.8),
    "truck": lambda L, W, H: vehicle(L, W, H, cabin=0.3, wheel=0.45),
    "bus": lambda L, W, H: vehicle(L, W, H, cabin=0.95, wheel=0.45),
    "motorcycle": two_wheeler, "bicycle": two_wheeler,
    "bed": bed, "desk": table, "table": table, "chair": chair, "bookshelf": shelf,
    "wardrobe": lambda L, W, H: cabinet(L, W, H, fronts=2 if L < 1.2 else 3),
    "chest of drawers": lambda L, W, H: cabinet(L, W, H, drawers=4),
    "nightstand": lambda L, W, H: cabinet(L, W, H, drawers=2),
    "sofa": sofa,
    "television": lambda L, W, H: [box(L, max(W, 0.04), H)],
}}

data = json.load(open(objs_path))
doc = FreeCAD.openDocument(out_fcstd) if from_fcstd else FreeCAD.newDocument("furniture")
meshes = []
for o in data["objects"]:
    if from_fcstd:
        obj = doc.getObject(o["id"])
        if obj is None:
            continue
    else:
        L, W, H = o["size"]
        make = TEMPLATES.get(o["class"], lambda L, W, H: [box(L, W, H)])
        shape = Part.makeCompound(make(L, W, H))
        obj = doc.addObject("Part::Feature", o["id"])
        obj.Shape = shape
        obj.Label = o["id"]
        obj.Placement = Placement(Vector(o["center"][0], o["center"][1], o["floor_z"]),
                                  Rotation(Vector(0, 0, 1), o["yaw_deg"]))
        for key, val in (("ObjectClass", o["class"]),):
            obj.addProperty("App::PropertyString", key, "vid2bim")
            setattr(obj, key, val)
    doc.recompute()
    world = obj.Shape.copy()          # Shape already carries the Placement
    v, f = world.tessellate(0.005)
    meshes.append({{"id": o["id"], "class": o["class"],
                    "verts": [[p.x, p.y, p.z] for p in v], "faces": [list(t) for t in f]}})
if not from_fcstd:
    doc.saveAs(out_fcstd)
else:
    doc.save()
json.dump({{"meshes": meshes}}, open(out_mesh, "w"))
print("[freecad] furniture objects:", len(meshes))
'''


def build_freecad(objdir: Path, from_fcstd: bool) -> Path:
    cmd = freecad_cmd()
    if not cmd:
        sys.exit("FreeCAD not found")
    script = objdir / "freecad_furniture.py"
    mesh = objdir / "furniture_meshes.json"
    script.write_text(FREECAD_TEMPLATES.format(objs=str(objdir / "objects.json"),
                                               fcstd=str(objdir / "furniture.FCStd"),
                                               mesh=str(mesh), from_fcstd=from_fcstd))
    r = subprocess.run(cmd + [str(script)], capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if "[freecad]" in l or "Error" in l or "Exception" in l))
    if not mesh.exists():
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("FreeCAD furniture build failed")
    return mesh


MATERIAL_BY_CLASS = {
    "bed": "Fabric", "sofa": "Fabric", "chair": "Fabric",
    "wardrobe": "Wood", "desk": "Wood", "table": "Wood", "bookshelf": "Wood",
    "chest of drawers": "Wood", "nightstand": "Wood", "television": "Plastic",
    "car": "Metal", "van": "Metal", "truck": "Metal", "bus": "Metal",
    "motorcycle": "Metal", "bicycle": "Metal",
}
VEHICLES = {"car", "van", "truck", "bus", "motorcycle", "bicycle"}


def add_to_ifc(room_ifc: Path, house_ifc: Path, objects: list, meshes: list) -> None:
    import ifcopenshell
    import ifcopenshell.api as api
    import ifcopenshell.guid

    f = ifcopenshell.open(str(room_ifc))
    storey = f.by_type("IfcBuildingStorey")[0]
    body = next((c for c in f.by_type("IfcGeometricRepresentationSubContext")
                 if c.ContextIdentifier == "Body"), None) or f.by_type("IfcGeometricRepresentationContext")[0]
    byid = {o["id"]: o for o in objects}
    for m in meshes:
        o = byid[m["id"]]
        # IFC has no vehicle class; a typed proxy keeps it out of the furniture tree
        ifc_class = "IfcBuildingElementProxy" if o["class"] in VEHICLES else "IfcFurniture"
        el = api.run("root.create_entity", f, ifc_class=ifc_class, name=m["id"])
        el.ObjectType = o["class"]
        pts = f.createIfcCartesianPointList3D([tuple(map(float, p)) for p in m["verts"]])
        tri = f.createIfcTriangulatedFaceSet(pts, None, None, [tuple(i + 1 for i in t) for t in m["faces"]], None)
        rep = f.createIfcShapeRepresentation(body, "Body", "Tessellation", [tri])
        el.Representation = f.createIfcProductDefinitionShape(None, None, [rep])
        el.ObjectPlacement = f.createIfcLocalPlacement(
            None, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None))
        api.run("spatial.assign_container", f, products=[el], relating_structure=storey)
        # material + measured colour
        mat = api.run("material.add_material", f, name=MATERIAL_BY_CLASS.get(o["class"], "Generic"))
        api.run("material.assign_material", f, products=[el], material=mat)
        r_, g_, b_ = o["colors"][0]
        style = api.run("style.add_style", f, name=f"{m['id']}_colour")
        api.run("style.add_surface_style", f, style=style, ifc_class="IfcSurfaceStyleShading",
                attributes={"SurfaceColour": {"Name": None, "Red": r_, "Green": g_, "Blue": b_}})
        api.run("style.assign_representation_styles", f, shape_representation=rep, styles=[style])
        pset = api.run("pset.add_pset", f, product=el, name="vid2bim_Measured")
        api.run("pset.edit_pset", f, pset=pset, properties={
            "Length": o["size"][0], "Depth": o["size"][1], "Height": o["size"][2],
            "YawDeg": o["yaw_deg"], "SourceVertices": o["n_vertices"]})
    f.write(str(house_ifc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--from-fcstd", action="store_true",
                    help="use the (agent-)edited objects/furniture.FCStd instead of rebuilding templates")
    a = ap.parse_args()

    scene = Path(a.scene).resolve()
    objdir = scene / "objects"
    objects = json.loads((objdir / "objects.json").read_text())["objects"]
    if not objects:
        sys.exit("no objects in objects.json")
    mesh_json = build_freecad(objdir, a.from_fcstd)
    meshes = json.loads(mesh_json.read_text())["meshes"]

    room_ifc, house_ifc = scene / "bim" / "room.ifc", scene / "bim" / "house.ifc"
    add_to_ifc(room_ifc, house_ifc, objects, meshes)
    log(f"house IFC with {len(meshes)} furniture items: {house_ifc}")
    ifc_meshes(house_ifc)
    to_freecad(house_ifc)


if __name__ == "__main__":
    main()
