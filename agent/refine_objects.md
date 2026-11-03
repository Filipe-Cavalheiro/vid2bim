# Refine furniture models (agent task)

You are refining the FreeCAD furniture models of one scanned scene so they match the real objects.
The pipeline has already detected, measured and roughly modelled each object. Your job is to make
the shape right, not to re-measure. **Never change an object's position, yaw or overall size by
more than 10 % unless the evidence clearly shows the measurement is wrong.** If you do, say why.

## Inputs (in `SCENE/objects/`)

- `objects.json` holds one entry per object, all in metres in the house frame (z up, floor at z = 0):
  - `id` and `class`
  - `center`, `size [length, depth, height]`, `yaw_deg` (rotation about z) and `floor_z`
  - `colors`, the dominant RGB colours in 0–1, largest share first
- `<id>.jpg` is the best photo crop of the object. **Look at it**, because it is the main evidence for the shape.
- `<id>.ply` holds the object's 3D points in the house frame. Use it to check proportions, such as where
  the headboard is and how high the seat is. A quick way to summarise it:
  `.venv/bin/python -c "import open3d as o3d, numpy as np; p=np.asarray(o3d.io.read_point_cloud('SCENE/objects/<id>.ply').points); print(p.min(0), p.max(0), np.percentile(p[:,2],[10,50,90]))"`
- `furniture.FCStd` holds one `Part::Feature` per object, named by `id`, with the template shape and placement.
  The template's local frame is: x = length, y = depth, z = up, with the origin at the centre of the footprint on the floor.

## How to edit

Pick one of these two ways:

1. **FreeCAD MCP** (`freecad` server in `.mcp.json`), when FreeCAD is open with the MCP add-on's RPC server running.
   Open `SCENE/objects/furniture.FCStd` and edit it with `execute_code`. Save with `doc.save()`.
2. **Headless.** Write a short Python script and run it with
   `flatpak run --command=FreeCADCmd org.freecad.FreeCAD /abs/path/script.py`, or with `freecadcmd` if that is installed.
   The script should be:
   ```python
   import FreeCAD, Part
   from FreeCAD import Vector
   doc = FreeCAD.openDocument("/abs/path/SCENE/objects/furniture.FCStd")
   obj = doc.getObject("bed_1")
   pl = obj.Placement                  # keep it
   # build the new shape in LOCAL coordinates (see frame above), e.g. with Part.makeBox(...).fuse(...)
   obj.Shape = new_shape
   obj.Placement = pl
   doc.recompute(); doc.save()
   ```
   Note that `obj.Shape = ...` resets the placement, so set `obj.Placement = pl` again afterwards.

Typical improvements:
- Which end the headboard is on, by comparing the point heights at both ends.
- Drawer and door counts, read from the photo.
- Leg style and count, and the seat or back height of chairs.
- Shelf count.
- Rounded edges where the object clearly has them, using `makeFillet`.
- Splitting an object into parts, for example a mattress on a frame.

Keep it simple and solid: closed solids, no zero-thickness faces, and roughly under 2,000 faces per object.

## Finish

1. Push the refined shapes into the house model and the Blender meshes:
   `.venv/bin/python furniture.py SCENE --from-fcstd`
2. Rebuild the Blender scene: `./vid2bim <the scene's videos> --rebuild-blend` (or ask the user which input produced it).
3. Report one line per object: what you changed and why.
