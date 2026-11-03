# vid2blend

This tool turns one or more videos into a **3D point cloud with the camera track, as a ready-to-open Blender file**.

It is a Unix port of [polyfjord's `batch_reconstruct.bat`](https://gist.github.com/polyfjord/4ed7e8988bdb9674145f1c270440200d) workflow, built on [COLMAP](https://colmap.github.io/) (through `pycolmap`) and ffmpeg. It writes the `.blend` directly, so you don't need an add-on. The COLMAP model it exports also works with [SBCV's Photogrammetry Importer](https://github.com/SBCV/Blender-Addon-Photogrammetry-Importer).

```
video.mp4 ──ffmpeg──> frames ──SIFT──> features ──sequential match──> matches
          ──incremental mapper──> sparse model ──Blender──> scene.blend
```

## Setup

You need `ffmpeg`, Python ≥ 3.9 and Blender ≥ 4.0 on your `PATH`. An NVIDIA GPU is optional but much faster.

```bash
./setup.sh            # creates .venv with pycolmap (CUDA build if a GPU is found)
```

## Usage

```bash
./vid2blend clip.mp4                       # one video  -> scenes/clip/clip.blend
./vid2blend a.mp4 b.mp4                    # two separate scenes
./vid2blend videos/                        # every video in a folder, one scene each
./vid2blend a.mp4 b.mp4 --merge street     # ALL videos in ONE shared scene
./vid2blend long.mp4 --fps 5               # extract only 5 frames per second (much faster)
```

Useful options (see `./vid2blend -h` for all of them):

| option | meaning |
|---|---|
| `--fps N` | Frames per second to extract. The default is every frame, as in the original script. For videos longer than about 30 s, 3–10 fps is usually plenty. |
| `--merge NAME` | Reconstruct all the inputs together. Keyframes of each video are matched against every other video (`--cross-keyframes`), so the clips must overlap visually. |
| `--overlap N` | How many neighbouring frames each frame is matched with. The default is 15. |
| `--max-image-size PX` | Downscale frames whose longest side is larger than this. |
| `--cpu` | Don't use CUDA. |
| `--rebuild-blend` | Only regenerate the `.blend` for scenes that are already reconstructed. |
| `--force` | Redo scenes that already exist. Existing scenes are skipped by default. |

## Output (`scenes/<name>/`)

| file | content |
|---|---|
| `<name>.blend` | The Blender scene, described below. |
| `model_txt/` | The COLMAP model as `cameras.txt`, `images.txt` and `points3D.txt`. You can import it with the SBCV add-on. |
| `points.ply` | The colored point cloud, for MeshLab, CloudCompare and similar tools. |
| `sparse/<n>/` | Raw COLMAP binary models. If tracking broke, there is more than one; the largest one is used. |
| `images/<video>/` | The extracted frames. |

The `.blend` contains:

- **`COLMAP`** is an empty that parents everything else. Rotate, scale and move it to level the scene and set real-world scale, because COLMAP's scale and orientation are arbitrary.
- **`PointCloud`** holds the colored sparse points. Change the size under *Modifiers → Points → Point Radius*. It shows in Solid mode, Eevee and Cycles.
- **`Cam_<video>`** is an animated camera for each video. It has the real focal length, and its background is the original frames, frame-accurate. Press *Numpad 0* and play the animation to check the solve.
- **`Track_<video>`** is the camera path, drawn as a line.

## Tips

- Move the camera; don't just pan it. Reconstruction needs parallax.
- Textured, static scenes work best. Moving people, cars, water, sky and reflections hurt the result.
- If you get several models, or few frames are registered, try a higher `--fps` (more overlap), a higher `--overlap`, or slower camera motion.
- The camera solve ignores lens distortion in Blender (the default model is `SIMPLE_RADIAL`). For wide or fisheye lenses, try `--camera-model OPENCV`.

## AI agent access to Blender and FreeCAD (MCP)

`.mcp.json` registers two MCP servers for this project, so Claude Code and other MCP-capable agents opened in this folder can drive the apps:

- `blender` is [MCP for Blender](https://github.com/ahujasid/mcp-for-blender), started with `uvx mcp-for-blender`. Its telemetry is turned off in `.mcp.json`.
- `freecad` is [freecad-mcp](https://github.com/neka-nat/freecad-mcp), started with `uvx freecad-mcp`.

Each server talks to an add-on that runs inside the app, so install the add-ons once:

```bash
# Blender add-on, then enable "Interface: MCP for Blender" in Preferences → Add-ons
DISABLE_TELEMETRY=true uvx mcp-for-blender install-addon

# FreeCAD add-on (the folder below is for the Flatpak install of FreeCAD 1.1;
# in FreeCAD's Python console, os.path.join(FreeCAD.getUserAppDataDir(), "Mod") prints yours)
git clone --depth 1 https://github.com/neka-nat/freecad-mcp.git /tmp/freecad-mcp
mkdir -p ~/.var/app/org.freecad.FreeCAD/data/FreeCAD/v1-1/Mod
cp -r /tmp/freecad-mcp/addon/FreeCADMCP ~/.var/app/org.freecad.FreeCAD/data/FreeCAD/v1-1/Mod/
```

Each app has to be open for its agent tools to work:

- **Blender:** the add-on starts its server on `localhost:9876` automatically. You can also start it from the **MCP for Blender** tab in the 3D view's `N` panel.
- **FreeCAD:** switch to the **MCP Addon** workbench and click **Start RPC Server**, which listens on `127.0.0.1:9875`. Enable **FreeCAD MCP → Auto-Start Server** to have it start on its own.

⚠️ Both servers let an agent run arbitrary Python inside the app, without authentication. Keep them bound to localhost, and save your work before letting an agent edit a scene.

## Credits and related projects

vid2blend builds on other people's work. If it doesn't fit your needs, one of these might:

- **[polyfjord](https://gist.github.com/polyfjord/4ed7e8988bdb9674145f1c270440200d)** wrote the original Windows `batch_reconstruct.bat` and the tutorial video it comes from. The pipeline and default parameters here follow that script.
- **[COLMAP](https://colmap.github.io/)** / **[pycolmap](https://github.com/colmap/pycolmap)** do all of the reconstruction work.
- **[SBCV's Photogrammetry Importer](https://github.com/SBCV/Blender-Addon-Photogrammetry-Importer)** is the Blender add-on the original workflow uses to import COLMAP models. The `model_txt/` output here stays compatible with it.
- **[oneShot](https://github.com/beingcarrot/oneShot)** by beingcarrot is a Blender add-on that does video → COLMAP → Blender from inside Blender. If you prefer working in Blender's UI over the command line, use this one.
- Unix ports of polyfjord's script:
  - [Chaos-man's Linux guide and script](https://gist.github.com/Chaos-man/cbe063cbfb0624026f022a0ae8de021f) (GLOMAP)
  - [Norgus's bash version](https://gist.github.com/Norgus/9e877924949d778db12c534ff1bafe36), adapted from celestial-33's macOS port
  - [souhaib101's Polyfjord-COLMAP-Workflow](https://zff.dev/souhaib101/Polyfjord-COLMAP-Workflow) (POSIX shell)
