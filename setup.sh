#!/usr/bin/env bash
# Creates .venv with COLMAP (pycolmap). Uses the CUDA build when an NVIDIA GPU is present.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
for t in ffmpeg ffprobe; do command -v "$t" >/dev/null || { echo "Please install ffmpeg (e.g. sudo apt install ffmpeg)"; exit 1; }; done
command -v blender >/dev/null || echo "Note: blender not on PATH - .blend export needs it (or pass --blender PATH)."

if command -v uv >/dev/null; then
    [[ -d .venv ]] || uv venv .venv
    PIP=(uv pip install --python .venv)
else
    [[ -d .venv ]] || python3 -m venv .venv
    PIP=(.venv/bin/pip install)
fi

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    "${PIP[@]}" pycolmap-cuda12 || "${PIP[@]}" pycolmap
else
    "${PIP[@]}" pycolmap
fi
.venv/bin/python -c "import pycolmap; print('pycolmap', pycolmap.__version__, '| CUDA:', pycolmap.has_cuda)"

# Optional BIM extras (./setup.sh --bim): MoGe-2 (MIT), Open3D (MIT), IfcOpenShell (LGPL), Cloud2BIM (MIT)
if [[ "${1:-}" == "--bim" ]]; then
    "${PIP[@]}" torch torchvision "git+https://github.com/microsoft/MoGe.git" open3d ifcopenshell \
        shapely scikit-image pyyaml pandas matplotlib plotly tqdm scipy opencv-python-headless
    if [[ ! -d third_party/Cloud2BIM ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/VaclavNezerka/Cloud2BIM third_party/Cloud2BIM
    fi
fi
chmod +x vid2blend reconstruct.py
echo "Ready:  ./vid2blend my_video.mp4"
