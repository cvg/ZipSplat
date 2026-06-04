"""Interactive viser viewer for ZipSplat — live compression slider + token-coloring toggle.

Requires the optional viewer dependencies: ``pip install "zipsplat[viewer]"`` (viser + nerfview).

Usage:
    python -m zipsplat.viewer path/to/images/         # a directory of images
    python -m zipsplat.viewer "scene/*.jpg"            # a glob of images
    python -m zipsplat.viewer pan.mp4 --num-frames 24  # a video clip

Author: Alexander Veicht
"""

import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import viser
from nerfview import Viewer
from zipsplat.camera import Camera
from zipsplat.pose import Pose
from zipsplat.predictor import ZipSplat
from zipsplat.utils import load_image, load_video

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(model: ZipSplat, images: List[torch.Tensor], port: int = 8080) -> None:
    """Launch the viewer on the given model and input views."""
    group_size = model.model.conf["gaussians_per_token"]
    state = {"gaussians": None, "colored": None, "by_token": False}

    def recompute(compression: float) -> None:
        g = model(images, compression=compression)[0]
        state["gaussians"], state["colored"] = g, g.color_by_group(group_size)

    recompute(1.0)

    server = viser.ViserServer(port=port, verbose=False)
    gs_count = server.gui.add_number("Gaussians", 0, disabled=True)
    compression = server.gui.add_slider(
        "Compression", min=0.05, max=1.0, step=0.05, initial_value=1.0
    )
    color_toggle = server.gui.add_checkbox("Color by token", initial_value=False)

    @torch.no_grad()
    def render_fn(camera_state, render_tab_state):
        w, h = render_tab_state.viewer_width, render_tab_state.viewer_height
        K = torch.from_numpy(camera_state.get_K((w, h))).float()
        c2w = torch.from_numpy(camera_state.c2w).float()
        scene = state["colored"] if state["by_token"] else state["gaussians"]
        rgb, _ = scene.render(Camera.from_K(K, w=w, h=h), Pose.from_4x4mat(c2w), mode="RGB")
        gs_count.value = scene.num_gaussians
        return rgb[0].clamp(0, 1).moveaxis(0, -1).cpu().numpy()  # (H, W, 3)

    viewer = Viewer(server, render_fn, output_dir=None, mode="rendering")

    @server.on_client_connect
    def _(client: viser.ClientHandle):  # start at the identity pose (reference view)
        client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        client.camera.position = np.array([0.0, 0.0, 0.0])

    @compression.on_update
    def _(_):
        recompute(compression.value)  # re-runs the model
        viewer.rerender(None)

    @color_toggle.on_update
    def _(_):
        state["by_token"] = color_toggle.value
        viewer.rerender(None)

    print(f"Viewer running at http://localhost:{port} — Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


def _load_inputs(path: str, num_frames: int) -> List[torch.Tensor]:
    """Load a directory of images, an image glob, or a video into a list of (3, H, W) tensors."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES:
        return load_video(p, num_frames=num_frames)
    files = (
        sorted(f for f in p.iterdir() if p.is_dir() and f.suffix.lower() in _IMAGE_SUFFIXES)
        if p.is_dir()
        else sorted(Path().glob(path))
    )
    if not files:
        raise ValueError(f"No images/video found at {path!r}.")
    return [load_image(f) for f in files]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Interactive ZipSplat viewer.")
    ap.add_argument("input", help="Directory of images, an image glob, or a video file.")
    ap.add_argument("--weights", default="zipsplat", help="Registry name, local path, or URL.")
    ap.add_argument("--num-frames", type=int, default=24, help="Frames to sample from a video.")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    images = _load_inputs(args.input, args.num_frames)
    print(f"Loaded {len(images)} view(s).")
    model = ZipSplat(weights=args.weights).cuda().eval()
    run(model, images, port=args.port)


if __name__ == "__main__":
    main()
