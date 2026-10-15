"""Interactive viser viewer for ZipSplat -- live compression + token-group coloring.

Requires the optional viewer dependencies: ``pip install "zipsplat[viewer]"`` (viser + nerfview).
The scene is shown translated so the median Gaussian sits at the viewer origin.

Toolbar controls:
    Run TokenBA       refine the current prediction on the input views (CUDA).
    Reset             back to the feed-forward prediction.
    TokenBA settings  rounds, joint refinement steps, early stopping (collapsed by default).
    Compression       fraction of tokens kept (re-runs the decoder, resets TokenBA).
    Color mode        RGB | Token groups.

Usage:
    python -m zipsplat.viewer path/to/images/           # a directory of images
    python -m zipsplat.viewer "scene/*.jpg"              # a glob of images
    python -m zipsplat.viewer pan.mp4 --num-frames 24    # a video clip
    python -m zipsplat.viewer frames/ --num-frames 24    # video frames as a folder (subsampled)

Author: Alexander Veicht
"""

import argparse
import time
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import viser
from nerfview import Viewer

from zipsplat.camera import Camera
from zipsplat.pose import Pose
from zipsplat.predictor import ZipSplat
from zipsplat.utils import load_image, load_video, remove_batch_dim

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_COLOR_MODES = ("RGB", "Token groups")


@torch.no_grad()
def _encode(model: ZipSplat, images: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
    """Run the compression-independent encoder once. Returns (model inputs, encoding)."""
    imgs = model.prepare_inputs(images)[0]  # (V, 3, H, W) on the model device
    with model.autocast():
        return imgs, model.model.encode(imgs[None])


@torch.no_grad()
def _decode(model: ZipSplat, imgs: torch.Tensor, enc: Dict, compression: float) -> Dict:
    """Cluster + fuse + decode at a compression: a prediction as ZipSplat.forward returns it
    (without TokenBA and with the model-input cameras), ready for ZipSplat.refine."""
    with model.autocast():
        pred = remove_batch_dim(model.model.decode(enc, imgs[None], compression))
    pred["model_images"], pred["model_cameras"] = imgs, pred["cameras"]
    return pred


def run(model: ZipSplat, images: List[torch.Tensor], port: int = 8080) -> None:
    """Launch the viewer on the given model and input views."""
    G = model.model.gaussian_head.gaussians_per_token
    # The encoder is compression-independent -> run it once, re-run only the decoder on change.
    imgs, enc = _encode(model, images)

    state = {
        "pred": None,  # feed-forward prediction at the current compression (TokenBA's input)
        "gaussians": None,  # Gaussians shown (unbatched, original RGB), feed-forward or refined
        "render": None,  # the colored Gaussians actually rendered
        "mode": "RGB",
    }

    def show(pred) -> None:
        """Cache a prediction's Gaussians for rendering."""
        state["gaussians"] = pred["gaussians"]

    def recompute(compression: float) -> None:
        """Re-run the decoder at a new compression (the feed-forward prediction)."""
        state["pred"] = _decode(model, imgs, enc, compression)
        show(state["pred"])

    def restyle() -> None:
        """Recolor the cached Gaussians for the current color mode (no head re-run)."""
        g = state["gaussians"]
        state["render"] = g if state["mode"] == "RGB" else g.color_by_group(G)

    recompute(1.0)
    restyle()
    # Viewer frame = scene frame translated so the median Gaussian is the origin (fixed once).
    center = state["gaussians"].means.float().median(dim=0).values.cpu()  # (3,)

    server = viser.ViserServer(port=port, verbose=False)
    gs_count = server.gui.add_number("Gaussians", 0, disabled=True)
    shown = server.gui.add_text("Prediction", "feed-forward", disabled=True)
    run_ba = server.gui.add_button("Run TokenBA")
    reset = server.gui.add_button("Reset")
    with server.gui.add_folder("TokenBA settings", expand_by_default=False):
        ba_rounds = server.gui.add_number("Rounds", initial_value=4, min=1, step=1)
        ba_steps = server.gui.add_number("Joint steps", initial_value=96, min=0, step=8)
        ba_early_stop = server.gui.add_checkbox("Early stopping", initial_value=True)
    compression = server.gui.add_slider(
        "Compression", min=0.05, max=1.0, step=0.05, initial_value=1.0
    )
    mode = server.gui.add_dropdown("Color mode", _COLOR_MODES, initial_value="RGB")

    @torch.no_grad()
    def render_fn(camera_state, render_tab_state):
        w, h = render_tab_state.viewer_width, render_tab_state.viewer_height
        K = torch.from_numpy(camera_state.get_K((w, h))).float()
        c2w = torch.from_numpy(camera_state.c2w).float()
        c2w[:3, 3] += center  # viewer frame -> scene frame
        scene = state["render"]
        rgb, _ = scene.render(Camera.from_K(K, w=w, h=h), Pose.from_4x4mat(c2w), mode="RGB")
        gs_count.value = scene.num_gaussians
        return rgb[0].clamp(0, 1).moveaxis(0, -1).cpu().numpy()  # (H, W, 3)

    viewer = Viewer(server, render_fn, output_dir=None, mode="rendering")
    viewer._rendering_folder.expand_by_default = False  # nerfview's render-trajectory panel

    @server.on_client_connect
    def _(client: viser.ClientHandle):  # start at the reference camera, orbiting the origin
        client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        client.camera.position = -center.numpy()
        client.camera.look_at = np.zeros(3)

    @compression.on_update
    def _(_):
        recompute(compression.value)  # re-runs the decoder
        shown.value = "feed-forward"
        restyle()
        viewer.rerender(None)

    @run_ba.on_click
    def _(_):
        run_ba.disabled = reset.disabled = True
        shown.value = "running TokenBA..."
        settings = {"rounds": int(ba_rounds.value), "joint_steps": int(ba_steps.value)}
        if not ba_early_stop.value:
            settings["joint_tol"] = None
        show(model.refine(state["pred"], verbose=True, **settings))
        shown.value = "TokenBA"
        restyle()
        viewer.rerender(None)
        run_ba.disabled = reset.disabled = False

    @reset.on_click
    def _(_):
        show(state["pred"])
        shown.value = "feed-forward"
        restyle()
        viewer.rerender(None)

    @mode.on_update
    def _(_):
        state["mode"] = mode.value
        restyle()  # recolor only -- no decoder re-run
        viewer.rerender(None)

    print(f"Viewer running at http://localhost:{port} -- Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


def _load_inputs(path: str, num_frames: Optional[int]) -> List[torch.Tensor]:
    """Load a directory of images, an image glob, or a video into a list of (3, H, W) tensors.

    `num_frames` evenly subsamples the (sorted) images just like video frames -- pass it when the
    folder holds extracted video frames. None keeps every image.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES:
        return load_video(p, num_frames=num_frames or 24)
    candidates = p.iterdir() if p.is_dir() else map(Path, glob(path))
    files = sorted(f for f in candidates if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No images/video found at {path!r}.")
    if num_frames is not None and len(files) > num_frames:
        idx = torch.linspace(0, len(files) - 1, num_frames).round().long().tolist()
        files = [files[i] for i in idx]
    return [load_image(f) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive ZipSplat viewer.")
    ap.add_argument("input", help="Directory of images, an image glob, or a video file.")
    ap.add_argument(
        "--weights",
        default="zipsplat",
        help="Registry name, local path, or URL (default: zipsplat).",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Evenly subsample this many frames (default: 24 for a video, all images otherwise).",
    )
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    images = _load_inputs(args.input, args.num_frames)
    print(f"Loaded {len(images)} view(s).")
    model = ZipSplat(weights=args.weights).cuda().eval()
    run(model, images, port=args.port)


if __name__ == "__main__":
    main()
