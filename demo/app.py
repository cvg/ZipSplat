"""Gradio demo of ZipSplat: a few unposed photos or a video in, 3D Gaussians out.

The scene is shown in an in-browser splat viewer (Spark, see viewer.html) that starts at the first
input view, and as a fly-through video along the predicted input cameras (fly_through.py). Runs
locally and on a Hugging Face ZeroGPU Space.

Usage (from the repo root):
    python demo/app.py

Author: Alexander Veicht
"""

import html
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import gradio_client.utils
import numpy as np
import torch
from fly_through import BG, fly_through, look_targets
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from zipsplat import Camera, Pose, ZipSplat, load_image, load_video

try:
    import spaces  # ZeroGPU on Hugging Face Spaces
except ImportError:
    spaces = None

# gradio 4's API schema parser fails on the boolean sub-schemas pydantic >= 2.11 emits.
_parse_schema = gradio_client.utils._json_schema_to_python_type
gradio_client.utils._json_schema_to_python_type = lambda schema, defs=None: (
    "Any" if isinstance(schema, bool) else _parse_schema(schema, defs)
)

ROOT = Path(__file__).parent
TMP = Path(tempfile.mkdtemp(prefix="zipsplat_demo_"))
VIEWER = (ROOT / "viewer.html").read_text()
# gradio 4 serves allowed files at file=, gradio 5+ at gradio_api/file=.
FILE_ROUTE = "file=" if gr.__version__.startswith("4.") else "gradio_api/file="
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_SIDE = 1008  # long edge the views are stored at, the model resizes them further
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV axes <-> three.js axes


def bf16_weights(model: ZipSplat) -> ZipSplat:
    """Halve the weights: bf16 for everything but the two heads the model runs in fp32 outside
    autocast, the Gaussian head and the camera decoder."""
    for name, module in model.model.named_children():
        if name == "backbone":
            module.backbone.to(torch.bfloat16)
            module.cam_enc.to(torch.bfloat16)
        elif name != "gaussian_head":
            module.to(torch.bfloat16)
    return model


# the release weights from the Hub cache, which the Space preloads (see README.md)
WEIGHTS = hf_hub_download("veichta/zipsplat", "zipsplat-da3g-504.tar")
MODEL = bf16_weights(ZipSplat(weights=WEIGHTS)).cuda().eval()
GROUP = MODEL.model.gaussian_head.gaussians_per_token
PATCH = MODEL.model.backbone.backbone.patch_size  # one token per ViT patch


def gpu(duration: int):
    """spaces.GPU on a ZeroGPU Space, a no-op elsewhere."""
    return spaces.GPU(duration=duration) if spaces else lambda fn: fn


def pil(frame: torch.Tensor) -> Image.Image:
    """A (3, H, W) float frame in [0, 1] as a PIL image."""
    return Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def prepare(files: Optional[List[str]], max_views: int) -> Tuple[List[str], List[str], int, int]:
    """Store the views the model sees as JPEGs, for the gallery and the state, and count the
    tokens the model will see (one per ViT patch of the input size) and the views, for the
    live Gaussian count.

    A single video is sampled evenly, a photo set larger than `max_views` subsampled evenly.
    """
    if not files:
        return [], [], 0, 0
    max_views = int(max_views)
    paths = sorted((Path(f) for f in files), key=lambda p: p.name)
    if len(paths) == 1 and paths[0].suffix.lower() in VIDEO_SUFFIXES:
        frames = [pil(f) for f in load_video(paths[0], num_frames=max_views)]
    else:
        if len(paths) > max_views:
            idx = np.linspace(0, len(paths) - 1, max_views).round().astype(int)
            paths = [paths[i] for i in idx]
        frames = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]
    out = Path(tempfile.mkdtemp(dir=TMP))
    views = []
    for i, im in enumerate(frames):
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        views.append(str(out / f"{i:03d}.jpg"))
        im.save(views[-1], quality=95)
    log_ar = sum(math.log(im.width / im.height) for im in frames) / len(frames)
    W, H = min(MODEL.input_sizes, key=lambda s: abs(math.log(s[0] / s[1]) - log_ar))
    return views, views, len(views) * (H // PATCH) * (W // PATCH), len(views)


# The Gaussians the model will decode at the compression, out of all of them: the tokens kept
# (rounded as the model does) times the Gaussians per token. Runs in the browser.
COUNT_JS = f"""(tokens, compression, views) => {{
  if (!tokens) return "";
  const kept = compression >= 100 ? tokens : Math.max(1, Math.floor(tokens * compression / 100));
  const fmt = (n) => (n * {GROUP}).toLocaleString("en-US");
  return `**${{fmt(kept)}}** / ${{fmt(tokens)}} Gaussians from ${{views}} views`;
}}"""


def viewer_config(ply: Path, camera: Camera, pose: Pose, target: torch.Tensor) -> Dict:
    """Viewer settings: the scene URL and the first input view in three.js coordinates."""
    R, t = pose.R.cpu().double().numpy(), pose.t.cpu().double().numpy()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = FLIP @ R @ FLIP, FLIP @ t
    depth = float((target - pose.t).norm())
    return {
        "url": FILE_ROUTE + str(ply),
        "fov": math.degrees(float(camera.vfov)),
        "near": 0.01 * depth,
        "far": 1000.0 * depth,
        "pose": T.flatten(order="F").tolist(),  # column-major, as three.js reads it
        "up": (FLIP @ R @ [0.0, -1.0, 0.0]).tolist(),
        "target": (FLIP @ target.cpu().double().numpy()).tolist(),
        "bg": "#{:02x}{:02x}{:02x}".format(*(round(255 * c) for c in BG)),
    }


def viewer_html(config: Optional[Dict]) -> str:
    """The viewer page in an iframe, or a placeholder before the first scene."""
    style = "width: 100%; aspect-ratio: 1 / 1; border: 0; border-radius: 12px; display: block;"
    if config is None:
        return (
            f'<div style="{style} display: flex; align-items: center; justify-content: center;'
            ' border: 1px dashed var(--block-border-color); color: var(--body-text-color-subdued);">'
            "The reconstructed scene appears here.</div>"
        )
    page = VIEWER.replace("__CONFIG__", json.dumps(config))
    return f'<iframe srcdoc="{html.escape(page)}" style="{style}"></iframe>'


@gpu(duration=120)
def reconstruct(
    files, views: List[str], max_views: int, compression: float, tokenba: bool, color: str
):
    """Run ZipSplat on the views, export the viewer scenes and render the fly-through."""
    if not views:  # Reconstruct was clicked before the upload was prepared
        views = prepare(files, max_views)[0]
    if not views:
        raise gr.Error("Upload a few images or a video first.")
    images = [load_image(p) for p in views]
    ba = {"joint_steps": 16} if tokenba else False  # a short joint refinement for the demo
    pred = MODEL(images, compression=compression / 100, ba=ba)

    out = Path(tempfile.mkdtemp(dir=TMP))
    gaussians, poses, cameras = pred["gaussians"].float(), pred["poses"], pred["model_cameras"]
    target = look_targets(gaussians, cameras[:1], poses[:1])[0]
    gaussians.save_ply(out / "scene.ply")
    gaussians.color_by_group(GROUP).save_ply(out / "tokens.ply")
    fly_through(gaussians, cameras, poses, pred["model_images"], out / "fly_through.mp4")
    scene = {
        "RGB": viewer_config(out / "scene.ply", cameras[0], poses[0], target),
        "Token groups": viewer_config(out / "tokens.ply", cameras[0], poses[0], target),
    }
    del pred, gaussians, images
    torch.cuda.empty_cache()  # the run's working memory goes back to the driver, not the cache
    return scene, viewer_html(scene[color]), str(out / "fly_through.mp4"), str(out / "scene.ply")


def show(scene: Optional[Dict], color: str) -> str:
    """Swap the viewer between RGB and token-group colors, without running the model."""
    return viewer_html(scene[color] if scene else None)


def load_example(thumb: str) -> List[str]:
    """The input files of the example whose thumbnail was clicked."""
    return EXAMPLES[Path(thumb).stem]


# name -> input files under assets/examples/: a folder of photos or one video
EXAMPLES = {
    name: sorted(str(q) for q in p.iterdir()) if p.is_dir() else [str(p)]
    for name in ("office", "motorcycle", "statues")
    for p in (ROOT.parent / "assets" / "examples").glob(f"{name}*")
}


def thumbnail(name: str) -> str:
    """A 320 px square crop of the example's middle view, for the examples strip."""
    files = EXAMPLES[name]
    if Path(files[0]).suffix.lower() in VIDEO_SUFFIXES:
        im = pil(load_video(files[0], num_frames=3)[1])
    else:
        im = ImageOps.exif_transpose(Image.open(files[len(files) // 2])).convert("RGB")
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    path = TMP / f"{name}.jpg"
    im.crop((left, top, left + s, top + s)).resize((320, 320), Image.LANCZOS).save(path, quality=90)
    return str(path)


# ------------------------------- page -------------------------------
# The look of the project page: Quicksand on an off-white background with the forest green.
GREEN = gr.themes.Color(
    "#f1f7f2",
    "#dfeee2",
    "#c0dcc6",
    "#94c29f",
    "#67a376",
    "#3a7a45",
    "#2f6639",
    "#27532f",
    "#214327",
    "#1b3720",
    "#0f2013",
    name="forest",
)
THEME = gr.themes.Base(
    primary_hue=GREEN,
    neutral_hue=gr.themes.colors.stone,
    font=(gr.themes.GoogleFont("Quicksand"), "ui-sans-serif", "system-ui", "sans-serif"),
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_sm,
    spacing_size=gr.themes.sizes.spacing_sm,
).set(
    body_background_fill="#f9f8f4",
    body_text_color="#1e211c",
    body_text_color_subdued="#76776c",
    block_background_fill="#ffffff",
    block_border_color="#dcdcd6",
    block_label_text_color="#76776c",
    block_title_text_color="#1e211c",
    button_primary_background_fill="#3a7a45",
    button_primary_background_fill_hover="#2f6639",
    button_primary_text_color="#ffffff",
    slider_color="#3a7a45",
    checkbox_background_color_selected="#3a7a45",
    checkbox_border_color_selected="#3a7a45",
    checkbox_label_background_fill_selected="#3a7a45",
    checkbox_label_text_color_selected="#ffffff",
    link_text_color="#3a7a45",
    link_text_color_hover="#2f6639",
    # the same palette on a dark page
    body_background_fill_dark="#181a17",
    body_text_color_dark="#e9e8e2",
    body_text_color_subdued_dark="#a2a399",
    block_background_fill_dark="#22251f",
    block_border_color_dark="#3b3e37",
    block_label_text_color_dark="#a2a399",
    block_title_text_color_dark="#e9e8e2",
    button_primary_background_fill_dark="#3a7a45",
    button_primary_background_fill_hover_dark="#2f6639",
    button_primary_text_color_dark="#ffffff",
    slider_color_dark="#3a7a45",
    checkbox_background_color_selected_dark="#3a7a45",
    checkbox_border_color_selected_dark="#3a7a45",
    checkbox_label_background_fill_selected_dark="#3a7a45",
    checkbox_label_text_color_selected_dark="#ffffff",
    link_text_color_dark="#8fc49b",
    link_text_color_hover_dark="#b4d8bc",
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; font-weight: 500; }
footer { display: none !important; }
/* gradio sizes the gallery grid itself, which clips it: size the scrolling wrapper instead */
.gallery-container .grid-wrap { max-height: 260px; overflow-y: auto !important; }
.gallery-container .grid-container { height: auto !important; }
.zs-count { min-height: 1.5rem; }
.zs-head { text-align: center; padding: 1.25rem 0 0.25rem; }
.zs-head h1 {
  font-size: 2.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em;
  line-height: 1.1; margin: 0 0 1rem !important;
}
.zs-authors {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 0.3rem 1.6rem;
  font-size: 1.15rem; margin: 0 0 0.4rem !important;
}
.zs-authors a { color: var(--body-text-color) !important; text-decoration: none !important; }
.zs-authors a:hover { color: var(--link-text-color) !important; }
.zs-head sup { color: var(--body-text-color-subdued); font-size: 0.62em; }
.zs-affil { color: var(--body-text-color-subdued); font-size: 1.05rem; margin: 0 0 1.25rem !important; }
.zs-affil sup + sup, .zs-affil sup:not(:first-child) { margin-left: 0.6rem; }
.zs-links {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 0.6rem; margin: 0 0 1.5rem;
}
.zs-links a {
  display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.5rem 1.05rem;
  border: 1px solid var(--block-border-color); border-radius: 999px;
  color: var(--body-text-color) !important; text-decoration: none !important;
  font-size: 1rem; font-weight: 500;
}
.zs-links a:hover { color: var(--link-text-color) !important; border-color: var(--link-text-color); }
.zs-pitch {
  max-width: 78ch; margin: 0 auto 0.5rem !important; font-size: 1.08rem; line-height: 1.65;
}
.zs-pitch strong { color: var(--link-text-color); font-weight: 600; }
"""

HEADER = """
<div class="zs-head">
  <h1>ZipSplat: Fewer Gaussians, Better Splats</h1>
  <p class="zs-authors">
    <a href="https://veichta.com" target="_blank">Alexander Veicht<sup>1</sup></a>
    <a href="https://sunghwanhong.github.io" target="_blank">Sunghwan Hong<sup>1</sup></a>
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ" target="_blank">
      D&aacute;niel Bar&aacute;th<sup>1</sup></a>
    <a href="https://people.inf.ethz.ch/marc.pollefeys/" target="_blank">
      Marc Pollefeys<sup>1,2</sup></a>
  </p>
  <p class="zs-affil"><sup>1</sup>ETH Z&uuml;rich <sup>2</sup>Microsoft</p>
  <div class="zs-links">
    <a href="https://arxiv.org/abs/2606.05102" target="_blank">&#x1F4C4; Paper</a>
    <a href="https://veichta.com/zipsplat" target="_blank">&#x1F310; Project Page</a>
    <a href="https://github.com/cvg/ZipSplat" target="_blank">&#x1F4BB; Code</a>
    <a href="https://huggingface.co/veichta/zipsplat" target="_blank">&#x1F917; Weights</a>
  </div>
  <p class="zs-pitch">
    Upload a few photos or a short video of a scene. ZipSplat predicts <strong>3D Gaussians and
    the cameras</strong>, without poses or intrinsics, and refines both with
    TokenBA. Keep fewer <strong>Gaussians</strong> to compress the scene, and color them by
    <strong>token groups</strong> to see how each token places its Gaussians.
  </p>
</div>
"""

with gr.Blocks(title="ZipSplat", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER)
    views, scene = gr.State([]), gr.State(None)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            files = gr.File(
                label="Images or a video", file_count="multiple", file_types=["image", "video"]
            )
            gallery = gr.Gallery(label="Input views", columns=4, height=260, object_fit="cover")
            max_views = gr.Slider(1, 32, value=24, step=1, label="Views")
            compression = gr.Slider(1, 100, value=100, step=1, label="Gaussians kept (%)")
            count = gr.Markdown(elem_classes="zs-count")
            tokens, nviews = gr.Number(visible=False), gr.Number(visible=False)
            with gr.Row():
                tokenba = gr.Checkbox(value=True, label="TokenBA", scale=0, min_width=120)
                run = gr.Button("Reconstruct", variant="primary", size="sm", scale=1)
            # the example thumbnails stand for their input files
            thumb = gr.Image(type="filepath", visible=False)
            gr.Examples(
                examples=[[thumbnail(name)] for name in EXAMPLES],
                inputs=[thumb],
                fn=load_example,
                outputs=[files],
                run_on_click=True,
                examples_per_page=len(EXAMPLES),
                label="Examples",
            )
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("3D viewer"):
                    viewer = gr.HTML(viewer_html(None))
                    color = gr.Radio(
                        ["RGB", "Token groups"], value="RGB", show_label=False, container=False
                    )
                with gr.Tab("Fly-through"):
                    video = gr.Video(show_label=False, autoplay=True, loop=True, interactive=False)
            ply = gr.File(label="Gaussians (.ply)", interactive=False)

    counting = dict(fn=None, inputs=[tokens, compression, nviews], outputs=count, js=COUNT_JS)
    prepared = [gallery, views, tokens, nviews]
    files.change(prepare, [files, max_views], prepared).then(**counting)
    max_views.release(prepare, [files, max_views], prepared).then(**counting)
    compression.change(**counting)
    run.click(
        reconstruct,
        [files, views, max_views, compression, tokenba, color],
        [scene, viewer, video, ply],
    )
    color.change(show, [scene, color], viewer)


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(TMP)])
