<p align="center">
  <h1 align="center">ZipSplat: Fewer Gaussians, Better Splats</h1>
  <p align="center">
    <a href="https://veichta.com">Alexander Veicht</a>
    &middot; <a href="https://sunghwanhong.github.io">Sunghwan Hong</a>
    &middot; <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">D&aacute;niel Bar&aacute;th</a>
    &middot; <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
    <br>
    <sub>ETH Z&uuml;rich &nbsp;&middot;&nbsp; Microsoft</sub>
  </p>
  <h2 align="center">
    <a href="https://arxiv.org/abs/2606.05102">Paper</a>
    |
    <a href="https://veichta.com/zipsplat">Project Page</a>
  </h2>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="ZipSplat teaser" width="100%">
  <br>
  <em>ZipSplat decouples Gaussian placement from the pixel grid, reconstructing a scene from a few
  unposed images in under a second, with higher quality from far fewer Gaussians.</em>
</p>

---

ZipSplat is a feed-forward model for 3D Gaussian Splatting: it reconstructs a scene from a few
unposed images in a single forward pass, with no poses, intrinsics, or per-scene optimization.
Instead of emitting one Gaussian per pixel, it compresses the input views into a compact set of
scene tokens and decodes each into a small group of Gaussians, reaching state-of-the-art quality
with far fewer Gaussians and a single knob to trade quality for size. This repository hosts the
[inference](#inference), [evaluation](#evaluation), and [training](#training) code for ZipSplat,
together with our pretrained weights.

## Setup and demo

We provide a standalone inference package [`zipsplat`](zipsplat) that requires only minimal
dependencies and Python >= 3.10. First clone the repository and install it:

```bash
git clone https://github.com/cvg/ZipSplat.git && cd ZipSplat
python -m pip install -e .
# optional: xformers' fused SwiGLU kernel the released checkpoint was trained with (small speedup)
python -m pip install -e ".[xformers]"
```

Rendering uses [gsplat](https://github.com/nerfstudio-project/gsplat) and requires a CUDA GPU. 
The pretrained weights are downloaded automatically on first use.

The easiest way to try ZipSplat is the interactive viewer: reconstruct a scene and explore it live
in your browser.
Drag the **compression** slider to trade Gaussians for quality in real time, and
toggle **token coloring** to see how each token's group of Gaussians is placed:

```bash
pip install "zipsplat[viewer]"
python -m zipsplat.viewer assets/examples/drone.mp4   # or assets/examples/office/, or your own dir/glob/video
```

## Inference

Here is a minimal usage example:

```python
import math, torch
from zipsplat import ZipSplat, Camera, Pose, load_image, viz

model = ZipSplat(weights="zipsplat").cuda().eval()
images = [load_image(p) for p in paths] # raw images, any size (auto-cropped and resized)
pred = model(images)                    # forward pass + test-time BA, no poses or intrinsics needed
gaussians = pred["gaussians"]           # 3D Gaussians
poses, cameras = pred["poses"], pred["cameras"]  # c2w poses + intrinsics of the input images

# render a novel view
camera = Camera.from_fov(math.radians(60), w=512, h=512)
pose = Pose.from_Rt(torch.eye(3), torch.zeros(3)) # identity pose = the first input view
rgb, info = gaussians.render(camera, pose)

# export
gaussians.save_ply("scene.ply")                             # open in any 3DGS viewer
viz.turntable(gaussians, "turntable.mp4", sweep_deg=None)   # wiggle orbit video
```

Everything lives in the frame of the first input view. `gaussians` is a
[`Gaussians`](zipsplat/gaussians.py) object. 
`gaussians.save_ply(...)` writes a standard 3DGS `.ply` you can drop into [SuperSplat](https://playcanvas.com/supersplat/editor) or any Gaussian-splat viewer. 

<details>
<summary><b>Input: images, a video, or a single image</b></summary>

```python
from zipsplat import load_image, load_video

# multiple images (any sizes: center-cropped to the nearest training aspect ratio and resized)
gaussians = model([load_image(p) for p in paths])["gaussians"]

# a single image
gaussians = model([load_image("photo.jpg")])["gaussians"]

# a video clip (evenly samples num_frames across the clip)
gaussians = model(load_video("assets/examples/drone.mp4", num_frames=24))["gaussians"]
```
The model handles 1-24+ views. More views give wider coverage, fewer give a tighter reconstruction.
</details>

<details>
<summary><b>Compression: fewer Gaussians</b></summary>

`compression` is the query-sampling ratio in `(0, 1]`. `1.0` (default) uses every token. Lower values
run k-means to keep a subset, shrinking the Gaussian count with graceful quality falloff.

```python
gaussians = model(images, compression=1.0)["gaussians"]   # full quality
gaussians = model(images, compression=0.25)["gaussians"]  # ~4x fewer Gaussians
print(gaussians.num_gaussians)
```
</details>

<details>
<summary><b>Rendering</b></summary>

```python
# a single view (scalar camera/pose) or a batch of views ([V])
rgb, info = gaussians.render(camera, pose)               # mode="RGB" by default
rgbd, _ = gaussians.render(camera, pose, mode="RGB+ED")  # append expected depth
```

Cameras/poses are moved to the scene's device automatically. Build cameras with `Camera.from_fov`, `Camera.from_focal`, or `Camera.from_K`.
</details>

<details>
<summary><b>Turntable video</b></summary>

```python
from zipsplat import viz

viz.turntable(gaussians, "orbit.mp4")                  # full 360-degree turntable
viz.turntable(gaussians, "wiggle.mp4", sweep_deg=None) # gentle wiggle (best for front-facing scenes)
```
</details>

<details>
<summary><b>TokenBA</b></summary>

By default (`ba=True`) the prediction is refined against the input views (TokenBA): the input
poses are re-solved by robust PnP on the token-patch correspondences plus a short photometric
polish, then the scene tokens and poses are optimized jointly through the frozen Gaussian head. It
stays pose-free and needs a CUDA GPU (pass `ba=False` on CPU). Pass a dict instead of `True` to
change TokenBA's settings (see [`zipsplat/token_ba.py`](zipsplat/token_ba.py)):

```python
pred = model(images, ba=False)                     # feed-forward only
pred = model(images, ba={"verbose": True})         # progress bars
pred = model(images, ba={"joint_steps": 32})       # shorter joint refinement (default 96)
pred = model(images, ba={"joint_tol": None})       # no early stopping: run all joint steps
pred = model(images, ba={"rounds": 8})             # more camera / token rounds (default 4)

pred = model(images, ba=False)                     # or refine later, e.g. with other settings
refined = model.refine(pred, joint_steps=32, verbose=True)
```
</details>

## Models

| `weights=`     | Backbone  | Input                          | Gaussians / token | Training data                                                                                            |
| -------------- | --------- | ------------------------------ | ----------------- | -------------------------------------------------------------------------------------------------------- |
| `zipsplat`     | DA3-Giant | long edge 504, 7 aspect ratios | 16                | RE10K, DL3DV, ScanNet++, ARKitScenes, Hypersim, UnrealStereo4K, BlendedMVS, CO3D, WildRGB-D, G-Objaverse |
| `zipsplat-252` | DA3-Giant | 252 x 252                      | 32                | RE10K, DL3DV                                                                                             |

`zipsplat` is the default and the model to use in the wild. It decodes 16 Gaussians per token
instead of the paper's 32 to keep the Gaussian count reasonable at the higher resolution. The
paper's numbers are from `zipsplat-252`. The weights are hosted on the [Hugging Face Hub](https://huggingface.co/veichta/zipsplat)
and fetched on first use. You can also pass a local path or URL: `ZipSplat(weights="path/to/checkpoint.tar")`.

## Training and evaluation

The training and evaluation code lives in [`splatfactory`](splatfactory), a general library for
feed-forward Gaussian Splatting. It streams large multi-view datasets, composes models and
losses through Hydra configs, trains across multiple GPUs, and benchmarks novel-view synthesis on a
suite of standard datasets. ZipSplat is the model we train with it, and the recipes below reproduce the
released checkpoint and the paper tables.

### Setup

`splatfactory` needs a CUDA toolchain to build two compiled extensions
([gsplat](https://github.com/JoannaCCJH/gsplat) for rasterization and
[fused-ssim](https://github.com/rahul-goel/fused-ssim) for the SSIM loss), so install in this order:

```bash
# from the repo root
conda create -n splatfactory python=3.12 -y && conda activate splatfactory
conda install -c nvidia cuda-toolkit -y                      # provides nvcc for the kernels below

# 1) torch first (CUDA build), then the training package + its core deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e splatfactory/

# 2) CUDA extensions need a built torch present, so disable build isolation
pip install --no-build-isolation --use-pep517 \
    git+https://github.com/JoannaCCJH/gsplat.git \
    git+https://github.com/rahul-goel/fused-ssim/
python -c "from gsplat.cuda._backend import _C"              # sanity check

# optional: xformers' fused SwiGLU kernel the released checkpoint was trained with
pip install -e "splatfactory/[xformers]"
```

We use a [gsplat fork](https://github.com/JoannaCCJH/gsplat) rather than upstream because it exposes
a per-Gaussian `activated` flag from the rasterizer. Training uses it to detach Gaussians that were
already rendered from the geometry (chamfer) loss, so that term only supervises the Gaussians the
images don't already constrain. Inference works with stock gsplat. This fork only matters for
training.

### Data

Every dataset is converted into the same format, tar shards of scenes under
`data/<dataset>/<HxW>/{train,test}-scenes/`, with one `download` and one `convert` script per
dataset and DA3 depth added by `extract_depth` where the source has none. The commands for every
dataset are in [`splatfactory/datasets/README.md`](splatfactory/datasets/README.md). Evaluation
needs only the test splits.

## Evaluation

With the data prepared, evaluate the **released model** with the `zipsplat_eval` config. Its weights
are downloaded automatically, so no local checkpoint is needed:

```bash
python -m splatfactory.eval.run <benchmark> --conf zipsplat_eval eval.num_views=<N>
```

Each benchmark renders held-out target views and reports PSNR, SSIM, and LPIPS, writing results to
`outputs/results/`. `eval.num_views=N` sets how many context views the model sees, while the scenes
and context/target views are fixed by the indices in `splatfactory/eval/indices/`, so runs are
exactly reproducible. To score one of your own training runs instead, pass `--checkpoint <exp>` in
place of `--conf zipsplat_eval`. The per-benchmark commands below cover the protocols we report:

<details>
<summary><b>DL3DV</b> (6 / 12 / 24 context views)</summary>

Evaluate the released model (append `model.eval_use_priors=true` to feed ground-truth camera priors):

```bash
python -m splatfactory.eval.run dl3dv_benchmark --conf zipsplat_eval eval.num_views=6
python -m splatfactory.eval.run dl3dv_benchmark --conf zipsplat_eval eval.num_views=12
python -m splatfactory.eval.run dl3dv_benchmark --conf zipsplat_eval eval.num_views=24
# with ground-truth camera priors:
python -m splatfactory.eval.run dl3dv_benchmark --conf zipsplat_eval eval.num_views=6 model.eval_use_priors=true
```

Evaluate your own checkpoint:

```bash
python -m splatfactory.eval.run dl3dv_benchmark --checkpoint <exp_name> eval.num_views=6
```

Results (PSNR / SSIM / LPIPS, pose-free, with DA3 and YoNoSplat for reference):

| Method                | 6 views               | 12 views              | 24 views              |
| --------------------- | --------------------- | --------------------- | --------------------- |
| DA3                   | 23.77 / 0.795 / 0.165 | 22.38 / 0.736 / 0.208 | 21.69 / 0.711 / 0.229 |
| YoNoSplat             | 24.10 / 0.783 / 0.160 | 22.73 / 0.736 / 0.200 | 22.01 / 0.710 / 0.223 |
| **ZipSplat**          | 25.24 / 0.804 / 0.172 | 24.27 / 0.767 / 0.197 | 24.14 / 0.768 / 0.198 |
| **ZipSplat + priors** | 25.34 / 0.810 / 0.169 | 24.37 / 0.773 / 0.194 | 24.23 / 0.773 / 0.194 |

</details>

<details>
<summary><b>RE10K</b> (6 context views)</summary>

Evaluate the released model (append `model.eval_use_priors=true` to feed ground-truth camera priors):

```bash
python -m splatfactory.eval.run re10k_benchmark --conf zipsplat_eval eval.num_views=6
# with ground-truth camera priors:
python -m splatfactory.eval.run re10k_benchmark --conf zipsplat_eval eval.num_views=6 model.eval_use_priors=true
```

Evaluate your own checkpoint:

```bash
python -m splatfactory.eval.run re10k_benchmark --checkpoint <exp_name> eval.num_views=6
```

Results (pose-free, with DA3 and YoNoSplat for reference):

| Method                | PSNR  | SSIM  | LPIPS |
| --------------------- | ----- | ----- | ----- |
| DA3                   | 20.90 | 0.724 | 0.234 |
| YoNoSplat             | 24.99 | 0.835 | 0.151 |
| **ZipSplat**          | 26.20 | 0.842 | 0.158 |
| **ZipSplat + priors** | 27.19 | 0.872 | 0.143 |

</details>

<details>
<summary><b>MipNeRF360</b> (32 / 64 / 128 context views)</summary>

Evaluate the released model (append `model.eval_use_priors=true` to feed ground-truth camera priors):

```bash
python -m splatfactory.eval.run mipnerf360_benchmark --conf zipsplat_eval eval.num_views=32
python -m splatfactory.eval.run mipnerf360_benchmark --conf zipsplat_eval eval.num_views=64
python -m splatfactory.eval.run mipnerf360_benchmark --conf zipsplat_eval eval.num_views=128
# with ground-truth camera priors:
python -m splatfactory.eval.run mipnerf360_benchmark --conf zipsplat_eval eval.num_views=32 model.eval_use_priors=true
```

Evaluate your own checkpoint:

```bash
python -m splatfactory.eval.run mipnerf360_benchmark --checkpoint <exp_name> eval.num_views=32
```

Results (PSNR / SSIM / LPIPS, pose-free, with DA3 and YoNoSplat for reference):

| Method                | 32 views              | 64 views              | 128 views             |
| --------------------- | --------------------- | --------------------- | --------------------- |
| DA3                   | 20.94 / 0.577 / 0.295 | 20.30 / 0.554 / 0.311 | 20.19 / 0.568 / 0.306 |
| YoNoSplat             | 17.62 / 0.409 / 0.465 | 17.77 / 0.413 / 0.466 | 17.16 / 0.409 / 0.509 |
| **ZipSplat**          | 21.72 / 0.594 / 0.325 | 22.18 / 0.615 / 0.298 | 22.29 / 0.624 / 0.290 |
| **ZipSplat + priors** | 22.95 / 0.655 / 0.276 | 23.31 / 0.675 / 0.260 | 23.37 / 0.683 / 0.255 |

</details>



## Training

ZipSplat is initialized from the DA3-Giant backbone, so first download those weights into `weights/`
and convert them to the `.pth` layout the backbone expects:

```bash
huggingface-cli download depth-anything/DA3-GIANT model.safetensors --local-dir weights
mv weights/model.safetensors weights/da3-giant.safetensor
python -m splatfactory.models.encoders.dav3_encoder --size giant   # -> weights/da3-giant.pth
```

With the data prepared (including the depth maps) and the backbone in place, start training from the
repo root:

```bash
# single GPU
python -m splatfactory.train zipsplat --conf zipsplat
# multiple GPUs on one node
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat --conf zipsplat
```

`zipsplat` is the experiment name (checkpoints are written to `outputs/training/zipsplat/`), use any
name you like. The released model trains on the 50/50 RE10K + DL3DV mix at 252px for 450K steps. Our
run used 16 GPUs (4 nodes x 4) at a global batch size of 384, so scale `data.batch_size` to your
hardware. Add `--restore` to resume an interrupted run. Configs are [Hydra](https://hydra.cc), so
anything can be overridden from the command line, e.g. a smaller backbone:

```bash
python -m splatfactory.models.encoders.dav3_encoder --size small   # -> weights/da3-small.pth
python -m splatfactory.train zipsplat-da3s --conf zipsplat model/backbone=da3s data.batch_size=96
```

To log training to [TensorBoard](https://www.tensorflow.org/tensorboard) or
[Weights & Biases](https://wandb.ai), set `train.writer`:

```bash
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat --conf zipsplat train.writer=tensorboard
```

The trained model can then be evaluated by its experiment name (see [Evaluation](#evaluation)):

```bash
python -m splatfactory.eval.run dl3dv_benchmark --checkpoint zipsplat --tag zipsplat-retrained eval.num_views=6
```

## BibTeX

```bibtex
@article{veicht2026zipsplat,
  title   = {ZipSplat: Fewer Gaussians, Better Splats},
  author  = {Veicht, Alexander and Hong, Sunghwan and Bar{\'a}th, D{\'a}niel and Pollefeys, Marc},
  journal = {arXiv preprint arXiv:2606.05102},
  year    = {2026}
}
```

## License

The **code** in this repository (the `zipsplat` and `splatfactory` packages) is released under the
[Apache-2.0 License](LICENSE).

The **pretrained weights** are released separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), **non-commercial use only**. This
is required by their dependencies: the released checkpoint is initialized from
[DA3-Giant](https://huggingface.co/depth-anything/DA3-GIANT) (CC BY-NC 4.0) and trained on
[DL3DV-10K](https://github.com/DL3DV-10K/Dataset) (CC BY-NC 4.0). See the
[Hugging Face model card](https://huggingface.co/veichta/zipsplat) for the full weights license and
dataset attribution.

## Acknowledgements

`splatfactory` is built mainly on the excellent [glue-factory](https://github.com/cvg/glue-factory)
training framework. Our model is initialized from the
[Depth Anything 3](https://huggingface.co/depth-anything/DA3-GIANT) backbone, and rendering uses
[gsplat](https://github.com/nerfstudio-project/gsplat). We thank the authors of these projects for
releasing their code and models.
