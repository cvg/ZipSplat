<h1 align="center">ZipSplat: Fewer Gaussians, Better Splats</h1>

<p align="center">
  <a href="https://veichta.com">Alexander Veicht</a><sup>1</sup>
  &nbsp;&middot;&nbsp;
  <a href="https://sunghwanhong.github.io">Sunghwan Hong</a><sup>1</sup>
  &nbsp;&middot;&nbsp;
  <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">D&aacute;niel Bar&aacute;th</a><sup>1</sup>
  &nbsp;&middot;&nbsp;
  <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>
  <br>
  <sup>1</sup>ETH Z&uuml;rich &nbsp;&nbsp; <sup>2</sup>Microsoft
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2606.05102">📄 Paper</a>
  &nbsp;|&nbsp;
  <a href="https://veichta.com/zipsplat">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/veichta/zipsplat">🤗 Weights</a>
</h3>

<p align="center">
  <img src="https://github.com/cvg/ZipSplat/releases/download/assets-v1/teaser.avif" alt="ZipSplat teaser" width="100%">
  <br>
  <em>ZipSplat turns unposed images into 3D Gaussians in seconds, with high quality and a Gaussian budget you choose at inference.</em>
</p>

---

ZipSplat reconstructs a 3D Gaussian Splatting scene from a handful of photos or a short video,
without camera poses or calibration. It predicts the scene and the cameras in one pass and then
refines both for a few seconds, reaching state-of-the-art quality. How many Gaussians to use is a
single knob you can turn at inference, from the full-quality scene down to a tiny one.
This repository contains a lightweight [inference](#inference) package with an interactive viewer,
our pretrained weights (the paper model and a higher-resolution model for images in the wild),
and the full [evaluation](#evaluation) and [training](#training) code.

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
python -m pip install -e ".[viewer]"
python -m zipsplat.viewer assets/examples/office/   # or your own dir/glob/video
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
gaussians = model(load_video("clip.mp4", num_frames=24))["gaussians"]
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

`splatfactory` adds the training and evaluation dependencies on top of the inference package:

```bash
# from the repo root
conda create -n splatfactory python=3.12 -y && conda activate splatfactory
conda install -c nvidia cuda-toolkit -y   # nvcc for gsplat, which compiles its kernels on first use
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e splatfactory/
# optional: xformers' fused SwiGLU kernel the released checkpoint was trained with
pip install -e "splatfactory/[xformers]"
```

### Data

Every dataset is converted into the same format, tar shards of scenes under
`data/<dataset>/<HxW>/{train,test}-scenes/`, with one `download` and one `convert` script per
dataset and DA3 depth added by `extract_depth` where the source has none. The depth model and the
ScanNet++ mesh renderer come with `pip install -e "splatfactory/[data]"`. The commands for every
dataset are in [`splatfactory/datasets/README.md`](splatfactory/datasets/README.md). Evaluation
needs only the test splits.

## Evaluation

With the data prepared, evaluate the **released model** with the `zipsplat_eval` config. It runs
`zipsplat-252`, the paper model, followed by TokenBA, and downloads its weights automatically:

```bash
python -m splatfactory.eval.run <benchmark> --conf zipsplat_eval eval.num_views=<N>
# feed-forward only, without TokenBA
python -m splatfactory.eval.run <benchmark> --conf zipsplat_eval eval.num_views=<N> model.token_ba=null
# with ground-truth camera priors
python -m splatfactory.eval.run <benchmark> --conf zipsplat_eval eval.num_views=<N> model.use_priors=true
```

Each benchmark renders held-out target views and reports PSNR, SSIM, and LPIPS together with the
pose accuracy of the predicted cameras, writing results to `outputs/results/`. The target cameras
are aligned to the predicted scene before scoring. `eval.num_views=N` selects the shipped view
indices in `splatfactory/eval/indices/`, so runs are exactly reproducible. To score one of your own
training runs, pass `--checkpoint <exp>` in place of `--conf zipsplat_eval` (add
`model.name=networks.zipsplat_eval` for TokenBA). The tables below are the paper's numbers
(PSNR / SSIM / LPIPS), with DA3 and YoNoSplat for reference.

<details>
<summary><b>DL3DV</b> (16 / 32 / 64 / 128 context views)</summary>

The 140 test scenes, every 8th frame a target and the context views picked by k-means over the
rest:

```bash
python -m splatfactory.eval.run dl3dv_benchmark --conf zipsplat_eval eval.num_views=16  # 32, 64, 128
```

| Method               | 16 views              | 32 views              | 64 views              | 128 views             |
| -------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| DA3                  | 20.34 / 0.652 / 0.276 | 20.82 / 0.673 / 0.255 | 20.59 / 0.666 / 0.257 | 20.09 / 0.646 / 0.271 |
| YoNoSplat            | 20.36 / 0.638 / 0.299 | 20.89 / 0.664 / 0.278 | 21.00 / 0.674 / 0.275 | 20.42 / 0.661 / 0.293 |
| ZipSplat w/o TokenBA | 22.61 / 0.725 / 0.213 | 23.65 / 0.763 / 0.182 | 24.25 / 0.786 / 0.165 | 24.56 / 0.800 / 0.155 |
| **ZipSplat**         | 24.79 / 0.808 / 0.175 | 27.11 / 0.865 / 0.133 | 28.99 / 0.900 / 0.106 | 30.22 / 0.920 / 0.090 |

</details>

<details>
<summary><b>RE10K</b> (6 context views)</summary>

The 6-view protocol of YoNoSplat on 1,600 test scenes with 8 targets each:

```bash
python -m splatfactory.eval.run re10k_benchmark --conf zipsplat_eval eval.num_views=6
```

| Method               | PSNR  | SSIM  | LPIPS |
| -------------------- | ----- | ----- | ----- |
| DA3                  | 21.37 | 0.726 | 0.229 |
| YoNoSplat            | 25.05 | 0.830 | 0.154 |
| ZipSplat w/o TokenBA | 26.86 | 0.856 | 0.134 |
| **ZipSplat**         | 28.50 | 0.882 | 0.120 |

</details>

<details>
<summary><b>MipNeRF360</b> (32 / 64 / 128 context views, zero-shot)</summary>

The 6 scenes, 8 targets each and the context views picked by farthest point sampling:

```bash
python -m splatfactory.eval.run mipnerf360_benchmark --conf zipsplat_eval eval.num_views=32  # 64, 128
```

| Method               | 32 views              | 64 views              | 128 views             |
| -------------------- | --------------------- | --------------------- | --------------------- |
| DA3                  | 21.48 / 0.603 / 0.266 | 21.03 / 0.585 / 0.277 | 20.62 / 0.579 / 0.287 |
| YoNoSplat            | 17.61 / 0.411 / 0.484 | 17.73 / 0.417 / 0.479 | 17.54 / 0.416 / 0.494 |
| ZipSplat w/o TokenBA | 22.30 / 0.606 / 0.256 | 22.53 / 0.620 / 0.244 | 22.81 / 0.634 / 0.231 |
| **ZipSplat**         | 24.43 / 0.696 / 0.210 | 25.24 / 0.723 / 0.185 | 25.81 / 0.739 / 0.169 |

</details>

<details>
<summary><b>ScanNet++</b> (32 / 64 / 128 context views, zero-shot)</summary>

The iPhone captures of the 46 NVS validation rooms (see the ScanNet++ section of the
[datasets README](splatfactory/datasets/README.md)), 8 targets each and the context views picked by
farthest point sampling:

```bash
python -m splatfactory.eval.run scannetpp_benchmark --conf zipsplat_eval eval.num_views=32  # 64, 128
```

| Method               | 32 views              | 64 views              | 128 views             |
| -------------------- | --------------------- | --------------------- | --------------------- |
| DA3 (trained on it)  | 18.48 / 0.683 / 0.394 | 19.15 / 0.702 / 0.362 | 19.12 / 0.703 / 0.361 |
| YoNoSplat            | 16.13 / 0.622 / 0.516 | 16.29 / 0.626 / 0.506 | 16.12 / 0.623 / 0.513 |
| ZipSplat w/o TokenBA | 20.34 / 0.729 / 0.324 | 21.42 / 0.758 / 0.278 | 21.76 / 0.770 / 0.265 |
| **ZipSplat**         | 22.36 / 0.779 / 0.298 | 24.36 / 0.826 / 0.249 | 25.40 / 0.848 / 0.235 |

</details>



## Training

ZipSplat is initialized from the DA3-Giant backbone. Its retrained 1.1 weights ship inside DA3's
nested checkpoint, so download that and convert the backbone to the `.pth` layout splatfactory
expects:

```bash
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1 model.safetensors --local-dir weights/da3
python -m splatfactory.models.encoders.dav3_encoder --size giant --nested \
    --input weights/da3/model.safetensors --output weights/da3/da3-giant-1.1.pth
```

With the data prepared (including the depth maps) and the backbone in place, train the paper model
(`zipsplat-252`) from the repo root:

```bash
# single GPU
python -m splatfactory.train zipsplat --conf zipsplat
# multiple GPUs on one node
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat --conf zipsplat
```

`zipsplat` is the experiment name (checkpoints are written to `outputs/training/zipsplat/`), use any
name you like. The recipe trains on the 50/50 RE10K + DL3DV mix at 252 px for 450K steps. Our run
used 16 GPUs (4 nodes x 4) at a global batch size of 384, so scale `data.batch_size` to your
hardware. Add `--restore` to resume an interrupted run.

The released `zipsplat` model finetunes a run of the same recipe with 16 Gaussians per token on the
504 px mix of ten datasets for 100K steps:

```bash
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat-g16 --conf zipsplat model.gaussians_per_token=16
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat-504 --conf zipsplat_ft504
```

Configs are [Hydra](https://hydra.cc), so anything can be overridden from the command line, e.g. a
smaller backbone:

```bash
huggingface-cli download depth-anything/DA3-SMALL model.safetensors --local-dir weights/da3
python -m splatfactory.models.encoders.dav3_encoder --size small \
    --input weights/da3/model.safetensors --output weights/da3/da3-small.pth
python -m splatfactory.train zipsplat-da3s --conf zipsplat model/backbone=da3s data.batch_size=96
```

To log training to [TensorBoard](https://www.tensorflow.org/tensorboard) or
[Weights & Biases](https://wandb.ai), set `train.writer`:

```bash
torchrun --nproc_per_node=4 -m splatfactory.train zipsplat --conf zipsplat train.writer=tensorboard
```

The trained model can then be evaluated by its experiment name (see [Evaluation](#evaluation)):

```bash
python -m splatfactory.eval.run dl3dv_benchmark --checkpoint zipsplat eval.num_views=16
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
is required by their dependencies: the released checkpoints are initialized from
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
