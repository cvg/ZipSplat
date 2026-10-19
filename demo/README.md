---
title: ZipSplat
emoji: ✨
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
python_version: "3.12"
app_file: demo/app.py
pinned: false
license: cc-by-nc-4.0
short_description: A few unposed photos or a video in, 3D Gaussians out
models:
- veichta/zipsplat
preload_from_hub:
- veichta/zipsplat zipsplat-da3g-504.tar
---

# ZipSplat demo

Upload a few photos or a short video of a scene. ZipSplat predicts 3D Gaussians and the cameras
in one forward pass, without poses or intrinsics, and refines both with TokenBA. The scene opens in
an in-browser splat viewer at the first input view, next to a fly-through video along the predicted
cameras. Keep fewer Gaussians to compress the scene, and color them by token groups to see how each
token places its Gaussians.

- Paper: [ZipSplat: Fewer Gaussians, Better Splats](https://arxiv.org/abs/2606.05102)
- Project page: [veichta.com/zipsplat](https://veichta.com/zipsplat)
- Code: [cvg/ZipSplat](https://github.com/cvg/ZipSplat)
- Weights: [veichta/zipsplat](https://huggingface.co/veichta/zipsplat), CC BY-NC 4.0

## Running locally

From the repository root, with the `zipsplat` package importable:

```bash
PYTHONPATH=. python demo/app.py
```

The weights (6 GB) download from the Hugging Face Hub on the first start and are cached. The
demo stores them in bf16 except for the two heads the model runs in fp32, which halves the GPU
memory at no visible change.

## The Space

The Space mirrors this layout: `demo/` (`app.py`, `fly_through.py`, `viewer.html`),
`assets/examples/`, this file as its `README.md` and `demo/requirements.txt` as its
`requirements.txt`. It runs on ZeroGPU (RTX PRO 6000, Python 3.12, torch 2.8).

gsplat compiles its CUDA kernels on first import and the Space has no compiler, so the
requirements install a prebuilt wheel from the weights repository
(`veichta/zipsplat`, `wheels/gsplat-1.5.3-cp312-cp312-linux_x86_64.whl`), built from the gsplat
1.5.3 source distribution with CUDA 12.8 for `TORCH_CUDA_ARCH_LIST="8.9;12.0"`.

## Examples

- `office`: five photos of an office, by the authors.
- `motorcycle`: sequence `motorcycle/106_12667_24728` of
  [CO3D](https://github.com/facebookresearch/co3d) (Reizenstein et al., ICCV 2021), CC BY-NC 4.0.
- `statues`: scene `59817e4a1bd4b175e7038d19` of
  [BlendedMVS](https://github.com/YoYo000/BlendedMVS) (Yao et al., CVPR 2020), CC BY 4.0. Its
  images are mesh renders blended with the photos, which leaves a light rim around the statues.
