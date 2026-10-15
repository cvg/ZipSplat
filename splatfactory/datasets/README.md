# Datasets

Every dataset is converted into the same format: tar shards of scenes (images, cameras, poses and
depth) under `data/<dataset>/<HxW>/{train,test}-scenes/`, which is where the configs expect them.

## Flow

Each dataset has a `download` step (fetch the raw data) and a `convert` step (pack it into shards).
Both take `--num-workers N`, and `convert --split test` is enough for evaluation:

```bash
python -m splatfactory.datasets.scripts.<dataset>.download
python -m splatfactory.datasets.scripts.<dataset>.convert --num-workers 8
```

Depth comes with the source for most datasets. Where it does not (RE10K, DL3DV), add it with DA3,
which fills in only the scenes without depth, needs a GPU and the `depth-anything-3` package, and
runs on several GPUs with `torchrun --nproc_per_node=N`:

```bash
python -m splatfactory.datasets.scripts.extract_depth data/<dataset>/<HxW>/train-scenes
```

Datasets sampled by covisibility (ScanNet++, ARKitScenes, Hypersim, UnrealStereo4K) also need view
rankings:

```bash
python -m splatfactory.datasets.scripts.precompute_rankings data/<dataset>/<HxW>/train-scenes
```

Images are resized to the training resolution on the fly. To save CPU and IO during training,
resize a dataset once and point the config at the copy:

```bash
python -m splatfactory.datasets.scripts.resize data/dl3dv/540x960/train-scenes \
    --output data/dl3dv/252x252/train-scenes --size 252 --num-workers 32
```

| Dataset                                                                               | `<dataset>`      | Shards                                     |
| ------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------ |
| [RE10K](https://google.github.io/realestate10k/)                                      | `re10k`          | `data/re10k/360x640`                       |
| [DL3DV](https://github.com/DL3DV-10K/Dataset) (gated, `huggingface-cli login` first)  | `dl3dv`          | `data/dl3dv/540x960`                       |
| [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) (see below)                       | `scannetpp`      | `data/scannetpp_v2/720x1080/{dslr,iphone}` |
| [ARKitScenes](https://github.com/apple/ARKitScenes)                                   | `arkitscenes`    | `data/arkitscenes/{480x640,640x480}`       |
| [Hypersim](https://github.com/apple/ml-hypersim)                                      | `hypersim`       | `data/ml-hypersim/768x1024`                |
| [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets)                             | `unrealstereo4k` | `data/unrealstereo4k/540x960`              |
| [BlendedMVS](https://github.com/YoYo000/BlendedMVS)                                   | `blendedmvs`     | `data/blendedmvs/576x768`                  |
| [CO3D](https://github.com/facebookresearch/co3d)                                      | `co3d`           | `data/co3d/720s`                           |
| [WildRGB-D](https://github.com/wildrgbd/wildrgbd)                                     | `wildrgbd`       | `data/wildrgbd/640x480`                    |
| [G-Objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse) | `gobjaverse`     | `data/gobjaverse/512x512`                  |

Each dataset comes with its own terms, which the scripts do not change: they only fetch from the
official hosts.

## ScanNet++

ScanNet++ is distributed under a signed agreement, so there is no download script. Fetch the scenes
with the official [toolkit](https://github.com/scannetpp/scannetpp) into `data/scannetpp_v2/`
(DSLR `resized_images`, iPhone `rgb.mkv`, the nerfstudio `transforms.json` files and
`mesh_aligned_0.05.ply`), then convert. The converter undistorts the frames and renders the mesh
depth itself (needs `open3d`):

```bash
python -m splatfactory.datasets.scripts.scannetpp.convert --device dslr --max-side 1080 --num-workers 12
python -m splatfactory.datasets.scripts.scannetpp.convert --device iphone --max-side 1080 --num-workers 12
```
