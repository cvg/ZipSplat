"""Convert GObjaverse per-object tars to canonical tar shards.

Reads the downloaded per-object tars directly (no extraction). Raw layout:
  <data-root>/<dict_id>/<inst_id>.tar with members
  <inst_id>/campos_512_v4/NNNNN/NNNNN{.json,.png,_nd.exr,...}, 40 views per object
  (24-view orbit + top + bottom + 14-view orbit, all looking at the origin).

Conventions applied (each verified on the data by cross-view depth reprojection):
  - Poses: json x/y/z axes are ALREADY OpenCV (x right, y down, z forward; the
    LGM-style loaders negate y/z to get OpenGL). c2w = [x y z | origin] ->
    Pose.data_ [R.flatten(), t].
  - Cameras: 512x512, f = (W/2) / tan(x_fov / 2) (full-angle fov; 0.6912 rad =
    Blender 50mm/36mm, x_fov == y_fov). The json has NO principal point; the renders
    are pp-centered: a cross-view depth-agreement scan over pp offsets has its
    minimum within 0.5 px of (256, 256) on both axes (< the 1.5 px tolerance of the
    live converts), so cx = cy = 256 exactly and no pp-centering crop/resize.
    Camera.data_ [w, h, fx, fy, cx, cy].
  - RGB: the PNG color is the tone-mapped PREMULTIPLIED render (edge pixels are
    color * alpha over black), so white compositing is rgb + (255 - alpha), not
    rgb * a + 255 * (1 - a) (that darkens silhouettes). Lossless WebP.
  - Depth: _nd.exr alpha channel = planar z (not ray distance), also premultiplied
    by alpha at partially covered pixels -> z = d / a where a >= 0.5, else -1
    (background is 0). Log-uint8 WebP.

Splits: per-object hash of (seed, key) -> ~--test-fraction test, stable when more
objects are downloaded later.

Usage:
    python -m splatfactory.datasets.scripts.gobjaverse.convert --num-workers 16
    python -m splatfactory.datasets.scripts.gobjaverse.convert --limit 5 --split test

Author: Alexander Veicht
"""

import argparse
import json
import os
import tarfile
import zlib
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # must be set before cv2 is imported

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from splatfactory import get_logger  # noqa: E402
from splatfactory.datasets.scripts.utils import parallel_map, write_shards  # noqa: E402
from splatfactory.datasets.utils.io import encode_depth, encode_image  # noqa: E402

logger = get_logger(__name__)

IMG_SIZE = 512
MIN_DEPTH_ALPHA = 128  # below: pixel is mostly background, premultiplied depth unreliable
SUFFIXES = (".json", ".png", "_nd.exr")


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def read_views(tar_path: Path) -> dict[str, dict[str, bytes]]:
    """view id -> {suffix: bytes} for the files we need, read straight from the tar."""
    views: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile() or "/campos_512_v4/" not in m.name:
                continue
            view, fname = m.name.split("/")[-2:]
            suffix = fname[len(view) :]
            if suffix in SUFFIXES:
                views.setdefault(view, {})[suffix] = tf.extractfile(m).read()
    return {v: f for v, f in sorted(views.items()) if all(s in f for s in SUFFIXES)}


def process_scene(task: tuple[str, str], data_root: Path) -> dict:
    dict_id, inst_id = task
    views = read_views(data_root / dict_id / f"{inst_id}.tar")
    if not views:
        raise ValueError(f"{dict_id}/{inst_id}: no complete views")

    images, depths, ranges, poses, cameras = [], [], [], [], []
    for files in views.values():
        cam = json.loads(files[".json"])
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = cam["x"], cam["y"], cam["z"]
        c2w[:3, 3] = cam["origin"]
        poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))
        fx = (IMG_SIZE / 2) / np.tan(cam["x_fov"] / 2)
        fy = (IMG_SIZE / 2) / np.tan(cam["y_fov"] / 2)
        cameras.append([IMG_SIZE, IMG_SIZE, fx, fy, IMG_SIZE / 2, IMG_SIZE / 2])

        bgra = _decode(files[".png"])
        assert bgra.shape == (IMG_SIZE, IMG_SIZE, 4) and bgra.dtype == np.uint8, bgra.shape
        alpha = bgra[..., 3:].astype(np.uint16)
        bgr = np.minimum(bgra[..., :3].astype(np.uint16) + (255 - alpha), 255).astype(np.uint8)
        images.append(encode_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

        depth = _decode(files["_nd.exr"])[..., 3].astype(np.float32)  # BGRA -> A = depth
        a = alpha[..., 0].astype(np.float32) / 255
        valid = (alpha[..., 0] >= MIN_DEPTH_ALPHA) & (depth > 0) & np.isfinite(depth)
        depth = np.where(valid, depth / np.maximum(a, 1e-6), -1.0).astype(np.float32)
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    return {
        "key": f"{dict_id}_{inst_id}",
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
    }


def split_of(key: str, seed: int, test_fraction: float) -> str:
    return (
        "test" if zlib.crc32(f"{seed}:{key}".encode()) % 10000 < test_fraction * 10000 else "train"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert GObjaverse to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/gobjaverse/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gobjaverse"))
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max objects per split")
    args = parser.parse_args()

    tasks = sorted((p.parent.name, p.stem) for p in args.data_root.glob("*/*.tar"))
    if not tasks:
        raise SystemExit(f"no object tars in {args.data_root}")
    logger.info(f"Found {len(tasks)} objects in {args.data_root}")

    for split in args.split:
        split_tasks = [
            t for t in tasks if split_of(f"{t[0]}_{t[1]}", args.seed, args.test_fraction) == split
        ]
        if args.limit:
            split_tasks = split_tasks[: args.limit]
        if not split_tasks:
            continue

        out_dir = args.output_dir / f"{IMG_SIZE}x{IMG_SIZE}" / f"{split}-scenes"
        index_path = out_dir / "index.json"
        if args.overwrite:
            for p in out_dir.glob("shard-*.tar*"):
                p.unlink()
            index_path.unlink(missing_ok=True)
        done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
        remaining = [t for t in split_tasks if f"{t[0]}_{t[1]}" not in done]
        if not remaining:
            logger.info(f"Skipping {split}: all {len(split_tasks)} objects done")
            continue
        logger.info(f"Converting {split}: {len(remaining)} objects -> {out_dir}")

        scenes = parallel_map(
            partial(process_scene, data_root=args.data_root),
            remaining,
            num_workers=args.num_workers,
            skip_label=lambda t: f"{t[0]}/{t[1]}",
        )
        write_shards(
            scenes,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"gobjaverse/{split}",
            total=len(split_tasks),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
