"""Convert ML-Hypersim scene zips to canonical tar shards.

Each (scene, camera trajectory) pair becomes one scene record keyed
"<scene>_<cam>". All Hypersim cameras share the same intrinsics (768x1024,
60 deg FOV, principal point exactly centered), so images need NO resampling:
the tone-mapped preview JPEGs are passed through verbatim (lossless policy).

Conventions:
  - Poses: keyframe position * meters_per_asset_unit + orientation, OpenGL c2w
    -> OpenCV (negate rotation columns 1,2) -> Pose.data_ [R.flatten(), t].
  - Depth: frame.NNNN.depth_meters.hdf5 holds EUCLIDEAN distance -> converted
    to planar z via the fixed-intrinsics ray-norm factor, log-uint8 WebP.
  - Frames excluded from the public release (metadata_images.csv) are dropped.

Splits from metadata_images_split_scene_v1.csv (val merged into train).

Usage:
    python -m splatfactory.datasets.scripts.hypersim.convert --num-workers 4

Author: Alexander Veicht
"""

import argparse
import csv
import io as _io
import json
import os
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from splatfactory import get_logger  # noqa: E402
from splatfactory.datasets.scripts.utils import parallel_map, write_shards  # noqa: E402
from splatfactory.datasets.utils.io import encode_depth  # noqa: E402

logger = get_logger(__name__)

IMG_H, IMG_W = 768, 1024
FOCAL = float(np.sqrt(3.0) * IMG_W / 2.0)  # 60 deg horizontal FOV
_py, _px = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
EUCLIDEAN_TO_Z = (
    1.0 / np.sqrt(((_px - IMG_W / 2) / FOCAL) ** 2 + ((_py - IMG_H / 2) / FOCAL) ** 2 + 1.0)
).astype(np.float32)


def load_splits(meta_dir: Path) -> dict[tuple[str, str], str]:
    """Split of each (scene, cam), with val merged into train."""
    out = {}
    with open(meta_dir / "metadata_images_split_scene_v1.csv") as f:
        for row in csv.DictReader(f):
            split = row["split_partition_name"]
            out[(row["scene_name"], row["camera_name"])] = "train" if split == "val" else split
    return out


def load_excluded(meta_dir: Path) -> set[tuple[str, str, int]]:
    path = meta_dir / "metadata_images.csv"
    if not path.exists():
        logger.warning("metadata_images.csv missing -- no frame exclusions applied")
        return set()
    excluded = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["included_in_public_release"] == "False":
                excluded.add((row["scene_name"], row["camera_name"], int(row["frame_id"])))
    return excluded


def _h5(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return h5py.File(_io.BytesIO(f.read()), "r")["dataset"][()]


def process_camera(
    task: tuple[str, str], zip_dir: Path, excluded: set[tuple[str, str, int]]
) -> dict:
    scene_name, cam = task
    with zipfile.ZipFile(zip_dir / f"{scene_name}.zip") as zf:
        with zf.open(f"{scene_name}/_detail/metadata_scene.csv") as f:
            row = next(csv.DictReader(_io.TextIOWrapper(f)))
            meters_per_asset = float(row["parameter_value"])

        detail = f"{scene_name}/_detail/{cam}"
        positions = _h5(zf, f"{detail}/camera_keyframe_positions.hdf5")
        orientations = _h5(zf, f"{detail}/camera_keyframe_orientations.hdf5")
        frame_ids = _h5(zf, f"{detail}/camera_keyframe_frame_indices.hdf5")
        names = set(zf.namelist())

        img_dir = f"{scene_name}/images/scene_{cam}_final_preview"
        dep_dir = f"{scene_name}/images/scene_{cam}_geometry_hdf5"
        images, depths, ranges, poses = [], [], [], []
        for kf, frame_id in sorted(enumerate(frame_ids.tolist()), key=lambda e: e[1]):
            frame_id = int(frame_id)
            if (scene_name, cam, frame_id) in excluded:
                continue
            img_name = f"{img_dir}/frame.{frame_id:04d}.tonemap.jpg"
            if img_name not in names:
                continue
            images.append(zf.read(img_name))  # verbatim passthrough

            c2w = np.eye(4)
            c2w[:3, :3] = orientations[kf]
            c2w[:3, 3] = positions[kf] * meters_per_asset
            c2w[:, 1:3] *= -1  # OpenGL -> OpenCV
            poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

            dep_name = f"{dep_dir}/frame.{frame_id:04d}.depth_meters.hdf5"
            if dep_name in names:
                depth = _h5(zf, dep_name).astype(np.float32) * EUCLIDEAN_TO_Z
                depth[~np.isfinite(depth) | (depth <= 0)] = -1.0
            else:
                depth = np.full((IMG_H, IMG_W), -1.0, dtype=np.float32)
            dep_bytes, d_min, d_max = encode_depth(depth)
            depths.append(dep_bytes)
            ranges.append((d_min, d_max))

    if not images:
        raise ValueError(f"{scene_name}/{cam}: no valid frames")

    cam_row = [IMG_W, IMG_H, FOCAL, FOCAL, IMG_W / 2, IMG_H / 2]
    return {
        "key": f"{scene_name}_{cam}",
        "cameras": np.asarray([cam_row] * len(images), dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert ML-Hypersim to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/ml-hypersim"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>")
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene-ids", nargs="+", default=None, help="Scene names (ai_XXX_XXX)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = args.output_dir or args.data_root
    zip_dir = args.data_root / "zip"
    meta_dir = args.data_root / "metadata"
    splits_map = load_splits(meta_dir)
    excluded = load_excluded(meta_dir)

    available = {p.stem for p in zip_dir.glob("ai_*.zip")}
    by_split: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (scene, cam), split in sorted(splits_map.items()):
        if scene in available:
            by_split[split].append((scene, cam))

    for split in args.split:
        tasks = by_split.get(split, [])
        if args.scene_ids:
            tasks = [t for t in tasks if t[0] in set(args.scene_ids)]
        if args.limit:
            tasks = tasks[: args.limit]
        if not tasks:
            logger.info(f"No {split} tasks (zips available: {len(available)})")
            continue

        out_dir = out_root / f"{IMG_H}x{IMG_W}" / f"{split}-scenes"
        index_path = out_dir / "index.json"
        if args.overwrite:
            for p in out_dir.glob("shard-*.tar*"):
                p.unlink()
            index_path.unlink(missing_ok=True)
        done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
        remaining = [t for t in tasks if f"{t[0]}_{t[1]}" not in done]
        if not remaining:
            logger.info(f"Skipping {split}: all {len(tasks)} scenes done")
            continue
        logger.info(f"Converting {split}: {len(remaining)} camera trajectories -> {out_dir}")

        fn = partial(process_camera, zip_dir=zip_dir, excluded=excluded)
        scenes = parallel_map(
            fn, remaining, num_workers=args.num_workers, skip_label=lambda t: f"{t[0]}/{t[1]}"
        )
        write_shards(
            scenes,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"hypersim/{split}",
            total=len(tasks),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
