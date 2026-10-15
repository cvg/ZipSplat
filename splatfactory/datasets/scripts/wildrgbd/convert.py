"""Convert WildRGB-D videos to canonical tar shards.

Raw layout, read directly from <data-root>/<category>.zip (split archives are merged by
download.py first):
  <category>/{camera_eval_list,nvs_list,types}.json
  <category>/scenes/scene_XXX/
      rgb/NNNNN.png      480x640 (WxH, portrait) RGB PNG, 30 fps iPhone video
      depth/NNNNN.png    uint16 depth in mm, same resolution, 0 = invalid
      masks/NNNNN.png    object masks (not stored)
      metadata           JSON {fps, w, h, K}, K stored column-major
      cam_poses.txt      rows "frame_id + 16 floats": c2w 4x4 row-major, OpenCV

Conventions, as in DUSt3R's preprocess_wildrgbd.py:
  - Frames subsampled to --num-frames evenly spaced indices, 0 keeps all.
  - Principal point centred per scene (crop_to_principal_point, resize_to_cover,
    crop_to_ar) back to exactly 640x480 (HxW).
  - Poses: OpenCV c2w Pose.data_ [R.flatten(), t].
  - RGB re-encoded as lossless WebP, since the PNG source is resampled.
  - Depth from mm to m, full frame (not masked to the object), log-uint8 WebP.
  - meta.json carries the category and the official video type (single / multi /
    hand) from types.json.

Splits: per category, scenes are shuffled (seeded), capped at --max-per-category,
and a --test-fraction of them becomes test-scenes.

Usage:
    python -m splatfactory.datasets.scripts.wildrgbd.convert --num-workers 8
    python -m splatfactory.datasets.scripts.wildrgbd.convert \
        --categories apple car --max-per-category 10

Author: Alexander Veicht
"""

import argparse
import json
import os
import zipfile
import zlib
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from splatfactory import get_logger  # noqa: E402
from splatfactory.datasets.scripts.utils import parallel_map, write_shards  # noqa: E402
from splatfactory.datasets.utils.io import encode_depth, encode_image  # noqa: E402
from splatfactory.utils.image import (  # noqa: E402
    _round_even,
    crop_to_ar,
    crop_to_principal_point,
    resize_to_cover,
)

logger = get_logger(__name__)

TARGET = (640, 480)  # (out_h, out_w) = native portrait resolution, pp-centered

_ZIPS: dict[Path, zipfile.ZipFile] = {}


def _zip(path: Path) -> zipfile.ZipFile:
    """Per-process zip handle cache (central directory parse takes ~1s).

    Holds one zip: tasks arrive grouped by category, and keeping every touched category's
    central directory open OOMs a 460 GB node at 32 workers."""
    if path not in _ZIPS:
        for zf in _ZIPS.values():
            zf.close()
        _ZIPS.clear()
        _ZIPS[path] = zipfile.ZipFile(path, "r")
    return _ZIPS[path]


def _read_image(zf: zipfile.ZipFile, name: str, flags: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), flags)


# --- geometry (same plan as arkitscenes convert) -----------------------------------


def _geometry_plan(w, h, fx, fy, cx, cy, target):
    probe = np.zeros((h, w, 3), np.uint8)
    probe, t = crop_to_principal_point(probe, cx, cy)
    th, tw = target
    probe, t_resize = resize_to_cover(probe, th, tw)
    probe, t_crop = crop_to_ar(probe, tw / th)
    t = t_crop @ t_resize @ t
    oh, ow = probe.shape[:2]
    k = t @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return ow, oh, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _apply_geometry(img, cx, cy, target):
    img, _ = crop_to_principal_point(img, cx, cy)
    th, tw = target
    img, _ = resize_to_cover(img, th, tw)
    img, _ = crop_to_ar(img, tw / th)
    return img


def _apply_geometry_depth(depth, cx, cy, target):
    d, _ = crop_to_principal_point(depth, cx, cy)
    th, tw = target
    h, w = d.shape
    scale = max(th / h, tw / w)
    hn, wn = _round_even(h * scale), _round_even(w * scale)
    if (hn, wn) != (h, w):
        d = cv2.resize(d, (wn, hn), interpolation=cv2.INTER_LINEAR)
    d, _ = crop_to_ar(d, tw / th)
    return d


# --- per-scene processing -----------------------------------------------------------


def process_scene(task: tuple[str, str, str], data_root: Path, num_frames: int) -> dict:
    category, scene, vtype = task
    zf = _zip(data_root / f"{category}.zip")
    prefix = f"{category}/scenes/{scene}"

    meta = json.loads(zf.read(f"{prefix}/metadata"))
    w, h = meta["w"], meta["h"]
    K = np.array(meta["K"], dtype=np.float64).reshape(3, 3).T
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rows = np.loadtxt(zf.read(f"{prefix}/cam_poses.txt").decode().splitlines(), ndmin=2)
    frame_ids = rows[:, 0].astype(int)
    c2ws = rows[:, 1:].reshape(-1, 4, 4)
    sel = np.arange(len(frame_ids))
    if num_frames and len(sel) > num_frames:
        sel = np.unique(np.round(np.linspace(0, len(sel) - 1, num_frames)).astype(int))

    ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, TARGET)
    assert (oh, ow) == TARGET, f"{task}: {oh}x{ow}"
    assert abs(ocx - ow / 2) < 1.5 and abs(ocy - oh / 2) < 1.5, f"{task}: pp not centered"

    images, depths, ranges, poses = [], [], [], []
    for i in sel:
        name = f"{frame_ids[i]:05d}.png"
        img = _read_image(zf, f"{prefix}/rgb/{name}", cv2.IMREAD_COLOR)
        assert img.shape[:2] == (h, w), f"{task}/{name}: rgb {img.shape}"
        out = _apply_geometry(img, cx, cy, TARGET)
        images.append(encode_image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))

        c2w = c2ws[i]
        poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

        raw = _read_image(zf, f"{prefix}/depth/{name}", cv2.IMREAD_UNCHANGED)
        assert raw.dtype == np.uint16 and raw.shape == (h, w), f"{task}/{name}: depth"
        depth = raw.astype(np.float32) / 1000.0
        depth[depth <= 0] = np.nan
        depth = np.nan_to_num(_apply_geometry_depth(depth, cx, cy, TARGET), nan=-1.0)
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    n = len(images)
    return {
        "key": f"{category}_{scene}",
        "cameras": np.tile(np.array([ow, oh, ofx, ofy, ocx, ocy], np.float32), (n, 1)),
        "poses": np.stack(poses),
        "images": images,
        "num_views": n,
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
        "meta_extra": {"category": category, "type": vtype},
    }


# --- main ---------------------------------------------------------------------------


def list_scenes(zip_path: Path, category: str) -> list[tuple[str, str]]:
    """[(scene, type)] for every scene with a cam_poses.txt.

    Opens its own handle (not the cache): forked workers must not share an fd offset.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        types = json.loads(zf.read(f"{category}/types.json"))
    suffix = "/cam_poses.txt"
    scenes = sorted(
        n.split("/")[2] for n in names if n.startswith(f"{category}/scenes/") and n.endswith(suffix)
    )
    vtype = {s.split("/")[-1]: t for t, seqs in types.items() for s in seqs}
    return [(s, vtype.get(s, "unknown")) for s in scenes]


def main():
    parser = argparse.ArgumentParser(description="Convert WildRGB-D to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/wildrgbd"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--num-frames", type=int, default=100, help="Evenly spaced frames per video (0 = all)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max scenes per split")
    args = parser.parse_args()

    out_root = args.output_dir or args.data_root
    categories = args.categories or sorted(p.stem for p in args.data_root.glob("*.zip"))
    if not categories:
        raise SystemExit(f"no category zips in {args.data_root}")

    splits: dict[str, list[tuple[str, str, str]]] = {"train": [], "test": []}
    for category in categories:
        zip_path = args.data_root / f"{category}.zip"
        if not zip_path.exists():
            logger.warning(f"{category}: missing {zip_path}, skipping")
            continue
        scenes = list_scenes(zip_path, category)
        rng = np.random.default_rng([args.seed, zlib.crc32(category.encode())])
        selected = [scenes[i] for i in rng.permutation(len(scenes))][: args.max_per_category]
        n_test = int(len(selected) * args.test_fraction)
        for i, (scene, vtype) in enumerate(selected):
            splits["test" if i < n_test else "train"].append((category, scene, vtype))
        logger.info(f"{category}: {len(selected)}/{len(scenes)} scenes ({n_test} test)")

    th, tw = TARGET
    for split, tasks in splits.items():
        if args.limit:
            tasks = tasks[: args.limit]
        if not tasks:
            continue
        out_dir = out_root / f"{th}x{tw}" / f"{split}-scenes"
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
        logger.info(f"Converting {split}: {len(remaining)} videos -> {out_dir}")

        fn = partial(process_scene, data_root=args.data_root, num_frames=args.num_frames)
        scenes = parallel_map(
            fn, remaining, num_workers=args.num_workers, skip_label=lambda t: f"{t[0]}/{t[1]}"
        )
        write_shards(
            scenes,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"wildrgbd/{split}",
            total=len(tasks),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
