"""Convert BlendedMVS (low-res, 113 scenes) to canonical tar shards.

Raw layout (extracted `<root>/BlendedMVS/<pid>/`):
  - blended_images/NNNNNNNN.jpg (+ *_masked.jpg, ignored) -- 768x576 JPEG
  - cams/NNNNNNNN_cam.txt -- text w2c 4x4 (OpenCV) + K 3x3 at native resolution
  - cams/pair.txt -- MVS neighbor rankings (not used here)
  - rendered_depth_maps/NNNNNNNN.pfm -- planar z-depth in meters (from the mesh)

Conventions:
  - Principal point centred per frame (crop_to_principal_point, resize_to_cover,
    crop_to_ar) to exactly 576x768. K varies per frame in some scenes.
  - Poses stored as OpenCV c2w Pose.data_ [R.flatten(), t] (the cam files are w2c).
  - Images re-encoded as JPEG q95, since they are resampled.
  - Depth gets the same geometry (invalid NaN, INTER_LINEAR, NaN to -1), then log-uint8
    WebP. Missing or empty PFMs give all-invalid depth.
  - orientation.json (29 sideways scenes) is not applied. Training uses relative poses, so
    the world orientation does not matter, and the images stay as captured.

Splits from project_lists/: train = BlendedMVS.txt minus validation_list.txt
(106), test = validation_list.txt (7).

Usage:
    python -m splatfactory.datasets.scripts.blendedmvs.convert --num-workers 12

Author: Alexander Veicht
"""

import argparse
import io as _io
import json
import os
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
from splatfactory.datasets.utils.io import encode_depth  # noqa: E402
from splatfactory.utils.image import (  # noqa: E402
    _round_even,
    crop_to_ar,
    crop_to_principal_point,
    resize_to_cover,
)

logger = get_logger(__name__)

TARGET = (576, 768)  # (out_h, out_w) = native resolution, pp-centered
JPEG_QUALITY = 95


# --- parsing ------------------------------------------------------------------


def parse_cam(text: str) -> tuple[np.ndarray, np.ndarray]:
    """BlendedMVS cam txt -> (w2c 4x4, K 3x3)."""
    lines = [ln.strip() for ln in text.splitlines()]
    e = lines.index("extrinsic")
    w2c = np.array([[float(x) for x in lines[e + 1 + i].split()] for i in range(4)])
    k = lines.index("intrinsic")
    K = np.array([[float(x) for x in lines[k + 1 + i].split()] for i in range(3)])
    return w2c, K


def parse_pfm(data: bytes) -> np.ndarray:
    """PFM bytes -> (H, W) float32 (single-channel 'Pf' only)."""
    f = _io.BytesIO(data)
    header = f.readline().decode("ascii").strip()
    if header != "Pf":
        raise ValueError(f"expected single-channel PFM, got {header}")
    w, h = (int(x) for x in f.readline().decode("ascii").split())
    scale = float(f.readline().decode("ascii").strip())
    arr = np.frombuffer(f.read(), dtype="<f" if scale < 0 else ">f").reshape(h, w)
    return np.flipud(arr).astype(np.float32) * abs(abs(scale) if scale != 0 else 1.0)


# --- geometry (mirrors scannetpp convert) --------------------------------------


def _geometry_plan(w, h, fx, fy, cx, cy, target):
    """Compose intrinsics through crop_to_principal_point -> resize_to_cover ->
    crop_to_ar. Returns the output camera (ow, oh, fx, fy, cx, cy)."""
    probe = np.zeros((h, w, 3), np.uint8)
    probe, t = crop_to_principal_point(probe, cx, cy)
    th, tw = target
    probe, t_resize = resize_to_cover(probe, th, tw)
    probe, t_crop = crop_to_ar(probe, tw / th)
    t = t_crop @ t_resize @ t
    oh, ow = probe.shape[:2]
    k = t @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return ow, oh, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _apply_geometry(img: np.ndarray, cx, cy, target) -> np.ndarray:
    img, _ = crop_to_principal_point(img, cx, cy)
    th, tw = target
    img, _ = resize_to_cover(img, th, tw)
    img, _ = crop_to_ar(img, tw / th)
    return img


def _apply_geometry_depth(depth: np.ndarray, cx, cy, target) -> np.ndarray:
    """_apply_geometry for depth, invalid pixels stay NaN through the resize."""
    d, _ = crop_to_principal_point(depth, cx, cy)
    th, tw = target
    h, w = d.shape
    scale = max(th / h, tw / w)
    hn, wn = _round_even(h * scale), _round_even(w * scale)
    if (hn, wn) != (h, w):
        d = cv2.resize(d, (wn, hn), interpolation=cv2.INTER_LINEAR)
    d, _ = crop_to_ar(d, tw / th)
    return d


# --- per-scene processing -------------------------------------------------------


def process_scene(scene_id: str, data_root: Path, target=TARGET) -> dict:
    scene = data_root / "BlendedMVS" / scene_id
    cam_files = sorted((scene / "cams").glob("[0-9]*_cam.txt"))
    if not cam_files:
        raise FileNotFoundError(f"{scene_id}: no cam files")

    th, tw = target
    images, depths, ranges, poses, cameras = [], [], [], [], []
    dropped = 0
    for cam_file in cam_files:
        idx = cam_file.name.removesuffix("_cam.txt")
        img_path = scene / "blended_images" / f"{idx}.jpg"
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            dropped += 1
            continue
        w2c, K = parse_cam(cam_file.read_text())
        h, w = img.shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, target)
        assert (oh, ow) == (th, tw), f"{scene_id}/{idx}: output {oh}x{ow} != {th}x{tw}"
        assert abs(ocx - ow / 2) < 1.5 and abs(ocy - oh / 2) < 1.5, f"{scene_id}/{idx}: pp"

        out = _apply_geometry(img, cx, cy, target)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"{scene_id}/{idx}: jpeg encode failed")
        images.append(buf.tobytes())
        cameras.append([ow, oh, ofx, ofy, ocx, ocy])

        c2w = np.linalg.inv(w2c)
        poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

        pfm_path = scene / "rendered_depth_maps" / f"{idx}.pfm"
        depth = None
        if pfm_path.exists():
            try:
                depth = parse_pfm(pfm_path.read_bytes())
            except Exception as e:  # noqa: BLE001 - malformed pfm -> invalid depth
                logger.warning(f"{scene_id}/{idx}: bad pfm ({e})")
        if depth is not None and depth.shape == (h, w):
            depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
            depth = _apply_geometry_depth(depth, cx, cy, target)
            depth = np.nan_to_num(depth, nan=-1.0)
        else:
            depth = np.full((th, tw), -1.0, dtype=np.float32)
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    if not images:
        raise ValueError(f"{scene_id}: no valid frames")
    if dropped:
        logger.warning(f"{scene_id}: dropped {dropped} frames (missing/empty image)")

    return {
        "key": scene_id,
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
    }


# --- main -----------------------------------------------------------------------


def _read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main():
    parser = argparse.ArgumentParser(description="Convert BlendedMVS to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/blendedmvs"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>")
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = args.output_dir or args.data_root
    all_ids = _read_list(args.data_root / "project_lists" / "BlendedMVS.txt")
    val_ids = set(_read_list(args.data_root / "project_lists" / "validation_list.txt"))
    splits = {
        "train": [s for s in all_ids if s not in val_ids],
        "test": [s for s in all_ids if s in val_ids],
    }

    th, tw = TARGET
    for split in args.split:
        scene_ids = args.scene_ids or splits[split]
        if args.scene_ids:
            scene_ids = [s for s in scene_ids if s in set(splits[split])]
        if args.limit:
            scene_ids = scene_ids[: args.limit]
        if not scene_ids:
            continue

        out_dir = out_root / f"{th}x{tw}" / f"{split}-scenes"
        index_path = out_dir / "index.json"
        if args.overwrite:
            for p in out_dir.glob("shard-*.tar*"):
                p.unlink()
            index_path.unlink(missing_ok=True)

        done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
        remaining = [s for s in scene_ids if s not in done]
        if not remaining:
            logger.info(f"Skipping {split}: all {len(scene_ids)} scenes done")
            continue
        logger.info(f"Converting {split}: {len(remaining)} scenes -> {out_dir}")

        fn = partial(process_scene, data_root=args.data_root)
        scenes = parallel_map(fn, remaining, num_workers=args.num_workers, skip_label=lambda s: s)
        write_shards(
            scenes,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"blendedmvs/{split}",
            total=len(scene_ids),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
