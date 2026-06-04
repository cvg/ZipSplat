"""Convert RE10K downloaded zip to tar shards.

Reads `.torch` files directly from re10k.zip (pixelSplat/MVSplat format).
Each `.torch` contains ~12 scenes with (N, 18) RE10K cameras and raw JPEG images.
Passes JPEG bytes through unchanged when --target-size is omitted; otherwise
re-encodes as WebP lossless after crop+resize.

Usage:
    python -m splatfactory.datasets.scripts.re10k.convert
    python -m splatfactory.datasets.scripts.re10k.convert --split train --num-workers 8
    python -m splatfactory.datasets.scripts.re10k.convert --target-size 252

Author: Alexander Veicht
"""

import argparse
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from einops import repeat

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import parallel_map, update_camera_intrinsics, write_shards
from splatfactory.datasets.utils.io import decode_image, encode_image
from splatfactory.geometry import Camera, Pose
from splatfactory.utils.image import crop_to_ar, crop_to_principal_point, resize_to_cover

logger = get_logger(__name__)


def convert_cameras(raw: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(N, 18) RE10K format -> Camera.data_ (N, 6) + Pose.data_ (N, 12)."""
    raw_t = torch.as_tensor(raw, dtype=torch.float32)
    b = raw_t.shape[0]

    fx, fy, cx, cy = raw_t[:, :4].T
    K = repeat(torch.eye(3), "h w -> b h w", b=b).clone()
    K[:, 0, 0] = fx * w
    K[:, 1, 1] = fy * h
    K[:, 0, 2] = cx * w
    K[:, 1, 2] = cy * h

    w2c = repeat(torch.eye(4), "h w -> b h w", b=b).clone()
    w2c[:, :3] = raw_t[:, 6:].reshape(b, 3, 4)

    cam = Camera.from_calibration_matrix(K).float().data_.numpy().astype(np.float32)
    pose = Pose.from_4x4mat(w2c).inv().data_.numpy().astype(np.float32)
    return cam, pose


def _resize_and_update(
    img_bytes: bytes, cam_data: np.ndarray, target_size: int
) -> tuple[bytes, np.ndarray]:
    """Crop to principal point -> crop to square -> resize to target. Returns (bytes, new cam)."""
    w, h, fx, fy, cx, cy = cam_data
    img = decode_image(img_bytes)

    transform = np.eye(3)
    img, t = crop_to_principal_point(img, cx, cy)
    transform = t @ transform
    img, t = resize_to_cover(img, int(h), int(w))
    transform = t @ transform
    img, t = crop_to_ar(img, 1.0)
    transform = t @ transform
    img, t = resize_to_cover(img, target_size, target_size)
    transform = t @ transform

    final_h, final_w = img.shape[:2]
    return encode_image(img), update_camera_intrinsics(fx, fy, cx, cy, transform, final_w, final_h)


def convert_torch_file(zip_path: Path, torch_name: str, target_size: int | None) -> list[dict]:
    """Read one .torch file from zip, return list of scene dicts."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(torch_name) as f:
        data = torch.load(BytesIO(f.read()), weights_only=False)

    scenes = []
    for scene in data:
        images_raw = [bytes(t.numpy()) for t in scene["images"]]
        orig_w, orig_h = decode_image(images_raw[0]).shape[1::-1]
        cam_data, pose_data = convert_cameras(scene["cameras"].numpy(), orig_h, orig_w)

        if target_size is None:
            images, cameras = images_raw, cam_data
        else:
            images, cameras = [], cam_data.copy()
            for i, raw in enumerate(images_raw):
                img_bytes, cameras[i] = _resize_and_update(raw, cam_data[i], target_size)
                images.append(img_bytes)

        scenes.append(
            {
                "key": scene["key"],
                "cameras": cameras,
                "poses": pose_data,
                "images": images,
                "num_views": len(images),
                "has_depth": False,
                "depths": None,
                "depth_ranges": None,
            }
        )
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Convert RE10K zip to tar shards")
    parser.add_argument("--input-zip", type=Path, default=Path("data/re10k/raw/re10k.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/re10k"))
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with zipfile.ZipFile(args.input_zip) as zf:
        names = zf.namelist()
    train = sorted(n for n in names if n.startswith("re10k/train/") and n.endswith(".torch"))
    test = sorted(n for n in names if n.startswith("re10k/test/") and n.endswith(".torch"))
    logger.info(f"Found {len(train)} train, {len(test)} test .torch files in zip")

    size_name = f"{args.target_size}x{args.target_size}" if args.target_size else "360x640"
    output_base = args.output_dir / size_name

    splits = [s for s in (("train", train), ("test", test)) if args.split in (s[0], "all") and s[1]]

    for split_name, torch_files in splits:
        shard_dir = output_base / f"{split_name}-scenes"
        if shard_dir.exists() and not args.overwrite:
            logger.info(f"Skipping {split_name}: {shard_dir} exists (use --overwrite)")
            continue
        logger.info(f"Converting {len(torch_files)} {split_name} .torch files -> {shard_dir}")

        # Each .torch produces ~12 scenes -> flatten with chain-like comprehension.
        batches = parallel_map(
            partial(convert_torch_file, args.input_zip, target_size=args.target_size),
            torch_files,
            num_workers=args.num_workers,
        )
        scenes = (scene for batch in batches for scene in batch)
        write_shards(
            scenes, shard_dir, shard_size_mb=args.shard_size_mb, desc=f"RE10K {split_name}"
        )


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
