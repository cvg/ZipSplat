"""Convert DL3DV downloaded zips to tar shards.

Reads directly from compressed zips (no extraction needed).
Converts nerfstudio/OpenGL poses to OpenCV convention.
Images are re-encoded as WebP lossless.

Usage:
    python -m splatfactory.datasets.scripts.dl3dv.convert
    python -m splatfactory.datasets.scripts.dl3dv.convert --target-size 252
    python -m splatfactory.datasets.scripts.dl3dv.convert --resolution 960P --split train
    python -m splatfactory.datasets.scripts.dl3dv.convert --target-size 252 --num-workers 8

Author: Alexander Veicht
"""

import argparse
import json
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import parallel_map, update_camera_intrinsics, write_shards
from splatfactory.datasets.utils.io import decode_image, encode_image
from splatfactory.utils.image import crop_to_ar, crop_to_principal_point, resize_to_cover

logger = get_logger(__name__)

SPLIT_PATH = Path(__file__).parent / "split.json"


def _load_test_hashes() -> set[str]:
    with open(SPLIT_PATH) as f:
        return set(json.load(f)["test"])


def opengl_c2w_to_opencv_c2w(c2w_4x4: np.ndarray) -> np.ndarray:
    """OpenGL c2w -> OpenCV c2w: flip Y and Z axes of camera frame."""
    c2w = c2w_4x4[:3, :].copy()
    c2w[:, 1] *= -1
    c2w[:, 2] *= -1
    return c2w


def _scale_intrinsics(tf: dict, img_w: int) -> tuple[float, float, float, float]:
    """Scale (fx, fy, cx, cy) from transforms.json resolution to actual image resolution."""
    scale = img_w / tf["w"]
    return tf["fl_x"] * scale, tf["fl_y"] * scale, tf["cx"] * scale, tf["cy"] * scale


def _resolve_image_paths(zf: zipfile.ZipFile, tf_path: str, frames: list[dict]) -> dict[str, str]:
    """Map each frame's `file_path` to the actual zip member.

    transforms.json paths say "images/..." but the zip may have "images_4/..." etc.
    Returns {frame['file_path']: zip_member_name}.
    """
    prefix = tf_path.rsplit("/", 1)[0] + "/" if "/" in tf_path else ""
    img_dirs = sorted(
        {n.rsplit("/", 1)[0] for n in zf.namelist() if "/images" in n and not n.endswith("/")}
    )
    if not img_dirs:
        raise ValueError("no image directory found")
    actual = img_dirs[0].split("/")[-1]
    json_dir = frames[0]["file_path"].split("/")[0]
    return {f["file_path"]: prefix + f["file_path"].replace(json_dir, actual, 1) for f in frames}


def _opencv_pose_12(c2w_gl_4x4: np.ndarray) -> np.ndarray:
    """OpenGL c2w 4x4 -> flat [R(9), t(3)] in OpenCV convention."""
    c2w = opengl_c2w_to_opencv_c2w(c2w_gl_4x4)
    return np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]])


def convert_scene(zip_path: Path, target_size: int | None) -> dict:
    """Convert a single DL3DV zip to a scene dict for shard writing."""
    with zipfile.ZipFile(zip_path) as zf:
        tf_candidates = [n for n in zf.namelist() if n.endswith("transforms.json")]
        if not tf_candidates:
            raise ValueError(f"no transforms.json in {zip_path}")
        tf = json.loads(zf.read(tf_candidates[0]))
        frames = tf["frames"]
        zip_paths = _resolve_image_paths(zf, tf_candidates[0], frames)

        orig_w, orig_h = Image.open(BytesIO(zf.read(zip_paths[frames[0]["file_path"]]))).size
        fx, fy, cx, cy = _scale_intrinsics(tf, orig_w)

        cameras, poses, images = [], [], []
        for frame in frames:
            try:
                img_bytes = zf.read(zip_paths[frame["file_path"]])
            except KeyError:
                continue
            img = decode_image(img_bytes)

            transform = np.eye(3)
            img, t = crop_to_principal_point(img, cx, cy)
            transform = t @ transform
            img, t = resize_to_cover(img, orig_h, orig_w)
            transform = t @ transform
            if target_size is not None:
                img, t = crop_to_ar(img, 1.0)
                transform = t @ transform
                img, t = resize_to_cover(img, target_size, target_size)
                transform = t @ transform

            h, w = img.shape[:2]
            cameras.append(update_camera_intrinsics(fx, fy, cx, cy, transform, w, h))
            poses.append(_opencv_pose_12(np.asarray(frame["transform_matrix"], dtype=np.float32)))
            images.append(encode_image(img))

    if not images:
        raise ValueError(f"no valid frames in {zip_path}")

    return {
        "key": zip_path.stem,
        "cameras": np.stack(cameras),
        "poses": np.stack(poses),
        "images": images,
        "num_views": len(images),
        "has_depth": False,
        "depths": None,
        "depth_ranges": None,
    }


def _infer_size_name(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        imgs = [n for n in zf.namelist() if n.endswith((".png", ".jpg"))]
        w, h = Image.open(BytesIO(zf.read(imgs[0]))).size
    return f"{h}x{w}"


def main():
    parser = argparse.ArgumentParser(description="Convert DL3DV zips to tar shards")
    parser.add_argument("--input-dir", type=Path, default=Path("data/dl3dv/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dl3dv"))
    parser.add_argument(
        "--resolution",
        choices=["480P", "960P", "2K", "4K"],
        default="480P",
    )
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir / args.resolution
    zip_files = sorted(input_dir.glob("**/*.zip"))
    if not zip_files:
        logger.error(f"No zip files found in {input_dir}")
        return

    test_hashes = _load_test_hashes()
    train_zips = [z for z in zip_files if z.stem not in test_hashes]
    test_zips = [z for z in zip_files if z.stem in test_hashes]
    logger.info(f"Found {len(zip_files)} zips: {len(train_zips)} train, {len(test_zips)} test")

    size_name = (
        f"{args.target_size}x{args.target_size}"
        if args.target_size
        else _infer_size_name(zip_files[0])
    )
    output_base = args.output_dir / size_name
    logger.info(f"Output base directory: {output_base}")

    splits = [
        s
        for s in (("train", train_zips), ("test", test_zips))
        if args.split in (s[0], "all") and s[1]
    ]

    for split_name, zips in splits:
        shard_dir = output_base / f"{split_name}-scenes"
        if shard_dir.exists() and not args.overwrite:
            logger.info(f"Skipping {split_name}: {shard_dir} exists (use --overwrite)")
            continue
        logger.info(f"Converting {len(zips)} {split_name} scenes -> {shard_dir}")

        scenes = parallel_map(
            partial(convert_scene, target_size=args.target_size),
            zips,
            num_workers=args.num_workers,
            skip_label=lambda z: z.stem,
        )
        write_shards(
            scenes, shard_dir, shard_size_mb=args.shard_size_mb, desc=f"DL3DV {split_name}"
        )


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
