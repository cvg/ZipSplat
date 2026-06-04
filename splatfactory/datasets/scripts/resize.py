"""Resize tar shards to a target square resolution.

Reads scenes from `shard-*.tar`, resizes images + depth + cameras via
ImagePreprocessor, and writes new shards of ~512 MB each. Images are
re-encoded as WebP lossless; depth is re-quantized with refreshed
log-uint8 ranges.

Usage:
    python -m splatfactory.datasets.scripts.resize \\
        data/dl3dv/540x960/test-scenes --output data/dl3dv/252x252/test-scenes --size 252
    python -m splatfactory.datasets.scripts.resize \\
        data/dl3dv/540x960/train-scenes --output data/dl3dv/252x252/train-scenes \\
        --size 252 --num-workers 32

Author: Alexander Veicht
"""

import argparse
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402

cv2.setNumThreads(1)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from splatfactory import get_logger  # noqa: E402
from splatfactory.datasets.scripts.utils import parallel_map, write_shards  # noqa: E402
from splatfactory.datasets.utils.io import (  # noqa: E402
    decode_depth,
    decode_image,
    encode_depth,
    encode_image,
    iter_scenes_from_tar,
)
from splatfactory.geometry import Camera  # noqa: E402
from splatfactory.utils.image import ImagePreprocessor, to_numpy  # noqa: E402

logger = get_logger(__name__)


def _resize_view(
    preprocessor: ImagePreprocessor,
    img_bytes: bytes,
    depth_bytes: bytes | None,
    depth_range: tuple[float, float] | None,
) -> tuple[bytes, bytes | None, tuple[float, float] | None, np.ndarray]:
    """Resize one view. Returns (img_bytes, depth_bytes|None, depth_range|None, transform)."""
    img = decode_image(img_bytes)
    depth = decode_depth(depth_bytes, *depth_range) if depth_bytes else None

    result = preprocessor(img, depth=depth, aspect_ratio=1.0)
    img_out = encode_image(to_numpy(result["image"]))

    if "depth" in result:
        dep_out, d_min, d_max = encode_depth(result["depth"].numpy())
        return img_out, dep_out, (d_min, d_max), result["transform"]
    return img_out, None, None, result["transform"]


def resize_scene(scene: dict, target_size: int) -> dict:
    """Resize one scene to target_size x target_size. Matches write_scene_to_tar input."""
    preprocessor = ImagePreprocessor({"resize": target_size})
    n = scene["meta"]["num_views"]
    has_depth = scene["meta"].get("has_depth", False) and bool(scene.get("depths"))

    images: list[bytes] = []
    depths: list[bytes] = []
    ranges: list[tuple[float, float]] = []
    transform: np.ndarray | None = None

    for i in range(n):
        dep_bytes = scene["depths"].get(i) if has_depth else None
        dep_rng = scene["depth_ranges"][i] if has_depth else None
        img_out, dep_out, rng_out, t = _resize_view(
            preprocessor, scene["images"][i], dep_bytes, dep_rng
        )
        if transform is None:
            transform = t
        images.append(img_out)
        if has_depth:
            depths.append(dep_out if dep_out is not None else b"")
            ranges.append(rng_out if rng_out is not None else (0.0, 0.0))

    assert transform is not None, f"Scene {scene['key']} has no views"
    cameras = Camera(torch.from_numpy(scene["cameras"].copy())).compose_image_transform(transform)

    return {
        "key": scene["key"],
        "cameras": cameras.data_.numpy().astype(np.float32),
        "poses": scene["poses"],
        "images": images,
        "num_views": n,
        "has_depth": has_depth,
        "depths": depths if has_depth else None,
        "depth_ranges": ranges if has_depth else None,
    }


def _iter_input_scenes(input_dir: Path):
    """Stream scenes from all shards in a directory, in order."""
    for shard in sorted(input_dir.glob("shard-*.tar")):
        yield from iter_scenes_from_tar(str(shard))


def main():
    parser = argparse.ArgumentParser(description="Resize tar shards to a target square resolution")
    parser.add_argument(
        "input_dir", type=Path, help="Source shard directory (contains shard-*.tar)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output shard directory")
    parser.add_argument("--size", type=int, default=252, help="Target square edge length")
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    if args.output.exists() and any(args.output.glob("shard-*.tar")) and not args.overwrite:
        logger.info(f"Skipping: {args.output} already has shards (use --overwrite)")
        return

    logger.info(f"Resizing {args.input_dir} -> {args.output} @ {args.size}x{args.size}")
    scenes = parallel_map(
        partial(resize_scene, target_size=args.size),
        _iter_input_scenes(args.input_dir),
        num_workers=args.num_workers,
        skip_label=lambda s: s.get("key", "?"),
    )
    write_shards(
        scenes, args.output, shard_size_mb=args.shard_size_mb, desc=f"Resize {args.size}px"
    )


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
