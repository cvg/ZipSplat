"""Extract DA3 depth for tar shards, in place.

Round-robin chunking when --max-views is set (each chunk spans the full scene
trajectory for globally consistent depth; validated in playground/test_depth_chunking.py).
In-place rewrite via .tmp + atomic rename. Per-rank shard partitioning - no
concurrent writes to the same tar.

Usage:
    # Single GPU
    python -m splatfactory.datasets.scripts.extract_depth \\
        data/re10k/360x640/train-scenes

    # Distributed
    torchrun --nproc_per_node=4 -m splatfactory.datasets.scripts.extract_depth \\
        data/dl3dv/540x960/train-scenes --max-views 150

Requires the `depth_anything_3` package to be pip-installed.

Author: Alexander Veicht
"""

import os

os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import argparse
import math
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.utils.io import (
    decode_image,
    encode_depth,
    iter_scenes_from_tar,
    write_scene_to_tar,
)
from splatfactory.geometry import Camera, Pose
from splatfactory.utils.tools import set_seed

logger = get_logger(__name__)

MODEL_NAME = "depth-anything/DA3NESTED-GIANT-LARGE"


def round_robin_chunks(N: int, max_views: int | None) -> list[list[int]]:
    """Split [0, N) into round-robin chunks of up to max_views each."""
    if max_views is None or N <= max_views:
        return [list(range(N))]
    nc = math.ceil(N / max_views)
    return [[i for i in range(N) if i % nc == c] for c in range(nc)]


def extract_depth_for_scene(model, scene: dict, max_views: int | None):
    """Run DA3 on a scene. Returns (webp_bytes_per_view, (d_min, d_max)_per_view)."""
    img_keys = sorted(scene["images"].keys())
    imgs = [decode_image(scene["images"][k]) for k in img_keys]
    N = len(imgs)
    H, W = imgs[0].shape[:2]

    K = Camera(torch.from_numpy(scene["cameras"][:N]).float()).K.numpy()
    w2c = Pose(torch.from_numpy(scene["poses"][:N]).float()).inv().Rt.numpy()

    depth_all = np.zeros((N, H, W), dtype=np.float32)
    for chunk in round_robin_chunks(N, max_views):
        pred = model.inference(
            image=[imgs[i] for i in chunk],
            intrinsics=K[chunk],
            extrinsics=w2c[chunk],
            align_to_input_ext_scale=True,
        )
        d = torch.from_numpy(pred.depth).unsqueeze(1).float()
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
        for j, i in enumerate(chunk):
            depth_all[i] = d[j]

    depths, ranges = [], []
    for i in range(N):
        webp, d_min, d_max = encode_depth(depth_all[i])
        depths.append(webp)
        ranges.append((d_min, d_max))
    return depths, ranges


def _scene_to_write_dict(scene: dict, depths=None, depth_ranges=None) -> dict:
    """Convert iter_scenes_from_tar output to write_scene_to_tar input."""
    img_keys = sorted(scene["images"].keys())
    has_depth = depths is not None
    return {
        "key": scene["key"],
        "cameras": scene["cameras"],
        "poses": scene["poses"],
        "images": [scene["images"][k] for k in img_keys],
        "num_views": scene["meta"]["num_views"],
        "has_depth": has_depth,
        "depths": depths,
        "depth_ranges": depth_ranges,
    }


def _keep_existing_depth(scene: dict):
    depth_keys = sorted(scene["depths"].keys())
    depths = [scene["depths"][k] for k in depth_keys]
    ranges = scene.get("depth_ranges") or []
    return depths, ranges


def shard_is_complete(shard_path: Path) -> bool:
    """True if every scene has a depth member for every image member."""
    images: dict[str, int] = {}
    depths: dict[str, int] = {}
    with tarfile.open(shard_path, "r") as tar:
        for name in tar.getnames():
            key = name.split(".")[0]
            if ".img" in name:
                images[key] = images.get(key, 0) + 1
            elif ".dep" in name and name.endswith(".webp"):
                depths[key] = depths.get(key, 0) + 1
    if not images:
        return False
    return all(depths.get(k, 0) == v for k, v in images.items())


def process_shard(shard_path: Path, model, max_views: int | None, overwrite: bool) -> dict:
    """Rewrite shard in place: extract depth for any scene missing it."""
    tmp_path = shard_path.with_name(shard_path.name + ".tmp")
    stats = {"extracted": 0, "passthrough": 0, "failed": 0}

    with tarfile.open(tmp_path, "w") as out_tar:
        for scene in iter_scenes_from_tar(str(shard_path)):
            n_views = scene["meta"]["num_views"]
            has_depth = len(scene["depths"]) == n_views
            need_extract = overwrite or not has_depth

            if not need_extract:
                depths, ranges = _keep_existing_depth(scene)
                write_scene_to_tar(out_tar, _scene_to_write_dict(scene, depths, ranges))
                stats["passthrough"] += 1
                continue

            try:
                depths, ranges = extract_depth_for_scene(model, scene, max_views)
                write_scene_to_tar(out_tar, _scene_to_write_dict(scene, depths, ranges))
                stats["extracted"] += 1
            except Exception as e:
                logger.error(f"Scene {scene['key']} extraction failed: {e}")
                write_scene_to_tar(out_tar, _scene_to_write_dict(scene))
                stats["failed"] += 1

    os.rename(tmp_path, shard_path)
    return stats


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    return rank, world, local


def cleanup_stale_tmp(shards: list[Path]) -> None:
    for shard in shards:
        tmp = shard.with_name(shard.name + ".tmp")
        if tmp.exists():
            logger.warning(f"Removing stale {tmp.name}")
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description="Extract DA3 depth into tar shards in place")
    parser.add_argument("input_dir", type=Path, help="Shard directory (contains shard-*.tar)")
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help="Max views per DA3 forward pass; round-robin chunks used if N exceeds this.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-extract depth if already present"
    )
    parser.add_argument("--show-all-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local = setup_distributed()
    set_seed(args.seed + rank)

    all_shards = sorted(args.input_dir.glob("*.tar"))
    my_shards = all_shards[rank::world]

    if rank == 0:
        logger.info(f"Input: {args.input_dir}")
        logger.info(f"World size: {world}, total shards: {len(all_shards)}")
        logger.info(f"Max views per chunk: {args.max_views or 'unlimited'}")

    cleanup_stale_tmp(my_shards)

    if args.overwrite:
        todo = my_shards
        already_done = 0
    else:
        todo = [s for s in my_shards if not shard_is_complete(s)]
        already_done = len(my_shards) - len(todo)

    if rank == 0:
        logger.info(f"Loading {MODEL_NAME}...")
    model = DepthAnything3.from_pretrained(MODEL_NAME).to(f"cuda:{local}").eval()

    pbar = tqdm(
        todo,
        desc=f"[Rank {rank}]",
        ncols=120,
        disable=(rank != 0 and not args.show_all_progress),
    )
    totals = {"extracted": 0, "passthrough": 0, "failed": 0}
    t0 = time.time()
    for shard in pbar:
        pbar.set_postfix_str(shard.name)
        try:
            st = process_shard(shard, model, args.max_views, args.overwrite)
        except Exception as e:
            logger.error(f"Shard {shard.name} failed: {e}")
            tmp = shard.with_name(shard.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            continue
        for k in totals:
            totals[k] += st[k]
    pbar.close()

    elapsed = time.time() - t0
    logger.warning(
        f"[Rank {rank}] Done: {len(todo)} processed, {already_done} already complete, "
        f"extracted={totals['extracted']} passthrough={totals['passthrough']} "
        f"failed={totals['failed']} elapsed={elapsed:.0f}s"
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
