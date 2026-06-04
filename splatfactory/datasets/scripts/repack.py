"""Repack tar shards to a target size, in place.

Reads scenes from existing shards and writes new ones sized ~target MB each.
Sources are deleted as soon as their scenes are durably committed to new shards,
so peak disk usage stays close to 1x the input dataset (not 2x).

Naming during processing:
    _new-NNNNNN.tar.tmp   - writing
    _new-NNNNNN.tar       - committed (scenes inside are durable)
After all sources consumed, _new-* are renumbered to shard-NNNNNN.tar.

Usage:
    python -m splatfactory.datasets.scripts.repack data/re10k/360x640/train-scenes
    python -m splatfactory.datasets.scripts.repack <dir> --target-size-mb 512

Author: Alexander Veicht
"""

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.utils.io import iter_scenes_from_tar, write_scene_to_tar

logger = get_logger(__name__)


def _scene_to_write_dict(scene: dict) -> dict:
    """Convert iter_scenes_from_tar output to write_scene_to_tar input."""
    img_keys = sorted(scene["images"].keys())
    has_depth = bool(scene.get("depths")) and len(scene["depths"]) == scene["meta"]["num_views"]
    return {
        "key": scene["key"],
        "cameras": scene["cameras"],
        "poses": scene["poses"],
        "images": [scene["images"][k] for k in img_keys],
        "num_views": scene["meta"]["num_views"],
        "has_depth": has_depth,
        "depths": (
            [scene["depths"][k] for k in sorted(scene["depths"].keys())] if has_depth else None
        ),
        "depth_ranges": scene.get("depth_ranges") if has_depth else None,
    }


def _scan_committed(input_dir: Path) -> tuple[set[str], int]:
    """Scan _new-*.tar for committed scene keys and next available index."""
    committed_keys: set[str] = set()
    existing = sorted(input_dir.glob("_new-*.tar"))
    for path in existing:
        with tarfile.open(path, "r") as tar:
            for name in tar.getnames():
                committed_keys.add(name.split(".")[0])
    next_idx = max((int(p.stem.split("-")[1]) for p in existing), default=-1) + 1
    return committed_keys, next_idx


def main():
    parser = argparse.ArgumentParser(description="Repack tar shards to a target size, in place")
    parser.add_argument("input_dir", type=Path, help="Shard directory (contains *.tar)")
    parser.add_argument("--target-size-mb", type=int, default=512, help="Target shard size in MB")
    args = parser.parse_args()

    target_bytes = args.target_size_mb * 1024 * 1024

    # Cleanup stale .tmp from previous crashed runs
    for tmp in args.input_dir.glob("_new-*.tar.tmp"):
        logger.warning(f"Removing stale {tmp.name}")
        tmp.unlink()

    committed_keys, out_idx = _scan_committed(args.input_dir)
    if committed_keys:
        logger.info(f"Resuming: {len(committed_keys)} scenes in {out_idx} committed _new-*.tar")

    sources = sorted(p for p in args.input_dir.glob("*.tar") if not p.name.startswith("_new-"))
    logger.info(f"Sources: {len(sources)} tars, target: {args.target_size_mb} MB")
    if not sources and not committed_keys:
        logger.info("Nothing to repack")
        return

    # Open first output tar
    tmp_path = args.input_dir / f"_new-{out_idx:06d}.tar.tmp"
    final_path = tmp_path.with_suffix("")
    current_tar = tarfile.open(tmp_path, "w")
    uncommitted: list[Path] = []
    n_written = 0
    n_skipped = 0
    t0 = time.time()

    def commit_current():
        """Close current tar, commit via rename, delete uncommitted sources."""
        nonlocal tmp_path, final_path, current_tar, out_idx, uncommitted
        current_tar.close()
        os.rename(tmp_path, final_path)
        for src in uncommitted:
            src.unlink()
        uncommitted = []
        out_idx += 1
        tmp_path = args.input_dir / f"_new-{out_idx:06d}.tar.tmp"
        final_path = tmp_path.with_suffix("")
        current_tar = tarfile.open(tmp_path, "w")

    pbar = tqdm(sources, desc="Repacking", ncols=120)
    for source in pbar:
        pbar.set_postfix_str(source.name)
        for scene in iter_scenes_from_tar(str(source)):
            if scene["key"] in committed_keys:
                n_skipped += 1
                continue
            if current_tar.offset > 0 and current_tar.offset >= target_bytes:
                commit_current()
            write_scene_to_tar(current_tar, _scene_to_write_dict(scene))
            current_tar.fileobj.flush()
            n_written += 1
        uncommitted.append(source)
    pbar.close()

    # Flush trailing tar
    current_tar.close()
    if current_tar.offset > 0:
        os.rename(tmp_path, final_path)
    else:
        tmp_path.unlink()
    for src in uncommitted:
        src.unlink()

    # Final renumber: _new-*.tar -> shard-*.tar
    new_tars = sorted(args.input_dir.glob("_new-*.tar"))
    index: dict[str, str] = {}
    for i, path in enumerate(new_tars):
        final_name = f"shard-{i:06d}.tar"
        os.rename(path, args.input_dir / final_name)
        with tarfile.open(args.input_dir / final_name, "r") as tar:
            for name in tar.getnames():
                index[name.split(".")[0]] = final_name

    with open(args.input_dir / "index.json", "w") as f:
        json.dump(index, f)

    elapsed = time.time() - t0
    logger.info(
        f"Done: {n_written} written, {n_skipped} skipped (already committed), "
        f"{len(new_tars)} shards, {elapsed:.0f}s"
    )


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
