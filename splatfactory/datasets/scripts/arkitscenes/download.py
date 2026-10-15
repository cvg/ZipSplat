"""Download ARKitScenes raw-variant assets from Apple's servers.

Assets per video (default set = what convert.py needs, ~2-4GB per 10 videos):
  - vga_wide.zip: 640x480 RGB frames (PNG, named <video>_<timestamp>.png)
  - vga_wide_intrinsics.zip: per-frame pinhole .pincam
  - lowres_depth.zip: 256x192 LiDAR depth (uint16 PNG, mm)
  - lowres_wide.traj: camera trajectory (timestamp, axis-angle, translation)

The full raw variant is multi-TB across 5047 videos -- use --max-scenes to
download a sample (first N Training + N Validation videos by metadata order).
Zips are kept compressed; convert.py reads them directly.

Usage:
    python -m splatfactory.datasets.scripts.arkitscenes.download --max-scenes 20
    python -m splatfactory.datasets.scripts.arkitscenes.download --video-ids 40753679

Author: Alexander Veicht
"""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import CURL_ROBUST

logger = get_logger(__name__)

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
SPLITS_CSV = "https://raw.githubusercontent.com/apple/ARKitScenes/main/raw/raw_train_val_splits.csv"
METADATA_CSV = (
    f"{BASE_URL}/raw/metadata.csv"  # video_id, sky_direction (Up/Left/Down/Right/NA), ...
)
DEFAULT_ASSETS = ["vga_wide", "vga_wide_intrinsics", "lowres_depth", "lowres_wide.traj"]

# Videos missing the trajectory (and other 3dod-derived files) at source
MISSING_TRAJ_IDS = {
    "47334522",
    "47334523",
    "42897421",
    "45261582",
    "47333152",
    "47333155",
    "48458535",
    "48018733",
    "47429677",
    "48458541",
    "42897848",
    "47895482",
    "47333960",
    "47430089",
    "42899148",
    "42897612",
    "42899153",
    "42446164",
    "48018149",
    "47332198",
    "47334515",
    "45663223",
    "45663226",
    "45663227",
}


def load_metadata(output_dir: Path) -> list[dict]:
    """Split rows; also fetches Apple's metadata.csv (sky_direction per video) for convert."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in (
        ("raw_train_val_splits.csv", SPLITS_CSV),
        ("metadata.csv", METADATA_CSV),
    ):
        path = output_dir / name
        if not path.exists():
            ret = subprocess.call(["curl", "-sSL", "--fail", url, "--output", str(path)])
            if ret != 0:
                raise RuntimeError(f"failed to fetch {name} ({ret})")
    with open(output_dir / "raw_train_val_splits.csv") as f:
        return list(csv.DictReader(f))


def download_file(url: str, path: Path) -> str:
    if path.exists():
        return "existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ret = subprocess.call(
        ["curl", "-sSL", "--fail", "-C", "-", "--connect-timeout", "30", *CURL_ROBUST]
        + [url, "--output", str(tmp)]
    )
    if ret != 0:
        return "failed"
    tmp.rename(path)
    return "downloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/arkitscenes/raw"))
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--video-ids", nargs="+", default=None)
    parser.add_argument("--max-scenes", type=int, default=None, help="First N videos per split")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    rows = load_metadata(args.output_dir)
    by_split: dict[str, list[str]] = {"Training": [], "Validation": []}
    for row in rows:
        fold = row.get("fold", "")
        vid = row.get("video_id", "")
        if fold in by_split and vid and vid not in MISSING_TRAJ_IDS:
            by_split[fold].append(vid)

    tasks: list[tuple[str, Path]] = []
    for split, vids in by_split.items():
        if args.video_ids:
            vids = [v for v in vids if v in set(args.video_ids)]
        elif args.max_scenes:
            vids = vids[: args.max_scenes]
        for vid in vids:
            for asset in args.assets:
                fname = asset if asset.endswith(".traj") else f"{asset}.zip"
                tasks.append(
                    (f"{BASE_URL}/raw/{split}/{vid}/{fname}", args.output_dir / split / vid / fname)
                )

    logger.info(f"{len(tasks)} files to fetch")
    counts = {"existing": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(download_file, u, p) for u, p in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks), desc="download", ncols=100):
            counts[f.result()] += 1
    logger.info(f"Done: {counts}")
    if counts["failed"]:
        logger.error("Some downloads failed; rerun to resume.")


if __name__ == "__main__":
    main()
