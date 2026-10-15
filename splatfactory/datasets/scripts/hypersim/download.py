"""Download ML-Hypersim per-scene zips from Apple's servers.

Scene list comes from metadata_images_split_scene_v1.csv (fetched from the
ml-hypersim GitHub repo alongside metadata_images.csv). The full dataset is
461 scenes / ~1.9TB -- use --max-scenes for a sample.

Usage:
    python -m splatfactory.datasets.scripts.hypersim.download --max-scenes 10
    python -m splatfactory.datasets.scripts.hypersim.download --scenes ai_001_001

Author: Alexander Veicht
"""

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger

logger = get_logger(__name__)

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
GITHUB_RAW = "https://raw.githubusercontent.com/apple/ml-hypersim/main"
METADATA = {
    "metadata_images_split_scene_v1.csv": (
        f"{GITHUB_RAW}/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
    ),
    "metadata_images.csv": f"{GITHUB_RAW}/evermotion_dataset/analysis/metadata_images.csv",
}


def fetch(url: str, path: Path) -> bool:
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ret = subprocess.call(
        ["curl", "-sSL", "--fail", "-C", "-", "--connect-timeout", "30", url, "--output", str(tmp)]
    )
    if ret != 0:
        return False
    tmp.rename(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/ml-hypersim"))
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=3)
    args = parser.parse_args()

    meta_dir = args.output_dir / "metadata"
    for name, url in METADATA.items():
        if not fetch(url, meta_dir / name):
            raise SystemExit(f"failed to fetch {name}")

    scenes = []
    with open(meta_dir / "metadata_images_split_scene_v1.csv") as f:
        for row in csv.DictReader(f):
            if row["scene_name"] not in scenes:
                scenes.append(row["scene_name"])
    if args.scenes:
        scenes = [s for s in scenes if s in set(args.scenes)]
    elif args.max_scenes:
        scenes = scenes[: args.max_scenes]

    logger.info(f"{len(scenes)} scene zips")
    counts = {True: 0, False: 0}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [
            pool.submit(fetch, f"{BASE_URL}/{s}.zip", args.output_dir / "zip" / f"{s}.zip")
            for s in scenes
        ]
        for f in tqdm(as_completed(futures), total=len(scenes), desc="download", ncols=100):
            counts[f.result()] += 1
    logger.info(f"Done: {counts[True]} ok, {counts[False]} failed")


if __name__ == "__main__":
    main()
