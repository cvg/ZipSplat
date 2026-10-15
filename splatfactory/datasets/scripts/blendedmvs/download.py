"""Download BlendedMVS low-res (768x576, 113 scenes) from the official GitHub release.

The release ships one zip split into 16 parts (~30GB total). All parts are
required (no per-scene sampling possible); downloads resume via curl -C -.
After download the parts are merged into BlendedMVS_combined.zip, which
convert.py can read directly -- or extract it for faster parallel conversion:

    unzip -o data/blendedmvs/BlendedMVS_combined.zip -d data/blendedmvs

Usage:
    python -m splatfactory.datasets.scripts.blendedmvs.download
    python -m splatfactory.datasets.scripts.blendedmvs.download --num-workers 4 --no-merge

Author: Alexander Veicht
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger

logger = get_logger(__name__)

BASE_URL = "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0"
PARTS = [f"BlendedMVS.z{i:02d}" for i in range(1, 16)] + ["BlendedMVS.zip"]


def download_part(part: str, output_dir: Path) -> bool:
    """Download one part with resume. Returns True on success."""
    path = output_dir / part
    ret = subprocess.call(
        [
            "curl",
            "-sSL",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            f"{BASE_URL}/{part}",
            "--output",
            str(path),
        ]
    )
    if ret != 0:
        logger.error(f"FAILED {part} (curl returned {ret})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/blendedmvs"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-merge", action="store_true", help="Skip merging the split zip")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(download_part, p, args.output_dir): p for p in PARTS}
        for f in tqdm(as_completed(futures), total=len(PARTS), desc="download", ncols=100):
            ok += f.result()
    logger.info(f"{ok}/{len(PARTS)} parts present")
    if ok < len(PARTS):
        logger.error("Missing parts; rerun to resume. Skipping merge.")
        return

    if not args.no_merge:
        combined = args.output_dir / "BlendedMVS_combined.zip"
        if combined.exists():
            logger.info(f"{combined} already exists, skipping merge")
            return
        logger.info("Merging split zip (zip -s 0)...")
        ret = subprocess.call(
            [
                "zip",
                "-q",
                "-s",
                "0",
                str(args.output_dir / "BlendedMVS.zip"),
                "--out",
                str(combined),
            ]
        )
        if ret != 0:
            logger.error(f"zip merge failed ({ret})")
            return
        logger.info(f"Merged -> {combined}. Extract with: unzip -o {combined} -d {args.output_dir}")


if __name__ == "__main__":
    main()
