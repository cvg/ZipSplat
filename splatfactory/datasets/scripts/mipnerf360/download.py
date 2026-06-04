"""Download MipNeRF 360 dataset.

Downloads the 360_v2 archive (~3.6 GB) with 7 public scenes
(bicycle, bonsai, counter, garden, kitchen, room, stump).
Note: flowers and treehill are restricted and not in this public release.

Usage:
    python -m splatfactory.datasets.scripts.mipnerf360.download
    python -m splatfactory.datasets.scripts.mipnerf360.download --extract
    python -m splatfactory.datasets.scripts.mipnerf360.download --output-dir data/mipnerf360/raw --overwrite

Author: Alexander Veicht
"""

import argparse
from pathlib import Path

from splatfactory.datasets.scripts.utils import add_common_args, download_archive

URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


def main():
    parser = argparse.ArgumentParser(description="Download MipNeRF 360 dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mipnerf360/raw"))
    add_common_args(parser)
    args = parser.parse_args()
    download_archive(URL, args.output_dir, "360_v2.zip", args.overwrite, args.extract)


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F401 - install handler

    main()
