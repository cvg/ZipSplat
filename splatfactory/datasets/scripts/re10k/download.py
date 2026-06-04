"""Download RE10K dataset.

Downloads the RE10K archive (360x640 .torch files) from a hosted zip.

Usage:
    python -m splatfactory.datasets.scripts.re10k.download
    python -m splatfactory.datasets.scripts.re10k.download --extract
    python -m splatfactory.datasets.scripts.re10k.download --output-dir data/re10k --overwrite

Author: Alexander Veicht
"""

import argparse
from pathlib import Path

from splatfactory.datasets.scripts.utils import add_common_args, download_archive

URL = "http://schadenfreude.csail.mit.edu:8000/re10k.zip"


def main():
    parser = argparse.ArgumentParser(description="Download RE10K dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/re10k/raw"))
    add_common_args(parser)
    args = parser.parse_args()
    download_archive(URL, args.output_dir, "re10k.zip", args.overwrite, args.extract)


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F401 - install handler

    main()
