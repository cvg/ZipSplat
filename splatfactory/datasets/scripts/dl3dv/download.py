"""Download DL3DV-10K dataset from HuggingFace Hub.

Each user must individually agree to the DL3DV terms on HuggingFace.
Redistribution is not permitted - this script downloads directly from HF.

Usage:
    # Download all subsets at 480P (default)
    python -m splatfactory.datasets.scripts.dl3dv.download

    # Download specific subset at higher res
    python -m splatfactory.datasets.scripts.dl3dv.download --subset 1K --resolution 960P

    # Download 50 sample scenes
    python -m splatfactory.datasets.scripts.dl3dv.download --sample

    # Parallel download with extraction
    python -m splatfactory.datasets.scripts.dl3dv.download --num-workers 4 --extract

    # Specific scenes by hash
    python -m splatfactory.datasets.scripts.dl3dv.download --scene-ids abc123 def456

Author: Alexander Veicht
"""

import argparse
import zipfile
from pathlib import Path

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import (
    DownloadResult,
    add_common_args,
    hf_download,
    parallel_download,
    verify_hf_access,
)

logger = get_logger(__name__)

ALL_SUBSETS = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K"]

RESOLUTION_TO_REPO = {
    "480P": "DL3DV/DL3DV-ALL-480P",
    "960P": "DL3DV/DL3DV-ALL-960P",
    "2K": "DL3DV/DL3DV-ALL-2K",
    "4K": "DL3DV/DL3DV-ALL-4K",
}

META_CSV_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"


def fetch_metadata(cache_dir: Path):
    """Fetch DL3DV metadata CSV (cached). Returns pandas DataFrame."""
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "DL3DV-valid.csv"

    if not csv_path.exists():
        logger.info(f"Downloading metadata CSV -> {csv_path}")
        urllib.request.urlretrieve(META_CSV_URL, csv_path)

    import pandas as pd

    return pd.read_csv(csv_path)


def build_download_list(
    df, subsets: list[str], resolution: str, scene_ids: list[str] | None = None
) -> list[dict]:
    """Build list of {repo, rel_path, hash} dicts from metadata."""
    repo = RESOLUTION_TO_REPO[resolution]

    if scene_ids:
        # Filter by specific hashes, ignore subset
        rows = df[df["hash"].isin(scene_ids)]
        missing = set(scene_ids) - set(rows["hash"])
        if missing:
            logger.warning(f"{len(missing)} scene IDs not found in metadata: {list(missing)[:5]}")
    else:
        rows = df[df["batch"].isin(subsets)]

    items = []
    for _, row in rows.iterrows():
        items.append(
            {
                "repo": repo,
                "rel_path": f"{row['batch']}/{row['hash']}.zip",
                "hash": row["hash"],
            }
        )
    return items


def main():
    parser = argparse.ArgumentParser(description="Download DL3DV-10K dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dl3dv/raw"),
        help="Output directory (default: data/dl3dv/raw)",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=ALL_SUBSETS,
        default=None,
        help="Subsets to download (default: all). Example: --subset 1K 2K",
    )
    parser.add_argument(
        "--resolution",
        choices=["480P", "960P", "2K", "4K"],
        default="480P",
        help="Resolution to download (default: 480P)",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="+",
        default=None,
        help="Download specific scene hashes",
    )
    add_common_args(parser)
    args = parser.parse_args()

    # Sample mode: default to 1K subset, keep user-specified resolution
    if args.sample:
        args.subset = args.subset or ["1K"]

    subsets = args.subset or ALL_SUBSETS
    repo = RESOLUTION_TO_REPO[args.resolution]
    download_dir = args.output_dir / args.resolution

    # Verify HF access
    logger.info(f"Verifying access to {repo}...")
    if not verify_hf_access(repo):
        logger.error(
            f"Access denied. Go to https://huggingface.co/datasets/{repo} "
            f"and agree to the terms first."
        )
        return

    # Fetch metadata and build download list
    cache_dir = args.output_dir / ".cache"
    df = fetch_metadata(cache_dir)
    items = build_download_list(df, subsets, args.resolution, args.scene_ids)

    if args.sample:
        items = items[:50]

    # Filter existing
    if not args.overwrite:
        before = len(items)
        items = [item for item in items if not (download_dir / item["rel_path"]).exists()]
        if before != len(items):
            logger.info(f"Skipping {before - len(items)} existing")

    logger.info(
        f"{len(items)} scenes to download " f"(subsets={subsets}, resolution={args.resolution})"
    )

    if not items:
        logger.info("Nothing to download.")
        return

    # Download
    def _download_one(item: dict) -> DownloadResult:
        return hf_download(item["repo"], item["rel_path"], download_dir)

    results = parallel_download(
        items,
        _download_one,
        num_workers=args.num_workers,
        desc=f"DL3DV {args.resolution}",
        failed_log=download_dir / "failed.log",
    )

    # Extract successfully downloaded files
    if args.extract:
        succeeded = {r.key for r in results if r.success}
        to_extract = [i for i in items if i["rel_path"] in succeeded]
        logger.info(f"Extracting {len(to_extract)} zips...")
        for item in to_extract:
            zip_path = download_dir / item["rel_path"]
            if zip_path.exists():
                extract_dir = zip_path.with_suffix("")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811 - shadow for __main__ handler

    main()
