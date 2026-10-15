"""Download CO3Dv2 category zips from the official links manifest.

Full dataset is ~5.5TB (51 categories, multi-chunk zips per category) -- too big
to hold locally. Sampling modes:
  - --single-sequence: official small variant (~9GB, ~2 sequences/category)
  - --max-chunks N: only the first N zip chunks per category (chunk 000 carries
    frame_annotations.jgz / set_lists; sequences split across missing chunks are
    dropped at convert time)
  - --categories: subset of categories

Zips are kept compressed; convert.py reads them directly.

Usage:
    python -m splatfactory.datasets.scripts.co3d.download --single-sequence
    python -m splatfactory.datasets.scripts.co3d.download \
        --categories hydrant teddybear bench --max-chunks 2 --num-workers 4

Author: Alexander Veicht
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import CURL_ROBUST

logger = get_logger(__name__)

LINKS_URL = "https://raw.githubusercontent.com/facebookresearch/co3d/main/co3d/links.json"


def load_links(output_dir: Path) -> dict:
    """Fetch (or reuse cached) links.json."""
    path = output_dir / "links.json"
    if not path.exists():
        ret = subprocess.call(["curl", "-sSL", "--fail", LINKS_URL, "--output", str(path)])
        if ret != 0:
            raise RuntimeError(f"failed to fetch links.json ({ret})")
    return json.loads(path.read_text())


def download_file(url: str, path: Path) -> str:
    """Resumable download with atomic rename. Returns status."""
    if path.exists():
        return "existing"
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/co3d"))
    parser.add_argument("--single-sequence", action="store_true", help="~9GB official subset")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="First N zips per category")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = load_links(args.output_dir)
    variant = links["singlesequence" if args.single_sequence else "full"]

    all_cats = sorted(k for k in variant if k != "METADATA")
    cats = args.categories or all_cats
    unknown = set(cats) - set(all_cats)
    if unknown:
        raise SystemExit(f"Unknown categories: {sorted(unknown)}\nAvailable: {all_cats}")

    tasks = [(u, args.output_dir / u.split("/")[-1]) for u in variant.get("METADATA", [])]
    for cat in cats:
        urls = variant[cat]
        if args.max_chunks:
            urls = urls[: args.max_chunks]
        tasks += [(u, args.output_dir / u.split("/")[-1]) for u in urls]

    logger.info(f"{len(cats)} categories, {len(tasks)} files")
    counts = {"existing": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(download_file, u, p): p for u, p in tasks}
        for f in tqdm(as_completed(futures), total=len(tasks), desc="download", ncols=100):
            counts[f.result()] += 1
    logger.info(f"Done: {counts}")
    if counts["failed"]:
        logger.error("Some downloads failed; rerun to resume.")


if __name__ == "__main__":
    main()
