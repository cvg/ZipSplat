"""Download GObjaverse (G-buffer Objaverse) per-object tars from the Alibaba OSS mirror.

280K Objaverse objects, 40 rendered views each (~6.5TB total, ~25MB per object).
The mirror serves one uncompressed tar per object; tars are kept as-is (never
extracted), convert.py reads them directly.

Object lists come from the per-category index files
gobjaverse_280k_split/gobjaverse_280k_<category>.json (items "<dict_id>/<inst_id>").
Default: every category except Poor-quality.

Raw layout written (`<output-dir>`, default data/gobjaverse/raw):
  - index/gobjaverse_280k_<category>.json
  - <dict_id>/<inst_id>.tar  (members <inst_id>/campos_512_v4/NNNNN/NNNNN{.json,.png,_nd.exr,...})

Resumable: existing tars are skipped (atomic .tmp -> rename), failures go to failed.tsv.

Usage:
    python -m splatfactory.datasets.scripts.gobjaverse.download --num-workers 16
    python -m splatfactory.datasets.scripts.gobjaverse.download \
        --categories Animals Furnitures Food --max-per-category 10 --num-workers 8

Author: Alexander Veicht
"""

import argparse
import json
from pathlib import Path

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import DownloadResult, atomic_download, parallel_download

logger = get_logger(__name__)

BASE_URL = "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d"
TAR_URL = f"{BASE_URL}/objaverse_tar"
INDEX_URL = f"{BASE_URL}/gobjaverse_280k_split/gobjaverse_280k_{{}}.json"
CATEGORIES = [
    "Human-Shape",
    "Animals",
    "Daily-Used",
    "Furnitures",
    "Buildings_Outdoor",
    "Transportations",
    "Plants",
    "Food",
    "Electronics",
    "Poor-quality",
]


def load_index(output_dir: Path, category: str) -> list[str]:
    """Fetch (or reuse cached) category index -> list of "<dict_id>/<inst_id>"."""
    path = output_dir / "index" / f"gobjaverse_280k_{category}.json"
    if not path.exists():
        result = atomic_download(INDEX_URL.format(category), path)
        if not result.success:
            raise RuntimeError(result.error)
    return json.loads(path.read_text())


def download_object(item: str, output_dir: Path, overwrite: bool) -> DownloadResult:
    dest = output_dir / f"{item}.tar"
    if dest.exists() and not overwrite:
        return DownloadResult(key=item, success=True, size_bytes=dest.stat().st_size)
    result = atomic_download(f"{TAR_URL}/{item}.tar", dest)
    result.key = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/gobjaverse/raw"))
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=CATEGORIES,
        default=[c for c in CATEGORIES if c != "Poor-quality"],
    )
    parser.add_argument("--max-per-category", type=int, default=None, help="First N per category")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing tars")
    args = parser.parse_args()

    items, seen = [], set()
    for category in args.categories:
        cat_items = load_index(args.output_dir, category)
        n_total = len(cat_items)
        if args.max_per_category:
            cat_items = cat_items[: args.max_per_category]
        new = [it for it in cat_items if it not in seen]
        seen.update(new)
        items += new
        logger.info(f"{category}: {len(new)} objects (of {n_total})")

    logger.info(f"{len(items)} objects -> {args.output_dir}")
    parallel_download(
        items,
        lambda it: download_object(it, args.output_dir, args.overwrite),
        num_workers=args.num_workers,
        desc="gobjaverse",
        failed_log=args.output_dir / "failed.tsv",
    )


if __name__ == "__main__":
    main()
