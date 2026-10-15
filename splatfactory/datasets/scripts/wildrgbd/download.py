"""Download WildRGB-D category zips from the official HuggingFace release.

The release (huggingface.co/hongchi/wildrgbd, ~3.7 TB, 46 categories) ships one zip per
category, large ones as split archives (<cat>.z01, <cat>.z02, ..., <cat>.zip). The parts are
merged (`zip -s 0`) into <output-dir>/<cat>.zip and removed. Zips stay compressed, convert.py
reads them directly. A category with a finished zip is skipped, so reruns resume, and
interrupted files resume where they stopped.

Sampling: --categories picks categories by name, --max-categories the first N
(alphabetical) of the selection.

Usage:
    python -m splatfactory.datasets.scripts.wildrgbd.download --categories apple car
    python -m splatfactory.datasets.scripts.wildrgbd.download --max-categories 4 --num-workers 4

Author: Alexander Veicht
"""

import argparse
import json
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import CURL_ROBUST, DownloadResult, parallel_download

logger = get_logger(__name__)

REPO = "hongchi/wildrgbd"
TREE_URL = f"https://huggingface.co/api/models/{REPO}/tree/main"
FILE_URL = f"https://huggingface.co/{REPO}/resolve/main"


def list_categories() -> dict[str, list[tuple[str, int]]]:
    """category -> [(filename, size)], parts first (.z01, .z02, ..., .zip)."""
    with urllib.request.urlopen(TREE_URL, timeout=60) as r:
        tree = json.load(r)
    cats = defaultdict(list)
    for e in tree:
        name = e["path"]
        stem, _, ext = name.rpartition(".")
        if e["type"] == "file" and ext.startswith("z"):
            cats[stem].append((name, e["size"]))
    return {c: sorted(f, key=lambda x: (x[0].endswith(".zip"), x[0])) for c, f in cats.items()}


def download_file(item: tuple[str, int, Path]) -> DownloadResult:
    """Resumable curl download (.tmp kept on failure) with size check + atomic rename."""
    name, size, path = item
    if path.exists() and path.stat().st_size == size:
        return DownloadResult(key=name, success=True, size_bytes=size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ret = 0
    if not (tmp.exists() and tmp.stat().st_size == size):
        # HTTP/1.1 since Hugging Face cancels HTTP/2 streams mid-file, and the speed limits
        # abort a stalled connection, both retried
        cmd = ["curl", "-sSL", "--fail", "-C", "-", "--connect-timeout", "30", *CURL_ROBUST]
        ret = subprocess.call(cmd + [f"{FILE_URL}/{name}", "--output", str(tmp)])
    if ret != 0 or not tmp.exists() or tmp.stat().st_size != size:
        return DownloadResult(key=name, success=False, error=f"curl {ret}, rerun to resume")
    tmp.rename(path)
    return DownloadResult(key=name, success=True, size_bytes=size)


def merge_split(category: str, parts_dir: Path, output_dir: Path) -> bool:
    """Merge <cat>.z01.. + <cat>.zip in parts_dir into output_dir/<cat>.zip."""
    out = output_dir / f"{category}.zip"
    tmp = parts_dir / f"{category}.merged.zip"  # zip -s 0 insists on a .zip suffix
    ret = subprocess.call(
        ["zip", "-s", "0", str(parts_dir / f"{category}.zip"), "--out", str(tmp)],
        stdout=subprocess.DEVNULL,
    )
    if ret != 0:
        logger.error(f"{category}: merge failed (zip returned {ret})")
        tmp.unlink(missing_ok=True)
        return False
    # zip -s 0 can silently corrupt entries spanning several parts: verify before cleanup
    ret = subprocess.call(["unzip", "-tq", str(tmp)], stdout=subprocess.DEVNULL)
    if ret != 0:
        logger.error(f"{category}: merged zip failed unzip -t ({ret}); parts kept")
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(out)
    for p in parts_dir.glob(f"{category}.z*"):
        p.unlink()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/wildrgbd"))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    files = list_categories()
    cats = sorted(args.categories or files)
    unknown = set(cats) - set(files)
    if unknown:
        raise SystemExit(f"Unknown categories: {sorted(unknown)}\nAvailable: {sorted(files)}")
    if args.max_categories:
        cats = cats[: args.max_categories]

    todo = [c for c in cats if not (args.output_dir / f"{c}.zip").exists()]
    logger.info(f"{len(cats)} categories, {len(cats) - len(todo)} already done")
    parts_dir = args.output_dir / "parts"
    items = []
    for c in todo:
        dest = args.output_dir if len(files[c]) == 1 else parts_dir
        items += [(name, size, dest / name) for name, size in files[c]]
    gb = sum(size for _, size, _ in items) / 1e9
    logger.info(f"{len(items)} files, {gb:.0f} GB total")

    results = parallel_download(items, download_file, num_workers=args.num_workers, desc="wildrgbd")
    ok = {r.key for r in results if r.success}
    for c in todo:
        if len(files[c]) > 1 and all(name in ok for name, _ in files[c]):
            logger.info(f"Merging {c} ({len(files[c])} parts)")
            merge_split(c, parts_dir, args.output_dir)
    if len(ok) < len(items):
        logger.error("Some downloads failed; rerun to resume.")


if __name__ == "__main__":
    main()
