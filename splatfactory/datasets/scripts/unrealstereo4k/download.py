"""Download UnrealStereo4K-Q scene zips plus the per-frame cameras of the 4K release.

UnrealStereo4K (SMD-Nets, Tosi et al., CVPR 2021, MIT) has 9 Unreal Engine scenes:
00000-00007 with 1000 stereo pairs each, 00008 with 200 (out-of-domain test).
Two sources, both stored compressed under --output-dir (default data/unrealstereo4k/raw):

  - <scene>.zip -- HuggingFace fabiotosi92/UnrealStereo4K, the quarter-resolution
    release (UnrealStereo4K-Q): <scene>/{Image0,Image1}/NNNNN.jpg (960x540) and
    <scene>/{Disp0,Disp1}/NNNNN.npy (disparity in 960x540 pixels), ~4 GB per scene.
    It ships NO cameras.
  - <scene>_extrinsics.zip -- <scene>/{Extrinsics0,Extrinsics1}/NNNNN.txt cut out of
    the original 4K release on S3 (~74 GB per scene) with HTTP range requests, so only
    those ~1 MB of members are transferred. Same frame indexing as the -Q release
    (verified: the 4K disparity area-downsampled by 4 matches -Q to ~0.1%).

Resumable: finished files are skipped (written to .tmp, renamed on success).

Usage:
    python -m splatfactory.datasets.scripts.unrealstereo4k.download --num-workers 4
    python -m splatfactory.datasets.scripts.unrealstereo4k.download --scenes 00008
    python -m splatfactory.datasets.scripts.unrealstereo4k.download --max-scenes 2
    # cameras only (e.g. for already-extracted -Q scene dirs)
    python -m splatfactory.datasets.scripts.unrealstereo4k.download --sample --cameras-only

Author: Alexander Veicht
"""

import argparse
import io
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import DownloadResult, hf_download, parallel_download

logger = get_logger(__name__)

HF_REPO = "fabiotosi92/UnrealStereo4K"
S3_URL = "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_{}.zip"
ALL_SCENES = [f"{i:05d}" for i in range(9)]
SAMPLE_SCENES = ["00000", "00001", "00008"]  # 2 train + the out-of-domain test scene


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_extrinsics(scene: str, out_dir: Path, num_threads: int = 16) -> DownloadResult:
    """Copy the Extrinsics0/1 members of the 4K scene zip into <scene>_extrinsics.zip."""
    dest = out_dir / f"{scene}_extrinsics.zip"
    if dest.exists():
        return DownloadResult(key=dest.name, success=True, size_bytes=dest.stat().st_size)
    url = S3_URL.format(scene)
    local = threading.local()

    def _read(name: str) -> bytes:
        if not hasattr(local, "zf"):  # zipfile handles are not thread-safe: one per thread
            local.zf = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 14))
        return local.zf.read(name)

    try:
        with zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 14)) as zf:
            names = sorted(n for n in zf.namelist() if "/Extrinsics" in n and n.endswith(".txt"))
        if not names:
            raise ValueError(f"no Extrinsics members in {url}")
        with ThreadPoolExecutor(num_threads) as pool:
            blobs = list(pool.map(_read, names))
        tmp = dest.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, blob in zip(names, blobs):
                zf.writestr(name, blob)
        tmp.rename(dest)
        return DownloadResult(key=dest.name, success=True, size_bytes=dest.stat().st_size)
    except Exception as e:
        logger.error(f"{scene} extrinsics: {e}")
        return DownloadResult(key=dest.name, success=False, error=str(e))


def fetch_images(scene: str, out_dir: Path) -> DownloadResult:
    """Download the -Q <scene>.zip from HuggingFace (kept zipped)."""
    dest = out_dir / f"{scene}.zip"
    if dest.exists():
        return DownloadResult(key=dest.name, success=True, size_bytes=dest.stat().st_size)
    return hf_download(HF_REPO, dest.name, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/unrealstereo4k/raw"))
    parser.add_argument("--scenes", nargs="+", default=None, choices=ALL_SCENES)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sample", action="store_true", help=f"Only {SAMPLE_SCENES}")
    parser.add_argument("--cameras-only", action="store_true", help="Skip the -Q image zips")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    scenes = args.scenes or (SAMPLE_SCENES if args.sample else ALL_SCENES)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(scenes)} scenes -> {args.output_dir}")

    for scene in tqdm(scenes, desc="extrinsics", ncols=100):
        res = fetch_extrinsics(scene, args.output_dir)
        if not res.success:
            logger.error(f"{scene}: {res.error}")

    if not args.cameras_only:
        parallel_download(
            scenes,
            lambda s: fetch_images(s, args.output_dir),
            num_workers=args.num_workers,
            desc="images",
            failed_log=args.output_dir / "failed.tsv",
        )


if __name__ == "__main__":
    main()
