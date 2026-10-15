"""Shared utilities for dataset download and conversion scripts.

Author: Alexander Veicht
"""

import itertools
import json
import os
import socket
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.utils.io import write_scene_to_tar

logger = get_logger(__name__)

# urllib downloads (atomic_download) have no timeout by default: a stalled connection
# hangs a worker thread forever. 120 s of silence raises instead, and the caller retries.
socket.setdefaulttimeout(120)

# Extra curl flags for large files from flaky hosts: HTTP/1.1 (HuggingFace cancels HTTP/2
# streams mid-file, curl 92), abort a connection that stalls below 10 KB/s for 2 min
# instead of hanging forever, and retry both on the server.
CURL_ROBUST = [
    "--http1.1",
    "--speed-limit",
    "10000",
    "--speed-time",
    "120",
    "--retry",
    "5",
    "--retry-delay",
    "10",
    "--retry-all-errors",
]


@dataclass
class DownloadResult:
    """Result of a single download operation."""

    key: str
    success: bool
    size_bytes: int = 0
    error: str | None = None


def add_common_args(parser):
    """Add shared download CLI arguments to an argparse parser."""
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel download workers (default: 0 = sequential)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract archives after downloading",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download a small subset of the dataset",
    )


def atomic_download(
    url: str, dest: Path, retries: int = 3, show_progress: bool = False
) -> DownloadResult:
    """Download a file atomically: write to .tmp, rename on success.

    Cleans up partial .tmp file on failure or interrupt.
    Retries with exponential backoff on transient errors.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        pbar = None
        try:

            def _reporthook(block_num, block_size, total_size):
                nonlocal pbar
                if pbar is None and show_progress and total_size > 0:
                    pbar = tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        ncols=120,
                        leave=False,
                    )
                if pbar is not None:
                    downloaded = block_num * block_size
                    pbar.update(min(block_size, total_size - pbar.n))

            urllib.request.urlretrieve(url, tmp, reporthook=_reporthook)
            if pbar is not None:
                pbar.close()
            tmp.rename(dest)
            return DownloadResult(
                key=dest.name,
                success=True,
                size_bytes=dest.stat().st_size,
            )
        except KeyboardInterrupt:
            if pbar is not None:
                pbar.close()
            tmp.unlink(missing_ok=True)
            raise
        except Exception:
            if pbar is not None:
                pbar.close()
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                wait = 2**attempt
                logger.debug(f"Retry {attempt + 1}/{retries} for {url} in {wait}s")
                time.sleep(wait)
            else:
                error = f"Failed after {retries} attempts: {url}"
                logger.error(error)
                return DownloadResult(key=dest.name, success=False, error=error)

    # unreachable, but satisfies type checkers
    return DownloadResult(key=dest.name, success=False, error="unexpected")


def verify_hf_access(repo_id: str) -> bool:
    """Check if the user has access to a HuggingFace dataset repo."""
    try:
        fs = HfFileSystem()
        fs.ls(f"datasets/{repo_id}")
        return True
    except Exception:
        return False


def hf_download(repo_id: str, filename: str, local_dir: Path, retries: int = 3) -> DownloadResult:
    """Download a file from HuggingFace Hub with retry and cache cleanup.

    Returns DownloadResult for compatibility with parallel_download.
    """
    local_dir = Path(local_dir)

    for attempt in range(retries):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
            downloaded = local_dir / filename
            size = downloaded.stat().st_size if downloaded.exists() else 0
            return DownloadResult(key=filename, success=True, size_bytes=size)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt < retries - 1:
                wait = 2**attempt
                logger.debug(f"Retry {attempt + 1}/{retries} for {repo_id}/{filename} in {wait}s")
                time.sleep(wait)
            else:
                error = f"Failed after {retries} attempts: {repo_id}/{filename} ({e})"
                logger.error(error)
                return DownloadResult(key=filename, success=False, error=error)

    return DownloadResult(key=filename, success=False, error="unexpected")


def parallel_download(
    items: list,
    download_fn,
    num_workers: int = 0,
    desc: str = "Downloading",
    failed_log: Path | None = None,
) -> list[DownloadResult]:
    """Download items with optional parallelism, progress, and failed log.

    Args:
        items: List of items to pass to download_fn.
        download_fn: Callable(item) -> DownloadResult.
        num_workers: 0 for sequential, >0 for thread pool.
        desc: Progress bar description.
        failed_log: If set, write failed items to this file (tsv: key \\t error).

    Returns:
        List of all DownloadResults.
    """
    # Silence HF progress bars when using parallel_download
    disable_progress_bars()

    results: list[DownloadResult] = []
    success_count = 0
    downloaded_bytes = 0

    pbar = tqdm(total=len(items), desc=desc, unit="file", ncols=120)

    def _update(result: DownloadResult):
        nonlocal success_count, downloaded_bytes
        results.append(result)
        if result.success:
            success_count += 1
            downloaded_bytes += result.size_bytes
        gb = downloaded_bytes / (1024**3)
        n_done = len(results)
        est_gb = (downloaded_bytes / n_done * len(items)) / (1024**3) if n_done else 0
        fail = n_done - success_count
        pbar.set_postfix_str(f"ok={success_count} fail={fail} {gb:.1f}/{est_gb:.1f}GB")
        pbar.update(1)

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(download_fn, item): item for item in items}
            for future in as_completed(futures):
                _update(future.result())
    else:
        for item in items:
            _update(download_fn(item))

    pbar.close()

    # Summary
    failed = [r for r in results if not r.success]
    gb = downloaded_bytes / (1024**3)
    logger.info(f"Done: {success_count}/{len(items)} files, {gb:.1f} GB, {len(failed)} failed")

    # Write failed log
    if failed and failed_log:
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log, "w") as f:
            for r in sorted(failed, key=lambda r: r.key):
                f.write(f"{r.key}\t{r.error}\n")
        logger.warning(f"{len(failed)} failures logged to {failed_log}")

    return results


# --- Shard writing ---------------------------------------------------------------


def write_shards(
    scenes,
    output_dir: Path,
    shard_size_mb: int = 512,
    desc: str = "Writing shards",
    total: int | None = None,
    resume: bool = False,
    stats: dict | None = None,
) -> dict[str, str]:
    """Write scene dicts to tar shards with size-based rotation and atomic commits.

    Each shard is written to ``shard-NNNNNN.tar.tmp`` and renamed on close, so committed
    shards are always complete, and ``index.json`` only lists committed shards.

    Args:
        scenes: Scene dicts in the io.write_scene_to_tar format.
        output_dir: Directory of the shards and index.json.
        shard_size_mb: Approximate shard size in MB.
        desc: Progress bar description.
        total: Expected scene count, for the progress and the projected size.
        resume: Keep the committed shards, continue their numbering and skip their scenes.
            Scenes of a partial shard are processed again.

    Returns:
        The scene key to shard file name mapping.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_limit = shard_size_mb * 1024 * 1024
    index_path = output_dir / "index.json"

    # Drop any partial shard left by a previous crash (never atomically committed).
    for stale in output_dir.glob("shard-*.tar.tmp"):
        logger.warning(f"Removing stale {stale.name}")
        stale.unlink()

    index: dict[str, str] = {}
    shard_idx = 0
    closed_bytes = 0
    committed_keys: set[str] = set()
    if resume and index_path.exists():
        index = json.load(open(index_path))
        committed_keys = set(index)
        committed_shards = sorted(set(index.values()))
        closed_bytes = sum(
            (output_dir / n).stat().st_size for n in committed_shards if (output_dir / n).exists()
        )
        shard_idx = max((int(n[6:12]) for n in committed_shards), default=-1) + 1
        logger.info(f"Resuming: {len(committed_keys)} scenes committed, next shard {shard_idx}")

    n_resumed = len(index)
    tar = None
    tmp_path: Path | None = None
    scenes_in_shard = 0
    n_done = 0

    def _open_shard() -> str:
        nonlocal tar, tmp_path, scenes_in_shard
        shard_name = f"shard-{shard_idx:06d}.tar"
        tmp_path = output_dir / f"{shard_name}.tmp"
        tar = tarfile.open(tmp_path, "w")
        scenes_in_shard = 0
        return shard_name

    def _commit_shard(shard_name: str):
        """Atomically commit the current shard: close .tmp -> rename -> flush index."""
        nonlocal closed_bytes, shard_idx
        tar.close()
        final = output_dir / shard_name
        os.rename(tmp_path, final)
        closed_bytes += final.stat().st_size
        with open(index_path, "w") as f:
            json.dump(index, f)
        shard_idx += 1

    shard_name = _open_shard()
    pbar = tqdm(scenes, desc=desc, ncols=100, total=total, initial=n_resumed)
    for scene in pbar:
        if scene["key"] in committed_keys:
            continue  # already durable in a committed shard
        write_scene_to_tar(tar, scene)
        tar.fileobj.flush()
        index[scene["key"]] = shard_name
        scenes_in_shard += 1
        n_done += 1

        written = closed_bytes + tar.offset
        accounted = n_resumed + n_done
        q = f"q={stats['ready']}/{stats['max']} " if stats and "ready" in stats else ""
        size = (
            f"{written / 1e9:.0f}/{written / accounted * total / 1e9:.0f}GB"
            if total
            else f"{written / 1e9:.1f}GB"
        )
        pbar.set_postfix_str(f"{q}sh{shard_idx} {size}")

        if tar.offset >= shard_limit:
            _commit_shard(shard_name)
            shard_name = _open_shard()

    pbar.close()

    # Commit the trailing shard if it has content, else drop the empty .tmp.
    if scenes_in_shard > 0:
        _commit_shard(shard_name)
    else:
        tar.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Wrote {len(index)} scenes to {shard_idx} shards in {output_dir}")
    return index


# --- Parallel scene conversion ---------------------------------------------------


def parallel_map(
    fn: Callable,
    items: Iterable,
    num_workers: int = 0,
    max_pending: int | None = None,
    skip_label: Callable = repr,
    stats: dict | None = None,
) -> Iterator:
    """Apply fn to items in a process pool, yielding results as they complete.

    Items whose fn raises are logged and skipped.

    Args:
        fn: Function of one item.
        items: The items.
        num_workers: Pool size, 0 runs sequentially.
        max_pending: Cap on items in flight. Default: 2 * num_workers.
        skip_label: Label of an item in the skip message.
        stats: Optional dict, kept updated with the window size ("max") and the finished
            results not yet consumed ("ready").
    """
    if num_workers == 0:
        for item in items:
            try:
                yield fn(item)
            except Exception as e:
                tqdm.write(f"Skipping {skip_label(item)}: {e}")
        return

    max_pending = max_pending or num_workers * 2
    if stats is not None:
        stats["max"] = max_pending
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        src = iter(items)
        futures = {pool.submit(fn, it): it for it in itertools.islice(src, max_pending)}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                item = futures.pop(f)
                if stats is not None:
                    stats["ready"] = sum(1 for g in futures if g.done())
                try:
                    yield f.result()
                except Exception as e:
                    tqdm.write(f"Skipping {skip_label(item)}: {e}")
                nxt = next(src, None)
                if nxt is not None:
                    futures[pool.submit(fn, nxt)] = nxt


# --- Archive downloads -----------------------------------------------------------


def download_archive(
    url: str, output_dir: Path, filename: str, overwrite: bool = False, extract: bool = False
) -> Path:
    """Download a single archive (with skip-if-exists and optional zip extract).

    Returns the local path to the archive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / filename

    if archive.exists() and not overwrite:
        logger.info(f"Already exists: {archive} ({archive.stat().st_size / 1e9:.1f} GB)")
        logger.info("Use --overwrite to re-download")
    else:
        logger.info(f"Downloading {url} -> {archive}")
        result = atomic_download(url, archive, show_progress=True)
        if not result.success:
            raise RuntimeError(result.error)
        logger.info(f"Downloaded {result.size_bytes / 1e9:.1f} GB")

    if extract:
        logger.info(f"Extracting {archive} -> {output_dir}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(output_dir)
        logger.info("Extraction complete")

    return archive


# --- Intrinsics updates ----------------------------------------------------------


def update_camera_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    transform_3x3: np.ndarray,
    new_w: int,
    new_h: int,
) -> np.ndarray:
    """Compose an image-space 3x3 transform with intrinsics.

    Returns (w, h, fx, fy, cx, cy) float32 - the Camera.data_ format.
    """
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    K_new = transform_3x3 @ K
    return np.array(
        [new_w, new_h, K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]],
        dtype=np.float32,
    )
