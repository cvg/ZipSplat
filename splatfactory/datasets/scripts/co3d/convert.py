"""Convert CO3Dv2 sequences to canonical tar shards.

Reads category zips (co3d/<category>_NNN.zip, optionally *_singlesequence.zip)
or extracted category dirs. Selection follows DUSt3R's preprocess_co3d.py:
sequences in the fewview_train split (fallback manyview_dev_0) with
viewpoint_quality_score > --min-quality, top --max-per-category by frame count;
a per-category random --test-fraction becomes test-scenes.

Conventions applied (matching scannetpp/blendedmvs converts):
  - Cameras: PyTorch3D NDC -> OpenCV pinhole per frame (DUSt3R math), then the
    principal point is centered via crop_to_principal_point -> resize_to_cover
    -> crop_to_ar. Target size is per scene: first frame's aspect with the
    short side capped at --short-side (sequences differ in size/orientation,
    so the output tree is named "<short-side>s", not <HxW>).
  - Poses: OpenCV c2w Pose.data_ [R.flatten(), t].
  - Images re-encoded JPEG q95 (resampled).
  - Depth: uint16 PNG bit-cast to float16 * scale_adjustment, depth_mask == 0
    and <= 0 -> invalid; same geometry as the image; log-uint8 WebP.

Usage:
    python -m splatfactory.datasets.scripts.co3d.convert --num-workers 8
    python -m splatfactory.datasets.scripts.co3d.convert --categories apple --limit 2

Author: Alexander Veicht
"""

import argparse
import gzip
import json
import os
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from splatfactory import get_logger  # noqa: E402
from splatfactory.datasets.scripts.utils import parallel_map, write_shards  # noqa: E402
from splatfactory.datasets.utils.io import encode_depth  # noqa: E402
from splatfactory.utils.image import (  # noqa: E402
    _round_even,
    crop_to_ar,
    crop_to_principal_point,
    resize_to_cover,
)

logger = get_logger(__name__)

JPEG_QUALITY = 95


# --- category source (zips or extracted dir) ------------------------------------


class CategorySource:
    """Reads category files from extracted dir or the category's zip chunks."""

    def __init__(self, root: Path, category: str):
        self.dir = root / category
        self.zfs = []
        self.names: dict[str, zipfile.ZipFile] = {}
        if not (self.dir / "frame_annotations.jgz").exists():
            zips = sorted(root.glob(f"{category}_[0-9]*.zip"))
            if not zips:
                raise FileNotFoundError(f"no extracted dir or zips for '{category}' in {root}")
            for zp in zips:
                zf = zipfile.ZipFile(zp, "r")
                self.zfs.append(zf)
                for n in zf.namelist():
                    self.names.setdefault(n, zf)

    def read_bytes(self, path: str) -> bytes | None:
        if self.zfs:
            zf = self.names.get(path)
            return zf.read(path) if zf else None
        p = self.dir.parent / path
        return p.read_bytes() if p.exists() else None

    def read_image(self, path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray | None:
        data = self.read_bytes(path)
        if data is None:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    def close(self):
        for zf in self.zfs:
            zf.close()


_SOURCES: dict[str, CategorySource] = {}


def _source(root: Path, category: str) -> CategorySource:
    """Per-process category source cache (zip central directories are pricey).

    Holds one category: tasks arrive grouped by category, and keeping every touched
    category's central directories open OOMs a 460 GB node at 32 workers."""
    if category not in _SOURCES:
        for src in _SOURCES.values():
            src.close()
        _SOURCES.clear()
        _SOURCES[category] = CategorySource(root, category)
    return _SOURCES[category]


# --- annotations ------------------------------------------------------------------


def load_frame_annotations(src: CategorySource, category: str) -> dict[str, list[dict]]:
    data = src.read_bytes(f"{category}/frame_annotations.jgz")
    if data is None:
        return {}
    by_seq = defaultdict(list)
    for ann in json.loads(gzip.decompress(data)):
        by_seq[ann["sequence_name"]].append(ann)
    for seq in by_seq.values():
        seq.sort(key=lambda a: a["frame_number"])
    return dict(by_seq)


def load_quality(src: CategorySource, category: str) -> dict[str, float]:
    data = src.read_bytes(f"{category}/sequence_annotations.jgz")
    if data is None:
        return {}
    anns = json.loads(gzip.decompress(data))
    return {a["sequence_name"]: a.get("viewpoint_quality_score", 0.0) for a in anns}


def load_split_seqs(src: CategorySource, category: str) -> dict[str, int]:
    """Sequences in the train split -> #frames (fewview_train, else manyview_dev_0)."""
    for name in ("set_lists_fewview_train.json", "set_lists_manyview_dev_0.json"):
        data = src.read_bytes(f"{category}/set_lists/{name}")
        if data is None:
            continue
        counts: dict[str, int] = defaultdict(int)
        for seq, _frame, _path in json.loads(data).get("train", []):
            counts[seq] += 1
        if counts:
            return dict(counts)
    return {}


def ndc_to_opencv(viewpoint: dict, w: int, h: int):
    """PyTorch3D NDC camera -> (fx, fy, cx, cy [px], w2c 4x4). DUSt3R math."""
    half = np.array([w, h], dtype=np.float64) / 2
    rescale = half.min()
    fx, fy = np.array(viewpoint["focal_length"], dtype=np.float64) * rescale
    cx, cy = half - np.array(viewpoint["principal_point"], dtype=np.float64) * rescale
    R = np.array(viewpoint["R"], dtype=np.float64)
    T = np.array(viewpoint["T"], dtype=np.float64)
    T[0] *= -1
    T[1] *= -1
    R[:, 0] *= -1
    R[:, 1] *= -1
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = T
    return float(fx), float(fy), float(cx), float(cy), w2c


# --- geometry (same plan as blendedmvs convert) -----------------------------------


def _geometry_plan(w, h, fx, fy, cx, cy, target):
    probe = np.zeros((h, w, 3), np.uint8)
    probe, t = crop_to_principal_point(probe, cx, cy)
    th, tw = target
    probe, t_resize = resize_to_cover(probe, th, tw)
    probe, t_crop = crop_to_ar(probe, tw / th)
    t = t_crop @ t_resize @ t
    oh, ow = probe.shape[:2]
    k = t @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return ow, oh, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _apply_geometry(img, cx, cy, target):
    img, _ = crop_to_principal_point(img, cx, cy)
    th, tw = target
    img, _ = resize_to_cover(img, th, tw)
    img, _ = crop_to_ar(img, tw / th)
    return img


def _apply_geometry_depth(depth, cx, cy, target):
    d, _ = crop_to_principal_point(depth, cx, cy)
    th, tw = target
    h, w = d.shape
    scale = max(th / h, tw / w)
    hn, wn = _round_even(h * scale), _round_even(w * scale)
    if (hn, wn) != (h, w):
        d = cv2.resize(d, (wn, hn), interpolation=cv2.INTER_LINEAR)
    d, _ = crop_to_ar(d, tw / th)
    return d


def _scene_target(w: int, h: int, short_side: int) -> tuple[int, int]:
    """(out_h, out_w): keep aspect, cap the short side, round even."""
    scale = min(1.0, short_side / min(w, h))
    return _round_even(h * scale), _round_even(w * scale)


# --- per-scene processing -----------------------------------------------------------


def _read_depth(src: CategorySource, ann: dict) -> np.ndarray | None:
    info = ann.get("depth") or {}
    if not info.get("path"):
        return None
    raw = src.read_image(info["path"], cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        return None
    depth = raw.view(np.float16).astype(np.float32) * float(info.get("scale_adjustment", 1.0))
    if info.get("mask_path"):
        mask = src.read_image(info["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == depth.shape:
            depth[mask == 0] = np.nan
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


def process_scene(task: tuple[str, str], data_root: Path, short_side: int) -> dict:
    category, seq_name = task
    src = _source(data_root, category)
    anns = load_frame_annotations(src, category)[seq_name]

    target = None
    images, depths, ranges, poses, cameras = [], [], [], [], []
    dropped = 0
    for ann in anns:
        img = src.read_image(ann["image"]["path"])
        if img is None or img.max() == 0:
            dropped += 1
            continue
        h, w = img.shape[:2]
        if target is None:
            target = _scene_target(w, h, short_side)
        fx, fy, cx, cy, w2c = ndc_to_opencv(ann["viewpoint"], w, h)

        ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, target)
        assert abs(ocx - ow / 2) < 1.5 and abs(ocy - oh / 2) < 1.5, f"{task}: pp not centered"

        out = _apply_geometry(img, cx, cy, target)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"{task}: jpeg encode failed")
        images.append(buf.tobytes())
        cameras.append([ow, oh, ofx, ofy, ocx, ocy])

        c2w = np.linalg.inv(w2c)
        poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

        depth = _read_depth(src, ann)
        if depth is not None and depth.shape == (h, w):
            depth = np.nan_to_num(_apply_geometry_depth(depth, cx, cy, target), nan=-1.0)
        else:
            depth = np.full((oh, ow), -1.0, dtype=np.float32)
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    if not images:
        raise ValueError(f"{task}: no valid frames")
    if dropped:
        logger.warning(f"{task}: dropped {dropped} frames")

    return {
        "key": f"{category}_{seq_name}",
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
    }


# --- main ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert CO3D to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/co3d"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>")
    parser.add_argument("--short-side", type=int, default=720)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--min-quality", type=float, default=0.5)
    parser.add_argument("--max-per-category", type=int, default=50)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max sequences per split")
    args = parser.parse_args()

    out_root = args.output_dir or args.data_root

    if args.categories:
        categories = args.categories
    else:
        found = {
            d.name
            for d in args.data_root.iterdir()
            if d.is_dir() and (d / "frame_annotations.jgz").exists()
        }
        for zp in args.data_root.glob("*_[0-9]*.zip"):
            found.add(zp.stem.split("_")[0])
        categories = sorted(found)
    if not categories:
        raise SystemExit(f"no categories found in {args.data_root}")

    # DUSt3R-style selection + per-category test holdout
    rng = np.random.RandomState(args.seed)
    splits: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
    for category in categories:
        try:
            src = _source(args.data_root, category)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
        split_counts = load_split_seqs(src, category)
        quality = load_quality(src, category)
        cand = [s for s in split_counts if quality.get(s, 0.0) > args.min_quality]
        cand.sort(key=lambda s: split_counts[s], reverse=True)
        selected = cand[: args.max_per_category]
        if not selected:
            logger.warning(f"{category}: no sequences pass the filter")
            continue
        # floor, not max(1, ...): tiny categories (e.g. the single-sequence sample)
        # keep their sequences in train instead of losing them all to test
        n_test = int(len(selected) * args.test_fraction)
        test_idx = set(rng.choice(len(selected), size=n_test, replace=False)) if n_test else set()
        for i, seq in enumerate(selected):
            splits["test" if i in test_idx else "train"].append((category, seq))
        logger.info(f"{category}: {len(selected)} sequences ({n_test} test)")

    # Workers are forked: drop the parent's open zip handles so each worker reopens its own
    # (shared handles share one file offset and corrupt concurrent reads).
    for src in _SOURCES.values():
        src.close()
    _SOURCES.clear()

    for split, tasks in splits.items():
        if args.limit:
            tasks = tasks[: args.limit]
        if not tasks:
            continue
        out_dir = out_root / f"{args.short_side}s" / f"{split}-scenes"
        index_path = out_dir / "index.json"
        if args.overwrite:
            for p in out_dir.glob("shard-*.tar*"):
                p.unlink()
            index_path.unlink(missing_ok=True)
        done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
        remaining = [t for t in tasks if f"{t[0]}_{t[1]}" not in done]
        if not remaining:
            logger.info(f"Skipping {split}: all {len(tasks)} scenes done")
            continue
        logger.info(f"Converting {split}: {len(remaining)} sequences -> {out_dir}")

        fn = partial(process_scene, data_root=args.data_root, short_side=args.short_side)
        scenes = parallel_map(
            fn, remaining, num_workers=args.num_workers, skip_label=lambda t: f"{t[0]}/{t[1]}"
        )
        write_shards(
            scenes,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"co3d/{split}",
            total=len(tasks),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
