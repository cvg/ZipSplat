"""Convert ARKitScenes raw videos (vga_wide + LiDAR depth) to canonical tar shards.

Raw layout (`<root>/{Training,Validation}/<video_id>/`, assets zipped or extracted):
  - vga_wide[.zip]: 640x480 RGB PNG frames named <video>_<timestamp>.png
  - vga_wide_intrinsics[.zip]: per-frame .pincam "w h fx fy cx cy"
  - lowres_depth[.zip]: 256x192 uint16 PNG LiDAR depth in mm
  - lowres_wide.traj: rows "ts ax ay az tx ty tz" -- world-to-camera axis-angle

Conventions:
  - Per-frame poses interpolated from the ~10 Hz trajectory (position lerp, rotation slerp).
    Frames outside the trajectory or across gaps over 0.3 s are skipped, the rest subsampled
    by --subsample (default 10, ~3 fps from the 30 fps stream).
  - Frames rotated upright per video from Apple's metadata.csv `sky_direction`, since
    vga_wide keeps the sensor frame. Up / Down videos go to <output>/480x640, Left / Right
    videos to <output>/640x480 (portrait), with intrinsics and camera frame rotated along.
  - Principal point centred per frame (crop_to_principal_point, resize_to_cover,
    crop_to_ar) to exactly the tree's shape.
  - Poses stored as OpenCV c2w Pose.data_ [R.flatten(), t].
  - RGB re-encoded as lossless WebP, since the PNG source is resampled.
  - Depth from mm to m, upscaled to the RGB size (INTER_LINEAR, NaN-marked), log-uint8 WebP.

Splits: Training -> train, Validation -> test.

Usage:
    python -m splatfactory.datasets.scripts.arkitscenes.convert --num-workers 8

Author: Alexander Veicht
"""

import argparse
import csv
import json
import os
import zipfile
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
from splatfactory.datasets.utils.io import encode_depth, encode_image  # noqa: E402
from splatfactory.utils.image import (  # noqa: E402
    _round_even,
    crop_to_ar,
    crop_to_principal_point,
    resize_to_cover,
)

logger = get_logger(__name__)

MAX_TRAJ_GAP = 0.3  # skip frames whose bracketing trajectory entries are further apart (s)

# vga_wide keeps the sensor frame whatever way the iPad was held, so most videos are stored
# sideways or upside down (Apple's metadata.csv `sky_direction`). Frames are rotated upright
# per video, as in DUSt3R. Left / Right videos become portrait 640x480 in their own tree.
ROT_K = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}  # np.rot90 quarter turns (CCW)
TARGETS = {0: (480, 640), 1: (640, 480), 2: (480, 640), 3: (640, 480)}  # (out_h, out_w)
# Camera-frame rotation matching np.rot90(img, k): columns = new axes in old camera coords.
ROT_M = {
    0: np.eye(3),
    1: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    2: np.diag([-1.0, -1.0, 1.0]),
    3: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def rotate_intrinsics(k: int, w, h, fx, fy, cx, cy):
    """Intrinsics after np.rot90(img, k) (continuous pixel coords, pp at w/2 = centered)."""
    if k == 0:
        return w, h, fx, fy, cx, cy
    if k == 1:  # u' = v, v' = w - u
        return h, w, fy, fx, cy, w - cx
    if k == 2:  # u' = w - u, v' = h - v
        return w, h, fx, fy, w - cx, h - cy
    return h, w, fy, fx, h - cy, cx  # k == 3: u' = h - v, v' = u


def sky_from_poses(c2ws: list[np.ndarray]) -> str:
    """DUSt3R's fallback for videos without a metadata label (ARKit world up = +z)."""
    R = np.stack([p[:3, :3] for p in c2ws])
    up = (R @ np.array([0.0, -1.0, 0.0])).mean(0)
    right = (R @ np.array([1.0, 0.0, 0.0])).mean(0)
    a_up = np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1, 1)))
    a_right = np.degrees(np.arccos(np.clip(right[2] / np.linalg.norm(right), -1, 1)))
    if abs(a_up - 90) < abs(a_right - 90):
        return "Left" if a_right > 90 else "Right"
    return "Down" if a_up > 90 else "Up"


def load_sky_directions(data_root: Path) -> dict[str, str]:
    """video_id -> Up/Left/Down/Right from Apple's metadata.csv ("NA" -> pose rule later)."""
    path = data_root / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: rerun arkitscenes.download to fetch it")
    with open(path) as f:
        return {r["video_id"]: r["sky_direction"] for r in csv.DictReader(f)}


class VideoSource:
    """Read a video's per-asset files from zips or extracted directories."""

    def __init__(self, video_dir: Path):
        self.dir = video_dir
        self._zips: dict[str, zipfile.ZipFile | None] = {}

    def _zip(self, asset: str) -> zipfile.ZipFile | None:
        if asset not in self._zips:
            zp = self.dir / f"{asset}.zip"
            self._zips[asset] = zipfile.ZipFile(zp, "r") if zp.exists() else None
        return self._zips[asset]

    def list_dir(self, asset: str) -> list[str]:
        zf = self._zip(asset)
        if zf is not None:
            prefix = f"{asset}/"
            return sorted(
                n.removeprefix(prefix)
                for n in zf.namelist()
                if n.startswith(prefix) and not n.endswith("/")
            )
        d = self.dir / asset
        return sorted(f.name for f in d.iterdir() if f.is_file()) if d.is_dir() else []

    def read_bytes(self, asset: str, name: str) -> bytes | None:
        zf = self._zip(asset)
        if zf is not None:
            try:
                return zf.read(f"{asset}/{name}")
            except KeyError:
                return None
        p = self.dir / asset / name
        return p.read_bytes() if p.exists() else None

    def close(self):
        for zf in self._zips.values():
            if zf is not None:
                zf.close()


def load_trajectory(text: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """.traj rows -> (sorted timestamps, w2c 4x4 list). Rows are w2c axis-angle
    (verified photometrically: c2w = inv(row) warps views onto each other)."""
    entries = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        ts = float(parts[0])
        R, _ = cv2.Rodrigues(np.array([float(x) for x in parts[1:4]]))
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = [float(x) for x in parts[4:7]]
        entries.append((ts, w2c))
    entries.sort(key=lambda e: e[0])
    return np.array([e[0] for e in entries]), [e[1] for e in entries]


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z]."""
    aa, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = (aa / angle).flatten()
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(np.clip(q[0], -1, 1))
    s = np.sqrt(max(1 - q[0] * q[0], 1e-24))
    R, _ = cv2.Rodrigues(np.ascontiguousarray(q[1:] / s * angle))
    return R


def interpolate_c2w(timestamps: np.ndarray, w2cs: list[np.ndarray], ts: float) -> np.ndarray | None:
    """c2w at ts: position lerp + rotation slerp between bracketing entries.

    The trajectory is ~10Hz while vga_wide frames are 30Hz, so nearest-neighbor
    matching would keep only ~9% of frames. Returns None when ts falls outside
    the trajectory or the bracketing gap exceeds MAX_TRAJ_GAP.
    """
    i = int(np.searchsorted(timestamps, ts))
    if i == 0 or i >= len(timestamps):
        return None
    t0, t1 = timestamps[i - 1], timestamps[i]
    if t1 - t0 > MAX_TRAJ_GAP:
        return None
    c0, c1 = np.linalg.inv(w2cs[i - 1]), np.linalg.inv(w2cs[i])
    a = float((ts - t0) / max(t1 - t0, 1e-9))
    q0, q1 = _quat(c0[:3, :3]), _quat(c1[:3, :3])
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = np.clip(np.dot(q0, q1), -1, 1)
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1 - a) * theta) * q0 + np.sin(a * theta) * q1) / np.sin(theta)
    c2w = np.eye(4)
    c2w[:3, :3] = _quat_to_R(q)
    c2w[:3, 3] = (1 - a) * c0[:3, 3] + a * c1[:3, 3]
    return c2w


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


# --- per-scene processing -----------------------------------------------------------


def process_scene(
    task: tuple[str, str, str], data_root: Path, subsample: int, skip_start: float
) -> dict:
    split, video_id, sky = task
    src = VideoSource(data_root / split / video_id)
    try:
        return _process(src, video_id, sky, subsample, skip_start)
    finally:
        src.close()


def _process(src: VideoSource, video_id: str, sky: str, subsample: int, skip_start: float) -> dict:
    traj_path = src.dir / "lowres_wide.traj"
    if not traj_path.exists():
        raise FileNotFoundError(f"{video_id}: no trajectory")
    timestamps, w2cs = load_trajectory(traj_path.read_text())
    if not len(timestamps):
        raise ValueError(f"{video_id}: empty trajectory")

    matched = []
    for fname in src.list_dir("vga_wide"):
        if not fname.endswith(".png"):
            continue
        try:
            ts = float(fname.removesuffix(".png").removeprefix(f"{video_id}_"))
        except ValueError:
            continue
        if ts - timestamps[0] < skip_start:
            continue  # ARKit tracking warm-up: first ~1s of poses is unreliable
        c2w = interpolate_c2w(timestamps, w2cs, ts)
        if c2w is not None:
            matched.append((ts, fname, c2w))
    if not matched:
        raise ValueError(f"{video_id}: no frames within the trajectory")
    matched.sort(key=lambda m: m[0])
    matched = matched[::subsample]

    if sky not in ROT_K:  # "NA" in the metadata: derive from the trajectory
        sky = sky_from_poses([m[2] for m in matched])
    k = ROT_K[sky]
    target = TARGETS[k]
    th, tw = target
    images, depths, ranges, poses, cameras = [], [], [], [], []
    for _ts, fname, c2w in matched:
        data = src.read_bytes("vga_wide", fname)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            continue
        h, w = img.shape[:2]

        pin = src.read_bytes("vga_wide_intrinsics", fname.replace(".png", ".pincam"))
        if pin is None:
            continue
        _, _, fx, fy, cx, cy = (float(x) for x in pin.decode().split())

        dep_data = src.read_bytes("lowres_depth", fname)
        depth = None
        if dep_data is not None:
            raw = cv2.imdecode(np.frombuffer(dep_data, np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.dtype == np.uint16:
                depth = raw.astype(np.float32) / 1000.0
                depth[depth <= 0] = np.nan
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Upright rotation (image, depth, intrinsics, camera frame) before pp centering
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            if depth is not None:
                depth = np.ascontiguousarray(np.rot90(depth, k))
            w, h, fx, fy, cx, cy = rotate_intrinsics(k, w, h, fx, fy, cx, cy)
            c2w = c2w.copy()
            c2w[:3, :3] = c2w[:3, :3] @ ROT_M[k]

        ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, target)
        assert (oh, ow) == (th, tw), f"{video_id}/{fname}: {oh}x{ow}"
        assert abs(ocx - ow / 2) < 1.5 and abs(ocy - oh / 2) < 1.5, f"{video_id}/{fname}: pp"

        out = _apply_geometry(img, cx, cy, target)
        images.append(encode_image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)))
        cameras.append([ow, oh, ofx, ofy, ocx, ocy])

        poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

        if depth is not None:
            depth = np.nan_to_num(_apply_geometry_depth(depth, cx, cy, target), nan=-1.0)
        else:
            depth = np.full((th, tw), -1.0, dtype=np.float32)
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    if not images:
        raise ValueError(f"{video_id}: no valid frames")

    return {
        "key": video_id,
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
        "meta_extra": {"sky_direction": sky},
    }


# --- main ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert ARKitScenes to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/arkitscenes/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/arkitscenes"))
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--subsample", type=int, default=10, help="Keep every Nth matched frame")
    parser.add_argument(
        "--skip-start", type=float, default=1.0, help="Drop frames in the first N seconds"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw_splits = {"train": "Training", "test": "Validation"}
    sky_dirs = load_sky_directions(args.data_root)

    def sky_of(split_raw: str, vid: str) -> str:
        sky = sky_dirs.get(vid, "NA")
        if sky in ROT_K:
            return sky
        traj = args.data_root / split_raw / vid / "lowres_wide.traj"
        _, w2cs = load_trajectory(traj.read_text())
        return sky_from_poses([np.linalg.inv(m) for m in w2cs])

    # Up/Down videos stay landscape (480x640), Left/Right become portrait (640x480): one
    # tree per orientation so every tree has a single frame shape.
    for split in args.split:
        split_dir = args.data_root / raw_splits[split]
        if not split_dir.is_dir():
            logger.warning(f"missing {split_dir}, skipping {split}")
            continue
        vids = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        if args.scene_ids:
            vids = [v for v in vids if v in set(args.scene_ids)]
        if args.limit:
            vids = vids[: args.limit]
        if not vids:
            continue
        skies = {v: sky_of(raw_splits[split], v) for v in vids}
        for th, tw in sorted(set(TARGETS.values())):
            tree_vids = [v for v in vids if TARGETS[ROT_K[skies[v]]] == (th, tw)]
            if not tree_vids:
                continue
            convert_tree(args, split, raw_splits[split], tree_vids, skies, th, tw)


def convert_tree(args, split: str, split_raw: str, vids: list[str], skies: dict, th: int, tw: int):
    out_dir = args.output_dir / f"{th}x{tw}" / f"{split}-scenes"
    index_path = out_dir / "index.json"
    if args.overwrite:
        for p in out_dir.glob("shard-*.tar*"):
            p.unlink()
        index_path.unlink(missing_ok=True)
    done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
    remaining = [(split_raw, v, skies[v]) for v in vids if v not in done]
    if not remaining:
        logger.info(f"Skipping {split} {th}x{tw}: all {len(vids)} scenes done")
        return
    logger.info(f"Converting {split} {th}x{tw}: {len(remaining)} videos -> {out_dir}")
    fn = partial(
        process_scene,
        data_root=args.data_root,
        subsample=args.subsample,
        skip_start=args.skip_start,
    )
    scenes = parallel_map(fn, remaining, num_workers=args.num_workers, skip_label=lambda t: t[1])
    write_shards(
        scenes,
        out_dir,
        shard_size_mb=args.shard_size_mb,
        desc=f"arkitscenes/{split}/{th}x{tw}",
        total=len(vids),
        resume=bool(done),
    )


if __name__ == "__main__":
    main()
