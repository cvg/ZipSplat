"""Convert UnrealStereo4K(-Q) stereo sequences to canonical tar shards.

Raw layout (`<raw-dir>`, written by download.py; zips are read directly, an
extracted `<scene>/` dir with the same members works too):
  - <scene>.zip: <scene>/{Image0,Image1}/NNNNN.jpg -- left/right renders, 960x540
    (HF quarter-res release; the 3840x2160 originals have the same layout as .png)
  - <scene>.zip: <scene>/{Disp0,Disp1}/NNNNN.npy -- float32 disparity in pixels of
    that resolution, aligned with the left/right image
  - <scene>_extrinsics.zip: <scene>/{Extrinsics0,Extrinsics1}/NNNNN.txt -- line 1 the
    3x3 K at 3840x2160 (fx = fy = 1920, pp exactly centered), line 2 the 3x4
    world-to-camera [R|t] (same parsing as CUT3R's preprocess_unreal4k.py)

Each sequence is a static Unreal Engine environment rendered from 1000 (00008: 200)
stereo pairs scattered through it (median step between consecutive frames ~1-4 m and
~23 deg, not a smooth video). Each sequence is split by frame index into consecutive
chunks of --chunk-pairs stereo pairs (default 100 -> 200 views, ~110 MB, since the
loader holds whole records in worker memory), one scene record per chunk keyed
`<scene>_<chunk:03d>`. Views are interleaved [L0, R0, L1, R1, ...]; cam_ids (0 = left,
1 = right) and time_ids (frame index) are stored so the stereo partner of a view is
recoverable.

Conventions applied:
  - Intrinsics: the 4K K scaled by (w / 3840, h / 2160) to the stored resolution.
  - Poses: the UE world is left-handed (det R = -1 for every frame), so the world
    Y axis is flipped (w2c @ diag(1, -1, 1)); the camera frame is already OpenCV
    (x right, y down, z forward; the right camera sits at +baseline along x).
    Stored as c2w Pose.data_ [R.flatten(), t]. Units as rendered (~meters).
  - Depth: z = fx * baseline / disparity with fx at the stored resolution and the
    per-frame baseline from the two extrinsics (0.2 in 00000, 0.5 in 00001/00008).
    Invalid (disp <= 0, non-finite) or sky / far plane (z > MAX_DEPTH; the renders
    put it at ~1e4, with an empty gap above ~500) -> -1, then log-uint8 WebP.
  - Geometry: per-frame pp from the Extrinsics K (not assumed centered), then
    pp-centering crop -> resize_to_cover -> crop_to_ar to --target-size, K composed
    through every transform and asserted centered within 1.5 px
    (default 540x960 = the -Q native size: an exact identity, so the JPEGs pass
    through verbatim; any other size re-encodes JPEG q95, depth goes through the
    same geometry with NaN-marked INTER_LINEAR).

Splits (by sequence): 00000-00007 train, 00008 (the official out-of-domain set) test.

Usage:
    python -m splatfactory.datasets.scripts.unrealstereo4k.convert --num-workers 9
    python -m splatfactory.datasets.scripts.unrealstereo4k.convert --max-frames 100

Author: Alexander Veicht
"""

import argparse
import io as _io
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
from splatfactory.datasets.scripts.utils import (  # noqa: E402
    parallel_map,
    update_camera_intrinsics,
    write_shards,
)
from splatfactory.datasets.utils.io import encode_depth  # noqa: E402
from splatfactory.utils.image import (  # noqa: E402
    _round_even,
    crop_to_ar,
    crop_to_principal_point,
    resize_to_cover,
)

logger = get_logger(__name__)

SPLITS = {"train": [f"{i:05d}" for i in range(8)], "test": ["00008"]}
REF_W, REF_H = 3840, 2160  # resolution the Extrinsics K refers to
MAX_DEPTH = 1000.0  # beyond -> sky / far plane (~1e4 in the renders)
JPEG_QUALITY = 95
FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])  # left-handed UE world -> right-handed


class SceneSource:
    """Read `<scene>/<rel>` from <scene>.zip / <scene>_extrinsics.zip or `<scene>/`."""

    def __init__(self, raw_dir: Path, scene: str):
        self.scene = scene
        self.dir = raw_dir / scene
        paths = [raw_dir / f"{scene}.zip", raw_dir / f"{scene}_extrinsics.zip"]
        self.zips = [zipfile.ZipFile(p) for p in paths if p.exists()]
        self.names = [set(zf.namelist()) for zf in self.zips]

    def read(self, rel: str) -> bytes:
        name = f"{self.scene}/{rel}"
        for zf, names in zip(self.zips, self.names):
            if name in names:
                return zf.read(name)
        return (self.dir / rel).read_bytes()

    def list(self, sub: str) -> list[str]:
        prefix = f"{self.scene}/{sub}/"
        out = {n.removeprefix(prefix) for names in self.names for n in names}
        out = {n for n in out if n and "/" not in n}
        if (self.dir / sub).is_dir():
            out |= {p.name for p in (self.dir / sub).iterdir()}
        return sorted(out)

    def close(self):
        for zf in self.zips:
            zf.close()


def parse_extrinsics(text: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (K 3x3 at REF_W x REF_H, raw left-handed w2c 4x4)."""
    lines = text.split("\n")
    K = np.array(lines[0].split(), dtype=np.float64).reshape(3, 3)
    w2c = np.eye(4)
    w2c[:3] = np.array(lines[1].split(), dtype=np.float64).reshape(3, 4)
    return K, w2c


# --- geometry (same plan as blendedmvs / arkitscenes converts) ----------------------


def _geometry_plan(w, h, fx, fy, cx, cy, target):
    probe = np.zeros((h, w, 3), np.uint8)
    probe, t = crop_to_principal_point(probe, cx, cy)
    th, tw = target
    probe, t_resize = resize_to_cover(probe, th, tw)
    probe, t_crop = crop_to_ar(probe, tw / th)
    t = t_crop @ t_resize @ t
    oh, ow = probe.shape[:2]
    return update_camera_intrinsics(fx, fy, cx, cy, t, ow, oh), np.allclose(t, np.eye(3))


def _apply_geometry(img, cx, cy, target):
    img, _ = crop_to_principal_point(img, cx, cy)
    th, tw = target
    img, _ = resize_to_cover(img, th, tw)
    img, _ = crop_to_ar(img, tw / th)
    return img


def _apply_geometry_depth(depth, cx, cy, target):
    """Same size math as _apply_geometry; invalid stays NaN through the resize."""
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


def chunk_tasks(raw_dir: Path, scene: str, max_frames: int | None, chunk_pairs: int) -> list:
    """-> [(scene, key, frame ids)] with consecutive frame-index chunks of chunk_pairs."""
    src = SceneSource(raw_dir, scene)
    try:
        frames = sorted(n.rsplit(".", 1)[0] for n in src.list("Image0"))
    finally:
        src.close()
    if max_frames:
        frames = frames[:max_frames]
    return [
        (scene, f"{scene}_{i // chunk_pairs:03d}", frames[i : i + chunk_pairs])
        for i in range(0, len(frames), chunk_pairs)
    ]


def process_scene(task: tuple[str, str, list[str]], raw_dir: Path, target) -> dict:
    scene, key, frames = task
    src = SceneSource(raw_dir, scene)
    try:
        return _process(src, scene, key, frames, target)
    finally:
        src.close()


def _process(src: SceneSource, scene: str, key: str, frames: list[str], target) -> dict:
    img_ext = src.list("Image0")[0].rsplit(".", 1)[1]

    th, tw = target
    images, depths, ranges, poses, cameras, cam_ids, time_ids = [], [], [], [], [], [], []
    for fid in frames:
        cams = [parse_extrinsics(src.read(f"Extrinsics{c}/{fid}.txt").decode()) for c in (0, 1)]
        baseline = float((cams[0][1] @ np.linalg.inv(cams[1][1]))[0, 3])
        for c, (K, w2c) in enumerate(cams):
            disp = np.load(_io.BytesIO(src.read(f"Disp{c}/{fid}.npy"))).astype(np.float32)
            h, w = disp.shape
            sx, sy = w / REF_W, h / REF_H
            fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy

            cam, identity = _geometry_plan(w, h, fx, fy, cx, cy, target)
            ow, oh = int(cam[0]), int(cam[1])
            assert (oh, ow) == (th, tw), f"{scene}/{fid}: output {oh}x{ow} != {th}x{tw}"
            assert abs(cam[4] - ow / 2) < 1.5 and abs(cam[5] - oh / 2) < 1.5, f"{scene}: pp"
            cameras.append(cam)

            data = src.read(f"Image{c}/{fid}.{img_ext}")
            if identity and img_ext == "jpg":
                images.append(data)  # verbatim passthrough
            else:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                assert img.shape[:2] == (h, w), f"{scene}/{fid}: image/disparity size mismatch"
                out = _apply_geometry(img, cx, cy, target)
                ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    raise RuntimeError(f"{scene}/{fid}: jpeg encode failed")
                images.append(buf.tobytes())

            c2w = FLIP_Y @ np.linalg.inv(w2c)  # = inv(w2c @ FLIP_Y)
            poses.append(np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32))

            with np.errstate(divide="ignore", invalid="ignore"):
                depth = (fx * baseline / disp).astype(np.float32)
            depth[~np.isfinite(disp) | (disp <= 0) | ~(depth <= MAX_DEPTH)] = np.nan
            if not identity:
                depth = _apply_geometry_depth(depth, cx, cy, target)
            depth = np.nan_to_num(depth, nan=-1.0)
            dep_bytes, d_min, d_max = encode_depth(depth)
            depths.append(dep_bytes)
            ranges.append((d_min, d_max))
            cam_ids.append(c)
            time_ids.append(int(fid))

    return {
        "key": key,
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": len(images),
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
        "cam_ids": np.asarray(cam_ids, dtype=np.int32),
        "time_ids": np.asarray(time_ids, dtype=np.int32),
    }


# --- main ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert UnrealStereo4K to canonical tar shards")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/unrealstereo4k/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/unrealstereo4k"))
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--target-size", type=int, nargs=2, default=[540, 960], metavar=("H", "W"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max sequences per split")
    parser.add_argument("--max-frames", type=int, default=None, help="Stereo pairs per sequence")
    parser.add_argument(
        "--chunk-pairs", type=int, default=100, help="Stereo pairs per scene record"
    )
    args = parser.parse_args()

    target = tuple(args.target_size)
    th, tw = target
    for split in args.split:
        scenes = [
            s
            for s in SPLITS[split]
            if ((args.raw_dir / f"{s}.zip").exists() or (args.raw_dir / s).is_dir())
            and (
                (args.raw_dir / f"{s}_extrinsics.zip").exists()
                or (args.raw_dir / s / "Extrinsics0").is_dir()
            )
        ]
        if args.scene_ids:
            scenes = [s for s in scenes if s in set(args.scene_ids)]
        if args.limit:
            scenes = scenes[: args.limit]
        if not scenes:
            logger.info(f"No {split} scenes with images + extrinsics in {args.raw_dir}")
            continue

        out_dir = args.output_dir / f"{th}x{tw}" / f"{split}-scenes"
        index_path = out_dir / "index.json"
        if args.overwrite:
            for p in out_dir.glob("shard-*.tar*"):
                p.unlink()
            index_path.unlink(missing_ok=True)
        done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
        tasks = [
            t
            for s in scenes
            for t in chunk_tasks(args.raw_dir, s, args.max_frames, args.chunk_pairs)
        ]
        remaining = [t for t in tasks if t[1] not in done]
        if not remaining:
            logger.info(f"Skipping {split}: all {len(tasks)} chunks done")
            continue
        logger.info(f"Converting {split}: {len(remaining)} chunks -> {out_dir}")

        fn = partial(process_scene, raw_dir=args.raw_dir, target=target)
        scenes_iter = parallel_map(
            fn, remaining, num_workers=args.num_workers, skip_label=lambda t: t[1]
        )
        write_shards(
            scenes_iter,
            out_dir,
            shard_size_mb=args.shard_size_mb,
            desc=f"unrealstereo4k/{split}",
            total=len(tasks),
            resume=bool(done),
        )


if __name__ == "__main__":
    main()
