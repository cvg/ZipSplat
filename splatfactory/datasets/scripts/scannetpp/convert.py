"""Convert ScanNet++ raw scenes to canonical tar shards (DSLR + iPhone).

No intermediate files: the mesh depth is rendered on the GPU with open3d's off-screen renderer
and encoded straight into the shards.

Per device:
  - DSLR: `resized_images/*.JPG` (OPENCV_FISHEYE) undistorted to PINHOLE in memory, exactly
    as the ScanNet++ toolkit does, plus the mesh depth at the new K.
  - iPhone: every registered frame of `rgb.mkv`, undistorted in memory (OPENCV to PINHOLE),
    plus the mesh depth. Its test split (the NVS validation rooms) is the ScanNet++ benchmark.

Conventions:
  - `cameras` = Camera.data_ = [w, h, fx, fy, cx, cy] from the transforms.
  - `poses` = Pose.data_ = [R.flatten(), t] of the OpenCV c2w (nerfstudio c2w with columns
    1 and 2 negated).
  - Depth is stored as view-space planar z, consistent with `poses`.

Output layout: <output-dir>/<HxW>/<device>/<split>-scenes/shard-*.tar, where <HxW> is the
output size: full resolution per device (dslr 1168x1752, iphone 1440x1920), or with
--max-side N both cropped to the DSLR aspect ratio (e.g. 1080 gives 720x1080). Splits:
train is nvs_sem_train, test is nvs_sem_val.

Usage:
    python -m splatfactory.datasets.scripts.scannetpp.convert --num-workers 12
    python -m splatfactory.datasets.scripts.scannetpp.convert --max-side 1080 --num-workers 12

Author: Alexander Veicht
"""

import argparse
import json
import os
import re
import tarfile
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

MESH_NAME = "scans/mesh_aligned_0.05.ply"
SPLIT_FILES = {"train": "nvs_sem_train.txt", "test": "nvs_sem_val.txt"}
JPEG_QUALITY = 95  # for iphone frames re-encoded from the video (dslr is passthrough)


# --- camera / pose conventions ----------------------------------------------


def _c2w_to_pose_data(c2w_opengl) -> np.ndarray:
    """nerfstudio c2w (OpenGL) -> Pose.data_ [R.flatten(), t] in OpenCV convention."""
    c2w = np.array(c2w_opengl, dtype=np.float64).copy()
    c2w[:, 1:3] *= -1  # OpenGL -> OpenCV (negate y, z columns)
    return np.concatenate([c2w[:3, :3].flatten(), c2w[:3, 3]]).astype(np.float32)


def _ns_c2w_to_colmap_w2c(c2w_opengl) -> np.ndarray:
    """nerfstudio c2w -> COLMAP world-to-camera (places the camera in the mesh frame)."""
    c2w = np.array(c2w_opengl, dtype=np.float64).copy()
    c2w[2, :] *= -1
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[0:3, 1:3] *= -1
    return np.linalg.inv(c2w)


# --- GPU mesh depth renderer (open3d Filament, EGL headless) -----------------


class MeshDepthRenderer:
    """Off-screen depth renderer, rebuilt only when the frame size changes."""

    def __init__(self):
        self._r = None
        self._mat = None
        self._wh = None

    def set_scene(self, mesh, w: int, h: int):
        from open3d.visualization import rendering  # lazy: optional dependency (open3d)

        if self._wh != (w, h):
            self._r = rendering.OffscreenRenderer(w, h)
            self._mat = rendering.MaterialRecord()
            self._mat.shader = "defaultUnlit"
            self._wh = (w, h)
        else:
            self._r.scene.clear_geometry()
        self._r.scene.add_geometry("mesh", mesh, self._mat)

    def render(self, w, h, fx, fy, cx, cy, c2w_opengl) -> np.ndarray:
        """View-space planar z depth in meters, -1 at invalid pixels."""
        import open3d as o3d  # lazy: optional dependency

        intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        self._r.setup_camera(intr, _ns_c2w_to_colmap_w2c(c2w_opengl))
        d = np.asarray(self._r.render_to_depth_image(z_in_view_space=True)).astype(np.float32)
        d[~np.isfinite(d)] = -1.0
        d[d == 0.0] = -1.0
        return d


_RENDERER = None  # per-process singleton (lazy, so worker forks/spawns stay clean)


def _renderer() -> MeshDepthRenderer:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = MeshDepthRenderer()
    return _RENDERER


# --- helpers -----------------------------------------------------------------


def _read_split(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_transforms(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _intrinsics(tr: dict) -> tuple[int, int, float, float, float, float]:
    return (
        int(tr["w"]),
        int(tr["h"]),
        float(tr["fl_x"]),
        float(tr["fl_y"]),
        float(tr["cx"]),
        float(tr["cy"]),
    )


def _read_mesh(scene_dir: Path):
    import open3d as o3d  # lazy: optional dependency

    return o3d.io.read_triangle_mesh(str(scene_dir / MESH_NAME))


def _dslr_image_reader(dslr_dir: Path):
    """Return read(file_path) -> BGR ndarray, from `resized_images.tar` if present
    (loaded once, keyed by basename) else from loose `resized_images/` files."""
    tar_path = dslr_dir / "resized_images.tar"
    if tar_path.exists():
        with tarfile.open(tar_path) as t:
            blobs = {
                Path(m.name).name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()
            }

        def read(file_path: str):
            buf = blobs.get(Path(file_path).name)
            if buf is None:
                raise FileNotFoundError(f"{file_path} not in {tar_path}")
            return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

        return read

    img_dir = dslr_dir / "resized_images"

    def read(file_path: str):
        img = cv2.imread(str(img_dir / file_path))
        if img is None:
            raise FileNotFoundError(img_dir / file_path)
        return img

    return read


def _geometry_plan(w, h, fx, fy, cx, cy, target):
    """Output camera (out_w, out_h, fx, fy, cx, cy) after centring the principal point.

    With a `target` (out_h, out_w), the frame is also resized and cropped to it. The
    intrinsics follow every step, and _apply_geometry replays the same steps per frame.
    """
    probe = np.zeros((h, w, 3), np.uint8)
    probe, t = crop_to_principal_point(probe, cx, cy)
    if target is not None:
        th, tw = target
        probe, t_resize = resize_to_cover(probe, th, tw)
        probe, t_crop = crop_to_ar(probe, tw / th)
        t = t_crop @ t_resize @ t
    oh, ow = probe.shape[:2]
    k = t @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return ow, oh, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])


def _apply_geometry(img: np.ndarray, cx, cy, target) -> np.ndarray:
    """crop_to_principal_point (center the PP) + optional resize_to_cover + crop_to_ar."""
    img, _ = crop_to_principal_point(img, cx, cy)
    if target is not None:
        th, tw = target
        img, _ = resize_to_cover(img, th, tw)
        img, _ = crop_to_ar(img, tw / th)
    return img


def _build_scene(scene_id, cameras, poses, images, depths, ranges) -> dict:
    n = len(images)
    return {
        "key": scene_id,
        "cameras": np.asarray(cameras, dtype=np.float32),
        "poses": np.stack(poses).astype(np.float32),
        "images": images,
        "num_views": n,
        "has_depth": True,
        "depths": depths,
        "depth_ranges": ranges,
    }


# --- per-scene processing ----------------------------------------------------


def _fisheye_undistort_maps(tr: dict):
    """Distorted OPENCV_FISHEYE transforms -> (new_K pinhole, map1, map2, w, h).

    Bit-identical to ScanNet++ toolkit `dslr/undistort.py`: estimate the new camera
    matrix (balance=0), force the principal point to the image center, then build
    CV_32FC1 remap tables. Verified exact (max diff 0) vs the toolkit's cv2 4.12.
    """
    w, h = int(tr["w"]), int(tr["h"])
    K = np.array(
        [[tr["fl_x"], 0, tr["cx"]], [0, tr["fl_y"], tr["cy"]], [0, 0, 1]], dtype=np.float64
    )
    dist = np.array([tr["k1"], tr["k2"], tr["k3"], tr["k4"]], dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, dist, (w, h), np.eye(3), balance=0.0
    )
    new_K[0, 2] = w / 2.0
    new_K[1, 2] = h / 2.0
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, dist, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )
    return new_K, map1, map2, w, h


def process_dslr_scene(scene_id: str, data_root: Path, target=None) -> dict:
    scene_dir = data_root / "data" / scene_id
    dslr = scene_dir / "dslr"
    # fisheye (OPENCV_FISHEYE) transforms, undistorted to PINHOLE below
    tr = _load_transforms(dslr / "nerfstudio" / "transforms.json")
    new_K, map1, map2, w, h = _fisheye_undistort_maps(tr)
    fx, fy = float(new_K[0, 0]), float(new_K[1, 1])
    cx, cy = float(new_K[0, 2]), float(new_K[1, 2])  # undistorted PP (already centered)
    ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, target)
    frames = sorted(
        (f for f in tr["frames"] if not f.get("is_bad", False)), key=lambda x: x["file_path"]
    )
    if not frames:
        raise ValueError(f"{scene_id}: no valid dslr frames")

    r = _renderer()
    r.set_scene(_read_mesh(scene_dir), ow, oh)

    read_image = _dslr_image_reader(dslr)
    images, depths, ranges, poses = [], [], [], []
    for f in frames:
        src = read_image(f["file_path"])  # distorted BGR (from resized_images.tar or loose)
        und = cv2.remap(
            src, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        und = _apply_geometry(und, cx, cy, target)
        ok, buf = cv2.imencode(".jpg", und, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"{scene_id}: jpeg encode failed for {f['file_path']}")
        images.append(buf.tobytes())
        poses.append(_c2w_to_pose_data(f["transform_matrix"]))
        depth = r.render(ow, oh, ofx, ofy, ocx, ocy, f["transform_matrix"])
        dep_bytes, d_min, d_max = encode_depth(depth)
        depths.append(dep_bytes)
        ranges.append((d_min, d_max))

    cameras = [[ow, oh, ofx, ofy, ocx, ocy]] * len(frames)
    return _build_scene(scene_id, cameras, poses, images, depths, ranges)


def process_iphone_scene(scene_id: str, data_root: Path, target=None) -> dict:
    scene_dir = data_root / "data" / scene_id
    iph = scene_dir / "iphone"
    tr = _load_transforms(iph / "nerfstudio" / "transforms.json")
    w, h, fx, fy, cx, cy = _intrinsics(tr)  # native PP (off-center)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(
        [tr.get("k1", 0.0), tr.get("k2", 0.0), tr.get("p1", 0.0), tr.get("p2", 0.0)],
        dtype=np.float64,
    )
    ow, oh, ofx, ofy, ocx, ocy = _geometry_plan(w, h, fx, fy, cx, cy, target)
    frames = sorted(
        (f for f in tr["frames"] if not f.get("is_bad", False)), key=lambda x: x["file_path"]
    )
    if not frames:
        raise ValueError(f"{scene_id}: no valid iphone frames")

    r = _renderer()
    r.set_scene(_read_mesh(scene_dir), ow, oh)

    cap = cv2.VideoCapture(str(iph / "rgb.mkv"))
    if not cap.isOpened():
        raise RuntimeError(f"{scene_id}: cannot open rgb.mkv")
    images, depths, ranges, poses = [], [], [], []
    try:
        for f in frames:
            frame_idx = int(re.findall(r"\d+", f["file_path"])[-1])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"{scene_id}: failed to read frame {frame_idx}")
            und = cv2.undistort(bgr, K, dist)  # OPENCV -> PINHOLE (PP still off-center)
            und = _apply_geometry(und, cx, cy, target)  # center PP + optional resize/crop
            ok, buf = cv2.imencode(".jpg", und, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"{scene_id}: jpeg encode failed for frame {frame_idx}")
            images.append(buf.tobytes())
            poses.append(_c2w_to_pose_data(f["transform_matrix"]))
            depth = r.render(ow, oh, ofx, ofy, ocx, ocy, f["transform_matrix"])
            dep_bytes, d_min, d_max = encode_depth(depth)
            depths.append(dep_bytes)
            ranges.append((d_min, d_max))
    finally:
        cap.release()

    cameras = [[ow, oh, ofx, ofy, ocx, ocy]] * len(frames)
    return _build_scene(scene_id, cameras, poses, images, depths, ranges)


def _process_scene(scene_id: str, *, device: str, data_root: Path, target=None) -> dict:
    if device == "dslr":
        return process_dslr_scene(scene_id, data_root, target)
    return process_iphone_scene(scene_id, data_root, target)


# --- main --------------------------------------------------------------------


def _peek_dims(data_root: Path, device: str, scene_id: str) -> tuple[int, int]:
    """(w, h) from a device's transforms.json (distorted for dslr, native for iphone)."""
    tr = _load_transforms(data_root / "data" / scene_id / device / "nerfstudio" / "transforms.json")
    return int(tr["w"]), int(tr["h"])


def _output_size(data_root: Path, device: str, scene_id: str, target) -> tuple[int, int]:
    """Actual (out_w, out_h) the converter produces -- after PP-centering crop (+ resize)."""
    tr = _load_transforms(data_root / "data" / scene_id / device / "nerfstudio" / "transforms.json")
    w, h = int(tr["w"]), int(tr["h"])
    if device == "dslr":
        cx, cy = w / 2.0, h / 2.0  # undistort centers the PP -> crop is a no-op
    else:
        cx, cy = float(tr["cx"]), float(tr["cy"])
    ow, oh, *_ = _geometry_plan(w, h, 1.0, 1.0, cx, cy, target)
    return ow, oh


def main():
    parser = argparse.ArgumentParser(description="Convert ScanNet++ to canonical tar shards")
    parser.add_argument("--data-root", type=Path, default=Path("data/scannetpp_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/scannetpp_v2"))
    parser.add_argument(
        "--device", nargs="+", choices=["dslr", "iphone"], default=["dslr", "iphone"]
    )
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="cap the long side at this many px (both devices cropped to DSLR's AR for a "
        "common size, e.g. 1080 -> 720x1080). Default: keep each device's full resolution.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-size-mb", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene-ids", nargs="+", default=None, help="Process only these scenes")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N scenes")
    args = parser.parse_args()

    # common target size: the DSLR frame scaled to max_side, the iPhone frames cropped to it
    target = None
    if args.max_side:
        ref = args.scene_ids or _read_split(args.data_root / "splits" / SPLIT_FILES[args.split[0]])
        dw, dh = _peek_dims(args.data_root, "dslr", ref[0])
        s = args.max_side / max(dw, dh)
        target = (_round_even(dh * s), _round_even(dw * s))  # (out_h, out_w)

    for device in args.device:
        for split in args.split:
            split_path = args.data_root / "splits" / SPLIT_FILES[split]
            if not split_path.exists():
                logger.warning(f"Split file missing, skipping: {split_path}")
                continue
            scene_ids = args.scene_ids or _read_split(split_path)
            if args.limit:
                scene_ids = scene_ids[: args.limit]

            ow, oh = _output_size(args.data_root, device, scene_ids[0], target)
            out_dir = args.output_dir / f"{oh}x{ow}" / device / f"{split}-scenes"
            index_path = out_dir / "index.json"

            if args.overwrite:
                for p in out_dir.glob("shard-*.tar*"):
                    p.unlink()
                index_path.unlink(missing_ok=True)

            # Shard-level resume: skip scenes already committed to a shard.
            done = set(json.loads(index_path.read_text())) if index_path.exists() else set()
            remaining = [s for s in scene_ids if s not in done]
            if not remaining:
                logger.info(f"Skipping {device}/{split}: all {len(scene_ids)} scenes done")
                continue
            if done:
                logger.info(f"Resuming {device}/{split}: {len(done)} done, {len(remaining)} to go")
            else:
                logger.info(f"Converting {device}/{split}: {len(remaining)} scenes -> {out_dir}")

            stats: dict = {}
            fn = partial(_process_scene, device=device, data_root=args.data_root, target=target)
            scenes = parallel_map(
                fn, remaining, num_workers=args.num_workers, skip_label=lambda s: s, stats=stats
            )
            write_shards(
                scenes,
                out_dir,
                shard_size_mb=args.shard_size_mb,
                desc=f"{device}/{split}",
                total=len(scene_ids),
                resume=bool(done),
                stats=stats,
            )


if __name__ == "__main__":
    import multiprocessing as mp

    from splatfactory import logger  # noqa: F811

    try:
        mp.set_start_method("spawn")  # EGL/Filament contexts are not fork-safe
    except RuntimeError:
        pass
    main()
