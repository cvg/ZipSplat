"""Convert MipNeRF 360 dataset to tar shards (one tar per scene).

Reads COLMAP sparse reconstructions directly from 360_v2.zip (no extraction).
Handles PINHOLE, SIMPLE_PINHOLE, OPENCV, and SIMPLE_RADIAL camera models.
Distorted images are decoded, undistorted, and re-encoded as WebP lossless.

Usage:
    python -m splatfactory.datasets.scripts.mipnerf360.convert
    python -m splatfactory.datasets.scripts.mipnerf360.convert --image-subdir images_4

Author: Alexander Veicht
"""

import argparse
import collections
import json
import struct
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.datasets.scripts.utils import update_camera_intrinsics
from splatfactory.datasets.utils.io import decode_image, encode_image, write_scene_to_tar
from splatfactory.geometry import Pose
from splatfactory.utils.image import crop_to_principal_point, resize_to_cover

logger = get_logger(__name__)

SCENES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
IMAGE_SUBDIR_SCALE = {"images": 1, "images_2": 2, "images_4": 4, "images_8": 8}

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
}

ColmapCamera = collections.namedtuple("ColmapCamera", ["id", "model", "width", "height", "params"])
ColmapImage = collections.namedtuple("ColmapImage", ["id", "qvec", "tvec", "camera_id", "name"])


def _read_next_bytes(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_cameras_binary(data: bytes) -> dict[int, ColmapCamera]:
    """Parse cameras.bin from raw bytes."""
    fid = BytesIO(data)
    cameras = {}
    num_cameras = _read_next_bytes(fid, 8, "Q")[0]
    for _ in range(num_cameras):
        camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
        num_params = CAMERA_MODEL_IDS[model_id].num_params
        params = np.array(_read_next_bytes(fid, 8 * num_params, "d" * num_params))
        cameras[camera_id] = ColmapCamera(
            id=camera_id,
            model=CAMERA_MODEL_IDS[model_id].model_name,
            width=width,
            height=height,
            params=params,
        )
    return cameras


def read_images_binary(data: bytes) -> dict[int, ColmapImage]:
    """Parse images.bin from raw bytes."""
    fid = BytesIO(data)
    images = {}
    num_images = _read_next_bytes(fid, 8, "Q")[0]
    for _ in range(num_images):
        props = _read_next_bytes(fid, 64, "idddddddi")
        image_id = props[0]
        qvec = np.array(props[1:5])
        tvec = np.array(props[5:8])
        camera_id = props[8]
        name = ""
        ch = _read_next_bytes(fid, 1, "c")[0]
        while ch != b"\x00":
            name += ch.decode("utf-8")
            ch = _read_next_bytes(fid, 1, "c")[0]
        num_points2D = _read_next_bytes(fid, 8, "Q")[0]
        if num_points2D > 0:
            _read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
        images[image_id] = ColmapImage(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name
        )
    return images


def qvec2rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def get_intrinsics(cam: ColmapCamera) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) in pixels from a COLMAP camera."""
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        f, cx, cy = cam.params[:3]
        return f, f, cx, cy
    if cam.model in ("PINHOLE", "OPENCV", "RADIAL"):
        fx, fy, cx, cy = cam.params[:4]
        return fx, fy, cx, cy
    raise ValueError(f"Unsupported camera model: {cam.model}")


def get_distortion_coeffs(cam: ColmapCamera) -> np.ndarray | None:
    """OpenCV distortion coefficients, or None for pinhole."""
    if cam.model in ("PINHOLE", "SIMPLE_PINHOLE"):
        return None
    if cam.model == "SIMPLE_RADIAL":
        return np.array([cam.params[3], 0, 0, 0], dtype=np.float64)
    if cam.model == "RADIAL":
        return np.array([cam.params[4], cam.params[5], 0, 0], dtype=np.float64)
    if cam.model == "OPENCV":
        return np.array(cam.params[4:8], dtype=np.float64)
    return None


def undistort_image(img, K, dist_coeffs):
    """Undistort, return (undistorted_img, new_K)."""
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h))
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_K)
    return undistorted, new_K


def _read_image_bytes(zf, scene_path: str, image_subdir: str, name: str) -> bytes | None:
    """Read an image by name, falling back to full-res images/ if the chosen subdir lacks it."""
    for subdir in (image_subdir, "images"):
        try:
            return zf.read(f"{scene_path}/{subdir}/{name}")
        except KeyError:
            continue
    return None


def _pose_12_from_colmap(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP (qvec, tvec) world-to-camera -> Pose.data_ (flat c2w)."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = qvec2rotmat(qvec)
    w2c[:3, 3] = tvec
    return Pose.from_4x4mat(torch.from_numpy(w2c)).inv().data_.numpy().astype(np.float32)


def convert_scene(zf: zipfile.ZipFile, zip_prefix: str, scene_name: str, image_subdir: str) -> dict:
    """Convert a single scene from zip to scene dict for write_scene_to_tar."""
    scene_path = f"{zip_prefix}{scene_name}"
    colmap_cameras = read_cameras_binary(zf.read(f"{scene_path}/sparse/0/cameras.bin"))
    colmap_images = read_images_binary(zf.read(f"{scene_path}/sparse/0/images.bin"))
    scale = 1.0 / IMAGE_SUBDIR_SCALE[image_subdir]

    images, cameras_list, poses_list = [], [], []
    for colmap_img in sorted(colmap_images.values(), key=lambda x: x.name):
        img_bytes = _read_image_bytes(zf, scene_path, image_subdir, colmap_img.name)
        if img_bytes is None:
            logger.warning(f"Image not found: {scene_name}/{colmap_img.name}, skipping")
            continue

        cam = colmap_cameras[colmap_img.camera_id]
        fx, fy, cx, cy = (v * scale for v in get_intrinsics(cam))

        img = decode_image(img_bytes)
        dist_coeffs = get_distortion_coeffs(cam)
        if dist_coeffs is not None:
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            img, new_K = undistort_image(img, K, dist_coeffs)
            fx, fy, cx, cy = new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]

        orig_h, orig_w = img.shape[:2]
        transform = np.eye(3)
        img, t = crop_to_principal_point(img, cx, cy)
        transform = t @ transform
        img, t = resize_to_cover(img, orig_h, orig_w)
        transform = t @ transform

        h, w = img.shape[:2]
        images.append(encode_image(img))
        cameras_list.append(update_camera_intrinsics(fx, fy, cx, cy, transform, w, h))
        poses_list.append(_pose_12_from_colmap(colmap_img.qvec, colmap_img.tvec))

    if not images:
        raise ValueError(f"No valid frames in {scene_name}")

    return {
        "key": scene_name,
        "cameras": np.stack(cameras_list),
        "poses": np.stack(poses_list),
        "images": images,
        "num_views": len(images),
        "has_depth": False,
        "depths": None,
        "depth_ranges": None,
    }


def find_zip_prefix(zf: zipfile.ZipFile) -> str:
    """Find the prefix (if any) before scene dirs in the zip."""
    for name in zf.namelist():
        if "bicycle/sparse/0/cameras.bin" in name:
            return name.split("bicycle/")[0]
    raise RuntimeError("Could not find 'bicycle/sparse/0/cameras.bin' in zip")


def main():
    parser = argparse.ArgumentParser(description="Convert MipNeRF 360 zip to tar shards")
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=Path("data/mipnerf360/raw/360_v2.zip"),
        help="Input zip (default: data/mipnerf360/raw/360_v2.zip)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mipnerf360"),
        help="Output base directory (default: data/mipnerf360)",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images_4",
        choices=list(IMAGE_SUBDIR_SCALE),
        help="Image resolution subdirectory (default: images_4)",
    )
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zf = zipfile.ZipFile(args.input_zip, "r")
    try:
        zip_prefix = find_zip_prefix(zf)

        first_scene = args.scenes[0]
        for name in zf.namelist():
            if f"{first_scene}/{args.image_subdir}/" in name and name.lower().endswith(
                (".jpg", ".png")
            ):
                sample = decode_image(zf.read(name))
                break
        else:
            raise RuntimeError(f"No images found for {first_scene}/{args.image_subdir}")

        # Output dims are rounded to even (resize_to_cover uses _round_even).
        h, w = sample.shape[:2]
        h, w = 2 * round(h / 2), 2 * round(w / 2)
        output_dir = args.output_dir / f"{h}x{w}" / "test-scenes"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output: {output_dir} ({w}x{h})")

        index = {}

        pbar = tqdm(args.scenes, desc="Converting", ncols=120)
        for scene_name in pbar:
            pbar.set_postfix_str(scene_name)
            tar_path = output_dir / f"{scene_name}.tar"
            if tar_path.exists() and not args.overwrite:
                tqdm.write(f"  skip  {scene_name} (exists)")
                index[scene_name] = tar_path.name
                continue

            scene = convert_scene(zf, zip_prefix, scene_name, args.image_subdir)
            with tarfile.open(tar_path, "w") as tar:
                write_scene_to_tar(tar, scene)
            index[scene_name] = tar_path.name
            tqdm.write(f"  done  {scene_name}: {scene['num_views']} views -> {tar_path.name}")
        pbar.close()

        with open(output_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Done: {len(index)} scenes in {output_dir}")
    finally:
        zf.close()


if __name__ == "__main__":
    from splatfactory import logger  # noqa: F811

    main()
