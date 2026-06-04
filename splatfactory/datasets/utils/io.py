"""Scene I/O for tar shard storage.

Tar shard format (one scene = group of files sharing a key prefix):
    {key}.meta.json          # {"num_views": N, "has_depth": bool}
    {key}.cameras.npy        # (N, D) float32 - Camera.data_
    {key}.poses.npy          # (N, 12) float32 - Pose.data_ (c2w)
    {key}.img{i}.{ext}       # image bytes (JPEG/PNG/WebP passthrough)
    {key}.dep{i}.webp        # log-uint8 depth, WebP lossless (optional)
    {key}.depth_ranges.json  # [[d_min, d_max], ...] per view (optional)

Provides:
- Image encode (lossless WebP) / decode (JPEG/PNG/WebP -> HWC numpy)
- Depth encode/decode (log-uint8 + WebP, <1% error)
- Tar shard read/write/iterate

Author: Alexander Veicht
"""

import io
import json
import math
import tarfile

import numpy as np
import torch
import torchvision.io as tvio
from PIL import Image

from splatfactory import get_logger

logger = get_logger(__name__)

_DEPTH_LEVELS = 254  # usable levels: uint8 [1..255], 0 = invalid sentinel
_DEPTH_EPS = 1e-6
INVALID_DEPTH = -1.0


# --- Image encoding ----------------------------------------------------------


def encode_image(img: np.ndarray) -> bytes:
    """Encode an HWC/HW uint8 image to WebP lossless bytes."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="WEBP", lossless=True)
    return buf.getvalue()


# --- Depth compression -------------------------------------------------------


def encode_depth(depth: np.ndarray) -> tuple[bytes, float, float]:
    """Compress a float depth map to WebP bytes via log-space uint8 quantization.

    1. Compute range from p0.1/p99.9 percentiles of valid pixels
    2. Pixels outside the range -> invalid (u8=0)
    3. Pixels inside -> log-quantize to u8 [1..255]
    4. Compress u8 map as WebP lossless

    Returns (webp_bytes, d_min, d_max).
    """
    mask = depth >= 0
    u8 = np.zeros(depth.shape, dtype=np.uint8)

    if not mask.any():
        return encode_image(u8), 0.0, 0.0

    valid = depth[mask].astype(np.float32)
    d_min = float(np.percentile(valid, 0.1))
    d_max = float(np.percentile(valid, 99.9))

    if d_max - d_min < _DEPTH_EPS:
        u8[mask] = 128
        return encode_image(u8), d_min, d_max

    in_range = mask & (depth >= d_min) & (depth <= d_max)
    log_min = math.log(d_min)
    log_range = math.log(d_max) - log_min
    norm = (np.log(depth[in_range].astype(np.float32)) - log_min) / log_range
    u8[in_range] = np.clip(np.round(norm * _DEPTH_LEVELS).astype(int) + 1, 1, 255).astype(np.uint8)

    return encode_image(u8), d_min, d_max


def decode_depth(data: bytes, d_min: float, d_max: float) -> np.ndarray:
    """Decompress WebP bytes back to a float depth map.

    Inverts the log quantization from encode_depth.
    u8=0 -> INVALID_DEPTH. All other values decoded via exp().
    """
    try:
        tensor_bytes = torch.as_tensor(np.frombuffer(data, dtype=np.uint8).copy())
        u8 = tvio.decode_image(tensor_bytes, mode=tvio.ImageReadMode.UNCHANGED)[0]
    except RuntimeError:
        # Fallback to PIL if torchvision lacks libwebp support
        u8_np = np.array(Image.open(io.BytesIO(data)))
        if u8_np.ndim == 3:
            u8_np = u8_np[..., 0]
        u8 = torch.from_numpy(u8_np)

    mask = u8 > 0
    if mask.any() and d_max - d_min >= _DEPTH_EPS:
        log_min = math.log(d_min)
        log_R = math.log(d_max / d_min)
        depth = ((u8.float() - 1) / _DEPTH_LEVELS * log_R + log_min).exp()
        depth = torch.where(mask, depth, INVALID_DEPTH)
    elif mask.any():
        depth = torch.where(mask, d_min, INVALID_DEPTH)
    else:
        depth = torch.full(u8.shape, INVALID_DEPTH)

    return depth.numpy()


# --- Image decoding -----------------------------------------------------------


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes (JPEG/PNG/WebP) to HWC uint8 numpy array."""
    try:
        tensor_bytes = torch.as_tensor(np.frombuffer(image_bytes, dtype=np.uint8).copy())
        decoded = tvio.decode_image(tensor_bytes, mode=tvio.ImageReadMode.RGB)
        return decoded.permute(1, 2, 0).numpy()
    except RuntimeError:
        # Fallback to PIL if torchvision lacks libwebp support
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


# --- Tar helpers --------------------------------------------------------------


def _tar_add(tar: tarfile.TarFile, name: str, data: bytes):
    """Add a bytes blob to a tar file."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to .npy format bytes."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _detect_ext(data: bytes) -> str:
    """Detect image format from magic bytes."""
    if data[:2] == b"\xff\xd8":
        return "jpg"
    if data[:4] == b"\x89PNG":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "bin"


# --- Scene encode / decode ----------------------------------------------------


def write_scene_to_tar(tar: tarfile.TarFile, scene: dict):
    """Write a single scene's files into an open tar archive.

    Args:
        tar: Open TarFile in write mode.
        scene: dict with:
            "key": str - scene identifier (used as filename prefix)
            "cameras": (N, D) float32 ndarray - Camera.data_
            "poses": (N, 12) float32 ndarray - Pose.data_ (c2w)
            "images": list[bytes] - image bytes per view (passthrough)
            "num_views": int
            "has_depth": bool
            "depths": list[bytes] | None - WebP bytes from encode_depth
            "depth_ranges": list[tuple[float, float]] | None
    """
    key = scene["key"]
    n = scene["num_views"]

    _tar_add(
        tar,
        f"{key}.meta.json",
        json.dumps(
            {
                "num_views": n,
                "has_depth": scene["has_depth"],
            }
        ).encode(),
    )
    _tar_add(tar, f"{key}.cameras.npy", _npy_bytes(scene["cameras"].astype(np.float32)))
    _tar_add(tar, f"{key}.poses.npy", _npy_bytes(scene["poses"].astype(np.float32)))

    for i in range(n):
        ext = _detect_ext(scene["images"][i])
        _tar_add(tar, f"{key}.img{i}.{ext}", scene["images"][i])

    if scene["has_depth"]:
        for i in range(n):
            _tar_add(tar, f"{key}.dep{i}.webp", scene["depths"][i])
        _tar_add(tar, f"{key}.depth_ranges.json", json.dumps(scene["depth_ranges"]).encode())


def _parse_tar_entry(name: str, data: bytes, scene: dict):
    """Parse a single tar entry into the scene dict."""
    if name.endswith(".meta.json"):
        scene["meta"] = json.loads(data)
    elif name.endswith(".cameras.npy"):
        scene["cameras"] = np.load(io.BytesIO(data))
    elif name.endswith(".poses.npy"):
        scene["poses"] = np.load(io.BytesIO(data))
    elif name.endswith(".depth_ranges.json"):
        scene["depth_ranges"] = [tuple(r) for r in json.loads(data)]
    elif ".img" in name:
        idx = int(name.split(".img")[1].split(".")[0])
        scene["images"][idx] = data
    elif ".dep" in name:
        idx = int(name.split(".dep")[1].split(".")[0])
        scene["depths"][idx] = data


def read_scenes_from_tar(path: str) -> list[dict]:
    """Read all scenes from a tar shard, grouped by scene prefix."""
    groups = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            key = member.name.split(".")[0]
            if key not in groups:
                groups[key] = {"images": {}, "depths": {}}
            _parse_tar_entry(member.name, tar.extractfile(member).read(), groups[key])
    return [{"key": k, **v} for k, v in groups.items()]


def iter_scenes_from_tar(path: str):
    """Iterate scenes from a tar shard one at a time.

    Yields raw scene dicts as each scene completes (all its files have been read).
    Assumes scenes are stored contiguously (guaranteed by write_scene_to_tar).
    """
    current_key = None
    current = None

    with tarfile.open(path, "r") as tar:
        for member in tar:
            key = member.name.split(".")[0]
            if key != current_key:
                if current is not None:
                    yield {"key": current_key, **current}
                current_key = key
                current = {"images": {}, "depths": {}}
            _parse_tar_entry(member.name, tar.extractfile(member).read(), current)

    if current is not None:
        yield {"key": current_key, **current}
