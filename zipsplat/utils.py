"""TensorWrapper base for batched per-tensor packed types (Camera, Pose, Gaussians),
plus shared utilities (k-means clustering, image loading).

Author: Alexander Veicht
"""

from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image


class TensorWrapper:
    """Single-tensor wrapper. Subclasses pack their parameters into a ``[..., D]`` tensor and
    expose named views over the trailing dimension; all leading dimensions are batch dims.

    The leading ``data_.shape[:-1]`` is the batch shape (``self.shape``). Indexing, dtype/device
    moves, and ``torch.cat`` / ``torch.stack`` operate on the batch dims and rewrap the result.
    """

    data_: torch.Tensor

    def __init__(self, data_: torch.Tensor):
        self.data_ = data_
        self.__post_init__()

    def __post_init__(self) -> None:
        self.batch_size = self.data_.shape[:-1]

    # ------------------------- shape / dtype / device -------------------------

    @property
    def shape(self) -> torch.Size:
        """Batch shape (everything but the trailing packed dimension)."""
        return self.data_.shape[:-1]

    @property
    def device(self) -> torch.device:
        return self.data_.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data_.dtype

    def __len__(self) -> int:
        return self.shape[0]

    # ------------------------- indexing / device moves -------------------------

    def __getitem__(self, idx) -> "TensorWrapper":
        """Index the batch dims (int, slice, tensor, or ``None`` to add a dim)."""
        return self.__class__(self.data_[idx])

    def to(self, *args, **kwargs) -> "TensorWrapper":
        return self.__class__(self.data_.to(*args, **kwargs))

    def cpu(self) -> "TensorWrapper":
        return self.__class__(self.data_.cpu())

    def cuda(self, *args, **kwargs) -> "TensorWrapper":
        return self.__class__(self.data_.cuda(*args, **kwargs))

    def clone(self) -> "TensorWrapper":
        return self.__class__(self.data_.clone())

    def detach(self) -> "TensorWrapper":
        return self.__class__(self.data_.detach())

    # ------------------------- torch dispatch -------------------------

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Make ``torch.cat`` / ``torch.stack`` / ``torch.concat`` over a sequence of wrappers
        rewrap the underlying ``data_`` (any ``dim`` arg is a batch dim, not the packed dim)."""
        kwargs = kwargs or {}
        if (
            func in (torch.cat, torch.concat, torch.stack)
            and args
            and isinstance(args[0], (list, tuple))
            and all(isinstance(x, cls) for x in args[0])
        ):
            return cls(func([x.data_ for x in args[0]], *args[1:], **kwargs))
        return NotImplemented

    def __repr__(self) -> str:
        return str(self)


def kmeans(
    x: torch.Tensor, K: int, n_iters: int = 5, chunk_size: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard k-means with chunked distance computation (memory-efficient).

    Uses chunked cdist + scatter_add to avoid materializing full [N, K] matrices.
    Peak memory: O(chunk x K) instead of O(N x K). Centroids are uniformly initialized
    via linspace over the input tokens.

    Args:
        x: [B, N, D] input features.
        K: number of clusters.
        n_iters: number of iterations (converges in 2-3; 5 leaves headroom).
        chunk_size: tokens per chunk for the distance computation.

    Returns:
        centroids: [B, K, D].
        nearest_idx: [B, K] index of the nearest original token per centroid.
        assignments: [B, N] cluster assignment per token from the last iteration.
    """
    B, N, D = x.shape
    idx = torch.linspace(0, N - 1, K, device=x.device).round().long().unsqueeze(0).expand(B, -1)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, K)
    centroids = x[batch_idx, idx]
    ones = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

    for _ in range(n_iters):
        assignments = torch.empty(B, N, dtype=torch.long, device=x.device)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            dists = torch.cdist(x[:, s:e], centroids)
            assignments[:, s:e] = dists.argmin(dim=-1)
            del dists

        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(1, assignments.unsqueeze(-1).expand(-1, -1, D), x)
        counts = torch.zeros(B, K, 1, device=x.device, dtype=x.dtype)
        counts.scatter_add_(1, assignments.unsqueeze(-1), ones)
        centroids = new_centroids / counts.clamp(min=1)

    nearest_idx = torch.empty(B, K, dtype=torch.long, device=x.device)
    for s in range(0, K, chunk_size):
        e = min(s + chunk_size, K)
        dists = torch.cdist(centroids[:, s:e], x)
        nearest_idx[:, s:e] = dists.argmin(dim=-1)
        del dists

    return centroids, nearest_idx, assignments


def load_image(path: Path) -> torch.Tensor:
    """Load an image to a (3, H, W) float tensor in [0, 1]."""
    return to_tensor(Image.open(path).convert("RGB"))


def load_video(
    path: Path, num_frames: Optional[int] = 8, stride: Optional[int] = None
) -> List[torch.Tensor]:
    """Load video frames as a list of (3, H, W) float tensors in [0, 1].

    By default evenly samples `num_frames` frames across the whole clip (widest baseline). Pass
    `stride` to take every Nth frame instead. Returns all frames when the clip is shorter than
    requested. Note: frame seeking is approximate for some codecs.
    """
    reader = imageio.get_reader(str(path))
    try:
        total = reader.count_frames()
        if stride is not None:
            indices = list(range(0, total, stride))
        else:
            n = min(num_frames, total)
            indices = torch.linspace(0, total - 1, n).round().long().tolist()
        frames = [to_tensor(reader.get_data(i)) for i in indices]
    except Exception:
        # Unreliable frame count / seeking: stream all frames, then subsample.
        frames = [to_tensor(f) for f in reader]
        if stride is not None:
            frames = frames[::stride]
        elif len(frames) > num_frames:
            idx = torch.linspace(0, len(frames) - 1, num_frames).round().long().tolist()
            frames = [frames[i] for i in idx]
    finally:
        reader.close()
    return frames


def to_tensor(image) -> torch.Tensor:
    """Convert HWC image (uint8 or float in [0, 1]) to (3, H, W) float in [0, 1]."""
    arr = np.asarray(image)
    arr = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)
    return chw_from_hwc(torch.from_numpy(arr)).contiguous()


def resize_image(image: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize image (..., 3, H, W) to exactly (h, w) using PIL LANCZOS."""
    *batch, C, H, W = image.shape
    if (H, W) == (h, w):
        return image
    flat = image.reshape(-1, C, H, W)
    out = []
    for i in range(flat.shape[0]):
        arr = (hwc_from_chw(flat[i]).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).resize((w, h), Image.LANCZOS)
        out.append(chw_from_hwc(torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)))
    return torch.stack(out).reshape(*batch, C, h, w).to(device=image.device, dtype=image.dtype)


def crop_to_ar(image: torch.Tensor, ar: float) -> torch.Tensor:
    """Center-crop image (..., 3, H, W) to aspect ratio `ar` (= w/h).

    ar > 1 = landscape, ar < 1 = portrait, ar = 1 = square.
    Whichever dimension exceeds the target is cropped; never resizes.
    """
    H, W = image.shape[-2:]
    target_w = round(H * ar)
    if target_w <= W:
        new_h, new_w = H, target_w
        top, left = 0, (W - target_w) // 2
    else:
        target_h = round(W / ar)
        new_h, new_w = target_h, W
        top, left = (H - target_h) // 2, 0
    if (new_h, new_w) == (H, W):
        return image
    return image[..., top : top + new_h, left : left + new_w]


# Native model resolution (square), divisible by the DA3 patch size (14).
IMAGE_SIZE = 252


def to_square(image: torch.Tensor, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Center-crop image (..., 3, H, W) to square and resize to (size, size)."""
    return resize_image(crop_to_ar(image, 1.0), size, size)


def chw_from_hwc(x):
    """...HWC → ...CHW (works on numpy arrays and torch tensors)."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -1, -3)
    return x.transpose(-2, -1).transpose(-3, -2)


def hwc_from_chw(x):
    """...CHW → ...HWC (works on numpy arrays and torch tensors)."""
    if isinstance(x, np.ndarray):
        return np.moveaxis(x, -3, -1)
    return x.transpose(-3, -2).transpose(-2, -1)
