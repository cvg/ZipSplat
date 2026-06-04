"""Pinhole camera intrinsics, packed as [w, h, fx, fy, cx, cy].

Author: Alexander Veicht
"""

from typing import Optional, Tuple, Union

import torch
from zipsplat.utils import TensorWrapper


def _focal_from_fov(fov: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return size / (2 * torch.tan(fov / 2))


def _fov_from_focal(focal: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    return 2 * torch.atan(size / (2 * focal))


class Camera(TensorWrapper):
    """Pinhole camera. data_ packs [w, h, fx, fy, cx, cy]."""

    def __init__(self, data_: torch.Tensor):
        assert data_.shape[-1] == 6
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def from_K(cls, K: torch.Tensor, w: int, h: int) -> "Camera":
        """Build from a 3x3 intrinsics matrix and image size (w, h)."""
        K = torch.as_tensor(K, dtype=torch.float32)
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        w_t = torch.full_like(fx, float(w))
        h_t = torch.full_like(fy, float(h))
        return cls(torch.stack([w_t, h_t, fx, fy, cx, cy], -1))

    @classmethod
    def from_focal(
        cls,
        fx: torch.Tensor,
        fy: Optional[torch.Tensor] = None,
        *,
        w: int,
        h: int,
        cx: Optional[torch.Tensor] = None,
        cy: Optional[torch.Tensor] = None,
    ) -> "Camera":
        """Build from focal length(s) and image size.

        fy defaults to fx (square pixels); principal point defaults to image center.
        """
        fx = torch.as_tensor(fx, dtype=torch.float32)
        fy = fx if fy is None else torch.as_tensor(fy, dtype=torch.float32)
        w_t = torch.full_like(fx, float(w))
        h_t = torch.full_like(fy, float(h))
        cx_t = torch.full_like(fx, 0.5 * w) if cx is None else torch.as_tensor(cx, dtype=fx.dtype)
        cy_t = torch.full_like(fy, 0.5 * h) if cy is None else torch.as_tensor(cy, dtype=fy.dtype)
        return cls(torch.stack([w_t, h_t, fx, fy, cx_t, cy_t], -1))

    @classmethod
    def from_fov(
        cls,
        fov_x: torch.Tensor,
        fov_y: Optional[torch.Tensor] = None,
        *,
        w: int,
        h: int,
    ) -> "Camera":
        """Build from FOV(s) in radians and image size.

        fov_y defaults to fov_x; principal point at image center.
        """
        fov_x = torch.as_tensor(fov_x, dtype=torch.float32)
        fov_y = fov_x if fov_y is None else torch.as_tensor(fov_y, dtype=torch.float32)
        w_t = torch.full_like(fov_x, float(w))
        h_t = torch.full_like(fov_y, float(h))
        fx = _focal_from_fov(fov_x, w_t)
        fy = _focal_from_fov(fov_y, h_t)
        return cls(torch.stack([w_t, h_t, fx, fy, 0.5 * w_t, 0.5 * h_t], -1))

    @property
    def K(self) -> torch.Tensor:
        """3x3 intrinsics matrix, shape (..., 3, 3)."""
        K = torch.zeros(*self.shape, 3, 3, device=self.device, dtype=self.dtype)
        K[..., 0, 0] = self.data_[..., 2]
        K[..., 1, 1] = self.data_[..., 3]
        K[..., 0, 2] = self.data_[..., 4]
        K[..., 1, 2] = self.data_[..., 5]
        K[..., 2, 2] = 1.0
        return K

    @property
    def size(self) -> torch.Tensor:
        """Image size (w, h), shape (..., 2)."""
        return self.data_[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy), shape (..., 2)."""
        return self.data_[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal point (cx, cy), shape (..., 2)."""
        return self.data_[..., 4:6]

    @property
    def hfov(self) -> torch.Tensor:
        """Horizontal FOV in radians."""
        return _fov_from_focal(self.f[..., 0], self.size[..., 0])

    @property
    def vfov(self) -> torch.Tensor:
        """Vertical FOV in radians."""
        return _fov_from_focal(self.f[..., 1], self.size[..., 1])

    def scale(self, factor: Union[float, Tuple[float, float], torch.Tensor]) -> "Camera":
        """Scale intrinsics + image size (for image resize)."""
        if isinstance(factor, (int, float)):
            factor = (float(factor), float(factor))
        s = torch.as_tensor(factor, dtype=self.dtype, device=self.device)
        return self.__class__(torch.cat([self.size * s, self.f * s, self.c * s], -1))

    def crop(self, pad: Union[Tuple[float, float], torch.Tensor]) -> "Camera":
        """Update camera after symmetric pad/crop of (dw, dh). Negative pad = crop."""
        pad = torch.as_tensor(pad, dtype=self.dtype, device=self.device)
        size = self.size + pad
        c = self.c + pad / 2
        return self.__class__(torch.cat([size, self.f, c], -1))

    def __str__(self):
        if self.shape == ():
            w, h, fx, fy, cx, cy = self.data_.tolist()
            return (
                f"Camera(w={w:.0f}, h={h:.0f}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f})"
            )
        return f"Camera({tuple(self.shape)} {self.dtype} {self.device})"
