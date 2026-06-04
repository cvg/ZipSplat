"""Pinhole camera as a packed tensor (TensorWrapper): intrinsics + image size,
with projection, unprojection, and image-resize/crop transforms.

Author: Alexander Veicht
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from splatfactory.utils.conversions import focal2fov, fov2focal, from_homogeneous, to_homogeneous
from splatfactory.utils.tensor import TensorWrapper, autocast


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data_: torch.Tensor):
        assert data_.shape[-1] in {6, 8, 10}
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def from_colmap(cls, camera: Union[Dict, NamedTuple]) -> "Camera":
        """Camera from a COLMAP Camera tuple or dictionary.
        We use the corner-convetion from COLMAP (center of top left pixel is (0.5, 0.5))
        """
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE", "RADIAL"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data = np.r_[camera["width"], camera["height"], fx, fy, cx, cy, params]
        return cls(data)

    @classmethod
    def data_from_K(cls, K: torch.Tensor) -> torch.Tensor:
        cx, cy = K[..., 0, 2], K[..., 1, 2]
        fx, fy = K[..., 0, 0], K[..., 1, 1]
        return torch.stack([2 * cx, 2 * cy, fx, fy, cx, cy], -1)

    @classmethod
    @autocast
    def from_calibration_matrix(cls, K: torch.Tensor) -> "Camera":
        return cls(Camera.data_from_K(K))

    @classmethod
    def from_fov(cls, fov_x: torch.Tensor, fov_y: torch.Tensor, w: int, h: int) -> "Camera":
        """Create camera from vertical field of view and image size.

        Args:
            fov_w: vertical field of view in radians, shape (...,).
            w: image width, shape (...,).
            h: image height, shape (...,).
        """
        w = fov_x.new_full(fov_x.shape, w) if isinstance(w, int) else w
        h = fov_y.new_full(fov_y.shape, h) if isinstance(h, int) else h

        fx = fov2focal(fov_x, w)
        fy = fov2focal(fov_y, h)
        cx, cy = 0.5 * w, 0.5 * h
        data_ = torch.cat([torch.stack([w, h], -1), torch.stack([fx, fy, cx, cy], -1)], -1)
        return cls(data_)

    @autocast
    def compose_image_transform(self, new_t_img: torch.Tensor, inplace: bool = False) -> "Camera":
        """Update the camera parameters after an image space transformation.

        Args:
            new_t_img: 3x3 image transformation matrix.
            inplace: whether to update the current camera or return a new one.
        """
        K = self.K
        new_K = new_t_img.to(K) @ K
        fx, fy = new_K[..., 0, 0], new_K[..., 1, 1]
        cx, cy = new_K[..., 0, 2], new_K[..., 1, 2]

        t = new_t_img.to(self.data_)
        w = self.w * t[..., 0, 0] + 2 * t[..., 0, 2]
        h = self.h * t[..., 1, 1] + 2 * t[..., 1, 2]
        newdata_ = torch.stack([w, h, fx, fy, cx, cy], -1)

        alldata_ = torch.cat([newdata_, self.dist], -1)
        if inplace:
            self.data_ = alldata_
            return self
        return Camera(alldata_)

    @property
    def K(self):
        """Get the calibration matrix, with shape (..., 3, 3)."""
        K = torch.zeros(
            *self.data_.shape[:-1], 3, 3, device=self.data_.device, dtype=self.data_.dtype
        )
        K[..., 0, 2] = self.data_[..., 4]
        K[..., 1, 2] = self.data_[..., 5]
        K[..., 0, 0] = self.data_[..., 2]
        K[..., 1, 1] = self.data_[..., 3]
        K[..., 2, 2] = 1.0
        return K

    def K_normalized(self):
        """Get the normalized calibration matrix, with f and p in relative pixel coordinates.

        Returns:
            torch.Tensor: Normalized calibration matrix with shape (..., 3, 3).
        """
        K = self.K
        K[..., 0, :] /= self.size[..., 0:1]
        K[..., 1, :] /= self.size[..., 1:2]
        return K

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self.data_[..., :2]

    @property
    def w(self) -> torch.Tensor:
        """Return image width without batch dimensions."""
        w = self.size[(0,) * (self.size.ndim - 1) + (0,)]
        return w

    @property
    def h(self) -> torch.Tensor:
        """Return image height without batch dimensions."""
        h = self.size[(0,) * (self.size.ndim - 1) + (1,)]
        return h

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self.data_[..., 2:4]

    @property
    def vfov(self) -> torch.Tensor:
        """Vertical field of view in radians."""
        return focal2fov(self.f[..., 1], self.size[..., 1])

    @property
    def hfov(self) -> torch.Tensor:
        """Horizontal field of view in radians."""
        return focal2fov(self.f[..., 0], self.size[..., 0])

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self.data_[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self.data_[..., 6:]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the self parameters after resizing an image."""
        scales = (scales, scales) if isinstance(scales, (int, float)) else scales
        s = scales if isinstance(scales, torch.Tensor) else self.new_tensor(scales)

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([self.size * s, self.f * s, self.c * s, dist], -1))

    def crop(self, pad: Tuple[float]):
        """Update the self parameters after cropping an image."""
        pad = pad if isinstance(pad, torch.Tensor) else self.new_tensor(pad)
        size = self.size + pad.to(self.size)
        c = self.c + pad.to(self.c) / 2

        dist = self.dist if hasattr(self, "dist") else self.new_zeros(self.f.shape)
        return self.__class__(torch.cat([size, self.f, c, dist], -1))

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the camera plane and check for visibility."""
        return from_homogeneous(p3d, self.eps), p3d[..., -1] > self.eps

    def J_project(self, p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        z = z.clamp(min=self.eps)
        J = torch.stack([1 / z, zero, -x / z**2, zero, 1 / z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def distort(self, p2d: torch.Tensor, return_scale: bool = False) -> Tuple[torch.Tensor]:
        """Distort normalized 2D coordinates and check for validity of the distortion model."""
        if self.dist.shape[-1] == 0:
            if return_scale:
                return p2d, None
            else:
                mask = torch.ones(p2d.shape[:-1], device=p2d.device, dtype=torch.bool)
                return p2d, mask

        assert self.dist.shape[-1] in {2, 4}, f"Distortion model not supported {self.dist.shape}"
        self.k1, self.k2 = self.dist[..., 0], self.dist[..., 1]

        r2 = torch.sum(p2d**2, -1, keepdim=True)
        r4 = r2**2
        radial = 1 + self.k1[..., None, None] * r2 + self.k2[..., None, None] * r4

        if return_scale:
            return radial, None

        return p2d * radial, self.check_valid(p2d)

    @autocast
    def undistort(self, p2d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates and check for validity of the distortion model."""
        if self.dist.shape[-1] == 0:
            mask = torch.ones(p2d.shape[:-1], device=p2d.device, dtype=torch.bool)
            return p2d, mask

        assert self.dist.shape[-1] in {2, 4}, f"Distortion model not supported {self.dist.shape}"
        self.k1, self.k2 = self.dist[..., 0], self.dist[..., 1]

        r2 = torch.sum(p2d**2, -1, keepdim=True)
        k1, k2 = self.k1[..., None, None], self.k2[..., None, None]
        b1, b2 = -k1, 3 * k1**2 - k2
        radial = 1 + b1 * r2 + b2 * r2**2
        return p2d * radial, self.check_valid(p2d)

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    def pixel_coordinates(self) -> torch.Tensor:
        """Pixel coordinates in camera frame [0, w-1] x [0, h-1].

        Returns:
            torch.Tensor: Pixel coordinates as a tensor of shape (..., h*w, 2)
                        where ... are the leading batch dimensions of the cameras.
        """
        w, h = self.w.round().long(), self.h.round().long()

        # Create a pixel grid of shape (h*w, 2)
        x = torch.arange(w, dtype=self.dtype, device=self.device)
        y = torch.arange(h, dtype=self.dtype, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)  # (h*w, 2)

        # Expand to match leading dims of the cameras
        # If cameras have shape (..., 2), then xy -> (..., h*w, 2)
        batch_shape = self.size.shape[:-1]  # leading dims
        xy = xy.reshape((1,) * len(batch_shape) + xy.shape)  # prepend dims
        xy = xy.expand(*batch_shape, -1, -1)  # tile across batch

        return xy

    def normalized_image_coordinates(self) -> torch.Tensor:
        """Normalized image coordinates in camera frame [-1, 1] x [-1, 1].

        Returns:
            torch.Tensor: Normalized image coordinates as a tensor of shape (B, h * w, 3).
        """
        xy = self.pixel_coordinates()
        uv1, _ = self.image2world(xy)
        return uv1

    @autocast
    def unproject_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Unproject depth map into 3D points in camera frame.

        Args:
            depth (torch.Tensor): Depth map as a tensor of shape (..., H, W).

        Returns:
            torch.Tensor: 3D points as a tensor of shape (..., H, W, 3).
        """
        w, h = self.w.round().long(), self.h.round().long()
        assert depth.shape[-2:] == (h, w), f"Depth shape {depth.shape[-2:]} != {(h, w)}."

        p3d = self.normalized_image_coordinates()  # (..., H*W, 2)
        p3d = rearrange(p3d, "... (h w) d -> ... h w d", h=depth.shape[-2], w=depth.shape[-1])
        return p3d * depth.unsqueeze(-1)

    @autocast
    def pixel_bearing_many(self, p3d: torch.Tensor) -> torch.Tensor:
        """Get the bearing vectors of pixel coordinates.

        Args:
            p2d (torch.Tensor): Pixel coordinates as a tensor of shape (..., 3).

        Returns:
            torch.Tensor: Bearing vectors as a tensor of shape (..., 3).
        """
        return F.normalize(p3d, dim=-1)

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.distort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    @autocast
    def image2world(self, p2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform point in the image plane to 3D world coordinates."""
        p2d = self.normalize(p2d)
        p2d, valid = self.undistort(p2d)
        # ones = p2d.new_ones(p2d.shape[:-1] + (1,))
        # p3d = torch.cat([p2d, ones], -1)
        p3d = to_homogeneous(p2d)
        return p3d, valid

    def to_cameradict(self, camera_model: Optional[str] = None) -> List[Dict]:
        data = self.data_.clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.dim() == 2
        b, d = data.shape
        if camera_model is None:
            camera_model = {6: "PINHOLE", 8: "RADIAL", 10: "OPENCV"}[d]
        cameras = []
        for i in range(b):
            if camera_model.startswith("SIMPLE_"):
                params = [x.item() for x in data[i, 3 : min(d, 7)]]
            else:
                params = [x.item() for x in data[i, 2:]]
            cameras.append(
                {
                    "model": camera_model,
                    "width": int(data[i, 0].item()),
                    "height": int(data[i, 1].item()),
                    "params": params,
                }
            )
        return cameras if self.data_.dim() == 2 else cameras[0]

    def __str__(self):
        if self.shape == ():
            param_names = ["w", "h", "fx", "fy", "cx", "cy"] + (
                ["k1", "k2"] if self.dist.shape[-1] in {2, 4} else []
            )
            param_str = ", ".join(
                f"{name}={value:.2f}" for name, value in zip(param_names, self.data_.tolist())
            )
            return f"Camera({param_str} - {self.dtype} - {self.device})"

        return f"Camera({self.shape} - {self.dtype} - {self.device})"
