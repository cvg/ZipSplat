"""Rigid 3D transform (SE(3)) as a packed tensor (TensorWrapper): rotation + translation,
with composition, inversion, and point/translation transforms.

Author: Alexander Veicht
"""

import math
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch

from splatfactory.geometry.utils import so3exp_map
from splatfactory.utils.conversions import rad2deg, rotmat2rad, skew_symmetric
from splatfactory.utils.tensor import TensorWrapper, autocast


class Pose(TensorWrapper):
    """SE(3) Pose storing cam2world conversion."""

    def __init__(self, data_: torch.Tensor):
        assert data_.shape[-1] == 12
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def identity(cls, device=None, dtype=None):
        R = torch.eye(3, device=device, dtype=dtype)
        t = torch.zeros(3, device=device, dtype=dtype)
        return cls.from_Rt(R, t)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor) -> "Pose":
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor) -> "Pose":
        """Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        """
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor) -> "Pose":
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple) -> "Pose":
        """Pose from a COLMAP Image."""
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self.data_[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self.data_[..., -3:]

    @property
    def Rt(self) -> torch.Tensor:
        mat = torch.zeros(
            (*self.data_.shape[:-1], 4, 4), dtype=self.data_.dtype, device=self.data_.device
        )
        mat[..., :3, :3] = self.R
        mat[..., :3, 3] = self.t
        mat[..., 3, 3] = 1
        return mat

    @property
    def Rt_inv(self) -> torch.Tensor:
        return self.inv().Rt

    def inv(self) -> "Pose":
        """Invert an SE(3) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Pose") -> "Pose":
        """Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B."""
        return self.transform(p3D)

    def __matmul__(self, other: Union["Pose", torch.Tensor]) -> Union["Pose", torch.Tensor]:
        """Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.
        or chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C."""
        if isinstance(other, self.__class__):
            return self.compose(other)
        else:
            return self.transform(other)

    @autocast
    def J_transform(self, p3d_out: torch.Tensor):
        # [[1,0,0,0,-pz,py],
        #  [0,1,0,pz,0,-px],
        #  [0,0,1,-py,px,0]]
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J  # N x 3 x 6

    def magnitude(self) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation angle in degrees.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def scale_translation(self, scale: float) -> "Pose":
        """Scale the translation component of the pose by a given factor."""
        scale = scale if not hasattr(scale, "shape") else scale[..., None]  # for broadcasting
        return Pose.from_Rt(self.R, self.t * scale)

    def __str__(self):
        if self.shape == ():
            param_str = f"RPY={rad2deg(rotmat2rad(self.R))}, t={self.t}"
            return f"Pose({param_str} - {self.dtype} - {self.device})"

        return f"Pose({self.shape} - {self.dtype} - {self.device})"

    @classmethod
    def orbit(
        cls,
        center: torch.Tensor,
        radius: float,
        azimuths: list,
        elevations: list,
    ) -> "Pose":
        """Create camera poses on sphere looking at center (OpenCV convention).

        Args:
            center: Scene center [3]
            radius: Distance from center
            azimuths: Azimuth angles in degrees
            elevations: Elevation angles in degrees

        Returns:
            Batched Pose with N poses
        """
        device, dtype = center.device, center.dtype
        up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)

        # Convert to radians
        az = torch.tensor([math.radians(a) for a in azimuths], device=device, dtype=dtype)
        el = torch.tensor([math.radians(e) for e in elevations], device=device, dtype=dtype)

        # Camera positions on sphere (az=0 at -Z, looking toward +Z)
        cos_el, sin_el = torch.cos(el), torch.sin(el)
        cos_az, sin_az = torch.cos(az), torch.sin(az)
        positions = (
            torch.stack(
                [
                    radius * cos_el * sin_az,
                    -radius * sin_el,
                    -radius * cos_el * cos_az,
                ],
                dim=-1,
            )
            + center
        )

        # Look-at rotation (cam2world) - OpenCV: x-right, y-down, z-forward
        forward = center - positions
        forward = forward / forward.norm(dim=-1, keepdim=True)

        right = torch.linalg.cross(forward, up.expand_as(forward))
        right_norm = right.norm(dim=-1, keepdim=True)
        # Handle top-down views (forward parallel to up)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(right)
        right = torch.where(right_norm < 1e-6, fallback, right / right_norm)

        down = torch.linalg.cross(forward, right)
        down = down / down.norm(dim=-1, keepdim=True)

        # R columns: right, down, forward
        Rs = torch.stack([right, down, forward], dim=-1)
        return cls.from_Rt(Rs, positions)
