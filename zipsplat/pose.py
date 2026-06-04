"""SE(3) cam-to-world pose, packed as flatten(R, 3x3) + t = [r00..r22, tx, ty, tz] (size 12).

OpenCV convention: x-right, y-down, z-forward. `t` is the camera origin in world coords.

Author: Alexander Veicht
"""

from typing import Tuple, Union

import torch
from zipsplat.utils import TensorWrapper


class Pose(TensorWrapper):
    """SE(3) cam-to-world pose."""

    def __init__(self, data_: torch.Tensor):
        assert data_.shape[-1] == 12
        self.data_ = data_
        super().__post_init__()

    @classmethod
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor) -> "Pose":
        """Build from cam-to-world rotation (..., 3, 3) and translation (..., 3)."""
        R = torch.as_tensor(R, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        return cls(torch.cat([R.flatten(start_dim=-2), t], -1))

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor) -> "Pose":
        """Build from a 4x4 cam-to-world matrix."""
        T = torch.as_tensor(T, dtype=torch.float32)
        assert T.shape[-2:] == (4, 4)
        return cls.from_Rt(T[..., :3, :3], T[..., :3, 3])

    @classmethod
    def identity(cls, batch_shape: Tuple[int, ...] = (), device=None, dtype=None) -> "Pose":
        """Identity pose, optionally batched."""
        R = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)
        t = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
        return cls.from_Rt(R, t)

    @property
    def R(self) -> torch.Tensor:
        """Rotation matrix, shape (..., 3, 3)."""
        return self.data_[..., :9].reshape(*self.shape, 3, 3)

    @property
    def t(self) -> torch.Tensor:
        """Translation, shape (..., 3)."""
        return self.data_[..., 9:]

    @property
    def quat(self) -> torch.Tensor:
        """Rotation as wxyz quaternion, shape (..., 4)."""
        return Pose.matrix_to_quaternion(self.R)

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        """3x3 rotation matrix → wxyz quaternion. Shape (..., 3, 3) → (..., 4)."""
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )
        q2 = torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
        q_abs = torch.where(q2 > 0, torch.sqrt(q2), torch.zeros_like(q2))
        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )
        flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
        indices = q_abs.argmax(dim=-1, keepdim=True)
        gather_indices = indices.unsqueeze(-1).expand(list(batch_dim) + [1, 4])
        out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
        return torch.where(out[..., 0:1] < 0, -out, out)

    @property
    def Rt(self) -> torch.Tensor:
        """4x4 cam-to-world matrix."""
        T = torch.zeros(*self.shape, 4, 4, dtype=self.dtype, device=self.device)
        T[..., :3, :3] = self.R
        T[..., :3, 3] = self.t
        T[..., 3, 3] = 1
        return T

    @property
    def Rt_inv(self) -> torch.Tensor:
        """4x4 world-to-cam matrix (gsplat viewmats convention)."""
        return self.inv().Rt

    def inv(self) -> "Pose":
        """Invert the pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Pose") -> "Pose":
        """Chain poses: T_B2C.compose(T_A2B) -> T_A2C."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform 3D points (..., N, 3)."""
        p3d = torch.as_tensor(p3d, dtype=self.dtype, device=self.device)
        assert p3d.shape[-1] == 3
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __matmul__(self, other: Union["Pose", torch.Tensor]) -> Union["Pose", torch.Tensor]:
        """`pose @ other_pose` -> compose; `pose @ p3d` -> transform points."""
        if isinstance(other, self.__class__):
            return self.compose(other)
        return self.transform(other)

    @classmethod
    def orbit(
        cls,
        center: torch.Tensor,
        radius: float,
        azimuths: torch.Tensor,
        elevations: torch.Tensor,
    ) -> "Pose":
        """Camera poses on a sphere looking at `center`. OpenCV convention.

        Args:
            center: scene center, shape (3,).
            radius: distance from center.
            azimuths: azimuth angles in degrees, shape (N,).
            elevations: elevation angles in degrees, shape (N,).
        """
        center = torch.as_tensor(center, dtype=torch.float32)
        device, dtype = center.device, center.dtype
        az = torch.deg2rad(torch.as_tensor(azimuths, dtype=dtype, device=device))
        el = torch.deg2rad(torch.as_tensor(elevations, dtype=dtype, device=device))
        up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype)

        cos_el, sin_el = torch.cos(el), torch.sin(el)
        cos_az, sin_az = torch.cos(az), torch.sin(az)
        positions = (
            torch.stack(
                [radius * cos_el * sin_az, -radius * sin_el, -radius * cos_el * cos_az],
                dim=-1,
            )
            + center
        )

        forward = center - positions
        forward = forward / forward.norm(dim=-1, keepdim=True)
        right = torch.linalg.cross(forward, up.expand_as(forward))
        right_norm = right.norm(dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(right)
        right = torch.where(right_norm < 1e-6, fallback, right / right_norm)
        down = torch.linalg.cross(forward, right)
        down = down / down.norm(dim=-1, keepdim=True)
        Rs = torch.stack([right, down, forward], dim=-1)
        return cls.from_Rt(Rs, positions)

    def __str__(self):
        return f"Pose({tuple(self.shape)} {self.dtype} {self.device})"
