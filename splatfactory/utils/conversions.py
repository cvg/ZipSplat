"""Utility functions for conversions between different representations.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from einops import rearrange

# Re-exported from image.py (handles both numpy and torch)
from splatfactory.utils.image import chw_from_hwc, hwc_from_chw  # noqa: F401


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(quaternions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions (torch.Tensor): quaternions of shape (..., 4) in (i, j, k, r) format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: rotation matrices of shape (..., 3, 3).
    """
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
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
    positive_mask = q2 > 0
    q_abs = torch.where(positive_mask, torch.sqrt(q2), torch.zeros_like(q2))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = q_abs.argmax(dim=-1, keepdim=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = indices.unsqueeze(-1).expand(expand_dims)
    out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
    return torch.where(out[..., 0:1] < 0, -out, out)


def build_covariance(scale: torch.Tensor, rotation_xyzw: torch.Tensor) -> torch.Tensor:
    """Build a covariance matrix from scale and rotation.

    Args:
        scale (torch.Tensor): scale of shape (..., 3).
        rotation_xyzw (torch.Tensor): rotation as quaternion of shape (..., 4) in (x, y, z, w) format.

    Returns:
        torch.Tensor: covariance matrix of shape (..., 3, 3).
    """
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def get_scale_quat_from_covariance(covariance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose a covariance matrix into scale and rotation.

    Args:
        covar (torch.Tensor): covariance matrix of shape (..., 3, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - scale of shape (..., 3)
            - rotation as quaternion of shape (..., 4) in (w, x, y, z) format (gsplat convention)
    """
    # Run eigh on CPU to avoid cusolver batch size limits with large batches
    covariance = (covariance + covariance.transpose(-1, -2)) / 2
    device = covariance.device
    eigvals, eigvecs = torch.linalg.eigh(covariance.cpu())
    eigvals, eigvecs = eigvals.to(device), eigvecs.to(device)
    scale = torch.sqrt(eigvals.clamp(min=0))

    det = torch.det(eigvecs)
    eigvecs = torch.where(det[..., None, None] < 0, -eigvecs, eigvecs)
    return scale, matrix_to_quaternion(eigvecs)


def reconstruct_feature_img(
    features: torch.Tensor, indices: torch.Tensor, target_shape: Tuple[int, int]
) -> torch.Tensor:
    """Reconstruct feature image from features and indices, resize to target size.

    Args:
        features (torch.Tensor): The feature tensor of shape (..., N, D).
        indices (torch.Tensor): The index tensor of shape (..., H, W).
        target_shape (Tuple[int, int]): The target shape (H, W) to resize the feature image.

    Returns:
        torch.Tensor: The reconstructed feature image of shape (..., D, H, W).
    """
    assert features.shape[:-2] == indices.shape[:-2], "Batch dimensions must match"
    *batch_dims, _, D = features.shape
    *batch_dims, H, W = indices.shape

    indices = rearrange(indices, "... h w -> ... (h w) 1").long()
    feature_img = torch.gather(features, -2, indices.expand(*batch_dims, H * W, D))
    feature_img = rearrange(feature_img, "... (h w) d -> ... d h w", h=H, w=W)

    if feature_img.shape[-2:] != target_shape:
        feature_img = torch.nn.functional.interpolate(
            feature_img, size=target_shape, mode="bilinear", align_corners=False
        )

    return feature_img


def polygon_to_mask(img_shape: Tuple[int, int], points_list: List[Tuple[int, int]]) -> np.ndarray:
    """Create dense mask given outline as points

    Args:
        img_shape (Tuple[int, int]): The shape of the image (H, W).
        points_list (List[Tuple[int, int]]): A list of points defining the polygon.

    Returns:
        np.ndarray: The binary mask with the same spatial dimensions as the input image.
    """
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3).

    Args:
        (torch.Tensor): Vector of size (..., 3).

    Returns:
        (torch.Tensor): Skew-symmetric matrix of size (..., 3, 3).
    """
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:].clamp(min=eps))


def batched_eye_like(x: torch.Tensor, n: int):
    """Create a batch of identity matrices.
    Args:
        x: a reference torch.Tensor whose batch dimension will be copied.
        n: the size of each identity matrix.
    Returns:
        A torch.Tensor of size (B, n, n), with same dtype and device as x.
    """
    return torch.eye(n).to(x)[None].repeat(len(x), 1, 1)


def rad2rotmat(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert (batched) roll, pitch, yaw angles (in radians) to rotation matrix.

    Args:
        roll (torch.Tensor): Roll angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        yaw (torch.Tensor, optional): Yaw angle in radians. Defaults to None.

    Returns:
        torch.Tensor: Rotation matrix of shape (..., 3, 3).
    """
    if yaw is None:
        yaw = roll.new_zeros(roll.shape)

    Rx = pitch.new_zeros(pitch.shape + (3, 3))
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = torch.cos(pitch)
    Rx[..., 1, 2] = torch.sin(pitch)
    Rx[..., 2, 1] = -torch.sin(pitch)
    Rx[..., 2, 2] = torch.cos(pitch)

    Ry = yaw.new_zeros(yaw.shape + (3, 3))
    Ry[..., 0, 0] = torch.cos(yaw)
    Ry[..., 0, 2] = -torch.sin(yaw)
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = torch.sin(yaw)
    Ry[..., 2, 2] = torch.cos(yaw)

    Rz = roll.new_zeros(roll.shape + (3, 3))
    Rz[..., 0, 0] = torch.cos(roll)
    Rz[..., 0, 1] = torch.sin(roll)
    Rz[..., 1, 0] = -torch.sin(roll)
    Rz[..., 1, 1] = torch.cos(roll)
    Rz[..., 2, 2] = 1

    return Rz @ Rx @ Ry


def rotmat2rad(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract roll, pitch, yaw from rotation matrix.

    Extracts Euler angles from the rotation matrix where:
    - Pitch: rotation around X-axis
    - Yaw: rotation around Y-axis
    - Roll: rotation around Z-axis

    Assumes ZYX rotation order (Roll -> Yaw -> Pitch).

    Args:
        degrees: If True, return angles in degrees. Default is radians.

    Returns:
        roll: Rotation around Z-axis with shape (...,).
        pitch: Rotation around X-axis with shape (...,).
        yaw: Rotation around Y-axis with shape (...,).
    """
    # Clamp to avoid numerical issues with arcsin
    sin_yaw = torch.clamp(-R[..., 2, 0], -1.0, 1.0)
    yaw = torch.asin(sin_yaw)

    # Check for gimbal lock (yaw near +/-90 degrees)
    cos_yaw = torch.cos(yaw)
    gimbal_lock = torch.abs(cos_yaw) < 1e-6

    # Extract pitch and roll
    pitch = torch.where(
        gimbal_lock,
        torch.zeros_like(yaw),  # Convention: set pitch to 0 at gimbal lock
        torch.atan2(R[..., 2, 1], R[..., 2, 2]),
    )

    roll = torch.where(
        gimbal_lock,
        # At gimbal lock, use alternative formula
        torch.atan2(-R[..., 0, 1], R[..., 1, 1]),
        torch.atan2(R[..., 1, 0], R[..., 0, 0]),
    )
    return torch.stack([roll, pitch, yaw], dim=-1)


def fov2focal(fov: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute focal length from (vertical/horizontal) field of view.

    Args:
        fov (torch.Tensor): Field of view in radians.
        size (torch.Tensor): Image height / width in pixels.

    Returns:
        torch.Tensor: Focal length in pixels.
    """
    return size / 2 / torch.tan(fov / 2)


def focal2fov(focal: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Compute (vertical/horizontal) field of view from focal length.

    Args:
        focal (torch.Tensor): Focal length in pixels.
        size (torch.Tensor): Image height / width in pixels.

    Returns:
        torch.Tensor: Field of view in radians.
    """
    return 2 * torch.arctan(size / (2 * focal))


def pitch2rho(pitch: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the distance from principal point to the horizon.

    Args:
        pitch (torch.Tensor): Pitch angle in radians.
        f (torch.Tensor): Focal length in pixels.
        h (torch.Tensor): Image height in pixels.

    Returns:
        torch.Tensor: Relative distance to the horizon.
    """
    return torch.tan(pitch) * f / h


def rho2pitch(rho: torch.Tensor, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute the pitch angle from the distance to the horizon.

    Args:
        rho (torch.Tensor): Relative distance to the horizon.
        f (torch.Tensor): Focal length in pixels.
        h (torch.Tensor): Image height in pixels.

    Returns:
        torch.Tensor: Pitch angle in radians.
    """
    return torch.atan(rho * h / f)


def rad2deg(rad: torch.Tensor) -> torch.Tensor:
    """Convert radians to degrees.

    Args:
        rad (torch.Tensor): Angle in radians.

    Returns:
        torch.Tensor: Angle in degrees.
    """
    return rad / torch.pi * 180


def deg2rad(deg: torch.Tensor) -> torch.Tensor:
    """Convert degrees to radians.

    Args:
        deg (torch.Tensor): Angle in degrees.

    Returns:
        torch.Tensor: Angle in radians.
    """
    return deg / 180 * torch.pi
