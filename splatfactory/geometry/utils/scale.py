"""Reference frames and scene scale for multi-view normalization.

Author: Alexander Veicht
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from splatfactory import get_logger
from splatfactory.geometry.cameras import Camera
from splatfactory.geometry.poses import Pose

logger = get_logger(__name__)


def mean_cam_transform(c2w: torch.Tensor) -> torch.Tensor:
    """World to average-camera transform [..., 4, 4] of OpenCV c2w poses [..., V, 4, 4].

    The average camera sits at the mean position, with the mean forward axis as primary.
    """
    pos = c2w[..., :3, 3].mean(dim=-2)  # [..., 3]
    fwd = F.normalize(c2w[..., :3, 2].mean(dim=-2), dim=-1)  # [..., 3]
    down = c2w[..., :3, 1].mean(dim=-2)  # [..., 3]
    down = F.normalize(down - (down * fwd).sum(-1, keepdim=True) * fwd, dim=-1)
    right = torch.linalg.cross(down, fwd, dim=-1)
    pos_avg = torch.stack([right, down, fwd, pos], dim=-1)  # [..., 3, 4]
    bottom = torch.tensor([0, 0, 0, 1], dtype=pos_avg.dtype, device=pos_avg.device)
    bottom = bottom.view(*([1] * (pos_avg.dim() - 2)), 1, 4).expand(*pos_avg.shape[:-2], 1, 4)
    pos_avg = torch.cat([pos_avg, bottom], dim=-2)  # [..., 4, 4]
    return torch.linalg.inv(pos_avg)


def mean_cam_down_transform(
    c2w: torch.Tensor, anchor_idx: int = 0, sigma0_deg: float = 10.0
) -> torch.Tensor:
    """World to average-camera transform [4, 4] with the mean down axis as primary.

    Unlike `mean_cam_transform`, it does not degenerate on inward or orbit captures, where the
    mean forward axis vanishes. The heading blends from the anchor view's toward the mean
    heading, weighted by how consistent the per-view headings are.

    Args:
        c2w: OpenCV c2w poses [V, 4, 4].
        anchor_idx: Context view that supplies the heading when the mean one is unreliable.
            View 0, the first_cam reference, is one the model can observe.
        sigma0_deg: Spread of the mean heading in degrees at which the mean and the anchor
            heading weigh equally.
    """
    d = F.normalize(c2w[:, :3, 1].mean(dim=0), dim=-1)
    h = c2w[:, :3, 2] - (c2w[:, :3, 2] @ d)[:, None] * d  # per-view horizontal headings
    valid = h.norm(dim=-1) > 1e-6
    h0 = h[anchor_idx] if valid[anchor_idx] else c2w[anchor_idx, :3, 0]  # straight-down: right
    u0 = F.normalize(h0, dim=-1)
    if valid.any():
        m = F.normalize(h[valid], dim=-1).mean(dim=0)
        r_c = m.norm().clamp(max=1.0 - 1e-9)
        sigma_mean = torch.sqrt((-2.0 * torch.log(r_c)).clamp(min=0.0)) / valid.sum() ** 0.5
        w = 1.0 / (1.0 + (torch.rad2deg(sigma_mean) / sigma0_deg) ** 4)
        um = F.normalize(m, dim=-1)
        a = w * torch.atan2(torch.linalg.cross(u0, um) @ d, u0 @ um)
        fwd = torch.cos(a) * u0 + torch.sin(a) * torch.linalg.cross(d, u0)
    else:
        fwd = u0
    right = torch.linalg.cross(d, fwd)
    pos = c2w[:, :3, 3].mean(dim=0)
    pos_avg = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    pos_avg[:3, 0], pos_avg[:3, 1], pos_avg[:3, 2], pos_avg[:3, 3] = right, d, fwd, pos
    return torch.linalg.inv(pos_avg)


def context_points(
    poses: Pose,
    cameras: Camera,
    depth: torch.Tensor,
    depth_mask: torch.Tensor,
) -> torch.Tensor:
    """Valid context depth points in world coordinates [N, 3]."""
    p3d = cameras.unproject_depth(depth)  # [S, H, W, 3]
    p3d = rearrange(p3d, "S H W D -> S (H W) D")
    p3d = poses.transform(p3d)  # per-view pose -> world
    p3d = rearrange(p3d, "S N D -> (S N) D")
    return p3d[rearrange(depth_mask, "S H W -> (S H W)")]


def context_point_centroid(
    poses: Pose,
    cameras: Camera,
    depth: torch.Tensor,
    depth_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Per-axis median [3] of the valid context depth points, None below 100 points."""
    if int(depth_mask.sum()) < 100:
        return None
    return context_points(poses, cameras, depth, depth_mask).median(dim=0).values  # [3]


def compute_scene_scale(
    poses: Pose,
    cameras: Camera,
    depth: torch.Tensor,
    depth_mask: torch.Tensor,
    num_context_views: int,
    normalize_baseline: bool | str,
) -> float:
    """Scene scale for normalization.

    Args:
        poses: Context c2w poses [V].
        cameras: Context cameras [V].
        depth: Context depth [V, H, W].
        depth_mask: Valid-depth mask [V, H, W].
        num_context_views: Number of context views.
        normalize_baseline: What to measure.
            False or None: median distance of the depth points from the frame origin.
            "medpoint": median distance of the depth points to their median point.
            True or "start_end": distance between the first and the last context camera.
            "max_pairwise_d": largest distance between two context cameras.
            "none": no rescaling, 1.0.
    """
    method = normalize_baseline
    if method is True:
        method = "start_end"

    if method == "none":
        return 1.0

    if method == "medpoint":
        if not depth_mask.any():
            raise ValueError("No valid depth for scale normalization")
        p3d = context_points(poses, cameras, depth, depth_mask)
        spread = (p3d - p3d.median(dim=0).values).norm(dim=-1)
        return float(spread.median().clamp(min=1e-3, max=1e3))

    if method:
        t = poses.t  # [V, 3]
        if method == "start_end":
            scale = (t[0] - t[-1]).norm().item()
        elif method == "max_pairwise_d":
            scale = torch.cdist(t, t).max().item()
        else:
            raise ValueError(f"Unknown pose-norm method: {method!r}")
        if num_context_views == 1:
            scale = 1.0
    else:
        if not depth_mask.any():
            raise ValueError("No valid depth for scale normalization")

        distances = context_points(poses, cameras, depth, depth_mask).norm(dim=-1)
        scale = distances.median().clamp(min=1e-3, max=1e3).item()

    if scale < 1e-6:
        logger.warning(f"Very small scale: {scale}")
    return scale
