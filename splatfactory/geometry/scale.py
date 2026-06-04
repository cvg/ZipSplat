"""Scene scale computation for multi-view normalization.

Author: Alexander Veicht
"""

import torch
from einops import rearrange

from splatfactory import get_logger
from splatfactory.geometry.cameras import Camera
from splatfactory.geometry.poses import Pose

logger = get_logger(__name__)


def compute_scene_scale(
    poses: Pose,
    cameras: Camera,
    depth: torch.Tensor,
    depth_mask: torch.Tensor,
    num_context_views: int,
    normalize_baseline: bool | str,
) -> float:
    """Compute scene scale for normalization.

    `normalize_baseline`:
        False / None            -> depth-based median distance (default)
        True / "start_end"      -> |t_0 - t_{-1}| across context cameras (NoPoSplat/MVSplat)
        "max_pairwise_d"        -> max pairwise context-camera distance (YoNoSplat)
        "none"                  -> return 1.0 (no rescaling - for baselines like DepthSplat
                                  that train on raw COLMAP-unit poses with absolute near/far)
    """
    method = normalize_baseline
    if method is True:
        method = "start_end"

    if method == "none":
        return 1.0

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

        p3d = cameras.unproject_depth(depth)
        p3d = rearrange(p3d, "S H W D -> S (H W) D")
        p3d = poses.transform(p3d)
        p3d = rearrange(p3d, "S N D -> (S N) D")

        mask = rearrange(depth_mask, "S H W -> (S H W)")
        distances = p3d[mask].norm(dim=-1)
        scale = distances.median().clamp(min=1e-3, max=1e3).item()

    if scale < 1e-6:
        logger.warning(f"Very small scale: {scale}")
    return scale
