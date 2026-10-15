"""Occlusion-aware multi-view overlap from GT depth.

Author: Alexander Veicht
"""

import torch
import torch.nn.functional as F


def pixel_overlap_degree(data, depth_tol: float = 0.05, idx: int = 0) -> torch.Tensor:
    """Number of context views that see each target pixel's GT 3D point [Tv, H, W].

    A view sees a point when its GT depth at the projection agrees within `depth_tol`
    (relative). Pixels without target depth are -1. `data` holds the "target" and "context"
    camera, pose and depth, `idx` selects the batch element.
    """
    tgt_cam = data["target"]["camera"][idx]
    tgt_pose = data["target"]["pose"][idx]
    tgt_depth = data["target"]["depth"][idx]  # [Tv, H, W]
    ctx_cam = data["context"]["camera"][idx]
    ctx_pose = data["context"]["pose"][idx]
    ctx_depth = data["context"]["depth"][idx]  # [V, H, W]
    Tv, H, W = tgt_depth.shape
    V = ctx_depth.shape[0]

    tgt_valid = tgt_depth > 0
    p_cam = tgt_cam.unproject_depth(tgt_depth * tgt_valid)  # [Tv, H, W, 3]
    pts = tgt_pose.transform(p_cam.reshape(Tv, H * W, 3)).reshape(-1, 3)  # world [N, 3]

    deg = torch.zeros(Tv * H * W, device=pts.device)
    for v in range(V):
        w2c = ctx_pose[v : v + 1].inv()
        pc = w2c.transform(pts.unsqueeze(0))[0]  # [N, 3] in context view v
        z = pc[..., 2]
        p2d, vis = ctx_cam[v : v + 1].world2image(pc.unsqueeze(0))
        p2d, vis = p2d[0], vis[0]
        # sample the context view's GT depth at the projected locations
        size = ctx_cam[v].size  # (w, h)
        grid = torch.stack(
            [2 * p2d[:, 0] / (size[0] - 1) - 1, 2 * p2d[:, 1] / (size[1] - 1) - 1], dim=-1
        )
        d_s = F.grid_sample(
            ctx_depth[v][None, None], grid[None, None], mode="nearest", align_corners=True
        ).reshape(-1)
        consistent = (z - d_s).abs() < depth_tol * d_s.clamp_min(1e-6)
        deg += (vis & (z > 0) & (d_s > 0) & consistent).float()
    deg = deg.reshape(Tv, H, W)
    deg[~tgt_valid] = -1
    return deg


def context_self_overlap_degree(data, depth_tol: float = 0.05, idx: int = 0) -> torch.Tensor:
    """Number of other context views that see each context pixel [V, H, W], -1 if invalid."""
    pair = {"target": data["context"], "context": data["context"]}
    deg = pixel_overlap_degree(pair, depth_tol, idx)
    return torch.where(deg >= 0, deg - 1, deg)
