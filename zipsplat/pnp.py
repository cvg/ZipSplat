"""Robust PnP of the input poses on ZipSplat's patch-Gaussian correspondences.

Training ties each token's Gaussians to the patch it was read from, so every Gaussian gives a
2D-3D correspondence and each pose is a render-free PnP problem.

Author: Alexander Veicht
"""

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.kernels import pnp_gnc
from zipsplat.pose import Pose

_Z_EPS = 1e-4
_INLIER_PX = 28.0  # inlier radius of the hypothesis ranking, two patches


def patch_correspondences(
    gaussians: Gaussians,
    image_hw: Sequence[int],
    num_views: int,
    assignments: Optional[Tensor] = None,
    gaussians_per_token: int = 32,
    patch_px: int = 14,
) -> Tuple[Tensor, Tensor]:
    """Gaussian to source-patch correspondences of a prediction.

    Under compression each token contributes its cluster's Gaussians, observed at its own patch
    centre.

    Args:
        gaussians: Gaussians (1, N).
        image_hw: Size (H, W) of the image the patch grid covers.
        num_views: Number of input views V.
        assignments: Per-token cluster ids (V * T) of a compressed prediction. Default: the
            token-major order of a full-quality one.
        gaussians_per_token: Gaussians per token G, used with `assignments`.
        patch_px: Patch size in pixels.

    Returns:
        pts: World-frame Gaussian means (V, M, 3), M = T * G for T patches per view.
        obs: Patch centre (x, y) each mean should project to (M, 2), shared across views.
    """
    H, W = image_hw
    pH, pW = H // patch_px, W // patch_px
    T = pH * pW
    means = gaussians.means[0]  # (N, 3)
    if assignments is None:
        G = means.shape[0] // (num_views * T)
        assert num_views * T * G == means.shape[0], "not token-major (pass assignments)"
        pts = means.reshape(num_views, T * G, 3)
    else:
        G = gaussians_per_token
        pts = means.reshape(-1, G, 3)[assignments.reshape(-1)].reshape(num_views, T * G, 3)
    idx = torch.arange(T, device=means.device, dtype=means.dtype)
    centres = torch.stack([(idx % pW + 0.5) * patch_px, (idx // pW + 0.5) * patch_px], -1)
    return pts, centres[:, None, :].expand(T, G, 2).reshape(T * G, 2)


def robust_pnp(
    pose: Pose,
    camera: Camera,
    pts: Tensor,
    obs: Tensor,
    scales: Sequence[float] = (64.0, 12.0, 4.0),
    iters: int = 4,
    subsample: Optional[int] = 2048,
    hypotheses_deg: float = 40.0,
    backend: str = "auto",
) -> Pose:
    """Robust multi-start PnP of the input poses on the patch correspondences.

    Seven rotation hypotheses per view (the init and +-`hypotheses_deg` about each camera axis)
    are solved in one batched call, and the hypothesis with the most inliers wins.

    Args:
        pose: Initial c2w poses (1, V).
        camera: Cameras of the patch grid (1, V).
        pts: World-frame points (V, M, 3) from `patch_correspondences`.
        obs: Their image observations (M, 2).
        scales: Robust-scale ladder in pixels, coarse to fine.
        iters: IRLS iterations per scale.
        subsample: Correspondences per solve, None for all.
        hypotheses_deg: Rotation offset of the extra hypotheses.
        backend: Solver backend, "auto", "triton" or "torch".

    Returns:
        The refined c2w poses (1, V).
    """
    dev = pts.device
    V, M = pts.shape[:2]
    axes = torch.eye(3, device=dev)
    aa = torch.stack(
        [torch.zeros(3, device=dev)]
        + [s * math.radians(hypotheses_deg) * axes[i] for i in range(3) for s in (1, -1)]
    )  # (K, 3)
    K = len(aa)
    # right-multiply so each hypothesis rotates in its own camera frame
    delta = Pose.from_aa(aa, torch.zeros(K, 3, device=dev))
    pose_k = pose[:, :, None].expand(1, V, K).reshape(1, V * K)[0]
    pose_k = pose_k @ delta[None].expand(V, K).reshape(V * K)
    cam_k = camera[:, :, None].expand(-1, V, K).reshape(1, V * K)
    pts_k = pts[:, None].expand(V, K, M, 3).reshape(V * K, M, 3)

    solve_pts, solve_obs = pts_k, obs
    if subsample is not None and subsample < M:  # fixed seed: the solve stays reproducible
        gen = torch.Generator(device="cpu").manual_seed(0)
        sel = torch.randperm(M, generator=gen)[:subsample].to(dev)
        solve_pts, solve_obs = pts_k[:, sel], obs[sel]

    w2c = pose_k.inv()
    R, t = pnp_gnc(
        w2c.R,
        w2c.t,
        cam_k.f[0],
        cam_k.c[0],
        solve_pts,
        solve_obs,
        scales=scales,
        iters=iters,
        backend=backend,
    )

    # rank the hypotheses on all correspondences
    P = pts_k @ R.transpose(-1, -2) + t[:, None]
    Z = P[..., 2].clamp(min=_Z_EPS)
    uv = P[..., :2] / Z[..., None] * cam_k.f[0][:, None] + cam_k.c[0][:, None]
    hits = ((uv - obs).norm(dim=-1) < _INLIER_PX) & (P[..., 2] > _Z_EPS)
    best = hits.float().mean(-1).reshape(V, K).argmax(-1)  # (V,)
    refined = Pose.from_Rt(R, t).inv().reshape(V, K)
    return refined[torch.arange(V, device=dev), best][None]
