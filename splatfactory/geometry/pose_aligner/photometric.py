"""Photometric residual and its analytic pose Jacobian.

Brightness constancy linearises r = render - target in the camera twist xi = (rot | trans):
J = -grad(I) . d(u, v)/d(xi), from one RGB + depth render and no autograd. The normal equations
run in a fused Triton kernel when available, else in torch.

All image tensors are [B, V, C, H, W].

Author: Alexander Veicht
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from splatfactory.gaussians import Gaussians
from splatfactory.geometry.cameras import Camera
from splatfactory.geometry.kernels.normal_equations import analytic_normal_eq
from splatfactory.geometry.poses import Pose

_ALPHA_EPS, _DEPTH_EPS = 1e-3, 1e-4  # validity thresholds of a rendered pixel
_BLOCK = 4096  # contraction block of the torch normal equations


def render_rgbd(
    gaussians: Gaussians, camera: Camera, pose: Pose, render_kw: Dict[str, Any]
) -> Tuple[Tensor, Tensor, Tensor]:
    """RGB [B, V, 3, H, W], expected depth [B, V, 1, H, W] and the mask of valid pixels."""
    out = gaussians.render_view(cameras=camera, poses=pose, rendering_mode="RGB+ED", **render_kw)
    rgb, depth = out["rendering"][:, :, :3], out["rendering"][:, :, 3:4]
    mask = ((out["alphas"] > _ALPHA_EPS) & (depth > _DEPTH_EPS)).to(rgb.dtype)
    return rgb, depth, mask


def image_gradient(img: Tensor) -> Tuple[Tensor, Tensor]:
    """Central-difference image gradient (d/du, d/dv), zero on the border."""
    gx, gy = torch.zeros_like(img), torch.zeros_like(img)
    gx[..., 1:-1] = (img[..., 2:] - img[..., :-2]) * 0.5
    gy[..., 1:-1, :] = (img[..., 2:, :] - img[..., :-2, :]) * 0.5
    return gx, gy


def interaction_rows(
    u: Tensor, v: Tensor, z: Tensor, fx: Tensor, fy: Tensor, cx: Tensor, cy: Tensor
) -> Tuple[Tensor, Tensor]:
    """d u / d xi and d v / d xi [..., 6] of pixel (u, v) at depth z. The inputs broadcast."""
    x, y, iz = (u - cx) * z / fx, (v - cy) * z / fy, 1.0 / z
    pu0, pu2 = fx * iz, -fx * x * iz * iz
    pv1, pv2 = fy * iz, -fy * y * iz * iz
    zero = torch.zeros_like(z)
    ju = torch.stack([-pu2 * y, -pu0 * z + pu2 * x, pu0 * y, -pu0, zero, -pu2], dim=-1)
    jv = torch.stack([pv1 * z - pv2 * y, pv2 * x, -pv1 * x, zero, -pv1, -pv2], dim=-1)
    return ju, jv


def pose_jacobian(gx: Tensor, gy: Tensor, depth: Tensor, camera: Camera) -> Tensor:
    """Residual Jacobian [B, V, H * W * 3, 6], rows ordered (pixel, channel)."""
    b, v, _, h, w = gx.shape
    vv, uu = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    per_view = lambda t: t.view(b, v, 1, 1)  # noqa: E731
    ju, jv = interaction_rows(
        uu,
        vv,
        depth[:, :, 0].clamp(min=_DEPTH_EPS),
        per_view(camera.f[..., 0]),
        per_view(camera.f[..., 1]),
        per_view(camera.c[..., 0]),
        per_view(camera.c[..., 1]),
    )  # [B, V, H, W, 6]
    gxp, gyp = gx.permute(0, 1, 3, 4, 2), gy.permute(0, 1, 3, 4, 2)  # [B, V, H, W, 3]
    J = -(gxp[..., None] * ju[..., None, :] + gyp[..., None] * jv[..., None, :])
    return J.reshape(b, v, h * w * 3, 6)


def _block_normal_equations(J: Tensor, r: Tensor) -> Tuple[Tensor, Tensor]:
    """H = JtJ [B, V, 6, 6] and g = Jtr [B, V, 6], reduced over pixel blocks."""
    b, v, m, p = J.shape
    nb = (m + _BLOCK - 1) // _BLOCK
    Jp = F.pad(J, (0, 0, 0, nb * _BLOCK - m)).reshape(b, v, nb, _BLOCK, p)
    rp = F.pad(r, (0, nb * _BLOCK - m)).reshape(b, v, nb, _BLOCK)
    H = torch.einsum("bvksi,bvksj->bvkij", Jp, Jp).sum(2)
    g = torch.einsum("bvksi,bvks->bvki", Jp, rp).sum(2)
    return H, g


def normal_equations(
    rgb: Tensor, target: Tensor, depth: Tensor, mask: Tensor, camera: Camera
) -> Tuple[Tensor, Tensor, Tensor]:
    """Masked residual r [B, V, 3, H, W] and its Gauss-Newton normal equations H, g."""
    gx, gy = image_gradient(rgb)
    r = (rgb - target) * mask
    h, w = rgb.shape[-2:]
    if analytic_normal_eq is not None:
        H, g = analytic_normal_eq(
            gx * mask,
            gy * mask,
            r,
            depth[:, :, 0],
            camera.f[..., 0],
            camera.f[..., 1],
            camera.c[..., 0],
            camera.c[..., 1],
            h,
            w,
        )
        return r, H, g
    b, v = mask.shape[:2]
    J = pose_jacobian(gx, gy, depth, camera)
    mask_rows = mask.expand(-1, -1, 3, -1, -1).permute(0, 1, 3, 4, 2).reshape(b, v, -1)
    r_rows = r.permute(0, 1, 3, 4, 2).reshape(b, v, -1)
    H, g = _block_normal_equations(J * mask_rows[..., None], r_rows)
    return r, H, g
