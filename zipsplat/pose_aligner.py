"""Photometric pose alignment by coarse-to-fine trust-region Levenberg-Marquardt, TokenBA's polish.

Brightness constancy linearises r = render - target in the camera twist xi = (rot | trans):
J = -grad(I) . d(u, v)/d(xi), from one RGB + depth render and no autograd. The normal equations
run in a fused Triton kernel when available, else in torch.

The splatfactory aligner trimmed to what TokenBA uses. All image tensors are (B, V, C, H, W).

Author: Alexander Veicht
"""

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.kernels import analytic_normal_eq
from zipsplat.pose import Pose

_ALPHA_EPS, _DEPTH_EPS = 1e-3, 1e-4  # validity thresholds of a rendered pixel
_BLOCK = 4096  # contraction block of the torch normal equations
_DAMPING = 1e-3  # initial Marquardt lambda
_MIN_STEP = 1e-4  # trust-region floor: below this a level is converged
_blur_kernels: dict = {}


# ==================================================================================================
# Photometric residual and its analytic pose Jacobian
# ==================================================================================================


def render_rgbd(gaussians: Gaussians, camera: Camera, pose: Pose, render_kw: Dict[str, Any]):
    """RGB (B, V, 3, H, W), expected depth (B, V, 1, H, W) and the mask of valid pixels."""
    rendering, info = gaussians.render(camera, pose, mode="RGB+ED", **render_kw)
    rgb, depth = rendering[:, :, :3], rendering[:, :, 3:4]
    mask = ((info["alphas"] > _ALPHA_EPS) & (depth > _DEPTH_EPS)).to(rgb.dtype)
    return rgb, depth, mask


def image_gradient(img: Tensor):
    """Central-difference image gradient (d/du, d/dv), zero on the border."""
    gx, gy = torch.zeros_like(img), torch.zeros_like(img)
    gx[..., 1:-1] = (img[..., 2:] - img[..., :-2]) * 0.5
    gy[..., 1:-1, :] = (img[..., 2:, :] - img[..., :-2, :]) * 0.5
    return gx, gy


def interaction_rows(u, v, z, fx, fy, cx, cy):
    """d u / d xi and d v / d xi (..., 6) of pixel (u, v) at depth z, xi = (rot | trans)."""
    x, y, iz = (u - cx) * z / fx, (v - cy) * z / fy, 1.0 / z
    pu0, pu2 = fx * iz, -fx * x * iz * iz
    pv1, pv2 = fy * iz, -fy * y * iz * iz
    zero = torch.zeros_like(z)
    ju = torch.stack([-pu2 * y, -pu0 * z + pu2 * x, pu0 * y, -pu0, zero, -pu2], dim=-1)
    jv = torch.stack([pv1 * z - pv2 * y, pv2 * x, -pv1 * x, zero, -pv1, -pv2], dim=-1)
    return ju, jv


def pose_jacobian(gx: Tensor, gy: Tensor, depth: Tensor, camera: Camera) -> Tensor:
    """Residual Jacobian (B, V, H * W * 3, 6), rows ordered (pixel, channel)."""
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
    )  # (B, V, H, W, 6)
    gxp, gyp = gx.permute(0, 1, 3, 4, 2), gy.permute(0, 1, 3, 4, 2)  # (B, V, H, W, 3)
    J = -(gxp[..., None] * ju[..., None, :] + gyp[..., None] * jv[..., None, :])
    return J.reshape(b, v, h * w * 3, 6)


def _block_normal_equations(J: Tensor, r: Tensor):
    """H = JtJ (B, V, 6, 6) and g = Jtr (B, V, 6) by a block-split reduction over pixels."""
    b, v, m, p = J.shape
    nb = (m + _BLOCK - 1) // _BLOCK
    Jp = F.pad(J, (0, 0, 0, nb * _BLOCK - m)).reshape(b, v, nb, _BLOCK, p)
    rp = F.pad(r, (0, nb * _BLOCK - m)).reshape(b, v, nb, _BLOCK)
    H = torch.einsum("bvksi,bvksj->bvkij", Jp, Jp).sum(2)
    g = torch.einsum("bvksi,bvks->bvki", Jp, rp).sum(2)
    return H, g


def normal_equations(rgb: Tensor, target: Tensor, depth: Tensor, mask: Tensor, camera: Camera):
    """Masked residual r (B, V, 3, H, W) and its Gauss-Newton normal equations H, g."""
    gx, gy = image_gradient(rgb)
    r = (rgb - target) * mask
    h, w = rgb.shape[-2:]
    if analytic_normal_eq is not None and rgb.is_cuda:
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


# ==================================================================================================
# PoseAligner
# ==================================================================================================


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur of (B, V, C, H, W)."""
    ks = int(2 * round(3 * sigma) + 1)
    key = (ks, sigma, x.device, x.dtype)
    if key not in _blur_kernels:
        c = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
        g = torch.exp(-(c**2) / (2 * sigma**2))
        g = g / g.sum()
        _blur_kernels[key] = (g[:, None] * g[None, :]).view(1, 1, ks, ks)
    b, v, c, h, w = x.shape
    k = _blur_kernels[key].expand(c, 1, ks, ks)
    return F.conv2d(x.reshape(b * v, c, h, w), k, padding=ks // 2, groups=c).reshape(b, v, c, h, w)


def _downscale(camera: Camera, target: Tensor, scale: float):
    """Camera and target of one pyramid level at `scale` of full resolution."""
    if scale == 1.0:
        return camera, target
    cam = camera.scale(torch.tensor([scale, scale], device=camera.device))
    w = int(round(cam.size[..., 0].flatten()[0].item()))
    h = int(round(cam.size[..., 1].flatten()[0].item()))
    b, v, c = target.shape[:3]
    t = F.interpolate(target.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False)
    return cam, t.reshape(b, v, c, h, w)


class PoseAligner:
    """Coarse-to-fine analytic Levenberg-Marquardt pose alignment, per view.

    Args:
        levels: Pyramid levels, coarse to fine, each (render scale, steps, blur sigma).
        verify: Accept or reject every step against the next render.
        abs_tol: With `verify`, a level converges once an accepted step lowers the cost by
            less than max(abs_tol, rel_tol * cost).
        rel_tol: Relative part of that tolerance.
        max_step: Trust-region radius on the pose twist.
        coarse_tile_size: gsplat tile_size below scale 0.5.
        fine_tile_size: gsplat tile_size from scale 0.5.
        coarse_radius_clip: Cull sub-pixel splats below scale 0.5.
    """

    def __init__(
        self,
        levels: Sequence[Sequence[float]] = ((0.5, 2, 0.0), (1.0, 1, 0.0)),
        verify: bool = False,
        abs_tol: float = 1e-7,
        rel_tol: float = 1e-3,
        max_step: float = 0.3,
        coarse_tile_size: int = 4,
        fine_tile_size: int = 16,
        coarse_radius_clip: float = 2.0,
    ) -> None:
        self.levels = [tuple(level) for level in levels]
        self.verify, self.abs_tol, self.rel_tol, self.max_step = verify, abs_tol, rel_tol, max_step
        self.coarse_tile_size, self.fine_tile_size = coarse_tile_size, fine_tile_size
        self.coarse_radius_clip = coarse_radius_clip

    def _render_kw(self, scale: float, bg: Optional[Tensor]) -> Dict[str, Any]:
        """Rasterizer arguments of a level."""
        kw = {"tile_size": self.coarse_tile_size if scale < 0.5 else self.fine_tile_size}
        if scale < 0.5:
            kw["radius_clip"] = self.coarse_radius_clip
        if bg is not None:
            kw["backgrounds"] = bg
        return kw

    @torch.no_grad()
    def __call__(
        self,
        gaussians: Gaussians,
        camera: Camera,
        pose: Pose,
        image: Tensor,
        background: Optional[Tensor] = None,
    ) -> Pose:
        """Align c2w poses (B, V) so the render of `gaussians` (B, N) matches `image`.

        Args:
            gaussians: Gaussians (B, N).
            camera: Cameras (B, V).
            pose: Initial c2w poses (B, V).
            image: Target images (B, V, 3, H, W).
            background: Optional background colors of the renders.

        Returns:
            The refined c2w poses (B, V), detached.
        """
        gaussians = gaussians.detach()
        b, v = camera.shape[:2]
        dev = pose.device
        for scale, steps, sigma in self.levels:
            cam_s, target_s = _downscale(camera, image, scale)
            if sigma > 0:
                target_s = _gaussian_blur(target_s, sigma)
            render_kw = self._render_kw(scale, background)
            # tried and accepted twists (rot, trans), relative to `pose`
            rot, trans = torch.zeros(b, v, 3, device=dev), torch.zeros(b, v, 3, device=dev)
            acc_rot, acc_trans = rot, trans
            lam = torch.full((b, v), _DAMPING, device=dev)
            radius = torch.full((b, v), self.max_step, device=dev)
            acc_cost = acc_H = acc_g = None
            for _ in range(int(steps)):
                rgb, depth, mask = render_rgbd(
                    gaussians, cam_s, pose @ Pose.from_aa(rot, trans), render_kw
                )
                if sigma > 0:
                    rgb = _gaussian_blur(rgb, sigma)
                r, H, g = normal_equations(rgb, target_s, depth, mask, cam_s)
                cost = (r**2).mean(dim=(-3, -2, -1))  # (B, V)

                # per view: a step that raises the cost is rolled back and retried with a
                # smaller radius and more damping
                if acc_cost is None or not self.verify:
                    better = torch.ones_like(cost, dtype=torch.bool)
                    improve = torch.full_like(cost, float("inf"))
                else:
                    better = cost < acc_cost
                    improve = torch.where(better, acc_cost - cost, torch.zeros_like(cost))
                    lam = (lam * torch.where(better, 0.3, 10.0)).clamp(1e-6, 1e3)
                    radius = torch.where(
                        better, (radius * 2).clamp(max=self.max_step), radius * 0.5
                    )
                acc_rot = torch.where(better[..., None], rot, acc_rot)
                acc_trans = torch.where(better[..., None], trans, acc_trans)
                acc_cost = cost if acc_cost is None else torch.where(better, cost, acc_cost)
                acc_H = H if acc_H is None else torch.where(better[..., None, None], H, acc_H)
                acc_g = g if acc_g is None else torch.where(better[..., None], g, acc_g)

                if self.verify:
                    tol = (self.rel_tol * acc_cost).clamp(min=self.abs_tol)
                    done = torch.where(better, improve <= tol, radius <= _MIN_STEP)
                    if bool(done.all()):
                        break

                # damped Gauss-Newton step from the accepted state, clamped to the trust region
                diag = acc_H.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)
                damped = acc_H + lam[..., None, None] * torch.diag_embed(diag)
                step = torch.linalg.solve(damped, (-acc_g).unsqueeze(-1)).squeeze(-1)
                norm = step.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                step = step * (radius[..., None] / norm).clamp(max=1.0)
                rot, trans = acc_rot + step[..., :3], acc_trans + step[..., 3:]
            if self.verify:
                pose = pose @ Pose.from_aa(acc_rot, acc_trans)
            else:  # unverified: apply the last solve
                pose = pose @ Pose.from_aa(rot, trans)
        return pose
