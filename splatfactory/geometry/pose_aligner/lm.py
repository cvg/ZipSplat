"""PoseAligner: coarse-to-fine trust-region Levenberg-Marquardt photometric pose alignment.

The benchmarks' target-pose aligner and TokenBA's photometric polish. A scan of the global scale
runs before the pyramid.

Author: Alexander Veicht
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.geometry.pose_aligner.photometric import normal_equations, render_rgbd
from splatfactory.models import BaseModel
from splatfactory.utils import mappings

_DAMPING = 1e-3  # initial Marquardt lambda
_MIN_STEP = 1e-4  # trust-region floor: below this a level is converged
_SCALE_MIN, _SCALE_MAX, _SCALE_STEPS = 0.5, 2.0, 9  # log-spaced gauge scan, includes s = 1
_blur_kernels: Dict[tuple, Tensor] = {}


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur of [B, V, C, H, W]."""
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


def _downscale(camera: Camera, target: Tensor, scale: float) -> Tuple[Camera, Tensor]:
    """Camera and target of one pyramid level at `scale` of full resolution."""
    if scale == 1.0:
        return camera, target
    cam = camera.scale(torch.tensor([scale, scale], device=camera.device))
    w = int(round(cam.size[..., 0].flatten()[0].item()))
    h = int(round(cam.size[..., 1].flatten()[0].item()))
    b, v, c = target.shape[:3]
    t = F.interpolate(target.flatten(0, 1), size=(h, w), mode="bilinear", align_corners=False)
    return cam, t.reshape(b, v, c, h, w)


class PoseAligner(BaseModel):
    """Coarse-to-fine analytic Levenberg-Marquardt pose alignment, per view."""

    default_conf = {
        # pyramid levels: [render scale, steps, blur sigma]
        "levels": [[0.125, 12, 0.7], [0.25, 10, 0.7], [0.5, 8, 0.0], [1.0, 6, 0.0]],
        # a level converges once an accepted step lowers the cost by less than
        # max(abs_tol, rel_tol * cost)
        "abs_tol": 1e-7,
        "rel_tol": 1e-3,
        "verify": True,  # accept or reject every step against the next render
        "full_res_guard": True,  # keep the best full-res pose across levels
        "max_step": 0.3,  # trust-region radius on the pose twist
        "coarse_tile_size": 4,  # gsplat tile_size below scale 0.5
        "fine_tile_size": 16,  # gsplat tile_size from scale 0.5
        "coarse_radius_clip": 2.0,  # cull sub-pixel splats below scale 0.5
        "global_scale": True,  # scan the scale before the pyramid
        "batch_size": None,  # views per chunk, None = all
    }
    required_data_keys = ["gaussians", "camera", "pose", "image"]

    def _init(self, conf):
        self.levels = [tuple(level) for level in conf.levels]

    def _render_kw(self, scale: float, bg: Optional[Tensor]) -> Dict[str, Any]:
        """Rasterizer arguments of a level."""
        kw = {"tile_size": self.conf.coarse_tile_size if scale < 0.5 else self.conf.fine_tile_size}
        if scale < 0.5:
            kw["radius_clip"] = self.conf.coarse_radius_clip
        if bg is not None:
            kw["backgrounds"] = bg
        return kw

    @torch.no_grad()
    def _global_scale(
        self,
        init_pose: Pose,
        gaussians: Gaussians,
        camera: Camera,
        target: Tensor,
        bg: Optional[Tensor],
    ) -> float:
        """Scale of the init translations with the lowest photometric cost, by brute force."""
        dev = init_pose.device
        grid = torch.exp(
            torch.linspace(math.log(_SCALE_MIN), math.log(_SCALE_MAX), _SCALE_STEPS, device=dev)
        )
        v = camera.shape[1]
        chunk = self.conf.batch_size or v
        best_cost, best_scale = None, 1.0
        for s in grid:
            cost = 0.0
            for a in range(0, v, chunk):
                sl = slice(a, min(a + chunk, v))
                rgb, _, mask = render_rgbd(
                    gaussians,
                    camera[:, sl],
                    init_pose[:, sl].scale_translation(float(s)),
                    self._render_kw(1.0, bg[sl] if bg is not None else None),
                )
                cost += float((((rgb - target[:, sl]) * mask) ** 2).sum())
            if best_cost is None or cost < best_cost:
                best_cost, best_scale = cost, float(s)
        return best_scale

    @torch.no_grad()
    def _align(
        self,
        init_pose: Pose,
        gaussians: Gaussians,
        camera: Camera,
        target: Tensor,
        bg: Optional[Tensor],
    ) -> Pose:
        """Coarse-to-fine trust-region LM from `init_pose`. Returns the refined poses [B, V]."""
        b, v = camera.shape[:2]
        dev = init_pose.device
        conf = self.conf
        pose = init_pose

        def full_res_cost(p):
            """Masked photometric cost [B, V] at full resolution."""
            rgb, _, mask = render_rgbd(gaussians, camera, p, self._render_kw(1.0, bg))
            return (((rgb - target) * mask) ** 2).mean(dim=(-3, -2, -1))

        if conf.full_res_guard:
            best_pose, best_cost = pose, full_res_cost(pose)
        for scale, steps, sigma in self.levels:
            cam_s, target_s = _downscale(camera, target, scale)
            if sigma > 0:
                target_s = _gaussian_blur(target_s, sigma)
            render_kw = self._render_kw(scale, bg)
            # tried and accepted twists (rot, trans), relative to `pose`
            rot, trans = torch.zeros(b, v, 3, device=dev), torch.zeros(b, v, 3, device=dev)
            acc_rot, acc_trans = rot, trans
            lam = torch.full((b, v), _DAMPING, device=dev)
            radius = torch.full((b, v), conf.max_step, device=dev)
            acc_cost = acc_H = acc_g = None
            for _ in range(steps):
                rgb, depth, mask = render_rgbd(
                    gaussians, cam_s, pose @ Pose.from_aa(rot, trans), render_kw
                )
                if sigma > 0:
                    rgb = _gaussian_blur(rgb, sigma)
                r, H, g = normal_equations(rgb, target_s, depth, mask, cam_s)
                cost = (r**2).mean(dim=(-3, -2, -1))  # [B, V]

                # per view: a step that raises the cost is rolled back and retried with a
                # smaller radius and more damping
                if acc_cost is None or not conf.verify:
                    better = torch.ones_like(cost, dtype=torch.bool)
                    improve = torch.full_like(cost, float("inf"))
                else:
                    better = cost < acc_cost
                    improve = torch.where(better, acc_cost - cost, torch.zeros_like(cost))
                    lam = (lam * torch.where(better, 0.3, 10.0)).clamp(1e-6, 1e3)
                    radius = torch.where(
                        better, (radius * 2).clamp(max=conf.max_step), radius * 0.5
                    )
                acc_rot = torch.where(better[..., None], rot, acc_rot)
                acc_trans = torch.where(better[..., None], trans, acc_trans)
                acc_cost = cost if acc_cost is None else torch.where(better, cost, acc_cost)
                acc_H = H if acc_H is None else torch.where(better[..., None, None], H, acc_H)
                acc_g = g if acc_g is None else torch.where(better[..., None], g, acc_g)

                if conf.verify:
                    tol = (conf.rel_tol * acc_cost).clamp(min=conf.abs_tol)
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
            if conf.verify:
                pose = pose @ Pose.from_aa(acc_rot, acc_trans)
            else:  # unverified: apply the last solve
                pose = pose @ Pose.from_aa(rot, trans)
            if conf.full_res_guard:  # keep the best full-res pose per view
                cost = full_res_cost(pose)
                best_pose = Pose.where(cost < best_cost, pose, best_pose)
                best_cost = torch.minimum(cost, best_cost)
                pose = best_pose
        return pose

    def _forward(self, data):
        """Align the target poses.

        Args:
            data: Batched or unbatched.
                gaussians: Gaussians [B, N].
                camera: Cameras [B, V].
                pose: Initial c2w poses [B, V].
                image: Target images [B, V, 3, H, W].
                render_kwargs: Optional rasterizer arguments, only `backgrounds` is used.

        Returns:
            pose: Refined c2w poses [B, V].
            global_scale: Gauge scale applied to the init, with `global_scale` on.
        """
        render_kwargs = data.get("render_kwargs", None)
        data = {k: v for k, v in data.items() if k != "render_kwargs"}
        if data["image"].dim() == 4:
            data = mappings.add_batch_dim(data)
        bg = render_kwargs.get("backgrounds") if render_kwargs else None
        init_pose, gaussians, camera, target = (
            data[k] for k in ("pose", "gaussians", "camera", "image")
        )
        gaussians = gaussians.detach()

        scale = None
        if self.conf.global_scale:  # one gauge for all views, so before the chunking
            scale = self._global_scale(init_pose, gaussians, camera, target, bg)
            init_pose = init_pose.scale_translation(scale)

        chunk, v = self.conf.batch_size, camera.shape[1]
        if chunk is None or v <= chunk:
            pose = self._align(init_pose, gaussians, camera, target, bg)
        else:
            parts = []
            for a in range(0, v, chunk):
                sl = slice(a, min(a + chunk, v))
                bg_sl = bg[sl] if bg is not None else None
                p = self._align(init_pose[:, sl], gaussians, camera[:, sl], target[:, sl], bg_sl)
                parts.append(p)
                torch.cuda.empty_cache()
            pose = torch.cat(parts, dim=1)
        out = {"pose": pose}
        if scale is not None:
            out["global_scale"] = scale
        return out

    def loss(self, pred, data):
        raise NotImplementedError("PoseAligner has no loss.")

    def metrics(self, pred, data):
        raise NotImplementedError("PoseAligner has no metrics.")
