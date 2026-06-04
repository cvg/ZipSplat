"""Test-time target-pose estimation for pose-free evaluation: optimizes each held-out
camera pose against its GT image (photometric + LPIPS) so novel views can be rendered
and scored in the predicted scene's frame.

Author: Alexander Veicht
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm

from splatfactory import get_logger
from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.models import BaseModel
from splatfactory.models.losses import LPIPSLoss
from splatfactory.models.metrics import calculate_psnr
from splatfactory.utils import mappings, tools

logger = get_logger(__name__)


class PoseEstimator(BaseModel):
    # Presets (override default_conf):
    #   fast:      lr=0.003, num_steps=50, lpips_weight=0  -> 6x faster, -0.12 PSNR
    #   very_fast: lr=0.003, num_steps=30, lpips_weight=0  -> 9x faster, -0.28 PSNR
    default_conf = {
        "rot_lr": 0.005,
        "trans_lr": 0.005,
        "num_steps": 200,
        "optimizer": "AdamW",  # Adam, AdamW, Adamax, etc.
        "opacity_threshold": 0.3,
        "atol": 1e-5,
        "patience_limit": 5,
        "lpips_weight": 0.05,
        "verbose": False,
        "batch_size": None,
    }

    def _init(self, conf):
        self.rot_lr = conf.rot_lr
        self.trans_lr = conf.trans_lr
        self.num_steps = conf.num_steps
        self.opacity_threshold = conf.opacity_threshold
        self.verbose = conf.verbose
        self.batch_size = conf.batch_size

        if conf.lpips_weight > 0:
            self.lpips = LPIPSLoss({"net_type": "vgg"})
        else:
            self.lpips = None

    @torch.no_grad()
    def estimate_initial_pose(self, gaussians: Gaussians, cameras: Camera):
        raise NotImplementedError("PnP RANSAC pose estimation is not yet implemented.")

    def _optimize_pose(
        self,
        init_pose: Pose,
        gaussians: Gaussians,
        camera: Camera,
        target_rgb: torch.Tensor,
        render_kwargs: dict | None = None,
    ) -> Pose:
        render_kwargs = dict(render_kwargs or {})

        # make sure gaussians, pose and camera do not require gradients
        gaussians = gaussians.detach()
        init_pose = init_pose.detach()
        camera = camera.detach()
        target_rgb = target_rgb.detach()

        b, v = camera.shape[:2]

        trans_delta = nn.Parameter(torch.zeros(b, v, 3, device=init_pose.data_.device))
        rot_delta = nn.Parameter(torch.zeros(b, v, 3, device=init_pose.data_.device))

        optimizer_cls = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_cls(
            [{"params": rot_delta, "lr": self.rot_lr}, {"params": trans_delta, "lr": self.trans_lr}]
        )

        prev_loss = None
        patience_counter = 0
        pbar = tqdm(
            range(self.num_steps),
            desc="Optimizing pose",
            ncols=120,
            disable=not self.verbose,
            leave=False,
        )
        pose = init_pose.clone()
        for step in range(self.num_steps):
            optimizer.zero_grad()

            pose_update = Pose.from_aa(rot_delta, trans_delta)

            views = gaussians.render_view(cameras=camera, poses=pose @ pose_update, **render_kwargs)

            loss, metrics = self.loss(
                {"rendering": views["rendering"], "alphas": views["alphas"]},
                {"image": target_rgb},
            )
            loss["total"].mean().backward()
            optimizer.step()

            postfix = {
                "loss": loss["total"].mean().item(),
                "psnr": metrics["psnr"].mean().item(),
            }
            if "lpips" in loss:
                postfix["lpips"] = loss["lpips"].mean().item()
            pbar.set_postfix(postfix)

            pbar.update()
            pose = pose @ Pose.from_aa(rot_delta, trans_delta).detach()

            rot_delta.data.fill_(0)
            trans_delta.data.fill_(0)

            # early stopping
            if prev_loss is not None:
                delta = abs(loss["total"].mean().item() - prev_loss)
                if delta < self.conf.atol:
                    patience_counter += 1
                    if patience_counter >= self.conf.patience_limit:
                        break
                else:
                    patience_counter = 0
            prev_loss = loss["total"].mean().item()

        pbar.close()
        return pose

    def _forward(self, data):
        # Extract render_kwargs before add_batch_dim to avoid
        # unsqueezing background tensors (gsplat expects 2D).
        render_kwargs = data.pop("render_kwargs", None)
        if len(data["image"].shape) == 4:
            data = mappings.add_batch_dim(data)

        init_pose = data.get("pose", None)
        if init_pose is None:
            init_pose = self.estimate_initial_pose(data["gaussians"], data["camera"])

        b, v = data["camera"].shape[:2]

        if self.batch_size is None or v <= self.batch_size:
            return {
                "pose": self._optimize_pose(
                    init_pose,
                    data["gaussians"],
                    data["camera"],
                    data["image"],
                    render_kwargs=render_kwargs,
                )
            }

        def _slice_render_kwargs(n):
            """Slice backgrounds to match the view batch size."""
            if not render_kwargs or "backgrounds" not in render_kwargs:
                return render_kwargs
            return {**render_kwargs, "backgrounds": render_kwargs["backgrounds"][:n]}

        poses_opt = []
        for start in range(0, v, self.batch_size):
            end = min(start + self.batch_size, v)
            pose_batch_opt = self._optimize_pose(
                init_pose[:, start:end],
                data["gaussians"],
                data["camera"][:, start:end],
                data["image"][:, start:end],
                render_kwargs=_slice_render_kwargs(end - start),
            )
            poses_opt.append(pose_batch_opt)
            torch.cuda.empty_cache()

        return {"pose": torch.cat(poses_opt, dim=1)}

    def metrics(self, pred, data):
        """Compute metrics between predicted and ground-truth images."""
        b, n = data["image"].shape[:2]
        gt_imgs = rearrange(data["image"], "b n c h w -> (b n) c h w")
        pred_imgs = rearrange(pred["rendering"], "b n c h w -> (b n) c h w")
        psnr = calculate_psnr(pred_imgs, gt_imgs)
        return {"psnr": rearrange(psnr, "(b n) -> b n", n=n, b=b).mean(dim=-1)}

    def loss(self, pred, data):
        """Compute photometric loss between rendered and ground truth images.

        When ``pred`` contains ``alphas`` (Gaussian accumulation buffer), background
        pixels are masked out so that they don't pollute the gradient signal.
        """
        b, n = data["image"].shape[:2]
        gt_imgs = rearrange(data["image"], "b n c h w -> (b n) c h w")
        pred_imgs = rearrange(pred["rendering"], "b n c h w -> (b n) c h w")

        # Build foreground mask from alpha (>0 means Gaussian coverage)
        if "alphas" in pred:
            mask = rearrange(pred["alphas"], "b n c h w -> (b n) c h w")  # [BN, 1, H, W]
        else:
            mask = torch.ones_like(pred_imgs[:, :1])

        # Masked L1: zero out background gradients, average over foreground only
        l1_loss = F.l1_loss(pred_imgs, gt_imgs, reduction="none")  # [BN, 3, H, W]
        l1_loss = (l1_loss * mask).sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) * 3 + 1e-8)
        l1_loss = rearrange(l1_loss, "(b n) -> b n", n=n, b=b).mean(dim=-1)

        losses = {"l1": l1_loss}
        total_loss = l1_loss

        if self.lpips is not None:
            # Mask by setting bg pixels to same value in both pred and gt
            masked_pred = pred_imgs * mask
            masked_gt = gt_imgs * mask
            lpips_loss = self.lpips({"rendering": masked_pred}, {"image": masked_gt})
            lpips_loss = rearrange(lpips_loss, "(b n) -> b n", n=n, b=b).mean(dim=-1)
            total_loss = total_loss + lpips_loss * self.conf.lpips_weight
            losses["lpips"] = lpips_loss

        losses["total"] = total_loss
        return losses, self.metrics(pred, data)
