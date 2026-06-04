"""Gaussian head: decodes scene tokens into 3D Gaussian parameters
(means, scales, quats, opacity, SH) with free 3D offsets, plus the rendering loss.

Author: Alexander Veicht
"""

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from splatfactory import get_logger
from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.models import BaseModel
from splatfactory.models.losses import ChamferLoss, LPIPSLoss
from splatfactory.models.metrics import calculate_psnr
from splatfactory.models.modules import transformer_block

logger = get_logger(__name__)


def inverse_log_transform(y, positive: bool = False) -> torch.Tensor:
    """Apply inverse log transform: sign(y) * (exp(|y|) - 1)."""
    sign = torch.sign(y) if not positive else torch.ones_like(y)
    return sign * (torch.expm1(torch.abs(y)))


class GaussianHead(BaseModel):
    default_conf = {
        "embed_dim": 512,
        "gaussians_per_token": 32,
        "sh_degree": 1,
        "random_background": True,
        "coupled_init": True,
        "max_scale": 1.0,
        "min_scale": 1e-6,
        # Penalty thresholds
        "opacity_penalty_threshold": 0.01,
        "location_penalty_range": 10.0,
        # Initialization biases (None = keep hardcoded defaults)
        "init_position_z_bias": None,  # default: 0.5
        "init_scale_bias": None,  # default: 0.5
        "init_opacity_bias": None,  # default: -1.5
        # Loss weights and metrics
        "detach_activated_gaussians": True,
        "loss_on_context": False,
        "use_l1_loss": True,
        "use_chamfer": True,
        "mse_weight": 1.0,
        "lpips_weight": 0.05,
        "chamfer_weight": 0.05,
        "depth_weight": 0.01,
    }

    required_data_keys = ["tokens"]

    def _init(self, conf: dict[str, Any]):
        self.num_sh_coeffs = (conf.sh_degree + 1) ** 2

        # G x (mean, scale, quat, opacity, sh)
        self.num_params = conf.gaussians_per_token * (3 + 3 + 4 + 1 + self.num_sh_coeffs * 3)

        hidden_dim = 2 * conf.embed_dim
        out_dim = conf.embed_dim
        self.gaussian_head = nn.Sequential(
            transformer_block.Mlp(
                in_features=conf.embed_dim,
                hidden_features=hidden_dim,
                out_features=out_dim,
            ),
            nn.Linear(out_dim, self.num_params),
        )

        self._init_linear_layers()

        # Loss functions
        self.lpips = LPIPSLoss({"net_type": "vgg"})
        self.chamfer_loss = ChamferLoss(
            {"reduction": None, "bidirectional": False, "reversed": False}
        )

    def _get_init_biases(self) -> tuple[float, float, float]:
        """Return (scale_bias, position_z_bias, opacity_bias) with config overrides applied."""
        c = self.conf
        scale_bias = c.init_scale_bias if c.init_scale_bias is not None else 0.5
        pos_z_bias = c.init_position_z_bias if c.init_position_z_bias is not None else 0.5
        opacity_bias = c.init_opacity_bias if c.init_opacity_bias is not None else -1.5
        return scale_bias, pos_z_bias, opacity_bias

    def _init_linear_layers(self):
        linear = self.gaussian_head[-1]
        params_per_gaussian = self.num_params // self.conf.gaussians_per_token
        scale_bias, pos_z_bias, opacity_bias = self._get_init_biases()

        if self.conf.coupled_init:
            linear_in = linear.in_features
            template_weight = torch.randn(params_per_gaussian, linear_in) * 0.02
            template_bias = torch.zeros(params_per_gaussian)
            template_bias[3:6] = scale_bias
            template_bias[2] = pos_z_bias
            template_bias[10] = opacity_bias

            with torch.no_grad():
                for g in range(self.conf.gaussians_per_token):
                    s = g * params_per_gaussian
                    e = s + params_per_gaussian
                    linear.weight.data[s:e] = template_weight
                    linear.bias.data[s:e] = template_bias
            return

        with torch.no_grad():
            for g in range(self.conf.gaussians_per_token):
                offset = g * params_per_gaussian
                linear.bias.data[offset + 3 : offset + 6].fill_(scale_bias)
                linear.bias.data[offset + 2].fill_(pos_z_bias)
                linear.bias.data[offset + 10].fill_(opacity_bias)

    def _render_view_pair(
        self, scenes: Gaussians, data: dict[str, Any], prefix: str
    ) -> tuple[dict[str, Any], float]:
        """Render RGB and depth for a view, returning results and activated mask."""
        if "pose" not in data or "camera" not in data:
            B = data["image"].shape[0]
            activated = torch.zeros(B, scenes.num_gaussians, device=scenes.means.device)
            return {}, activated

        B = data["image"].shape[0]
        # Render RGB+D
        rgbd_results = scenes.render_view(data["camera"], data["pose"], rendering_mode="RGB+D")
        rgb, depth = rgbd_results["rendering"].split([3, 1], dim=2)

        # Add random background during training -> encourage opacities to be correct
        if self.conf.random_background and self.training:
            b, n = rgb.shape[:2]
            background = torch.rand(b, n, 3, 1, 1, device=rgb.device)
            rgb = rgb + background * (1 - rgbd_results["alphas"])

        results = {f"{prefix}_rgb": rgb, f"{prefix}_depth": depth[..., 0, :, :]}

        activated = torch.zeros(B, scenes.num_gaussians, device=scenes.means.device)
        if "activated" in rgbd_results.get("info", {}):
            activated = rgbd_results["info"]["activated"].any(dim=1).float().detach()
        elif not getattr(self, "_warned_no_activated", False):
            self._warned_no_activated = True
            logger.warning(
                "Renderer returned no 'activated' mask - this likely means an incompatible "
                "gsplat version. Falling back to an empty mask (detach_activated_gaussians "
                "becomes a no-op)."
            )

        return results, activated

    def _render_results(self, scenes: Gaussians, data: dict[str, Any]) -> dict[str, Any]:
        """Render context and target views from the gaussian scene."""
        results = {}

        B = data["context"]["image"].shape[0]
        activated = torch.zeros(B, scenes.num_gaussians, device=scenes.means.device)

        # Render context views
        context_results, context_activated = self._render_view_pair(
            scenes, data["context"], "context"
        )
        results |= context_results
        activated += context_activated

        if "target" not in data:
            results["activated_gaussians"] = (activated > 0).float()
            return results

        # Render target views
        target_results, target_activated = self._render_view_pair(scenes, data["target"], "target")
        results |= target_results
        activated += target_activated

        results["activated_gaussians"] = (activated > 0).float()

        return results

    def _forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Forward pass through the gaussian head and rendering of views."""
        G = self.conf.gaussians_per_token

        params = self.gaussian_head(data["tokens"])  # [B, N, num_params]

        # Compute means
        gaussians = rearrange(params, "B N (G C) -> B (N G) C", G=G, C=self.num_params // G)
        clamped = gaussians[..., 0:3].clamp(min=-5.0, max=5.0)
        means = inverse_log_transform(clamped)

        # Extract and activate remaining parameters
        # Scale parameterization: softplus(x - 4)
        scales = F.softplus(gaussians[..., 3:6] - 4.0)
        scales = scales.clamp(1e-6, 15.0)
        quats = F.normalize(gaussians[..., 6:10], dim=-1)
        opacities = gaussians[..., 10:11].sigmoid()[..., 0]
        sh_coeffs = gaussians[..., 11:]
        sh_coeffs = rearrange(sh_coeffs, "B N (K C) -> B N K C", K=self.num_sh_coeffs, C=3)

        # Create gaussian scenes
        scenes = Gaussians.from_parameters(
            means=means, scales=scales, quats=quats, opacities=opacities, sh_coeffs=sh_coeffs
        )

        return self._render_results(scenes, data) | {"gaussians": scenes}

    def _get_loss_context(
        self, pred: dict[str, Any], data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the predicted and ground truth images for loss computation."""
        gt_imgs = data["target"]["image"]
        pred_imgs = pred["target_rgb"]

        cameras: Camera = data["target"]["camera"]
        poses: Pose = data["target"]["pose"]
        depth = data["target"]["depth"]
        pred_depth = pred["target_depth"]

        if self.conf.loss_on_context:
            # add context views for loss computation as well
            gt_imgs = torch.cat([data["context"]["image"], data["target"]["image"]], dim=1)
            pred_imgs = torch.cat([pred["context_rgb"], pred["target_rgb"]], dim=1)

            cameras = torch.cat([data["context"]["camera"], data["target"]["camera"]], dim=1)
            poses = torch.cat([data["context"]["pose"], data["target"]["pose"]], dim=1)
            depth = torch.cat([data["context"]["depth"], data["target"]["depth"]], dim=1)
            pred_depth = torch.cat([pred["context_depth"], pred["target_depth"]], dim=1)

        return {
            "pred_imgs": pred_imgs,
            "gt_imgs": gt_imgs,
            "cameras": cameras,
            "poses": poses,
            "depth": depth,
            "pred_depth": pred_depth,
        }

    @torch.no_grad()
    def metrics(self, pred: dict[str, Any], data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute metrics between predicted and ground truth images."""
        context = self._get_loss_context(pred, data)
        pred_imgs, gt_imgs = context["pred_imgs"], context["gt_imgs"]

        b, n = gt_imgs.shape[:2]
        gt_imgs = rearrange(gt_imgs, "b n c h w -> (b n) c h w")
        pred_imgs = rearrange(pred_imgs, "b n c h w -> (b n) c h w")

        psnr = calculate_psnr(pred_imgs, gt_imgs)

        # gaussian stats
        def _get_stats(tensor: torch.Tensor, prefix: str) -> dict:
            """Gaussian statistics of shape (B,)."""
            return {
                f"{prefix}_min": tensor.min(dim=-1).values,
                f"{prefix}_max": tensor.max(dim=-1).values,
                f"{prefix}_mean": tensor.mean(dim=-1),
                f"{prefix}_median": tensor.median(dim=-1).values,
                f"{prefix}_std": tensor.std(dim=-1),
            }

        B = pred["gaussians"].means.shape[0]
        stats = {
            **_get_stats(pred["gaussians"].means.reshape(B, -1), "means"),
            **_get_stats(pred["gaussians"].scales.reshape(B, -1), "scales"),
            **_get_stats(pred["gaussians"].opacities, "opacities"),
        }

        return {
            "psnr": rearrange(psnr, "(b n) -> b n", n=n, b=b).mean(dim=-1),
            "activated_gaussians": pred["activated_gaussians"].sum(dim=-1),
            "activated_pct": pred["activated_gaussians"].mean(dim=-1),
            **stats,
        }

    def _rendering_loss(
        self, pred: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Compute rendering-related losses."""
        context = self._get_loss_context(pred, data)
        losses = self._compute_image_losses(
            context,
            mse_weight=self.conf.mse_weight,
            lpips_weight=self.conf.lpips_weight,
            depth_weight=0.0,
        )
        penalties = self._penalties(pred)

        rendering_loss = (
            losses["total"]
            + penalties["scale_penalty"]
            + penalties["opacity_penalty"]
            + penalties["location_penalty"]
        )

        return {
            "rendering_loss": rendering_loss,
            "mse_loss": losses["mse_loss"],
            "lpips_loss": losses["lpips_loss"],
            **penalties,
        }

    def _penalties(self, pred: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute regularization penalties on gaussians."""
        scale_penalty = (
            F.relu(pred["gaussians"].scales - self.conf.max_scale).pow(2).mean(dim=(-1, -2))
        )
        scale_penalty += (
            F.relu(self.conf.min_scale - pred["gaussians"].scales).pow(2).mean(dim=(-1, -2))
        )
        opacity_penalty = (
            F.relu(self.conf.opacity_penalty_threshold - pred["gaussians"].opacities)
            .pow(2)
            .mean(dim=-1)
        )
        location_penalty = (
            F.relu(torch.abs(pred["gaussians"].means) - self.conf.location_penalty_range)
            .pow(2)
            .mean(dim=(-1, -2))
        )
        return {
            "scale_penalty": scale_penalty,
            "opacity_penalty": opacity_penalty,
            "location_penalty": location_penalty,
        }

    def _compute_image_losses(
        self,
        context: dict[str, Any],
        mse_weight: float,
        lpips_weight: float,
        depth_weight: float,
    ) -> dict[str, torch.Tensor]:
        """Shared loss computation for image reconstruction (L1/MSE + LPIPS + depth)."""
        pred_imgs, gt_imgs = context["pred_imgs"], context["gt_imgs"]
        b, n = gt_imgs.shape[:2]
        gt_flat = rearrange(gt_imgs, "b n c h w -> (b n) c h w")
        pred_flat = rearrange(pred_imgs, "b n c h w -> (b n) c h w")

        # L1 / MSE
        loss_fn = F.l1_loss if self.conf.use_l1_loss else F.mse_loss
        mse_loss = loss_fn(pred_flat, gt_flat, reduction="none")
        mse_loss = rearrange(mse_loss, "(b n) c h w -> b (n c h w)", n=n, b=b).mean(dim=-1)

        # LPIPS (skip if weight == 0)
        lpips_loss = mse_loss.new_zeros(b)
        if lpips_weight > 0:
            lpips_loss = self.lpips({"rendering": pred_flat}, {"image": gt_flat})
            lpips_loss = rearrange(lpips_loss, "(b n) -> b n", n=n, b=b).mean(dim=-1)

        # Depth (per-pixel masking for sparse depth support)
        depth_loss = mse_loss.new_zeros(b)
        if depth_weight > 0:
            gt_depth = context["depth"]
            pred_depth = context["pred_depth"]
            valid = gt_depth >= 0  # [B, S, H, W]
            pixel_loss = F.l1_loss(pred_depth, gt_depth, reduction="none") * valid
            num_valid = valid.reshape(b, -1).sum(dim=-1).clamp(min=1)
            depth_loss = pixel_loss.reshape(b, -1).sum(dim=-1) / num_valid

        total = mse_loss * mse_weight + lpips_loss * lpips_weight + depth_loss * depth_weight
        return {
            "total": total,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss,
            "depth_loss": depth_loss,
        }

    def _geometry_loss(self, pred: dict[str, Any], data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute geometry-related losses."""

        if "depth" not in data["context"] or "depth" not in data.get("target", {}):
            B = data["context"]["image"].shape[0]
            zeros = data["context"]["image"].new_zeros(B)
            return {
                "geometry_loss": zeros,
                "location_loss": zeros,
                "location_loss_scaled": zeros,
                "depth_loss": zeros,
                "depth_loss_scaled": zeros,
            }
        context = self._get_loss_context(pred, data)
        gt_depth = context["depth"]  # [B, S, H, W]
        pred_depth = context["pred_depth"]
        cameras: Camera = context["cameras"]
        poses: Pose = context["poses"]

        B = gt_depth.shape[0]

        # Per-pixel valid mask for sparse depth support
        valid = gt_depth >= 0  # [B, S, H, W]
        any_valid = valid.reshape(B, -1).any(dim=1)  # [B]

        # If no valid depth in entire batch, return zeros
        if not any_valid.any():
            zeros = data["context"]["image"].new_zeros(B)
            return {
                "geometry_loss": zeros,
                "location_loss": zeros,
                "location_loss_scaled": zeros,
                "depth_loss": zeros,
                "depth_loss_scaled": zeros,
            }

        # Chamfer loss for gaussian centers to 3D points - only unrendered gaussians
        gs_means = pred["gaussians"].means  # [B, num_gaussians, 3]
        if self.conf.detach_activated_gaussians:
            if "activated_gaussians" not in pred:
                logger.warning(
                    "detach_activated_gaussians=True but 'activated_gaussians' is not in pred "
                    "(rendering likely skipped) - skipping detach of activated gaussians."
                )
            else:
                activated = pred["activated_gaussians"].bool()
                gs_means = torch.where(activated.unsqueeze(-1), gs_means.detach(), gs_means)

        if not self.conf.use_chamfer:
            # encourage location close to [0,0,1] (in front of camera 0)
            target_location = torch.tensor([0, 0, 1], device=gs_means.device)
            target_location = repeat(
                target_location, "D -> B N D", B=gs_means.shape[0], N=gs_means.shape[1]
            )
            location_loss = F.l1_loss(gs_means, target_location, reduction="none")
            location_loss = location_loss.mean(dim=(-1, -2))
        else:
            # Get 3D points from depth maps, pack only valid points
            p3d = cameras.unproject_depth(gt_depth)  # [B, S, H, W, 3]
            p3d = rearrange(p3d, "B S H W D -> B S (H W) D")
            p3d = poses.transform(p3d)
            p3d = rearrange(p3d, "B S N D -> B (S N) D")  # [B, M, 3]
            valid_flat = rearrange(valid, "B S H W -> B (S H W)")  # [B, M]

            # Pack valid points, fill remaining slots with random duplicates
            target_lengths = valid_flat.sum(dim=1).long()  # [B]
            max_valid = target_lengths.max().item()
            packed = p3d.new_zeros(B, max_valid, 3)
            for i in range(B):
                n = target_lengths[i].item()
                if n == 0:
                    continue
                vi = p3d[i][valid_flat[i]]  # [n, 3]
                packed[i, :n] = vi
                if n < max_valid:
                    fill_idx = torch.randint(0, n, (max_valid - n,), device=p3d.device)
                    packed[i, n:] = vi[fill_idx]

            location_loss = self.chamfer_loss(pred={"points": gs_means}, data={"points": packed})

        # Zero out loss for samples with no valid depth at all
        location_loss = location_loss * any_valid.float()
        scaled_location_loss = location_loss * self.conf.chamfer_weight

        pixel_loss = F.l1_loss(pred_depth, gt_depth, reduction="none") * valid
        num_valid = valid.reshape(B, -1).sum(dim=-1).clamp(min=1)
        depth_loss = pixel_loss.reshape(B, -1).sum(dim=-1) / num_valid
        scaled_depth_loss = depth_loss * self.conf.depth_weight

        return {
            "geometry_loss": scaled_location_loss + scaled_depth_loss,
            "location_loss": location_loss,
            "location_loss_scaled": scaled_location_loss,
            "depth_loss": depth_loss,
            "depth_loss_scaled": scaled_depth_loss,
        }

    def loss(self, pred: dict[str, Any], data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute losses for the model."""
        losses = self._rendering_loss(pred, data) | self._geometry_loss(pred, data)
        losses["total"] = losses["rendering_loss"] + losses["geometry_loss"]
        metrics = self.metrics(pred, data) | {"lpips": losses["lpips_loss"]}
        return losses, metrics
