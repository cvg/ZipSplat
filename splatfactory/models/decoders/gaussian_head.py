"""Gaussian head: decodes each scene token into G Gaussians with free 3D offsets, plus the
rendering loss.

Author: Alexander Veicht
"""

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn

from splatfactory import get_logger
from splatfactory.gaussians import Gaussians
from splatfactory.geometry import Camera, Pose
from splatfactory.models import BaseModel
from splatfactory.models.losses import LPIPSLoss
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
        "random_background": False,  # training renders over a random background color
        "coupled_init": True,  # every Gaussian of a token starts from the same readout
        "max_scale": 0.1,  # scale penalty above this
        "min_scale": 1e-6,
        "render_max_scale": 15.0,  # hard upper clamp on the activated scale
        "render_kwargs": {},  # extra gsplat arguments for the training render
        # Penalty thresholds
        "opacity_penalty_threshold": 0.01,
        "location_penalty_range": 10.0,
        "scale_penalty_weight": 1.0,
        "opacity_penalty_weight": 1.0,
        # Initialization biases (None = keep hardcoded defaults)
        "init_position_z_bias": None,  # default: 0.5
        "init_scale_bias": None,  # default: 0.5
        "init_opacity_bias": None,  # default: -1.5
        # Loss weights
        "loss_on_context": False,  # also supervise the context renders
        "use_l1_loss": True,  # the photometric term is L1 (MSE when false)
        "mse_weight": 1.0,
        "lpips_weight": 0.1,
        "depth_weight": 0.0,  # L1 on the rendered depth where GT depth is valid
        "sh_dropout": 0.0,  # probability to zero a scene's degree > 0 SH in training
        "compile_loss": False,
    }

    required_data_keys = ["tokens"]

    def _init(self, conf: dict[str, Any]):
        self.num_sh_coeffs = (conf.sh_degree + 1) ** 2

        # per-gaussian channel offsets: means(3) | scales(3) | quats(4) | opacity(1) | sh(K*3)
        self._sc0 = 3
        self._q0 = self._sc0 + 3
        self._op0 = self._q0 + 4
        self._sh0 = self._op0 + 1

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

        self._decode_fn = self._decode_gaussians  # swapped by _compile()

        self.lpips = LPIPSLoss({"net_type": "vgg"})

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
            template_bias[self._sc0 : self._q0] = scale_bias
            template_bias[2] = pos_z_bias
            template_bias[self._op0] = opacity_bias

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
                linear.bias.data[offset + self._sc0 : offset + self._q0].fill_(scale_bias)
                linear.bias.data[offset + 2].fill_(pos_z_bias)
                linear.bias.data[offset + self._op0].fill_(opacity_bias)

    def _render_view_pair(
        self, scenes: Gaussians, data: dict[str, Any], prefix: str
    ) -> tuple[dict[str, Any], float]:
        """Render RGB and depth for one set of views. Returns (renders, activated (B, N))."""
        if "pose" not in data or "camera" not in data:
            B = data["image"].shape[0]
            activated = torch.zeros(B, scenes.num_gaussians, device=scenes.means.device)
            return {}, activated

        B = data["image"].shape[0]
        rgbd_results = scenes.render_view(
            data["camera"],
            data["pose"],
            rendering_mode="RGB+D",
            **OmegaConf.to_container(self.conf.render_kwargs, resolve=True),
        )
        rgb, depth = rgbd_results["rendering"].split([3, 1], dim=2)

        # A random background in training forces correct opacities.
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
                "Renderer meta has no 'activated' mask (no radii in rasterization meta). "
                "Falling back to an empty mask (the activated_* metrics read 0)."
            )

        return results, activated

    def _render_results(self, scenes: Gaussians, data: dict[str, Any]) -> dict[str, Any]:
        """Render the context and target views."""
        results = {}

        B = data["context"]["image"].shape[0]
        activated = torch.zeros(B, scenes.num_gaussians, device=scenes.means.device)

        # Training skips the context renders (render_context=False from the model).
        if data.get("render_context", True) or "target" not in data:
            context_results, context_activated = self._render_view_pair(
                scenes, data["context"], "context"
            )
            results |= context_results
            activated += context_activated

        if "target" not in data:
            results["activated_gaussians"] = (activated > 0).float()
            return results

        target_results, target_activated = self._render_view_pair(scenes, data["target"], "target")
        results |= target_results
        activated += target_activated

        results["activated_gaussians"] = (activated > 0).float()

        return results

    def _compile(self, *args, **kwargs) -> None:
        """Compile the decode only (dynamic, the token count varies per step)."""
        self._decode_fn = torch.compile(self._decode_gaussians, dynamic=True, *args, **kwargs)

    def decode(self, tokens, half_mlp=False):
        """Tokens (B, K, D) -> activated Gaussian parameters, each (B, K * G, ...).

        Args:
            tokens: (B, K, D) scene tokens.
            half_mlp: run the MLP body in fp16 (TokenBA). The readout stays fp32.

        Returns:
            (means, scales, quats, opacities, sh_coeffs).
        """
        return self._decode_fn(tokens, half_mlp)

    def _decode_gaussians(self, tokens, half_mlp=False):
        """decode() as a pure tensor chain (the compile target)."""
        G = self.conf.gaussians_per_token

        if half_mlp:
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = self.gaussian_head[0](tokens)
            params = self.gaussian_head[1](hidden.float())
        else:
            params = self.gaussian_head(tokens)  # [B, N, num_params]

        gaussians = rearrange(params, "B N (G C) -> B (N G) C", G=G, C=self.num_params // G)
        clamped = gaussians[..., 0:3].clamp(min=-5.0, max=5.0)
        means = inverse_log_transform(clamped)

        scales = F.softplus(gaussians[..., self._sc0 : self._q0] - 4.0).clamp(
            1e-6, self.conf.render_max_scale
        )
        quats = F.normalize(gaussians[..., self._q0 : self._op0], dim=-1)
        opacities = gaussians[..., self._op0 : self._op0 + 1].sigmoid()[..., 0]
        sh_coeffs = gaussians[..., self._sh0 :]
        sh_coeffs = rearrange(sh_coeffs, "B N (K C) -> B N K C", K=self.num_sh_coeffs, C=3)
        return means, scales, quats, opacities, sh_coeffs

    def _drop_sh(self, sh_coeffs: torch.Tensor) -> torch.Tensor:
        """Training-only SH dropout: zero a scene's degree > 0 SH so sh0 carries the color."""
        if not (self.training and self.conf.sh_dropout > 0):
            return sh_coeffs
        B = sh_coeffs.shape[0]
        drop = torch.rand(B, device=sh_coeffs.device) < self.conf.sh_dropout  # per scene
        mask = torch.ones_like(sh_coeffs[:, :1, :, :1])  # [B, 1, K, 1]
        mask[drop, :, 1:] = 0
        return sh_coeffs * mask

    def _forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decode the tokens into Gaussians and render the views."""
        means, scales, quats, opacities, sh_coeffs = self.decode(data["tokens"])
        sh_coeffs = self._drop_sh(sh_coeffs)

        scenes = Gaussians.from_parameters(
            means=means, scales=scales, quats=quats, opacities=opacities, sh_coeffs=sh_coeffs
        )

        return self._render_results(scenes, data) | {"gaussians": scenes}

    def _get_loss_context(
        self, pred: dict[str, Any], data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted and GT images, cameras, poses and depths for the loss."""
        gt_imgs = data["target"]["image"]
        pred_imgs = pred["target_rgb"]

        cameras: Camera = data["target"]["camera"]
        poses: Pose = data["target"]["pose"]
        depth = data["target"]["depth"]
        pred_depth = pred["target_depth"]

        if self.conf.loss_on_context:
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
        """PSNR, activated counts and Gaussian statistics, each (B,)."""
        context = self._get_loss_context(pred, data)
        pred_imgs, gt_imgs = context["pred_imgs"], context["gt_imgs"]

        b, n = gt_imgs.shape[:2]
        gt_imgs = rearrange(gt_imgs, "b n c h w -> (b n) c h w")
        pred_imgs = rearrange(pred_imgs, "b n c h w -> (b n) c h w")

        psnr = calculate_psnr(pred_imgs, gt_imgs)

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
        """Image terms plus the scale, opacity and location penalties, each (B,)."""
        context = self._get_loss_context(pred, data)
        losses = self._image_losses(context["pred_imgs"], context["gt_imgs"])
        penalties = self._penalties(pred)
        rendering_loss = (
            losses["total"]
            + self.conf.scale_penalty_weight * penalties["scale_penalty"]
            + self.conf.opacity_penalty_weight * penalties["opacity_penalty"]
            + penalties["location_penalty"]
        )
        return {
            "rendering_loss": rendering_loss,
            "mse_loss": losses["mse_loss"],
            "lpips_loss": losses["lpips_loss"],
            **penalties,
        }

    def _penalties(self, pred: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale, opacity and location penalties on the Gaussians, each (B,)."""
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

    def _image_losses(self, pred_imgs, gt_imgs) -> dict[str, torch.Tensor]:
        """Photometric (L1 or MSE) and LPIPS terms and their weighted sum, each (B,)."""
        b, n = gt_imgs.shape[:2]
        gt_flat = rearrange(gt_imgs, "b n c h w -> (b n) c h w")
        pred_flat = rearrange(pred_imgs, "b n c h w -> (b n) c h w")

        loss_fn = F.l1_loss if self.conf.use_l1_loss else F.mse_loss
        mse_loss = loss_fn(pred_flat, gt_flat, reduction="none")
        mse_loss = rearrange(mse_loss, "(b n) c h w -> b (n c h w)", n=n, b=b).mean(dim=-1)

        lpips_loss = mse_loss.new_zeros(b)
        if self.conf.lpips_weight > 0:
            lpips_loss = self.lpips({"rendering": pred_flat}, {"image": gt_flat})
            lpips_loss = rearrange(lpips_loss, "(b n) -> b n", n=n, b=b).mean(dim=-1)

        total = mse_loss * self.conf.mse_weight + lpips_loss * self.conf.lpips_weight
        return {"total": total, "mse_loss": mse_loss, "lpips_loss": lpips_loss}

    def _depth_loss(self, pred: dict[str, Any], data: dict[str, Any]) -> torch.Tensor:
        """L1 between the rendered and the GT depth over the valid GT pixels, (B,)."""
        context = self._get_loss_context(pred, data)
        gt_depth, pred_depth = context["depth"], context["pred_depth"]
        B = gt_depth.shape[0]
        valid = gt_depth >= 0  # [B, S, H, W]
        pixel_loss = F.l1_loss(pred_depth, gt_depth, reduction="none") * valid
        num_valid = valid.reshape(B, -1).sum(dim=-1).clamp(min=1)
        return pixel_loss.reshape(B, -1).sum(dim=-1) / num_valid

    def loss(self, pred: dict[str, Any], data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Rendering loss (+ depth loss when depth_weight > 0). Returns (losses, metrics)."""
        losses = self._rendering_loss(pred, data)
        losses["total"] = losses["rendering_loss"]
        has_depth = "depth" in data["context"] and "depth" in data.get("target", {})
        if self.conf.depth_weight > 0 and has_depth:
            losses["depth_loss"] = self._depth_loss(pred, data)
            losses["total"] = losses["total"] + self.conf.depth_weight * losses["depth_loss"]
        metrics = self.metrics(pred, data) | {"lpips": losses["lpips_loss"]}
        return losses, metrics
