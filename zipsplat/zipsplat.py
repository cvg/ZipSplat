"""ZipSplat model: feed-forward 3DGS from multi-view images.

Pipeline: backbone features → optional k-means clustering → cross/self-attention
over scene-tokens → color skip → Gaussian head.

Author: Alexander Veicht
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from zipsplat import modules
from zipsplat.backbone import DAV3Encoder
from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.pose import Pose
from zipsplat.utils import kmeans

logger = logging.getLogger(__name__)


def _inverse_log_transform(y: Tensor) -> Tensor:
    """sign(y) * (exp(|y|) - 1) — expand 'log-space' position predictions."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class _GaussianHead(nn.Module):
    """Mlp + Linear stack producing packed Gaussian parameters and decoding to a Gaussians scene.

    The inner `self.gaussian_head` Sequential matches the splatfactory checkpoint structure
    (`gaussian_head.gaussian_head.{0,1}.*`).
    """

    def __init__(self, head_dim: int, gaussians_per_token: int, sh_degree: int) -> None:
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.params_per_gaussian = 3 + 3 + 4 + 1 + self.num_sh_coeffs * 3
        self.gaussian_head = nn.Sequential(
            modules.Mlp(in_features=head_dim, hidden_features=2 * head_dim, out_features=head_dim),
            nn.Linear(head_dim, gaussians_per_token * self.params_per_gaussian),
        )

    def forward(self, tokens: Tensor) -> Gaussians:
        """Project tokens to packed Gaussian params, activate, return a Gaussians scene."""
        params = self.gaussian_head(tokens)  # (B, N, G * params_per_g)
        params = rearrange(
            params,
            "B N (G C) -> B (N G) C",
            G=self.gaussians_per_token,
            C=self.params_per_gaussian,
        )
        means = _inverse_log_transform(params[..., 0:3].clamp(-5.0, 5.0))
        scales = F.softplus(params[..., 3:6] - 4.0).clamp(1e-6, 15.0)
        quats = F.normalize(params[..., 6:10], dim=-1)
        opacities = params[..., 10].sigmoid()
        sh_coeffs = rearrange(params[..., 11:], "B N (K C) -> B N K C", K=self.num_sh_coeffs, C=3)
        return Gaussians.from_parameters(
            means=means, scales=scales, quats=quats, opacities=opacities, sh_coeffs=sh_coeffs
        )


class ZipSplat(nn.Module):
    """Feed-forward 3DGS model: images → Gaussians (DA3-Giant variant)."""

    # Defaults match the released DA3-Giant checkpoint.
    default_conf = {
        # backbone
        "vit_name": "vitg",
        "patch_size": 14,
        "out_layers": (19, 29, 39),
        # gaussian head
        "gaussians_per_token": 32,
        "sh_degree": 1,
        # color skip
        "color_skip_dim": 128,
        "color_mlp_ratio": 1.0,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}

        self.backbone = DAV3Encoder(
            vit_name=self.conf["vit_name"],
            out_layers=self.conf["out_layers"],
            with_camera_enc=True,
            patch_size=self.conf["patch_size"],
        )
        D = self.backbone.embed_dim
        L = len(self.conf["out_layers"])

        # Per-layer prep: LN(local) + LN(global) + Linear(2D→D) + skip residual
        self.pre_norm_local = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.pre_norm_global = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])
        self.downscale = nn.ModuleList([nn.Linear(2 * D, D) for _ in range(L)])

        # Geometry CA + SA stack (delta mode: CA returns gated delta only)
        self.cross_attention = nn.ModuleList(
            [
                modules.CrossAttentionBlock(
                    dim=D, num_heads=D // 64, mlp_ratio=4.0, qk_norm=True, init_values=0.01
                )
                for _ in range(L)
            ]
        )
        self.self_attention = nn.ModuleList(
            [
                modules.SelfAttentionBlock(
                    dim=D, num_heads=D // 64, mlp_ratio=4.0, qk_norm=True, init_values=0.01
                )
                for _ in range(L)
            ]
        )

        # Color skip: PatchEmbed → CA in color_skip_dim (4th symmetric layer)
        self.color_embed = modules.PatchEmbed(
            img_size=252,
            patch_size=self.conf["patch_size"],
            in_chans=3,
            embed_dim=self.conf["color_skip_dim"],
        )
        self.color_cross_attention = modules.CrossAttentionBlock(
            dim=self.conf["color_skip_dim"],
            num_heads=max(1, self.conf["color_skip_dim"] // 64),
            mlp_ratio=self.conf["color_mlp_ratio"],
            qk_norm=True,
            init_values=0.01,
        )

        # Gaussian head: tokens → packed params → activate → Gaussians
        self.gaussian_head = _GaussianHead(
            head_dim=D + self.conf["color_skip_dim"],
            gaussians_per_token=self.conf["gaussians_per_token"],
            sh_degree=self.conf["sh_degree"],
        )

    def flexible_load(self, state_dict: Dict[str, Any]) -> None:
        """Load a state dict tolerantly: strip DDP `module.` prefix, non-strict, warn on diffs."""
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected = [
            k for k in unexpected if "running" not in k and "num_batches_tracked" not in k
        ]
        if missing:
            logger.warning("missing %d keys (e.g. %s)", len(missing), list(missing)[:3])
        if unexpected:
            logger.warning("unexpected %d keys (e.g. %s)", len(unexpected), list(unexpected)[:3])

    # ------------------------- per-stage helpers -------------------------

    def _backbone_features(
        self, images: Tensor, cameras: Optional[Camera], poses: Optional[Pose]
    ) -> List[Tensor]:
        """Run backbone, reverse so features[0] = deepest layer, drop camera token."""
        feats = self.backbone(images, pose=poses, camera=cameras)
        return [t[0] for t in feats[::-1]]

    def _prepare(self, features: List[Tensor]) -> List[Tensor]:
        """Per-layer pre_norm + downscale + fuse_skip (always on)."""
        D = self.backbone.embed_dim
        out = []
        for l, raw in enumerate(features):  # raw: (B, V, T, 2D)
            local = self.pre_norm_local[l](raw[..., :D])
            global_ = self.pre_norm_global[l](raw[..., D:])
            tok = self.downscale[l](torch.cat([local, global_], dim=-1)) + (local + global_) / 2
            out.append(tok)
        return out

    def _cluster(self, tokens: Tensor, compression: float) -> Tensor:
        """K-means on the deepest layer; returns nearest-token indices [B, K].

        At compression=1.0 returns torch.arange(V*T) — k-means is skipped.
        """
        B, V, T, _ = tokens.shape
        VT = V * T
        K = max(1, int(VT * compression))
        if K >= VT:
            return torch.arange(VT, device=tokens.device).unsqueeze(0).expand(B, -1)
        flat = rearrange(tokens, "B V T D -> B (V T) D")
        with torch.no_grad(), torch.autocast(tokens.device.type, dtype=torch.bfloat16):
            _, nearest_idx, _ = kmeans(flat, K)
        return nearest_idx

    def _fuse(self, layer_tokens: List[Tensor], nearest_idx: Tensor) -> Tensor:
        """Iterative scene_tokens stream: gather + CA delta + SA per layer."""
        D = self.backbone.embed_dim
        idx = nearest_idx.unsqueeze(-1).expand(-1, -1, D)
        # Initialize scene from the deepest layer's gathered tokens
        scene = torch.gather(rearrange(layer_tokens[0], "B V T D -> B (V T) D"), 1, idx)
        for l, blk_ca in enumerate(self.cross_attention):
            keys = rearrange(layer_tokens[l], "B V T D -> B (V T) D")
            queries = torch.gather(keys, 1, idx)
            scene = scene + blk_ca(queries, keys, return_delta=True)
            scene = self.self_attention[l](scene)
        return scene

    def _color(self, images: Tensor, nearest_idx: Tensor) -> Tensor:
        """Color skip: PatchEmbed all views, gather at shared k-means indices, CA in color_skip_dim."""
        B = images.shape[0]
        flat_imgs = rearrange(images, "B V C H W -> (B V) C H W")
        color_tokens = rearrange(self.color_embed(flat_imgs), "(B V) T D -> B (V T) D", B=B)
        D_c = color_tokens.shape[-1]
        idx = nearest_idx.unsqueeze(-1).expand(-1, -1, D_c)
        return self.color_cross_attention(torch.gather(color_tokens, 1, idx), color_tokens)

    # ------------------------- public forward -------------------------

    def forward(
        self,
        images: Tensor,
        cameras: Optional[Camera] = None,
        poses: Optional[Pose] = None,
        use_priors: bool = False,
        compression: float = 1.0,
    ) -> Gaussians:
        """Predict Gaussians from multi-view images.

        Args:
            images: (B, V, 3, H, W) float in [0, 1].
            cameras: (B, V) Camera. Required when `use_priors=True`.
            poses: (B, V) Pose. Required when `use_priors=True`.
            use_priors: if True, inject pose+intrinsics tokens at the backbone's `alt_start`.
            compression: query-sampling ratio in (0, 1]. 1.0 disables k-means (all tokens used).

        Returns:
            Gaussians with batch shape (B, N_gaussians).
        """
        prior_pose, prior_camera = (poses, cameras) if use_priors else (None, None)
        features = self._backbone_features(images, prior_camera, prior_pose)
        layer_tokens = self._prepare(features)
        # K-means runs on the deepest (last) layer.
        nearest_idx = self._cluster(layer_tokens[0], compression)
        scene = self._fuse(layer_tokens, nearest_idx)
        color_feats = self._color(images, nearest_idx)
        return self.gaussian_head(torch.cat([scene, color_feats], dim=-1))
