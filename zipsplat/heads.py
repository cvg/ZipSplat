"""ZipSplat's decoders: the Gaussian head (scene tokens -> Gaussians) and DA3's camera head
(camera token -> pose and intrinsics).

CameraDec is a port of DA3's cam_dec (c 2025 ByteDance Ltd., Apache-2.0).

Author: Alexander Veicht
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from zipsplat.camera import Camera
from zipsplat.gaussians import Gaussians
from zipsplat.modules import Mlp
from zipsplat.pose import Pose, quaternion_to_matrix


def _inverse_log_transform(y: Tensor) -> Tensor:
    """sign(y) * (exp(|y|) - 1): expand the log-space position predictions."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class GaussianHead(nn.Module):
    """MLP decoder: each scene token -> a group of Gaussians with free 3D offsets.

    The inner `gaussian_head` Sequential matches the training checkpoint
    (`gaussian_head.gaussian_head.{0,1}.*`).
    """

    def __init__(self, embed_dim: int, gaussians_per_token: int, sh_degree: int = 1) -> None:
        super().__init__()
        self.gaussians_per_token = gaussians_per_token
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.params_per_gaussian = 3 + 3 + 4 + 1 + self.num_sh_coeffs * 3
        self.gaussian_head = nn.Sequential(
            Mlp(in_features=embed_dim, hidden_features=2 * embed_dim, out_features=embed_dim),
            nn.Linear(embed_dim, gaussians_per_token * self.params_per_gaussian),
        )

    def decode(self, tokens: Tensor, half_mlp: bool = False) -> Tuple[Tensor, ...]:
        """(B, K, D) tokens -> activated (means, scales, quats, opacities, sh_coeffs) of the
        B x (K * G) Gaussians.

        half_mlp runs the MLP body in fp16 (TokenBA); the readout and the activations always
        stay fp32 (the positions would quantize).
        """
        if half_mlp:
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = self.gaussian_head[0](tokens)
            params = self.gaussian_head[1](hidden.float())
        else:
            params = self.gaussian_head(tokens)
        params = rearrange(
            params, "B N (G C) -> B (N G) C", G=self.gaussians_per_token, C=self.params_per_gaussian
        )
        means = _inverse_log_transform(params[..., 0:3].clamp(-5.0, 5.0))
        scales = F.softplus(params[..., 3:6] - 4.0).clamp(1e-6, 15.0)
        quats = F.normalize(params[..., 6:10], dim=-1)
        opacities = params[..., 10:11].sigmoid()[..., 0]
        sh_coeffs = rearrange(params[..., 11:], "B N (K C) -> B N K C", K=self.num_sh_coeffs, C=3)
        return means, scales, quats, opacities, sh_coeffs

    def forward(self, tokens: Tensor, half_mlp: bool = False) -> Gaussians:
        """(B, K, D) tokens -> Gaussians with batch shape (B, K * G)."""
        means, scales, quats, opacities, sh_coeffs = self.decode(tokens, half_mlp)
        return Gaussians.from_parameters(
            means=means, scales=scales, quats=quats, opacities=opacities, sh_coeffs=sh_coeffs
        )


class CameraDec(nn.Module):
    """DA3's camera decoder: camera token -> pose encoding [t (3), quat_xyzw (4), fov_h, fov_w]."""

    def __init__(self, dim_in: int = 1536) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(dim_in, 3)
        self.fc_qvec = nn.Linear(dim_in, 4)
        self.fc_fov = nn.Sequential(nn.Linear(dim_in, 2), nn.ReLU())

    def forward(self, feat: Tensor) -> Tensor:
        """(B, V, dim_in) camera tokens -> (B, V, 9) c2w pose encoding."""
        B, V = feat.shape[:2]
        feat = self.backbone(feat.reshape(B * V, -1)).float()
        out_t = self.fc_t(feat).reshape(B, V, 3)
        out_qvec = self.fc_qvec(feat).reshape(B, V, 4)
        out_fov = self.fc_fov(feat).reshape(B, V, 2)
        return torch.cat([out_t, out_qvec, out_fov], dim=-1)


def decode_pose_encoding(pose_enc: Tensor, image_hw: Tuple[int, int]) -> Tuple[Pose, Camera]:
    """Pose encoding (..., 9) [t | quat_xyzw | fov_h, fov_w] -> (c2w Pose, Camera)."""
    t, quat, fov = pose_enc[..., :3], pose_enc[..., 3:7], pose_enc[..., 7:]
    pose = Pose.from_Rt(quaternion_to_matrix(F.normalize(quat, dim=-1)), t)
    H, W = image_hw
    fov = fov.clamp(min=0.02, max=3.12)  # the ReLU fov head can hit 0 -> inf focal
    return pose, Camera.from_fov(fov[..., 1], fov[..., 0], w=W, h=H)
