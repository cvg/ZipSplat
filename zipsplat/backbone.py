"""Depth-Anything-V3 ViT backbone with multi-view alternating local/global attention.

Vendored and trimmed from:
  - DINOv2 ViT (© Meta Platforms, Apache-2.0)
  - DA3 multi-view extensions and camera-prior encoder
    (© 2025 ByteDance Ltd., Apache-2.0; Lin et al. "Depth Anything 3", 2025)

Includes the optional camera-prior encoder (`CameraEnc`) that injects pose + intrinsics as a token
at the `alt_start` layer, replacing the model's learned camera token.

Author: Alexander Veicht
"""

import math
from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from zipsplat import modules
from zipsplat.camera import Camera
from zipsplat.pose import Pose

# ---------------------------------------------------------------------------
# Camera prior encoder
# ---------------------------------------------------------------------------


class CameraEnc(nn.Module):
    """Encodes (Pose, Camera) into per-view tokens injected at the ViT's `alt_start` layer.

    Pose encoding is derived from world-to-cam (DA3 training convention):
    [t_w2c (3), quat_xyzw_w2c (4), vfov, hfov].
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            *[
                modules.SelfAttentionBlock(
                    dim=dim_out, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = modules.Mlp(
            in_features=dim_in, hidden_features=dim_out // 2, out_features=dim_out
        )

    def forward(self, pose: Pose, camera: Camera) -> Tensor:
        w2c = pose.inv()
        quat_xyzw = w2c.quat[..., [1, 2, 3, 0]]
        fov = torch.stack([camera.vfov, camera.hfov], dim=-1)
        encoding = torch.cat([w2c.t, quat_xyzw, fov], dim=-1).float()
        return self.trunk_norm(self.trunk(self.token_norm(self.pose_branch(encoding))))


# ---------------------------------------------------------------------------
# DA3 ViT
# ---------------------------------------------------------------------------


_FFN_LAYERS = {
    "mlp": modules.Mlp,
    "swiglufused": modules.SwiGLUFFNFused,
    "swiglu": modules.SwiGLUFFN,
}


class DAV3DinoVisionTransformer(nn.Module):
    """DA3 DINO Vision Transformer with multi-view alternating local/global attention."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        init_values: float = 1.0,
        ffn_layer: str = "mlp",
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        rope_freq: float = 100.0,
        out_layers: Sequence[int] = (5, 7, 9, 11),
        cat_token: bool = True,
        patch_start_idx: int = 1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.rope_freq = rope_freq
        self.cat_token = cat_token
        self.out_layers = list(out_layers)
        self.patch_start_idx = patch_start_idx
        self.num_tokens = 1

        self.patch_embed = modules.PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if alt_start != -1:
            self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if ffn_layer not in _FFN_LAYERS:
            raise ValueError(f"Unknown ffn_layer={ffn_layer!r}")
        ffn_class = _FFN_LAYERS[ffn_layer]

        if rope_start != -1:
            self.rope = modules.RotaryPositionEmbedding2D(frequency=rope_freq)
            self.position_getter = modules.PositionGetter()
        else:
            self.rope = None
            self.position_getter = None

        self.blocks = nn.ModuleList(
            [
                modules.SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    norm_layer=norm_layer,
                    ffn_layer=ffn_class,
                    init_values=init_values,
                    qk_norm=(i >= qknorm_start) if qknorm_start != -1 else False,
                    rope=self.rope if (i >= rope_start and rope_start != -1) else None,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            kwargs["scale_factor"] = (
                (w0 + self.interpolate_offset) / M,
                (h0 + self.interpolate_offset) / M,
            )
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x: Tensor) -> Tensor:
        B, S, _, w, h = x.shape
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B * S, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1
            )
        return rearrange(x, "(b s) n c -> b s n c", b=B, s=S)

    def _prepare_rope(
        self, B: int, S: int, H: int, W: int, device: torch.device
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Build RoPE positions for local (`pos`) and global (`pos_nodiff`) attention.

        For global attention, positions are constant within a view so RoPE acts as a
        view-id rather than a spatial encoding. Special (CLS) tokens get position 0.
        """
        if self.rope is None:
            return None, None
        h_p, w_p = H // self.patch_size, W // self.patch_size
        pos = self.position_getter(B * S, h_p, w_p, device=device)
        pos = rearrange(pos, "(b s) n c -> b s n c", b=B) + 1
        pos_nodiff = torch.ones_like(pos)
        if self.patch_start_idx > 0:
            zeros = torch.zeros(B, S, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
            pos = torch.cat([zeros, pos], dim=2)
            pos_nodiff = torch.cat([zeros, pos_nodiff], dim=2)
        return pos, pos_nodiff

    @staticmethod
    def _process_attention(
        x: Tensor, block: nn.Module, is_global: bool, pos: Optional[Tensor]
    ) -> Tensor:
        """Run a block under either per-view local or cross-view global attention."""
        b, s, n = x.shape[:3]
        if is_global:
            x = rearrange(x, "b s n c -> b (s n) c")
            pos = rearrange(pos, "b s n c -> b (s n) c") if pos is not None else None
            x = block(x, pos=pos)
            return rearrange(x, "b (s n) c -> b s n c", b=b, s=s, n=n)
        x = rearrange(x, "b s n c -> (b s) n c")
        pos = rearrange(pos, "b s n c -> (b s) n c") if pos is not None else None
        x = block(x, pos=pos)
        return rearrange(x, "(b s) n c -> b s n c", b=b, s=s)

    def _default_cam_token(self, B: int, S: int) -> Tensor:
        """Build the learned (ref + src) camera-token bank when no priors are provided."""
        ref = self.camera_token[:, :1].expand(B, -1, -1)
        src = self.camera_token[:, 1:].expand(B, S - 1, -1)
        return torch.cat([ref, src], dim=1)

    def _apply_norm(self, x: Tensor) -> Tensor:
        """Final LN. With `cat_token`, only the global half is normed (DA3 convention)."""
        if not self.cat_token:
            return self.norm(x)
        D = self.embed_dim
        return torch.cat([x[..., :D], self.norm(x[..., D:])], dim=-1)

    def forward(
        self, x: Tensor, cam_token: Optional[Tensor] = None
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Run blocks and return per-output-layer (full_features, camera_token).

        Args:
            x: (B, S, 3, H, W) input.
            cam_token: optional (B, S, embed_dim) tensor injected at `alt_start` to override
                the learned camera token (used by the prior path).

        Returns:
            Tuple of (output, camera_token) per layer in `self.out_layers`. `output` shape is
            (B, S, T, D) or (B, S, T, 2D) when `cat_token=True`.
        """
        B, S, _, H, W = x.shape
        x = self.prepare_tokens(x)
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
        outputs: List[Tuple[Tensor, Tensor]] = []
        local_x: Optional[Tensor] = None

        for i, blk in enumerate(self.blocks):
            if i == self.alt_start:
                x[:, :, 0] = cam_token if cam_token is not None else self._default_cam_token(B, S)

            is_global = self.alt_start != -1 and i >= self.alt_start and i % 2 == 1
            rope_active = self.rope is not None and i >= self.rope_start
            blk_pos = (pos_nodiff if is_global else pos) if rope_active else None
            x = self._process_attention(x, blk, is_global=is_global, pos=blk_pos)

            if not is_global:
                local_x = x

            if i in self.out_layers:
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                outputs.append((out_x[:, :, 0], out_x))

        strip = 1 + self.num_register_tokens
        return tuple((self._apply_norm(full)[..., strip:, :], cam_tok) for cam_tok, full in outputs)


# ---------------------------------------------------------------------------
# DA3 variant factories
# ---------------------------------------------------------------------------


def vit_small(out_layers: Sequence[int] = (5, 7, 9, 11), **kwargs) -> DAV3DinoVisionTransformer:
    return DAV3DinoVisionTransformer(
        embed_dim=384,
        depth=12,
        num_heads=6,
        out_layers=out_layers,
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        ffn_layer="mlp",
        **kwargs,
    )


def vit_base(out_layers: Sequence[int] = (5, 7, 9, 11), **kwargs) -> DAV3DinoVisionTransformer:
    return DAV3DinoVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_layers=out_layers,
        alt_start=4,
        qknorm_start=4,
        rope_start=4,
        ffn_layer="mlp",
        **kwargs,
    )


def vit_large(out_layers: Sequence[int] = (11, 15, 19, 23), **kwargs) -> DAV3DinoVisionTransformer:
    return DAV3DinoVisionTransformer(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        out_layers=out_layers,
        alt_start=8,
        qknorm_start=8,
        rope_start=8,
        ffn_layer="mlp",
        **kwargs,
    )


def vit_giant2(out_layers: Sequence[int] = (19, 27, 33, 39), **kwargs) -> DAV3DinoVisionTransformer:
    return DAV3DinoVisionTransformer(
        embed_dim=1536,
        depth=40,
        num_heads=24,
        out_layers=out_layers,
        alt_start=13,
        qknorm_start=13,
        rope_start=13,
        ffn_layer="swiglufused",
        **kwargs,
    )


_VARIANT_FACTORIES = {
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vitg": vit_giant2,
}


# ---------------------------------------------------------------------------
# DA3 encoder (backbone + optional camera prior path)
# ---------------------------------------------------------------------------


class DAV3Encoder(nn.Module):
    """DA3 backbone with an optional camera-prior token encoder.

    When `pose` and `intrinsics` are passed to forward and `with_camera_enc=True`, the
    `CameraEnc` encodes them into per-view tokens that override the model's learned
    camera token at `alt_start`.
    """

    def __init__(
        self,
        vit_name: str = "vitg",
        out_layers: Sequence[int] = (19, 29, 39),
        with_camera_enc: bool = True,
        **vit_kwargs,
    ) -> None:
        super().__init__()
        if vit_name not in _VARIANT_FACTORIES:
            raise ValueError(
                f"Unknown vit_name={vit_name!r}; choose from {list(_VARIANT_FACTORIES)}"
            )
        self.backbone = _VARIANT_FACTORIES[vit_name](out_layers=out_layers, **vit_kwargs)
        self.embed_dim = self.backbone.embed_dim
        self.cam_enc = CameraEnc(dim_out=self.embed_dim) if with_camera_enc else None

    def forward(
        self, image: Tensor, pose: Optional[Pose] = None, camera: Optional[Camera] = None
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Run the backbone, optionally injecting camera priors.

        Args:
            image: (B, S, 3, H, W) float input in [0, 1].
            pose: cam-to-world Pose with batch shape (B, S) for prior injection. Optional.
            camera: Camera with matching batch shape for prior injection. Optional.

        Returns:
            Tuple of (output, camera_token) per output layer.
        """
        cam_token = None
        if pose is not None and camera is not None and self.cam_enc is not None:
            cam_token = self.cam_enc(pose, camera)
        return self.backbone(image, cam_token=cam_token)
